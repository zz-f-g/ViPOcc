"""
NeRF differentiable renderer.
References:
https://github.com/bmild/nerf
https://github.com/kwea123/nerf_pl
"""
import torch
from dotmap import DotMap
from torch.nn import functional as F


class _RenderWrapper(torch.nn.Module):
    def __init__(self, net, renderer, simple_output):
        super().__init__()
        self.net = net
        self.renderer = renderer
        self.simple_output = simple_output

    def forward(self, rays, bboxes_3d=None, want_weights=False, want_alphas=False, want_z_samps=False, want_rgb_samps=False, depth_only=False):
        if rays.shape[0] == 0:
            return (
                torch.zeros(0, 3, device=rays.device),
                torch.zeros(0, device=rays.device),
            )

        outputs = self.renderer(
            self.net,
            rays,
            bboxes_3d,
            want_weights=want_weights and not self.simple_output,
            want_alphas=want_alphas and not self.simple_output,
            want_z_samps=want_z_samps and not self.simple_output,
            want_rgb_samps=want_rgb_samps and not self.simple_output,
            depth_only=depth_only,
        )
        if depth_only:
            return outputs
        if self.simple_output:
            if self.renderer.using_fine:
                rgb = outputs.fine.rgb
                depth = outputs.fine.depth
            else:
                rgb = outputs.coarse.rgb
                depth = outputs.coarse.depth
            return rgb, depth
        else:
            # Make DotMap to dict to support DataParallel
            return outputs.toDict()


class NeRFRenderer(torch.nn.Module):
    """
    NeRF differentiable renderer
    :param n_coarse number of coarse (binned uniform) samples
    :param n_fine number of fine (importance) samples
    :param n_fine_depth number of expected depth samples
    :param noise_std noise to add to sigma. We do not use it
    :param depth_std noise for depth samples
    :param eval_batch_size ray batch size for evaluation
    :param white_bkgd if true, background color is white; else black
    :param lindisp if to use samples linear in disparity instead of distance
    :param sched ray sampling schedule. list containing 3 lists of equal length.
    sched[0] is list of iteration numbers,
    sched[1] is list of coarse sample numbers,
    sched[2] is list of fine sample numbers
    """

    def __init__(
            self,
            n_coarse=128,
            n_fine=0,
            n_fine_depth=0,
            noise_std=0.0,
            depth_std=0.01,
            eval_batch_size=100000,
            white_bkgd=False,
            lindisp=False,
            sched=None,  # ray sampling schedule for coarse and fine rays
            hard_alpha_cap=False
    ):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth

        self.noise_std = noise_std
        self.depth_std = depth_std

        self.eval_batch_size = eval_batch_size
        self.white_bkgd = white_bkgd
        self.lindisp = lindisp
        self.using_fine = n_fine > 0
        self.sched = sched
        if sched is not None and len(sched) == 0:
            self.sched = None
        self.register_buffer(
            "iter_idx", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "last_sched", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.hard_alpha_cap = hard_alpha_cap

        # self.sampler = ray_samplers.LinearDisparitySampler(num_samples=n_coarse)

    def sample_coarse(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays: ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

        step = 1.0 / self.n_coarse  # 1/64
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # 0~1, shape=64
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, 64)
        z_steps += torch.rand_like(z_steps) * step
        if not self.lindisp:  # Use linear sampling in depth space
            samples = near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space √
            samples = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)
        return samples

    def composite(self, model, rays, z_samp, bboxes_3d, coarse=True, sb=0):
        """
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays: ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp: z positions sampled for each ray (B, K) // (65536, 64), values in 3~80
        :param coarse whether to evaluate using coarse NeRF
        :param sb: super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """
        B, K = z_samp.shape

        deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)
        deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)
        # deltas = z_samp.deltas.squeeze(-1)

        # (B, K, 3) // o + td
        points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
        # points = z_samp.frustums.get_positions()
        points = points.reshape(-1, 3)  # (B*K, 3)

        rgbs_all, invalid_all, sigmas_all = [], [], []
        if sb > 0:  # superbatch, 16
            points = points.reshape(sb, -1, 3)  # (SB, B'*K, 3) B' is real ray batch size
            eval_batch_size = (self.eval_batch_size - 1) // sb + 1
            eval_batch_dim = 1
        else:
            eval_batch_size = self.eval_batch_size
            eval_batch_dim = 0

        split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)

        for pnts in split_points:
            rgbs, invalid, sigmas = model(pnts, bboxes_3d, coarse=coarse)
            rgbs_all.append(rgbs)
            invalid_all.append(invalid)
            sigmas_all.append(sigmas)

        # (B*K, 4) OR (SB, B'*K, 4)
        rgbs = torch.cat(rgbs_all, dim=eval_batch_dim)
        invalid = torch.cat(invalid_all, dim=eval_batch_dim)
        sigmas = torch.cat(sigmas_all, dim=eval_batch_dim)

        rgbs = rgbs.reshape(B, K, -1)  # (B, K, 12)
        invalid = invalid.reshape(B, K, -1)
        sigmas = sigmas.reshape(B, K)
        sigmas_valid = torch.where(torch.all(invalid, dim=-1), 0.0, sigmas).view(-1)
        sigmas_sharpened = F.softmax(sigmas_valid) * sigmas_valid.sum().detach()
        sigmas_sharpened = sigmas_sharpened - sigmas_sharpened.min()
        loss_sigma = ((sigmas_valid - sigmas_sharpened)**2).mean()

        if self.training and self.noise_std > 0.0:
            sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

        alphas = 1 - torch.exp(-deltas.abs() * torch.relu(sigmas))  # (B, 64) (delta should be positive anyway)

        if self.hard_alpha_cap:
            alphas[:, -1] = 1

        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
        )  # (B, K+1) = [1, a1, a2, ...]
        T = torch.cumprod(alphas_shifted, -1)  # (B)
        weights = alphas * T[:, :-1]  # (B, K)
        # alphas = None

        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3*4)
        # z_samp = (z_samp.frustums.starts + z_samp.frustums.ends) / 2
        # z_samp = torch.squeeze(z_samp, dim=-1)
        depth_final = torch.sum(weights * z_samp, -1)  # (B)
        # renderers.DepthRenderer()

        if self.white_bkgd:
            # White background
            pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
            rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)
        return (
            weights,
            rgb_final,
            depth_final,
            alphas,
            invalid,
            z_samp,
            rgbs,
            loss_sigma,
        )
    def composite_depth(self, model, rays, z_samp, bboxes_3d, coarse=True, sb=0):
        """
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays: ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp: z positions sampled for each ray (B, K) // (65536, 64), values in 3~80
        :param coarse whether to evaluate using coarse NeRF
        :param sb: super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """
        B, K = z_samp.shape

        deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)
        deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

        # (B, K, 3) // o + td
        points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
        # points = z_samp.frustums.get_positions()
        points = points.reshape(-1, 3)  # (B*K, 3)

        sigmas_all = []
        if sb > 0:  # superbatch, 16
            points = points.reshape(sb, -1, 3)  # (SB, B'*K, 3) B' is real ray batch size
            eval_batch_size = (self.eval_batch_size - 1) // sb + 1
            eval_batch_dim = 1
        else:
            eval_batch_size = self.eval_batch_size
            eval_batch_dim = 0

        split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)

        for pnts in split_points:
            rgbs, invalid, sigmas = model(pnts, bboxes_3d, coarse=coarse)
            sigmas_all.append(sigmas)

        # (B*K, 4) OR (SB, B'*K, 4)
        sigmas = torch.cat(sigmas_all, dim=eval_batch_dim)

        sigmas = sigmas.reshape(B, K)
        alphas = 1 - torch.exp(-deltas.abs() * torch.relu(sigmas))  # (B, 64) (delta should be positive anyway)

        if self.hard_alpha_cap:
            alphas[:, -1] = 1

        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
        )  # (B, K+1) = [1, a1, a2, ...]
        T = torch.cumprod(alphas_shifted, -1)  # (B)
        weights = alphas * T[:, :-1]  # (B, K)

        depth_final = torch.sum(weights * z_samp, -1)  # (B)

        return depth_final

    def forward(
            self, model, rays, bboxes_3d, want_weights=False, want_alphas=False, want_z_samps=False, want_rgb_samps=False, depth_only=False
    ):
        """
        :model: nerf model, should return (SB, B, (r, g, b, sigma))
        when called with (SB, B, (x, y, z)), for multi-object:
        SB = 'super-batch' = size of object batch,
        B  = size of per-object ray batch.
        Should also support 'coarse' boolean argument for coarse NeRF.
        :param rays: ray spec [origins (3), directions (3), near (1), far (1)] (SB, B, 8)
        :param want_weights if true, returns compositing weights (SB, B, K)
        :return render dict
        """
        if self.sched is not None and self.last_sched.item() > 0:
            self.n_coarse = self.sched[1][self.last_sched.item() - 1]
            self.n_fine = self.sched[2][self.last_sched.item() - 1]

        assert len(rays.shape) == 3
        superbatch_size = rays.shape[0]  # 16
        rays = rays.reshape(-1, 8)  # [16, 4096, 8] --> [16*4096=65536, 8]

        z_coarse = self.sample_coarse(rays)  # [65536, 8] --> [65536, 64]
        if depth_only:
            depth = self.composite_depth(model, rays, z_coarse, bboxes_3d, coarse=True, sb=superbatch_size)
            depth = depth.reshape(superbatch_size, -1)
            return depth
        else:
            coarse_composite = self.composite(model, rays, z_coarse, bboxes_3d, coarse=True, sb=superbatch_size)

        outputs = DotMap(
            coarse=self._format_outputs(
                coarse_composite[:-1], superbatch_size, want_weights=want_weights, want_alphas=want_alphas,
                want_z_samps=want_z_samps, want_rgb_samps=want_rgb_samps
            ),
        )
        outputs.coarse.loss_sigma = coarse_composite[-1]

        return outputs

    def _format_outputs(
            self, rendered_outputs, superbatch_size, want_weights=False, want_alphas=False, want_z_samps=False,
            want_rgb_samps=False
    ):
        weights, rgb_final, depth, alphas, invalid, z_samps, rgb_samps = rendered_outputs
        n_smps = weights.shape[-1]
        out_d_rgb = rgb_final.shape[-1]
        out_d_i = invalid.shape[-1]
        if superbatch_size > 0:
            rgb_final = rgb_final.reshape(superbatch_size, -1, out_d_rgb)
            depth = depth.reshape(superbatch_size, -1)
            weights = weights.reshape(superbatch_size, -1, n_smps)
            alphas = alphas.reshape(superbatch_size, -1, n_smps)
            invalid = invalid.reshape(superbatch_size, -1, n_smps, out_d_i)
            z_samps = z_samps.reshape(superbatch_size, -1, n_smps)
            rgb_samps = rgb_samps.reshape(superbatch_size, -1, n_smps, out_d_rgb)
        ret_dict = DotMap(rgb=rgb_final, depth=depth, invalid=invalid)
        if want_weights:
            ret_dict.weights = weights
        if want_alphas:
            ret_dict.alphas = alphas
        if want_z_samps:
            ret_dict.z_samps = z_samps
        if want_rgb_samps:
            ret_dict.rgb_samps = rgb_samps
        return ret_dict

    def sched_step(self, steps=1):
        """
        Called each training iteration to update sample numbers
        according to schedule
        """
        if self.sched is None:
            return
        self.iter_idx += steps
        while (
                self.last_sched.item() < len(self.sched[0])
                and self.iter_idx.item() >= self.sched[0][self.last_sched.item()]
        ):
            self.n_coarse = self.sched[1][self.last_sched.item()]
            self.n_fine = self.sched[2][self.last_sched.item()]
            print(
                "INFO: NeRF sampling resolution changed on schedule ==> c",
                self.n_coarse,
                "f",
                self.n_fine,
            )
            self.last_sched += 1

    @classmethod
    def from_conf(cls, conf, white_bkgd=False, eval_batch_size=100000):
        return cls(
            conf.get("n_coarse", 128),
            conf.get("n_fine", 0),
            n_fine_depth=conf.get("n_fine_depth", 0),
            noise_std=conf.get("noise_std", 0.0),
            depth_std=conf.get("depth_std", 0.01),
            white_bkgd=conf.get("white_bkgd", white_bkgd),
            lindisp=conf.get("lindisp", True),
            eval_batch_size=conf.get("eval_batch_size", eval_batch_size),
            sched=conf.get("sched", None),
            hard_alpha_cap=conf.get("hard_alpha_cap", False)
        )

    def bind_parallel(self, net, gpus=None, simple_output=False):
        """
        Returns a wrapper module compatible with DataParallel.
        Specifically, it renders rays with this renderer
        but always using the given network instance.
        Specify a list of GPU ids in 'gpus' to apply DataParallel automatically.
        :param net A PixelNeRF network
        :param gpus list of GPU ids to parallize to. If length is 1,
        does not parallelize
        :param simple_output only returns rendered (rgb, depth) instead of the 
        full render output map. Saves data transfer cost.
        :return torch module
        """
        wrapped = _RenderWrapper(net, self, simple_output=simple_output)
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            wrapped = torch.nn.DataParallel(wrapped, gpus, dim=1)
        return wrapped
