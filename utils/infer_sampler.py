import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt


def make_infer_sampler(
    hw: tuple[int, int],
    points_on_ray: int,
    z_range: tuple[float, float],
    sample_mode: str,
    device: torch.device,
):
    H, W = hw
    z_near, z_far = z_range

    z_steps = torch.linspace(0, 1 - 1 / points_on_ray, points_on_ray, device=device)
    z_samps = 1 / (1 / z_near * (1 - z_steps) + 1 / z_far * z_steps)

    uv_h = torch.stack(
        (
            torch.linspace(-1, 1, W, device=device).view(1, W).expand(H, W),
            torch.linspace(-1, 1, H, device=device).view(H, 1).expand(H, W),
            torch.ones((H, W), device=device),
        ),
        dim=-1,
    )

    def make_points(proj: Tensor):
        n, _, _ = proj.shape
        return (
            F.normalize(
                torch.matmul(
                    uv_h.view(1, H * W, 3).expand(n, -1, -1),
                    torch.inverse(proj).mT,
                ),
                dim=-1,
            ).view(n, H, W, 1, 3)
            * z_samps.view(points_on_ray, 1),
            z_samps.unsqueeze(0),
        )

    def sample(alpha: Tensor, xyz: Tensor, proj: Tensor):
        n, X, Y, Z, _ = xyz.shape
        assert alpha.shape == (n, H, W, points_on_ray)
        assert proj.shape == (n, 3, 3)
        xyz_projected = torch.matmul(
            xyz.view(n, X * Y * Z, 3),
            proj.mT,
        )
        disp = (
            1 / torch.norm(xyz.view(n, X * Y * Z, 3), dim=-1, keepdim=True) - 1 / z_near
        ) / (1 / z_far - 1 / z_near)
        # NOTE: bias on disp +1/2N because each point alpha calls for the occ of segment behind the point
        uvd = torch.cat(
            (
                xyz_projected[..., :2] / (xyz_projected[..., 2:] + 1e-10),
                (disp + 1 / (2 * points_on_ray)) * 2 - 1,
            ),
            dim=-1,
        ).view(n, X, Y, Z, 3)
        occ = F.grid_sample(
            alpha.view(n, 1, H, W, points_on_ray).permute(0, 1, 4, 2, 3),
            uvd.view(n, 1, 1, X * Y * Z, 3),
            sample_mode,
            "border",
            align_corners=True,
        ).view(n, X, Y, Z)
        return occ
    return make_points, sample


if __name__ == "__main__":
    hw = (3, 3)
    p = 64
    z_range = (3, 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj = torch.eye(3, device=device).view(1, 3, 3)
    make_points, infer_sampler = make_infer_sampler(
        hw,
        p,
        z_range,
        "nearest",
        device,
    )
    xyz_infer, z_samp = make_points(proj)
    alpha = torch.bernoulli(torch.full(xyz_infer.shape[:-1], 0.3, device=device))
    xyz_voxel = (
        torch.stack(
            torch.meshgrid(
                (
                    torch.arange(-40, 41, dtype=torch.float32, device=device),
                    torch.zeros([1], dtype=torch.float32, device=device),
                    torch.arange(1, 70, dtype=torch.float32, device=device),
                ),
                indexing="ij",
            ),
            dim=-1,
        )
    ).unsqueeze(0)
    occ_prob = infer_sampler(alpha, xyz_voxel, proj)

    plt.figure(figsize=(10, 9))
    plt.imshow(occ_prob[0, :, 0].cpu().numpy(), interpolation="nearest")
    u = xyz_infer[0, 1, ..., 0].view(-1).cpu().numpy() + 40
    v = xyz_infer[0, 1, ..., 2].view(-1).cpu().numpy()
    occ = alpha[0, 1].view(-1).cpu().numpy()
    plt.scatter(v[occ == 1], u[occ == 1], c="blue", marker='o', label="occ=1")
    plt.scatter(v[occ == 0], u[occ == 0], c="red", marker='*', label="occ=0")
    plt.legend()
    plt.grid(True)
    plt.show()
