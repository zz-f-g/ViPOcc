import os
import time
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from datasets.kitti_360.sscbench import load_sscbench
from datasets.kitti_360.voxel import read_calib
from datasets.kitti_360.annotation import KITTI360Bbox3D
from datasets.kitti_360.labels import id2label
from datasets.kitti_360.process_bbox3d import convert_vertices
from utils.augmentation import get_color_aug_fn
import random


class FisheyeToPinholeSampler:
    def __init__(self, K_target, target_image_size, calibs, rotation=None):
        self._compute_transform(K_target, target_image_size, calibs, rotation)

    def _compute_transform(self, K_target, target_image_size, calibs, rotation=None):
        x = torch.linspace(-1, 1, target_image_size[1]).view(1, -1).expand(target_image_size)
        y = torch.linspace(-1, 1, target_image_size[0]).view(-1, 1).expand(target_image_size)
        z = torch.ones_like(x)
        xyz = torch.stack((x, y, z), dim=-1).view(-1, 3)

        # Unproject
        xyz = (torch.inverse(torch.tensor(K_target)) @ xyz.T).T

        if rotation is not None:
            xyz = (torch.tensor(rotation) @ xyz.T).T

        # Backproject into fisheye
        xyz = xyz / torch.norm(xyz, dim=-1, keepdim=True)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        xi_src = calibs["mirror_parameters"]["xi"]
        x = x / (z + xi_src)
        y = y / (z + xi_src)

        k1 = calibs["distortion_parameters"]["k1"]
        k2 = calibs["distortion_parameters"]["k2"]

        r = x * x + y * y
        factor = (1 + k1 * r + k2 * r * r)
        x = x * factor
        y = y * factor

        gamma0 = calibs["projection_parameters"]["gamma1"]
        gamma1 = calibs["projection_parameters"]["gamma2"]
        u0 = calibs["projection_parameters"]["u0"]
        v0 = calibs["projection_parameters"]["v0"]

        x = x * gamma0 + u0
        y = y * gamma1 + v0

        xy = torch.stack((x, y), dim=-1).view(1, *target_image_size, 2)
        self.sample_pts = xy

    def resample(self, img):
        img = img.unsqueeze(0)
        resampled_img = F.grid_sample(img, self.sample_pts, align_corners=True).squeeze(0)
        return resampled_img


class Kitti360Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        pose_path: str,
        split_path: Optional[str],
        target_image_size=(192, 640),
        return_stereo=False,
        return_fisheye=True,
        return_segmentation=False,
        return_voxel=False,
        return_gt_depth=False,
        return_pseudo_depth=False,
        return_samples=False,
        return_3d_bboxes=False,
        bboxes_semantic_labels=["car"],
        bbox_all_verts_in_frustum=True,
        frame_count=2,
        keyframe_offset=0,
        dilation=1,
        fisheye_rotation=0,
        fisheye_offset=0,
        color_aug=False,
        is_preprocessed=False,
        variant_fisheye_offset=False,
        low_fisheye_offset=None,
        up_fisheye_offset=None,
    ):
        self.data_path = data_path
        self.pose_path = pose_path
        self.split_path = split_path
        self.target_image_size = target_image_size

        self.return_stereo = return_stereo
        self.return_fisheye = return_fisheye
        self.return_segmentation = return_segmentation
        self.return_voxel = return_voxel
        self.return_gt_depth = return_gt_depth

        self.return_pseudo_depth = return_pseudo_depth
        self.return_samples = return_samples

        self.return_3d_bboxes = return_3d_bboxes
        self.bboxes_semantic_labels = bboxes_semantic_labels
        self.bbox_all_verts_in_frustum = bbox_all_verts_in_frustum

        self.frame_count = frame_count
        self.keyframe_offset = keyframe_offset
        self.dilation = dilation
        self.fisheye_rotation = fisheye_rotation
        self.fisheye_offset = fisheye_offset
        self.color_aug = color_aug
        self.is_preprocessed = is_preprocessed
        # if true, sample fisheye cameras within the lower and uppper bound
        self.variant_fisheye_offset = variant_fisheye_offset
        self.low_fisheye_offset = low_fisheye_offset
        self.up_fisheye_offset = up_fisheye_offset

        if isinstance(self.fisheye_rotation, float) or isinstance(self.fisheye_rotation, int):
            self.fisheye_rotation = (0, self.fisheye_rotation)
        self.fisheye_rotation = tuple(self.fisheye_rotation)

        self._sequences = self._get_sequences(self.data_path)

        self._calibs = self._load_calibs(self.data_path, self.fisheye_rotation)
        self._resampler_02, self._resampler_03 = self._get_resamplers(self._calibs, self._calibs["K_fisheye"],
                                                                      self.target_image_size)
        self._img_ids, self._poses = self._load_poses(self.pose_path, self._sequences)
        self._left_offset = ((self.frame_count - 1) // 2 + self.keyframe_offset) * self.dilation

        if not self.is_preprocessed:
            self._perspective_folder = "data_rect"
            self._fisheye_folder = "data_rgb"
        else:
            self._perspective_folder = f"data_{self.target_image_size[0]}x{self.target_image_size[1]}"
            self._fisheye_folder = (f"data_{self.target_image_size[0]}x{self.target_image_size[1]}_"
                                    f"{self.fisheye_rotation[0]}x{self.fisheye_rotation[1]}")

        if self.split_path is not None:
            self._datapoints = self._load_split(self.split_path, self._img_ids)
        elif self.return_segmentation:
            self._datapoints = self._semantics_split(self._sequences, self.data_path, self._img_ids)
        else:
            self._datapoints = self._full_split(self._sequences, self._img_ids, self.check_file_integrity)

        if self.return_3d_bboxes:
            self._3d_bboxes = self._load_3d_bboxes(Path(data_path) / "data_3d_bboxes" / "train", self._sequences)

        if self.return_segmentation:
            # Segmentations are only provided for the left camera
            self._datapoints = [dp for dp in self._datapoints if not dp[2]]

        if self.return_voxel:
            self.ssc_root_path = Path(self.data_path) / ".." / "sscbench-kitti360"
            assert self.ssc_root_path.exists()
            self.img_ids_voxel, self.img_ids_test, self.imgid2sscid, self.sscid2imgid = load_sscbench(
                self.ssc_root_path / "kitti_360_match.txt",
                self._datapoints,
                self._img_ids,
            )

        self._skip = 0
        self.length = len(self._datapoints)

    def check_file_integrity(self, seq, id):
        dp = Path(self.data_path)
        image_00 = dp / "data_2d_raw" / seq / "image_00" / self._perspective_folder
        image_01 = dp / "data_2d_raw" / seq / "image_01" / self._perspective_folder
        image_02 = dp / "data_2d_raw" / seq / "image_02" / self._fisheye_folder
        image_03 = dp / "data_2d_raw" / seq / "image_03" / self._fisheye_folder

        seq_len = self._img_ids[seq].shape[0]

        ids = [id] + [max(min(i, seq_len - 1), 0) for i in
                      range(id - self._left_offset, id - self._left_offset + self.frame_count * self.dilation,
                            self.dilation) if i != id]
        ids_fish = [max(min(id + self.fisheye_offset, seq_len - 1), 0)] + [max(min(i, seq_len - 1), 0) for i in range(
            id + self.fisheye_offset - self._left_offset,
            id + self.fisheye_offset - self._left_offset + self.frame_count * self.dilation, self.dilation) if
                                                                           i != id + self.fisheye_offset]

        img_ids = [self.get_img_id_from_id(seq, id) for id in ids]
        img_ids_fish = [self.get_img_id_from_id(seq, id) for id in ids_fish]

        for img_id in img_ids:
            if not ((image_00 / f"{img_id:010d}.png").exists() and (image_01 / f"{img_id:010d}.png").exists()):
                return False
        if self.return_fisheye:
            for img_id in img_ids_fish:
                if not ((image_02 / f"{img_id:010d}.png").exists() and (image_03 / f"{img_id:010d}.png").exists()):
                    return False
        return True

    @staticmethod
    def _get_sequences(data_path):
        all_sequences = []

        seqs_path = Path(data_path) / "data_2d_raw"
        for seq in seqs_path.iterdir():
            if not seq.is_dir():
                continue
            all_sequences.append(seq.name)

        return all_sequences

    @staticmethod
    def _full_split(sequences, img_ids, check_integrity):
        datapoints = []
        for seq in sorted(sequences):
            ids = [id for id in range(len(img_ids[seq])) if check_integrity(seq, id)]
            datapoints_seq = [(seq, id, False) for id in ids] + [(seq, id, True) for id in ids]
            datapoints.extend(datapoints_seq)
        return datapoints

    @staticmethod
    def _semantics_split(sequences, data_path, img_ids):
        datapoints = []
        for seq in sorted(sequences):
            datapoints_seq = [(seq, id, False) for id in range(len(img_ids[seq]))]
            datapoints_seq = [dp for dp in datapoints_seq if os.path.exists(
                os.path.join(data_path, "data_2d_semantics", "train", seq, "image_00", "semantic_rgb",
                             f"{img_ids[seq][dp[1]]:010d}.png"))]
            datapoints.extend(datapoints_seq)
        return datapoints

    @staticmethod
    def _load_split(split_path, img_ids):
        """
        Create datapoints.
        self._datapoints = [
        (sequence, id, True/False),
        ...
        ]
        False for left image. True for right image.
        """
        img_id2id = {seq: {id: i for i, id in enumerate(ids)} for seq, ids in img_ids.items()}

        with open(split_path, "r") as f:
            lines = f.readlines()

        def split_line(l):
            segments = l.split(" ")
            seq = segments[0]
            id = img_id2id[seq][int(segments[1])]
            return seq, id, segments[2][0] == "r"

        return list(map(split_line, lines))

    @staticmethod
    def _load_calibs(data_path, fisheye_rotation=0):
        data_path = Path(data_path)

        calib_folder = data_path / "calibration"
        cam_to_pose_file = calib_folder / "calib_cam_to_pose.txt"
        cam_to_velo_file = calib_folder / "calib_cam_to_velo.txt"
        intrinsics_file = calib_folder / "perspective.txt"
        fisheye_02_file = calib_folder / "image_02.yaml"
        fisheye_03_file = calib_folder / "image_03.yaml"

        cam_to_pose_data = {}
        with open(cam_to_pose_file, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    cam_to_pose_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                except ValueError:
                    pass

        cam_to_velo_data = None
        with open(cam_to_velo_file, 'r') as f:
            line = f.readline()
            try:
                cam_to_velo_data = np.array([float(x) for x in line.split()], dtype=np.float32)
            except ValueError:
                pass

        intrinsics_data = {}
        with open(intrinsics_file, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    intrinsics_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                except ValueError:
                    pass

        with open(fisheye_02_file, 'r') as f:
            f.readline()  # Skips first line that defines the YAML version
            fisheye_02_data = yaml.safe_load(f)

        with open(fisheye_03_file, 'r') as f:
            f.readline()  # Skips first line that defines the YAML version
            fisheye_03_data = yaml.safe_load(f)

        im_size_rect = (int(intrinsics_data["S_rect_00"][1]), int(intrinsics_data["S_rect_00"][0]))  # 1408, 376
        im_size_fish = (fisheye_02_data["image_height"], fisheye_02_data["image_width"])

        # Projection matrices
        # We use these projection matrices also when resampling the fisheye cameras.
        # This makes downstream processing easier, but it could be done differently.
        P_rect_00 = np.reshape(intrinsics_data['P_rect_00'], (3, 4))
        P_rect_01 = np.reshape(intrinsics_data['P_rect_01'], (3, 4))

        # Rotation matrices from raw to rectified -> Needs to be inverted later
        R_rect_00 = np.eye(4, dtype=np.float32)
        R_rect_01 = np.eye(4, dtype=np.float32)
        R_rect_00[:3, :3] = np.reshape(intrinsics_data['R_rect_00'], (3, 3))
        R_rect_01[:3, :3] = np.reshape(intrinsics_data['R_rect_01'], (3, 3))

        # Rotation matrices from resampled fisheye to raw fisheye
        fisheye_rotation = np.array(fisheye_rotation).reshape((1, 2))
        R_02 = np.eye(4, dtype=np.float32)
        R_03 = np.eye(4, dtype=np.float32)
        R_02[:3, :3] = Rotation.from_euler("xy", fisheye_rotation[:, [1, 0]], degrees=True).as_matrix().astype(
            np.float32)
        R_03[:3, :3] = Rotation.from_euler("xy", fisheye_rotation[:, [1, 0]] * np.array([[1, -1]]),
                                           degrees=True).as_matrix().astype(np.float32)

        # Load cam to pose transforms
        T_00_to_pose = np.eye(4, dtype=np.float32)
        T_01_to_pose = np.eye(4, dtype=np.float32)
        T_02_to_pose = np.eye(4, dtype=np.float32)
        T_03_to_pose = np.eye(4, dtype=np.float32)
        T_00_to_velo = np.eye(4, dtype=np.float32)

        T_00_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_00"], (3, 4))
        T_01_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_01"], (3, 4))
        T_02_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_02"], (3, 4))
        T_03_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_03"], (3, 4))
        T_00_to_velo[:3, :] = np.reshape(cam_to_velo_data, (3, 4))

        # Compute cam to pose transforms for rectified perspective cameras
        T_rect_00_to_pose = T_00_to_pose @ np.linalg.inv(R_rect_00)
        T_rect_01_to_pose = T_01_to_pose @ np.linalg.inv(R_rect_01)

        # Compute cam to pose transform for fisheye cameras
        T_02_to_pose = T_02_to_pose @ R_02
        T_03_to_pose = T_03_to_pose @ R_03

        # Compute velo to cameras and velo to pose transforms
        T_velo_to_rect_00 = R_rect_00 @ np.linalg.inv(T_00_to_velo)
        T_velo_to_pose = T_rect_00_to_pose @ T_velo_to_rect_00
        T_velo_to_rect_01 = np.linalg.inv(T_rect_01_to_pose) @ T_velo_to_pose

        # Calibration matrix is the same for both perspective cameras
        K = P_rect_00[:3, :3]

        # Normalize calibration
        # f_x =
        # f_y =
        # c_x =
        # c_y =

        # Change to image coordinates [-1, 1]
        K[0, 0] = (K[0, 0] / im_size_rect[1]) * 2.
        K[1, 1] = (K[1, 1] / im_size_rect[0]) * 2.
        K[0, 2] = (K[0, 2] / im_size_rect[1]) * 2. - 1
        K[1, 2] = (K[1, 2] / im_size_rect[0]) * 2. - 1

        # Convert fisheye calibration to [-1, 1] image dimensions
        fisheye_02_data["projection_parameters"]["gamma1"] = (fisheye_02_data["projection_parameters"]["gamma1"] /
                                                              im_size_fish[1]) * 2.
        fisheye_02_data["projection_parameters"]["gamma2"] = (fisheye_02_data["projection_parameters"]["gamma2"] /
                                                              im_size_fish[0]) * 2.
        fisheye_02_data["projection_parameters"]["u0"] = (fisheye_02_data["projection_parameters"]["u0"] / im_size_fish[
            1]) * 2. - 1.
        fisheye_02_data["projection_parameters"]["v0"] = (fisheye_02_data["projection_parameters"]["v0"] / im_size_fish[
            0]) * 2. - 1.

        fisheye_03_data["projection_parameters"]["gamma1"] = (fisheye_03_data["projection_parameters"]["gamma1"] /
                                                              im_size_fish[1]) * 2.
        fisheye_03_data["projection_parameters"]["gamma2"] = (fisheye_03_data["projection_parameters"]["gamma2"] /
                                                              im_size_fish[0]) * 2.
        fisheye_03_data["projection_parameters"]["u0"] = (fisheye_03_data["projection_parameters"]["u0"] / im_size_fish[
            1]) * 2. - 1.
        fisheye_03_data["projection_parameters"]["v0"] = (fisheye_03_data["projection_parameters"]["v0"] / im_size_fish[
            0]) * 2. - 1.

        # Use same camera calibration as perspective cameras for resampling
        # K_fisheye = np.eye(3, dtype=np.float32)
        # K_fisheye[0, 0] = 2
        # K_fisheye[1, 1] = 2

        K_fisheye = K

        calibs = {
            "K_perspective": K,
            "K_fisheye": K_fisheye,
            "T_cam_to_pose": {
                "00": T_rect_00_to_pose,
                "01": T_rect_01_to_pose,
                "02": T_02_to_pose,
                "03": T_03_to_pose,
            },
            "T_velo_to_cam": {
                "00": T_velo_to_rect_00,
                "01": T_velo_to_rect_01,
            },
            "T_velo_to_pose": T_velo_to_pose,
            "fisheye": {
                "calib_02": fisheye_02_data,
                "calib_03": fisheye_03_data,
                "R_02": R_02[:3, :3],
                "R_03": R_03[:3, :3]
            },
            "im_size": im_size_rect
        }

        return calibs

    @staticmethod
    def _get_resamplers(calibs, K_target, target_image_size):
        resampler_02 = FisheyeToPinholeSampler(K_target, target_image_size, calibs["fisheye"]["calib_02"],
                                               calibs["fisheye"]["R_02"])
        resampler_03 = FisheyeToPinholeSampler(K_target, target_image_size, calibs["fisheye"]["calib_03"],
                                               calibs["fisheye"]["R_03"])

        return resampler_02, resampler_03

    @staticmethod
    def _load_poses(pose_path, sequences):
        ids = {}
        poses = {}

        for seq in sequences:
            pose_file = Path(pose_path) / seq / f"poses.txt"

            try:
                pose_data = np.loadtxt(pose_file)
            except FileNotFoundError:
                print(f'Ground truth poses are not avaialble for sequence {seq}.')

            ids_seq = pose_data[:, 0].astype(int)
            poses_seq = pose_data[:, 1:].astype(np.float32).reshape((-1, 3, 4))
            poses_seq = np.concatenate((poses_seq, np.zeros_like(poses_seq[:, :1, :])), axis=1)
            poses_seq[:, 3, 3] = 1

            ids[seq] = ids_seq
            poses[seq] = poses_seq
        return ids, poses

    @staticmethod
    def _load_3d_bboxes(bbox_path, sequences):
        bboxes = {}
        for seq in sequences:
            with open(Path(bbox_path) / f"{seq}.xml", "rb") as f:
                tree = ET.parse(f)
            root = tree.getroot()
            objects = defaultdict(list)
            num_bbox = 0
            for child in root:
                if child.find('transform') is None:
                    continue
                obj = KITTI360Bbox3D()
                if child.find("semanticId") is not None:
                    obj.parseBbox(child)
                else:
                    obj.parseStuff(child)
                objects[obj.timestamp].append(obj)
                num_bbox +=1
            bboxes[seq] = objects
        return bboxes

    def get_img_id_from_id(self, sequence, id):
        return self._img_ids[sequence][id]

    def load_images(self, seq, img_ids, load_left, load_right, img_ids_fish=None):
        """
        Load images from a sequence.
        :param seq: str, Sequence name.
        :param load_left: bool, load left image or not.
        :param load_right: bool, load right image or not.
        :param img_ids_fish: list, list of fisheye image ids.

        Return 4 images lists (imgs_p_left, imgs_f_left, imgs_p_right, imgs_f_right)
        """
        imgs_p_left = []
        imgs_f_left = []
        imgs_p_right = []
        imgs_f_right = []

        if img_ids_fish is None:
            img_ids_fish = img_ids

        for id in img_ids:
            if load_left:
                img_perspective = cv2.cvtColor(cv2.imread(
                    os.path.join(self.data_path, "data_2d_raw", seq, "image_00", self._perspective_folder,
                                 f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                imgs_p_left += [img_perspective]

            if load_right:
                img_perspective = cv2.cvtColor(cv2.imread(
                    os.path.join(self.data_path, "data_2d_raw", seq, "image_01", self._perspective_folder,
                                 f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                imgs_p_right += [img_perspective]

        for id in img_ids_fish:
            if load_left:
                img_fisheye = cv2.cvtColor(cv2.imread(
                    os.path.join(self.data_path, "data_2d_raw", seq, "image_02", self._fisheye_folder,
                                 f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                imgs_f_left += [img_fisheye]
            if load_right:
                img_fisheye = cv2.cvtColor(cv2.imread(
                    os.path.join(self.data_path, "data_2d_raw", seq, "image_03", self._fisheye_folder,
                                 f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                imgs_f_right += [img_fisheye]

        return imgs_p_left, imgs_f_left, imgs_p_right, imgs_f_right

    def process_img(self, img: np.array, color_aug_fn=None, resampler: FisheyeToPinholeSampler = None):
        if resampler is not None and not self.is_preprocessed:
            img = torch.tensor(img).permute(2, 0, 1)
            img = resampler.resample(img)
        else:
            if self.target_image_size:
                img = cv2.resize(img, (self.target_image_size[1], self.target_image_size[0]),
                                 interpolation=cv2.INTER_LINEAR)
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img)
        if color_aug_fn is not None:
            img = color_aug_fn(img)

        img = img * 2 - 1

        return img  # [-1,1]

    def get_3d_bboxes(self, seq, img_id, pose, projs):
        seq_3d_bboxes = self._3d_bboxes[seq]
        pose_w2c = np.linalg.inv(pose)

        def filter_bbox(bbox):
            verts = bbox.vertices
            verts = (projs @ (pose_w2c[:3, :3] @ verts.T + pose_w2c[:3, 3, None])).T
            verts[:, :2] /= verts[:, 2:3]
            valid = ((verts[:, 0] >= -1) & (verts[:, 0] <= 1)) & ((verts[:, 1] >= -1) & (verts[:, 1] <= 1)) & ((verts[:, 2] > 3) & (verts[:, 2] <= 80))
            valid = np.all(valid, axis=-1) if self.bbox_all_verts_in_frustum else np.any(valid, axis=-1)
            return valid

        transform_w2c = lambda verts: (
            pose_w2c[:3, :3] @ verts.T + pose_w2c[:3, 3, None]
        ).T
        vertices = []
        semanticId = []
        vertices_world = []
        for bbox in seq_3d_bboxes[-1] + seq_3d_bboxes[img_id]:
            if not filter_bbox(bbox):
                continue
            if self.bboxes_semantic_labels:
                if id2label[bbox.semanticId].name not in self.bboxes_semantic_labels:
                    continue
            vertices.append(transform_w2c(bbox.vertices))
            vertices_world.append(bbox.vertices)
            semanticId.append(bbox.semanticId)
        # vertices, semanticId = zip(
        #     *map(
        #         lambda bbox: (transform_w2c(bbox.vertices), bbox.semanticId),
        #         filter(
        #             (
        #                 (
        #                     lambda bbox: id2label[bbox.semanticId].name
        #                     in self.bboxes_semantic_labels
        #                 )
        #                 if self.bboxes_semantic_labels
        #                 else (lambda _: True)
        #             ),
        #             filter(filter_bbox, ),
        #         ),
        #     )
        # )

        if vertices:
            return {
                "vertices": torch.tensor(np.stack(list(vertices), axis=0), dtype=torch.float32),
                "vertices_world": torch.tensor(np.stack(vertices_world, axis=0), dtype=torch.float32),
                "semanticId": torch.tensor(np.array(list(semanticId)), dtype=torch.float32),
            }
        else:
            return {}

    def load_segmentation(self, seq, img_id):
        seg = cv2.imread(os.path.join(self.data_path, "data_2d_semantics", "train", seq, "image_00", "semantic",
                                      f"{img_id:010d}.png"), cv2.IMREAD_UNCHANGED)
        seg = cv2.resize(seg, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_NEAREST)
        return seg

    def load_pseudo_depth(self, seq, img_id, is_right):
        cam_dir = "image_01" if is_right else "image_00"
        pseudo_depth_path = os.path.join(self.data_path, "depth", seq, cam_dir,
                                         self._perspective_folder, f"{img_id:010d}.npy")
        depth = np.load(pseudo_depth_path).astype(np.float32)  # [192, 640]
        # depth = cv2.imread(pseudo_depth_path, -1).astype(np.float32)  # [192, 640]
        # depth = depth / 255.0 * 80
        return depth[None, :, :]

    def load_gt_depth(self, seq, img_id):
        gt_depth_path = os.path.join(self.data_path, "depth_gt", seq, f"{img_id:010d}.npy")
        gt_depth = np.load(gt_depth_path).astype(np.float32)
        return gt_depth[None, :, :]

    def load_depth(self, seq, img_id, is_right):    # load depth from data_3d_raw
        points = np.fromfile(
            os.path.join(self.data_path, "data_3d_raw", seq, "velodyne_points", "data", f"{img_id:010d}.bin"),
            dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0

        T_velo_to_cam = self._calibs["T_velo_to_cam"]["00" if not is_right else "01"]
        K = self._calibs["K_perspective"]

        # project the points to the camera
        velo_pts_im = np.dot(K @ T_velo_to_cam[:3, :], points.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., None]

        # the projection is normalized to [-1, 1] -> transform to [0, height-1] x [0, width-1]
        velo_pts_im[:, 0] = np.round((velo_pts_im[:, 0] * .5 + .5) * self.target_image_size[1])
        velo_pts_im[:, 1] = np.round((velo_pts_im[:, 1] * .5 + .5) * self.target_image_size[0])

        # check if in bounds
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < self.target_image_size[1]) & (
                velo_pts_im[:, 1] < self.target_image_size[0])
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros(self.target_image_size)
        depth[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = velo_pts_im[:, 1] * (self.target_image_size[1] - 1) + velo_pts_im[:, 0] - 1
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0

        return depth[None, :, :]

    def load_samples(self, seq, img_id, is_right):
        if is_right:
            cam_dir = ["image_01", "image_00"]
        else:
            cam_dir = ["image_00", "image_01"]
        sample_path = os.path.join(self.data_path, "samples", seq, cam_dir[0], self._perspective_folder,
                                   f"{img_id[0]:010d}.npy")
        samples0 = np.load(sample_path).astype(np.int32)
        sample_path = os.path.join(self.data_path, "samples", seq, cam_dir[0], self._perspective_folder,
                                   f"{img_id[1]:010d}.npy")
        samples1 = np.load(sample_path).astype(np.int32)
        sample_path = os.path.join(self.data_path, "samples", seq, cam_dir[1], self._perspective_folder,
                                   f"{img_id[0]:010d}.npy")
        samples2 = np.load(sample_path).astype(np.int32)
        sample_path = os.path.join(self.data_path, "samples", seq, cam_dir[1], self._perspective_folder,
                                   f"{img_id[1]:010d}.npy")
        samples3 = np.load(sample_path).astype(np.int32)

        rand_epoch = np.random.randint(0, 25)
        rand_anc = np.random.choice(64, 16, replace=False)

        samples0 = np.hstack([np.zeros((16, 1), dtype=np.int32), samples0[rand_epoch, rand_anc, :][:, ::-1] - 4])
        samples1 = np.hstack([np.ones((16, 1), dtype=np.int32), samples1[rand_epoch, rand_anc, :][:, ::-1] - 4])
        samples2 = np.hstack([np.zeros((16, 1), dtype=np.int32), samples2[rand_epoch, rand_anc, :][:, ::-1] - 4])
        samples3 = np.hstack([np.ones((16, 1), dtype=np.int32), samples3[rand_epoch, rand_anc, :][:, ::-1] - 4])

        return [samples0, samples1, samples2, samples3]

    def __getitem__(self, index: int):
        """
        return a dict with keys below:
        - 'imgs': list of 8 images, 2 left imgs + 2 right imgs + 2 left fisheye imgs + 2 right fisheye imgs
        - 'projs': list of 8 intrinsics matrix
        - 'poses': list of 8 camera poses
        - 'gt_depth': []
        - 'pseudo_depth': []
        - 'ts': tensor of shape (B, 8), the index of imgs
        - '3d_bboxes': []
        - 'segs': []

        """
        _start_time = time.time()

        if index >= self.length:
            raise IndexError()

        if self._skip != 0:
            index += self._skip

        sequence, id, is_right = self._datapoints[index]
        seq_len = self._img_ids[sequence].shape[0]
        load_left = (not is_right) or self.return_stereo
        load_right = is_right or self.return_stereo
        # id: 4180 --> ids: [4180, 4181]
        ids = [id] + [max(min(i, seq_len - 1), 0) for i in
                      range(id - self._left_offset, id - self._left_offset + self.frame_count * self.dilation,
                            self.dilation) if i != id]
        if not self.variant_fisheye_offset:
            ids_fish = ([max(min(id + self.fisheye_offset, seq_len - 1), 0)] +
                        [max(min(i, seq_len - 1), 0) for i in range(
                            id + self.fisheye_offset - self._left_offset,
                            id + self.fisheye_offset - self._left_offset + self.frame_count * self.dilation,
                            self.dilation) if
                         i != id + self.fisheye_offset])
        else:
            curr_fisheye_offset = random.randint(self.low_fisheye_offset, self.up_fisheye_offset)
            ids_fish = ([max(min(id + curr_fisheye_offset, seq_len - 1), 0)] +
                        [max(min(i, seq_len - 1), 0) for i in range(
                            id + curr_fisheye_offset - self._left_offset,
                            id + curr_fisheye_offset - self._left_offset + self.frame_count * self.dilation,
                            self.dilation) if i != id + curr_fisheye_offset])

        img_ids = [self.get_img_id_from_id(sequence, id) for id in ids]  # real image id
        img_ids_fish = [self.get_img_id_from_id(sequence, id) for id in ids_fish]

        if not self.return_fisheye:
            ids_fish = []
            img_ids_fish = []

        if self.color_aug:
            color_aug_fn = get_color_aug_fn(
                ColorJitter.get_params(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2],
                                       hue=[-0.1, 0.1])
            )
        else:
            color_aug_fn = None

        _start_time_loading = time.time()
        imgs_p_left, imgs_f_left, imgs_p_right, imgs_f_right = self.load_images(sequence, img_ids, load_left,
                                                                                load_right, img_ids_fish=img_ids_fish)
        _loading_time = np.array(time.time() - _start_time_loading)
        _start_time_processing = time.time()
        # self.process_img: resize + to tensor + color aug + normalization
        imgs_p_left = [self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs_p_left]
        imgs_f_left = [self.process_img(img, color_aug_fn=color_aug_fn, resampler=self._resampler_02) for img in
                       imgs_f_left]
        imgs_p_right = [self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs_p_right]
        imgs_f_right = [self.process_img(img, color_aug_fn=color_aug_fn, resampler=self._resampler_03) for img in
                        imgs_f_right]
        _processing_time = np.array(time.time() - _start_time_processing)

        # These poses are camera to world !!
        poses_p_left = [self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["00"] for i in
                        ids] if load_left else []
        poses_f_left = [self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["02"] for i in
                        ids_fish] if load_left else []
        poses_p_right = [self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["01"] for i in
                         ids] if load_right else []
        poses_f_right = [self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["03"] for i in
                         ids_fish] if load_right else []

        projs_p_left = [self._calibs["K_perspective"] for _ in ids] if load_left else []
        projs_f_left = [self._calibs["K_fisheye"] for _ in ids_fish] if load_left else []
        projs_p_right = [self._calibs["K_perspective"] for _ in ids] if load_right else []
        projs_f_right = [self._calibs["K_fisheye"] for _ in ids_fish] if load_right else []

        # imgs: 左x2 + 右x2 + 左鱼眼x2 + 右鱼眼x2 = 8张
        imgs = imgs_p_left + imgs_p_right + imgs_f_left + imgs_f_right if not is_right \
            else imgs_p_right + imgs_p_left + imgs_f_right + imgs_f_left
        projs = projs_p_left + projs_p_right + projs_f_left + projs_f_right if not is_right \
            else projs_p_right + projs_p_left + projs_f_right + projs_f_left
        poses = poses_p_left + poses_p_right + poses_f_left + poses_f_right if not is_right \
            else poses_p_right + poses_p_left + poses_f_right + poses_f_left
        ids = np.array(ids + ids + ids_fish + ids_fish, dtype=np.int32)

        if self.return_pseudo_depth:
            if len(img_ids) == 1:
                pseudo_depth = [self.load_pseudo_depth(sequence, img_ids[0], is_right)]
            else:
                pseudo_depth = [self.load_pseudo_depth(sequence, img_ids[0], is_right),
                                self.load_pseudo_depth(sequence, img_ids[1], is_right)]
        else:
            pseudo_depth = []
        if self.return_gt_depth:
            if len(img_ids) == 1:
                gt_depth = [self.load_gt_depth(sequence, img_ids[0])]
            else:
                gt_depth = [self.load_gt_depth(sequence, img_ids[0]),
                            self.load_gt_depth(sequence, img_ids[1])]
        else:
            gt_depth = []

        if self.return_samples:
            samples = self.load_samples(sequence, img_ids, is_right)  # list
        else:
            samples = []

        if self.return_segmentation:
            segs = [self.load_segmentation(sequence, img_ids[0])]
        else:
            segs = []

        if self.return_3d_bboxes:
            bboxes_3d = self.get_3d_bboxes(sequence, img_ids[0], poses[0], projs[0])
            if bboxes_3d: # whether 3dbbox exists in this camera view
                bboxes_3d.update(convert_vertices(bboxes_3d["vertices"]))

        if self.return_voxel:
            """
            self._img_ids       {seq: pose_id2img_id}
            self._poses         {seq: pose_id2T}
            self._datapoints    [(seq, pose_id, is_right)]
            self.img_ids_test   {seq: [img_id_test]}
            self.img_ids_voxel  {seq: [img_id_test + offset]}
            """
            img_id = self.get_img_id_from_id(sequence, id)
            get_first_id = lambda A, k: np.where(A == k)[0][0]
            img_id_voxel = self.img_ids_voxel[sequence][get_first_id(self.img_ids_test[sequence], img_id)]
            id_in_sscbench = self.imgid2sscid[sequence][img_id_voxel]
            voxel_file_path = self.ssc_root_path / "preprocess" / "labels" / sequence / f"{id_in_sscbench:0>6}_1_1.npy"
            voxel = np.load(voxel_file_path).astype(np.int32)

            pose_id_voxel = get_first_id(self._img_ids[sequence], img_id_voxel)
            c2w = self._poses[sequence][id, :, :] @ self._calibs["T_cam_to_pose"]["01" if is_right else "00"]
            voxelcam2w = self._poses[sequence][pose_id_voxel, :, :] @ self._calibs["T_cam_to_pose"]["00"]
            voxellidar2voxelcam = read_calib()["velo2cam"] # FIXME: wrong velo2cam
            voxellidar2c = (np.linalg.inv(c2w) @ voxelcam2w @ voxellidar2voxelcam).astype(np.float32)

        _proc_time = np.array(time.time() - _start_time)

        # print(f"_loading_time:      {_loading_time:.5f}")
        # print(f"_processing_time:   {_processing_time:.5f}")
        # print(f"all:                {_proc_time:.5f}")
        # print()

        data = {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "depths": gt_depth,
            "pseudo_depth": pseudo_depth,
            "samples": samples,
            "ts": ids,
            "segs": segs,
            "t__get_item__": np.array([_proc_time]),
            "index": np.array([index]),
            "is_right": is_right,
            "seq": sequence,
            "img_id": img_ids[0]
        }
        if self.return_3d_bboxes: # empty dict if no 3dbbox exists in this camera view
            data.update({"3d_bboxes": bboxes_3d})
        if self.return_voxel:
            data.update(
                {
                    "voxellidar2c": voxellidar2c,
                    "voxel": voxel,
                    "voxel_file_path": voxel_file_path.as_posix(),
                }
            )
        return data

    def __len__(self) -> int:
        return self.length
