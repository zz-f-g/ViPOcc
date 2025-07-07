import os

from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset


def make_datasets(config):
    type = config.get("type", "KITTI_Raw")  # 'KITTI_360'
    if type == "KITTI_Raw":
        train_dataset = KittiRawDataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=os.path.join(config["split_path"], "train_files.txt"),
            target_image_size=config.get("image_size", (192, 640)),
            frame_count=config.get("data_fc", 1),
            return_stereo=config.get("return_stereo", False),
            keyframe_offset=config.get("keyframe_offset", 0),
            dilation=config.get("dilation", 1),
            color_aug=config.get("color_aug", False),
            return_pseudo_depth=config.get("return_pseudo_depth", False),
        )
        test_dataset = KittiRawDataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=os.path.join(config["split_path"], "val_files.txt"),
            target_image_size=config.get("image_size", (192, 640)),
            frame_count=config.get("data_fc", 1),
            return_stereo=config.get("return_stereo", False),
            keyframe_offset=config.get("keyframe_offset", 0),
            dilation=config.get("dilation", 1),
            return_pseudo_depth=config.get("return_pseudo_depth", False),
            return_gt_depth=True,
        )
        return train_dataset, test_dataset

    elif type == "KITTI_360":
        if config.get("split_path", None) is None:
            train_split_path = None
            test_split_path = None
        else:
            train_split_path = os.path.join(config["split_path"], "train_files.txt")  # 'datasets/kitti_360/splits/seg'
            test_split_path = os.path.join(config["split_path"], "test_files.txt")

        train_dataset = Kitti360Dataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=train_split_path,
            target_image_size=tuple(config.get("image_size", (192, 640))),
            frame_count=config.get("data_fc", 3),
            return_stereo=config.get("return_stereo", True),
            return_fisheye=config.get("return_fisheye", True),
            return_gt_depth=config.get("return_gt_depth", False),
            return_pseudo_depth=config.get("return_pseudo_depth", False),
            return_all_pseudo_depth=config.get("return_all_pseudo_depth", False),
            return_3d_bboxes=config.get("return_3d_bboxes", False),
            bboxes_semantic_labels=config.get("bboxes_semantic_labels", []),
            return_segmentation=config.get("data_segmentation", False),
            keyframe_offset=config.get("keyframe_offset", 0),
            dilation=config.get("dilation", 1),
            fisheye_rotation=config.get("fisheye_rotation", 0),  # [0, -15]
            fisheye_offset=config.get("fisheye_offset", 1),  # 10
            color_aug=config.get("color_aug", False),  # True
            is_preprocessed=config.get("is_preprocessed", False),  # True
            return_samples=config.get("return_samples", False),
            variant_fisheye_offset=config.get("variant_fisheye_offset", False),
            low_fisheye_offset=config.get("low_fisheye_offset", False),
            up_fisheye_offset=config.get("up_fisheye_offset", False)
        )
        test_dataset = Kitti360Dataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=test_split_path,
            target_image_size=tuple(config.get("image_size", (192, 640))),
            frame_count=config.get("data_fc", 3),
            return_stereo=config.get("return_stereo", True),
            return_fisheye=config.get("data_fisheye", True),
            return_gt_depth=True,
            return_pseudo_depth=config.get("return_pseudo_depth", False),
            return_3d_bboxes=config.get("return_3d_bboxes", False),
            bboxes_semantic_labels=config.get("bboxes_semantic_labels", []),
            return_segmentation=config.get("data_segmentation", False),
            return_voxel=config.get("return_voxel", False),
            keyframe_offset=config.get("keyframe_offset", 0),
            fisheye_rotation=config.get("fisheye_rotation", 0),
            fisheye_offset=config.get("fisheye_offset", 1),
            dilation=config.get("dilation", 1),
            is_preprocessed=config.get("is_preprocessed", False)
        )
        return train_dataset, test_dataset

    else:
        raise NotImplementedError(f"Unsupported dataset type: {type}")


def make_test_dataset(config):
    type = config.get("type", "KITTI_Raw")
    if type == "KITTI_Raw":
        test_dataset = KittiRawDataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=os.path.join(config["split_path"], "test_files.txt"),
            target_image_size=config.get("image_size", (192, 640)),
            return_gt_depth=True,
            frame_count=1,
            return_stereo=config.get("return_stereo", False),
            keyframe_offset=0
        )
        return test_dataset
    elif type == "KITTI_360":
        test_dataset = Kitti360Dataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=os.path.join(config.get("split_path", None), "test_files.txt"),
            target_image_size=tuple(config.get("image_size", (192, 640))),
            frame_count=config.get("data_fc", 1),
            return_stereo=config.get("return_stereo", False),
            return_fisheye=config.get("data_fisheye", False),
            return_gt_depth=config.get("return_gt_depth", False),
            return_pseudo_depth=config.get("return_pseudo_depth", False),
            return_3d_bboxes=config.get("return_3d_bboxes", False),
            bboxes_semantic_labels=config.get("bboxes_semantic_labels", []),
            return_segmentation=config.get("data_segmentation", False),
            return_voxel=config.get("return_voxel", False),
            keyframe_offset=0,
            fisheye_rotation=config.get("fisheye_rotation", 0),
            fisheye_offset=config.get("fisheye_offset", 1),
            dilation=config.get("dilation", 1),
            is_preprocessed=config.get("is_preprocessed", False)
        )
        return test_dataset

    else:
        raise NotImplementedError(f"Unsupported dataset type: {type}")


def make_demo_dataset(config):
    test_dataset = Kitti360Dataset(
        data_path=config["data_path"],
        pose_path=config["pose_path"],
        split_path=config["split_path"],
        target_image_size=tuple(config.get("image_size", (192, 640))),
        frame_count=config.get("data_fc", 1),
        return_stereo=config.get("return_stereo", False),
        return_fisheye=config.get("data_fisheye", False),
        return_gt_depth=config.get("return_gt_depth", False),
        return_pseudo_depth=config.get("return_pseudo_depth", False),
        return_segmentation=config.get("data_segmentation", False),
        keyframe_offset=0,
        fisheye_rotation=config.get("fisheye_rotation", 0),
        fisheye_offset=config.get("fisheye_offset", 1),
        dilation=config.get("dilation", 1),
        is_preprocessed=config.get("is_preprocessed", False)
    )
    return test_dataset
