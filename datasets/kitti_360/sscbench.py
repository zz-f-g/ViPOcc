from pathlib import Path
import numpy as np
from numpy.typing import NDArray


def load_match_file(file_path: Path):
    assert file_path.exists()
    with open(file_path, "r") as f:
        match_raw = f.readlines()
    match_info: dict[str: list[int]] = {}
    for line in match_raw:
        seq, kitti360_filename, sscbench_filename = line.split(" ")
        if seq not in match_info:
            match_info[seq] = []
        match_info[seq].append(int(kitti360_filename[:-4]))
    return {
        seq: np.array(sscid2kitti360id) for seq, sscid2kitti360id in match_info.items()
    }, {
        seq: {kitti360id: sscid for sscid, kitti360id in enumerate(sscid2kitti360id)}
        for seq, sscid2kitti360id in match_info.items()
    }


def find_next_id_w_voxel(id_test: NDArray[int], id_w_voxel: NDArray[int]):
    """
    assert both input are incremental
    """
    assert id_test.max() < id_w_voxel.max(), f"can not find id with voxel bigger than {id_test.max()}"
    return id_w_voxel[np.searchsorted(id_w_voxel, id_test)]

def load_sscbench(
    match_file_path: Path,
    data_points: list[tuple[str, int, bool]],
    ids_w_pose: dict[str, NDArray[int]],
):
    id_test = {}
    for seq, id_in_pose, is_right in data_points:
        if seq not in id_test:
            id_test[seq] = []
        id_test[seq].append(ids_w_pose[seq][id_in_pose])
    for seq in id_test:
        id_test[seq] = np.array(id_test[seq])

    sscid2kitti360id, kitti360id2sscid = load_match_file(match_file_path)
    id_w_voxel_for_test = {}
    for seq, id_w_pose in ids_w_pose.items():
        assert np.setdiff1d(sscid2kitti360id[seq][::5], id_w_pose).size == 0
        id_w_voxel_for_test[seq] = find_next_id_w_voxel(id_test[seq], sscid2kitti360id[seq][::5])
    return id_w_voxel_for_test, id_test, kitti360id2sscid
