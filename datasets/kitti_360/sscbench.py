from pathlib import Path
import numpy as np
from numpy.typing import NDArray


def load_match_file(file_path: Path):
    assert file_path.exists()
    with open(file_path, "r") as f:
        match_raw = f.readlines()
    match_info: dict[str, list[int]] = {}
    for line in match_raw:
        seq, kitti360_filename, sscbench_filename = line.split(" ")
        if seq not in match_info:
            match_info[seq] = []
        match_info[seq].append(int(kitti360_filename[:-4]))
    return {seq: np.array(sscid2imgid) for seq, sscid2imgid in match_info.items()}, {
        seq: {imgid: sscid for sscid, imgid in enumerate(sscid2imgid)}
        for seq, sscid2imgid in match_info.items()
    }


def find_next_id_w_voxel(id_test: NDArray[int], id_w_voxel: NDArray[int]):
    """
    Find the next ID in id_w_voxel for each ID in id_test.
    For each value in id_test, find the smallest value in id_w_voxel that is greater than or equal to it.
    Both input arrays must be sorted in ascending order.

    Parameters:
    -----------
    id_test : NDArray[int]
        A sorted array of IDs to look up
    id_w_voxel : NDArray[int]
        A sorted array of IDs with voxel data

    Returns:
    --------
    NDArray[int]
        An array containing, for each ID in id_test, the next ID from id_w_voxel

    Examples:
    ---------
    >>> find_next_id_w_voxel(np.array([1, 3, 5]), np.array([2, 4, 6, 8]))
    array([2, 4, 6])

    >>> find_next_id_w_voxel(np.array([0, 2, 4]), np.array([2, 4, 6, 8]))
    array([2, 2, 4])

    >>> find_next_id_w_voxel(np.array([1, 3, 7]), np.array([1, 3, 5, 7, 9]))
    array([1, 3, 7])

    >>> # When id_test values are exact matches in id_w_voxel
    >>> find_next_id_w_voxel(np.array([2, 6, 8]), np.array([2, 4, 6, 8]))
    array([2, 6, 8])

    >>> # When all id_test values are smaller than smallest id_w_voxel
    >>> find_next_id_w_voxel(np.array([1, 2, 3]), np.array([5, 10, 15]))
    array([5, 5, 5])

    >>> # When arrays have single elements
    >>> find_next_id_w_voxel(np.array([3]), np.array([5]))
    array([5])

    >>> # Test the assertion error when max of id_test is larger than max of id_w_voxel
    >>> find_next_id_w_voxel(np.array([10, 20]), np.array([5, 15]))
    Traceback (most recent call last):
    ...
    AssertionError: can not find id with voxel bigger than 20

    >>> # Empty id_test case
    >>> find_next_id_w_voxel(np.array([]), np.array([2, 4, 6]))
    array([], dtype=int64)
    """
    assert (
        id_test.max() <= id_w_voxel.max() if id_test.size > 0 else True
    ), f"can not find id with voxel bigger than {id_test.max()}"
    return id_w_voxel[np.searchsorted(id_w_voxel, id_test)]


def load_sscbench(
    match_file_path: Path,
    data_points: list[tuple[str, int, bool]],
    ids_w_pose: dict[str, NDArray[int]],
):
    img_ids_test = {}
    for seq, id_in_pose, is_right in data_points:
        if seq not in img_ids_test:
            img_ids_test[seq] = []
        img_ids_test[seq].append(ids_w_pose[seq][id_in_pose])
    for seq in img_ids_test:
        img_ids_test[seq] = np.array(img_ids_test[seq])

    sscid2imgid, imgid2sscid = load_match_file(match_file_path)
    img_ids_voxel = {}
    for seq, id_w_pose in ids_w_pose.items():
        assert np.setdiff1d(sscid2imgid[seq][::5], id_w_pose).size == 0
        img_ids_voxel[seq] = find_next_id_w_voxel(
            img_ids_test[seq], sscid2imgid[seq][::5]
        )
    return img_ids_voxel, img_ids_test, imgid2sscid, sscid2imgid


if __name__ == "__main__":
    import doctest

    doctest.testmod()
