import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from warp_rgb import find_calibration_path, get_intrinsics_extrinsics, load_calibration_yaml, warp_rgb_to_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute warped DSEC RGB images.")
    parser.add_argument("--dataset-root", type=str, default="datasets/DSEC/data")
    parser.add_argument("--split", type=str, default="all", choices=["train", "validation", "all"])
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-images", type=int, default=0)
    return parser.parse_args()


def find_closest(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(reference, query)
    indices = np.clip(indices, 1, len(reference) - 1)
    left_values = reference[indices - 1]
    right_values = reference[indices]
    return np.where(
        np.abs(query - left_values) <= np.abs(query - right_values),
        indices - 1,
        indices,
    )


def disparity_to_depth(disparity: np.ndarray, q_event: np.ndarray) -> np.ndarray:
    focal = abs(q_event[2, 3])
    baseline = abs(1.0 / q_event[3, 2])
    return (focal * baseline) / (disparity + 1e-6)


def get_sequence_dirs(dataset_root: str, split: str, sequence: Optional[str]) -> list[str]:
    splits = ["train", "validation"] if split == "all" else [split]
    sequence_dirs = []
    for split_name in splits:
        split_dir = os.path.join(dataset_root, split_name)
        if not os.path.isdir(split_dir):
            continue
        if sequence is not None:
            seq_dir = os.path.join(split_dir, sequence)
            if os.path.isdir(seq_dir):
                sequence_dirs.append(seq_dir)
            continue
        for entry in sorted(os.listdir(split_dir)):
            seq_dir = os.path.join(split_dir, entry)
            if os.path.isdir(seq_dir):
                sequence_dirs.append(seq_dir)
    return sequence_dirs


def precompute_sequence(sequence_path: str, overwrite: bool = False, limit_images: int = 0) -> None:
    seq_name = os.path.basename(sequence_path)
    image_dir = os.path.join(sequence_path, f"{seq_name}_images_rectified_left")
    disparity_dir = os.path.join(sequence_path, f"{seq_name}_disparity_event")
    output_dir = os.path.join(sequence_path, f"{seq_name}_images_warped_left")
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(entry for entry in os.listdir(image_dir) if entry.endswith(".png"))
    disparity_files = sorted(entry for entry in os.listdir(disparity_dir) if entry.endswith(".png"))
    image_timestamps = np.loadtxt(os.path.join(sequence_path, "image_timestamps.txt"), dtype="int64")
    disparity_timestamps = np.loadtxt(os.path.join(sequence_path, "disparity_timestamps.txt"), dtype="int64")

    calib = load_calibration_yaml(find_calibration_path(sequence_path))
    event_K, rgb_K, T_10 = get_intrinsics_extrinsics(calib)
    q_event = np.array(calib["disparity_to_depth"]["cams_03"])

    if len(image_files) != len(image_timestamps):
        raise ValueError(f"{sequence_path}: image file count does not match image timestamps")
    if len(disparity_files) != len(disparity_timestamps):
        raise ValueError(f"{sequence_path}: disparity file count does not match disparity timestamps")

    disparity_indices = find_closest(image_timestamps, disparity_timestamps)
    if limit_images > 0:
        image_files = image_files[:limit_images]
        disparity_indices = disparity_indices[:limit_images]

    current_disparity_idx = None
    current_depth = None

    for image_file, disparity_idx in tqdm.tqdm(
        zip(image_files, disparity_indices),
        total=len(image_files),
        desc=f"Warping {seq_name}",
    ):
        output_path = os.path.join(output_dir, image_file)
        if not overwrite and os.path.isfile(output_path):
            continue

        disparity_idx = int(disparity_idx)
        if current_disparity_idx != disparity_idx:
            disparity_path = os.path.join(disparity_dir, disparity_files[disparity_idx])
            disp_16bit = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH)
            if disp_16bit is None:
                raise ValueError(f"Failed to load disparity map: {disparity_path}")
            disparity = disp_16bit.astype("float32") / 256.0
            current_depth = disparity_to_depth(disparity, q_event)
            current_disparity_idx = disparity_idx

        image_path = os.path.join(image_dir, image_file)
        rgb_img = cv2.imread(image_path)
        if rgb_img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        warped = warp_rgb_to_event(rgb_img, current_depth, event_K, rgb_K, T_10)
        if not cv2.imwrite(output_path, warped):
            raise ValueError(f"Failed to save warped image: {output_path}")


def main() -> None:
    args = parse_args()
    sequence_dirs = get_sequence_dirs(args.dataset_root, args.split, args.sequence)
    if not sequence_dirs:
        raise FileNotFoundError("No matching DSEC sequence directories found.")

    for sequence_path in sequence_dirs:
        precompute_sequence(
            sequence_path=sequence_path,
            overwrite=args.overwrite,
            limit_images=args.limit_images,
        )


if __name__ == "__main__":
    main()
