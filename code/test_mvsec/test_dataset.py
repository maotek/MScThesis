import argparse
import os
from pathlib import Path
from typing import List

import numpy as np

from datasets.MVSEC.constants import MVSEC_ALL_DATA_FOLDERS
from util import save_depth_colormap_with_cbar

BASE_DIR = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(BASE_DIR / p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect MVSEC depth GT .npy files to check value range (metric vs disparity)."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="datasets/MVSEC/data",
        help="Root path to MVSEC dataset.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="test/outdoor_day1",
        help="Sequence folder (e.g., train/outdoor_day2). Defaults to first available.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=5,
        help="Number of depth files to inspect.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific indices of depth files to inspect (overrides --num-files).",
    )
    parser.add_argument(
        "--print-array",
        action="store_true",
        help="Print full array values (can be large).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_mvsec_output/test_dataset",
        help="Directory to save depth visualizations (PNG).",
    )
    return parser.parse_args()


def pick_sequence(datapath: str, requested: str) -> str:
    if requested:
        return requested
    # Default to first known sequence that exists
    for seq in MVSEC_ALL_DATA_FOLDERS:
        if os.path.isdir(os.path.join(datapath, seq)):
            return seq
    # Fallback: just use the first entry if nothing exists
    return MVSEC_ALL_DATA_FOLDERS[0] if MVSEC_ALL_DATA_FOLDERS else ""


def list_depth_files(depth_dir: str) -> List[str]:
    files = [f for f in os.listdir(depth_dir) if f.endswith(".npy")]
    files.sort()
    return files


def print_stats(path: str, arr: np.ndarray, print_array: bool) -> None:
    finite = np.isfinite(arr)
    if finite.any():
        vals = arr[finite]
        stats = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
        }
    else:
        stats = {"min": None, "max": None, "mean": None, "median": None, "p95": None}

    print(f"\nFile: {path}")
    print(f"  shape={arr.shape} dtype={arr.dtype}")
    print(
        "  stats: min={min} max={max} mean={mean} median={median} p95={p95}".format(**stats)
    )
    if print_array:
        print(arr)


def save_depth_image(path: str, depth: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    depth_vis = depth.copy()
    if depth_vis.ndim == 3:
        depth_vis = depth_vis[0]
    depth_vis = np.nan_to_num(depth_vis, nan=0.0, posinf=0.0, neginf=0.0)
    save_depth_colormap_with_cbar(path, depth_vis)


def main() -> None:
    args = parse_args()
    args.datapath = resolve_path(args.datapath)
    args.output_dir = resolve_path(args.output_dir)

    seq = pick_sequence(args.datapath, args.sequence)
    if not seq:
        raise FileNotFoundError("No MVSEC sequences found.")

    depth_dir = os.path.join(args.datapath, seq, "depth", "data")
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    depth_files = list_depth_files(depth_dir)
    if not depth_files:
        raise FileNotFoundError(f"No .npy depth files found in: {depth_dir}")

    if args.indices:
        indices = args.indices
    else:
        indices = list(range(min(args.num_files, len(depth_files))))

    print(f"Dataset root: {args.datapath}")
    print(f"Sequence: {seq}")
    print(f"Depth dir: {depth_dir}")
    print(f"Output dir: {args.output_dir}")

    for idx in indices:
        if idx < 0 or idx >= len(depth_files):
            print(f"[WARN] index {idx} out of range (0..{len(depth_files)-1}), skipping")
            continue
        path = os.path.join(depth_dir, depth_files[idx])
        arr = np.load(path)
        print_stats(path, arr, args.print_array)

        out_name = os.path.splitext(depth_files[idx])[0] + ".png"
        out_path = os.path.join(args.output_dir, seq.replace("/", "_"), out_name)
        save_depth_image(out_path, arr)


if __name__ == "__main__":
    main()
