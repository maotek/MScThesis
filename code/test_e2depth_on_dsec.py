import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events.events_representations import E2DepthVoxelGrid
from networks.e2depth_wrapper import load_e2depth
from util import save_depth_colormap, save_voxelgrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2Depth on a DSEC sequence and visualize depth prediction.")
    parser.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default="datasets/DSEC/data/validate/interlaken_00_f",
        help="Path to a DSEC sequence root",
    )
    parser.add_argument("--index", type=int, default=0, help="Index within the sequence to visualize (depth-aligned events).")
    parser.add_argument("--time-window-ms", type=int, default=50, help="Event window size for building voxel-grid representations.")
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of bins for the voxel grid; must match the checkpoint's num_bins (E2Depth default uses 5).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional E2Depth checkpoint path. Defaults to models/rpg_e2depth/pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar",
    )
    parser.add_argument("--output-dir", type=str, default="output/test_e2depth_on_dsec", help="Where to save visualizations.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/mps/cpu). Defaults to auto-selection.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def visualize(events_voxel: torch.Tensor, depth: torch.Tensor, out_dir: str, idx: int) -> None:
    # events_voxel: (C,H,W), depth: (1,H,W)
    save_voxelgrid(os.path.join(out_dir, f"{idx:05d}_events.png"), events_voxel)
    save_depth_colormap(os.path.join(out_dir, f"{idx:05d}_depth.png"), depth)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    rep = E2DepthVoxelGrid(channels=args.num_bins, height=DSEC_HEIGHT, width=DSEC_WIDTH)
    dataset = DsecSequence(
        sequence_path=args.sequence,
        event_representation=rep,
        time_window_ms=args.time_window_ms,
        augmentator=None,
        load_images=False,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        split="train",
        self_supervised=False,
        postfix="",
    )

    assert 0 <= args.index < len(dataset), f"Index {args.index} out of range for sequence of length {len(dataset)}"
    sample = dataset[args.index]

    model = load_e2depth(weights_path=args.checkpoint, device=device)
    # E2Depth expects (B,C,H,W)
    events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)
    depth = model(events)  # (B,1,H,W)

    out_dir = ensure_dir(args.output_dir)
    visualize(sample["depth_aligned_events"][0], depth[0], out_dir, args.index)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
