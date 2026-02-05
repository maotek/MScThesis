import argparse
import os
from pathlib import Path

import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events.events_representations import VoxelGrid, ETNetVoxelGrid
from networks.etnet import load_etnet
from util import save_grayscale, save_voxelgrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ET-Net on a DSEC sequence and visualize intensity reconstruction.")
    parser.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default="datasets/DSEC/data/validation/interlaken_00_f",
        help="Path to a DSEC sequence root",
    )
    parser.add_argument("--index", type=int, default=0, help="Index within the sequence to visualize (depth-aligned events).")
    parser.add_argument("--time-window-ms", type=int, default=50, help="Event window size for building voxel-grid representations.")
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of bins for the voxel grid; must match the checkpoint's expected num_bins.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("models", "etnet", "checkpoints", "etnet.pth"),
        help="ET-Net checkpoint path",
    )
    parser.add_argument("--use-minmax-norm", action="store_true", help="Apply ET-Net's optional min-max normalization to outputs.")
    parser.add_argument("--output-dir", type=str, default="output/test_etnet_on_dsec", help="Where to save visualizations.")
    return parser.parse_args()


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def visualize(events_voxel: torch.Tensor, recon: torch.Tensor, out_dir: str, idx: int) -> None:
    # events_voxel: (C,H,W), recon: (1,H,W)
    save_voxelgrid(os.path.join(out_dir, f"{idx:05d}_events.png"), events_voxel)
    save_grayscale(os.path.join(out_dir, f"{idx:05d}_recon.png"), recon)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rep = ETNetVoxelGrid(channels=args.num_bins, height=DSEC_HEIGHT, width=DSEC_WIDTH)
    # rep = ETNetVoxelGrid(channels=args.num_bins, height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True)
    dataset = DsecSequence(
        sequence_path=args.sequence,
        event_representation=rep,
        time_window_ms=args.time_window_ms,
        augmentator=None,
        load_images=False,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        self_supervised=False,
        postfix="",
    )

    assert 0 <= args.index < len(dataset), f"Index {args.index} out of range for sequence of length {len(dataset)}"
    sample = dataset[args.index]

    model = load_etnet(checkpoint_path=args.checkpoint, device=device, use_minmax_norm=args.use_minmax_norm)
    # ET-Net expects (B,C,H,W)
    events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)
    print(events.shape)
    print("min:", events.min().item(), "max:", events.max().item())
    recon = model(events)  # (B,1,H,W)

    out_dir = ensure_dir(args.output_dir)
    visualize(sample["depth_aligned_events"][0], recon[0], out_dir, args.index)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
