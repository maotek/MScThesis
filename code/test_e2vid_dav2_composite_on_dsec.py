import argparse
import os
from pathlib import Path
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events.events_representations import VoxelGrid, E2vidVoxelGrid
from networks.e2vid_dav2_composite import E2VIDDav2Composite, E2VIDDav2Composite2
from util import save_depth_colormap, save_voxelgrid

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2VID->DAV2 composite on a DSEC sequence and visualize input, composite, and depth."
    )
    parser.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default="datasets/DSEC/data/train/zurich_city_01_e",
        help="Path to a DSEC sequence root",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=10,
        help="Index within the sequence to visualize (depth-aligned events).",
    )
    parser.add_argument(
        "--time-window-ms",
        type=int,
        default=50,
        help="Event window size for building voxel-grid representations.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of bins for the voxel grid; must match the E2VID checkpoint (lightweight uses 5).",
    )
    parser.add_argument(
        "--e2vid-checkpoint",
        type=str,
        default=None,
        help="Optional E2VID checkpoint path. Defaults to models/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar",
    )
    parser.add_argument(
        "--dav2-checkpoint",
        type=str,
        default=os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vits.pth"),
        help="DAV2 checkpoint path (default: models/dav2/checkpoints/depth_anything_v2_vits.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_e2vid_dav2_composite_on_dsec",
        help="Where to save visualizations.",
    )
    parser.add_argument(
        "--clip-distance",
        type=float,
        default=80.0,
        help="Max depth value (meters) for visualization clipping.",
    )
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


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    rep = E2vidVoxelGrid(channels=args.num_bins, height=DSEC_HEIGHT, width=DSEC_WIDTH)
    dataset = DsecSequence(
        sequence_path=args.sequence,
        event_representation=rep,
        time_window_ms=args.time_window_ms,
        augmentator=None,
        load_images=False,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        split="validation",
        self_supervised=False,
        postfix="",
    )

    assert 0 <= args.index < len(dataset), f"Index {args.index} out of range for sequence of length {len(dataset)}"
    sample = dataset[args.index]

    model = E2VIDDav2Composite(
        e2vid_weights=args.e2vid_checkpoint,
        dav2_encoder="vits",
        dav2_checkpoint=args.dav2_checkpoint,
        device=device,
    )

    events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)

    # Run E2VID/DAV2 pipeline
    depth, composite = model(events)  # composite: (B,3,H,W)

    out_dir = ensure_dir(args.output_dir)
    # Save inputs and outputs
    save_voxelgrid(os.path.join(out_dir, f"{args.index:05d}_events.png"), sample["depth_aligned_events"][0])
    save_depth_colormap(os.path.join(out_dir, f"{args.index:05d}_depth.png"), depth[0])

    comp = composite[0].detach().cpu()
    comp_uint8 = (comp.clamp(0.0, 1.0) * 255.0).permute(1, 2, 0).numpy()
    plt.imsave(os.path.join(out_dir, f"{args.index:05d}_composite.png"), comp_uint8.astype("uint8"))

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
