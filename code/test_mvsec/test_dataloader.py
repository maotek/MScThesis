import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from datasets.MVSEC.mvsec_dataset import fetch_dataloader
from util import (
    grayscale_to_uint8,
    rgb_to_uint8,
    save_depth_colormap_with_cbar,
    save_image,
    save_rgb,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_LOADER_CONFIG = {
    "dataset": "mvsec",
    "datapath": "datasets/MVSEC/data",
    "split": "train",
    "concatenate_sequences": False,
    "event_representation": {
        "representation_type": "voxelgrid",
        "channels": 5,
        "height": 260,
        "width": 346,
        "normalize": True,
    },
    "preprocessing": [
        {
            "preprocessing_type": "Crop",
            "height": 260,
            "width": 346,
        }
    ],
    "load_images": True,
    "batch_size": 10,
    "num_workers": 1,
    "pin_memory": True,
    "shuffle": True,
    "sequence_window": 1,
    "sequence_step": 1,
    "time_window_ms": 50,
}


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(BASE_DIR / p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize an MVSEC sample using a hardcoded MVSEC fetch_dataloader "
            "and save the same outputs as test_mvsec/mvsec_sequence.py."
        )
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="test/outdoor_day1",
        help="Full sequence key, e.g. test/outdoor_day1 or train/outdoor_day2.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=100,
        help="Visualize every Nth sample in the selected sequence.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting sample index.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on the number of saved samples (0 = no cap).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_mvsec_output/test_dataloader",
        help="Where to save PNG outputs.",
    )
    return parser.parse_args()
def fetch_sample(data_loader, sample_index: int):
    dataset = data_loader.dataset
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"Index {sample_index} out of range for dataset of size {len(dataset)}")
    return dataset[sample_index]


def visualize_events(events) -> np.ndarray:
    if events.ndim != 3:
        raise ValueError(f"Expected events with shape (C,H,W), got {tuple(events.shape)}")

    if events.shape[0] == 3:
        return rgb_to_uint8(events)

    events_np = events.cpu().numpy() if hasattr(events, "cpu") else np.asarray(events)
    mean_intensity = np.mean(np.abs(events_np), axis=0)
    if mean_intensity.max() > 0:
        mean_intensity = mean_intensity / mean_intensity.max()
    return grayscale_to_uint8(mean_intensity)


def ensure_three_channels(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


def print_depth_stats(depth_np: np.ndarray) -> None:
    finite_mask = np.isfinite(depth_np)
    nonzero_mask = finite_mask & (depth_np > 0)

    print("Depth stats:")
    if finite_mask.any():
        finite_vals = depth_np[finite_mask]
        print(f"  finite min/max: {finite_vals.min():.6f}/{finite_vals.max():.6f}")
        print(f"  finite mean/median: {finite_vals.mean():.6f}/{np.median(finite_vals):.6f}")
    else:
        print("  finite min/max: none")
        print("  finite mean/median: none")

    if nonzero_mask.any():
        nonzero_vals = depth_np[nonzero_mask]
        print(f"  nonzero min/max: {nonzero_vals.min():.6f}/{nonzero_vals.max():.6f}")
        print(f"  nonzero mean/median: {nonzero_vals.mean():.6f}/{np.median(nonzero_vals):.6f}")
    else:
        print("  nonzero min/max: none")
        print("  nonzero mean/median: none")

    print(f"  zero pixels: {np.size(depth_np) - int(nonzero_mask.sum())}")
    print(f"  valid nonzero pixels: {int(nonzero_mask.sum())}/{depth_np.size}")


def save_sample_visualization(
    sample: dict,
    sequence_name: str,
    sample_index: int,
    output_dir: str,
) -> None:
    events = sample["depth_aligned_events"][0]
    depth = sample["depth"][0]
    print(f"Sequence: {sequence_name}")
    print(f"Sample {sample_index}: events shape {events.shape}, depth shape {depth.shape}")
    print(f"Event min/max: {events.min()}/{events.max()}, Depth min/max: {depth.min()}/{depth.max()}")

    events_vis = visualize_events(events)

    depth_np = depth.cpu().numpy()
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    print_depth_stats(depth_np)

    depth_vis = depth_np.copy()
    if depth_vis.max() > depth_vis.min():
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())
    else:
        depth_vis = np.zeros_like(depth_vis)
    depth_vis = grayscale_to_uint8(depth_vis)

    rgb_vis = None
    if "rgb" in sample:
        rgb = sample["rgb"][0].cpu().numpy()
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_vis = rgb_to_uint8(rgb)

    sequence_dir = os.path.join(output_dir, sequence_name.replace("/", "_"))
    sample_dir = os.path.join(sequence_dir, f"{sample_index:05d}")
    os.makedirs(sample_dir, exist_ok=True)

    combined_path = os.path.join(sample_dir, "combined.png")
    if rgb_vis is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(events_vis, cmap="gray" if events_vis.ndim == 2 else None)
        axes[0].set_title("Events")
        axes[0].axis("off")
        axes[1].imshow(depth_vis, cmap="gray")
        axes[1].set_title("Depth")
        axes[1].axis("off")
        axes[2].imshow(rgb_vis)
        axes[2].set_title("RGB")
        axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(combined_path, dpi=150)
        plt.close(fig)
        print(f"Saved combined view to {combined_path}")

        if rgb_vis.shape[:2] != events_vis.shape[:2]:
            events_for_overlay = cv2.resize(
                ensure_three_channels(events_vis),
                (rgb_vis.shape[1], rgb_vis.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            events_for_overlay = ensure_three_channels(events_vis)
        overlay = 0.5 * events_for_overlay.astype(np.float32) + 0.5 * rgb_vis.astype(np.float32)
        overlay = np.clip(overlay / 255.0, 0.0, 1.0)
        overlay_path = os.path.join(sample_dir, "overlay_events_rgb.png")
        save_image(overlay_path, overlay)
        print(f"Saved overlay to {overlay_path}")

    events_path = os.path.join(sample_dir, "events.png")
    depth_path = os.path.join(sample_dir, "depth.png")

    if depth_vis.shape[:2] != events_vis.shape[:2]:
        events_for_overlay = cv2.resize(
            ensure_three_channels(events_vis),
            (depth_vis.shape[1], depth_vis.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        events_for_overlay = ensure_three_channels(events_vis)

    if depth_vis.ndim == 2:
        depth_for_overlay = np.repeat(depth_vis[..., None], 3, axis=2)
    elif depth_vis.ndim == 3 and depth_vis.shape[2] == 1:
        depth_for_overlay = np.repeat(depth_vis, 3, axis=2)
    else:
        depth_for_overlay = depth_vis

    depth_overlay = 0.5 * events_for_overlay.astype(np.float32) + 0.5 * depth_for_overlay.astype(
        np.float32
    )
    depth_overlay = np.clip(depth_overlay / 255.0, 0.0, 1.0)
    depth_overlay_path = os.path.join(sample_dir, "overlay_events_depth.png")
    save_image(depth_overlay_path, depth_overlay)

    save_image(events_path, events_vis, cmap="gray" if events_vis.ndim == 2 else None)
    save_depth_colormap_with_cbar(depth_path, depth_np)
    print(f"Saved overlay (events on depth) to {depth_overlay_path}")

    if "rgb" in sample:
        rgb_path = os.path.join(sample_dir, "rgb.png")
        save_rgb(rgb_path, sample["rgb"][0])
        print(f"Saved rgb to {rgb_path}")

    print(f"Saved events to {events_path}")
    print(f"Saved depth with colorbar to {depth_path}")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    sequence_name = args.sequence

    if args.every <= 0:
        raise ValueError("--every must be > 0")
    if "/" not in sequence_name:
        raise ValueError("--sequence must be a full key like test/outdoor_day1 or train/outdoor_day2")

    data_loader_config = dict(DATA_LOADER_CONFIG)
    data_loader_config["split"] = sequence_name.split("/", 1)[0]
    dataloaders = fetch_dataloader(data_loader_config, test=True)
    if sequence_name not in dataloaders:
        available = ", ".join(sorted(dataloaders.keys()))
        raise KeyError(f"Sequence '{sequence_name}' not found. Available sequences: {available}")

    print("Using hardcoded MVSEC data_loader config.")
    print(f"Sequence: {sequence_name}")
    print(f"Saving every {args.every} sample(s) starting from index {args.start_index}.")

    data_loader = dataloaders[sequence_name]
    dataset = data_loader.dataset
    saved = 0

    for sample_index in range(args.start_index, len(dataset), args.every):
        sample = fetch_sample(data_loader, sample_index)
        save_sample_visualization(sample, sequence_name, sample_index, output_dir)
        saved += 1
        if args.max_samples > 0 and saved >= args.max_samples:
            break

    print(f"Saved {saved} sample(s) for {sequence_name}.")


if __name__ == "__main__":
    main()
