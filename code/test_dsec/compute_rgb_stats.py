import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.dsec_dataset import fetch_dataloader as fetch_dsec_dataloader

DATA_LOADER_CONFIG = {
    "dataset": "dsec",
    "datapath": "datasets/DSEC/data",
    "split": "validation",
    "concatenate_sequences": False,
    "event_representation": {
        "representation_type": "tencode",
        "normalize": True,
        "white_frame": False,
        "height": DSEC_HEIGHT,
        "width": DSEC_WIDTH,
    },
    "preprocessing": [
        {
            "preprocessing_type": "CenterCrop",
            "height": 320,
            "width": 640,
        }
    ],
    "load_images": True,
    "batch_size": 1,
    "num_workers": 0,
    "pin_memory": False,
    "shuffle": False,
    "sequence_window": 1,
    "sequence_step": 1,
    "time_window_ms": 50,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OUTPUT_DIR = Path("test_dsec_output/rgb_stats")


def plot_rgb_hist(hist_counts: np.ndarray, bins: np.ndarray, title: str, out_path: Path, total_pixels: int) -> None:
    centers = (bins[:-1] + bins[1:]) / 2.0
    labels = ["R", "G", "B"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, ax in enumerate(axes):
        y = hist_counts[i] / float(total_pixels)
        ax.plot(centers, y)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Value")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    dataloaders = fetch_dsec_dataloader(DATA_LOADER_CONFIG, test=True)
    if not dataloaders:
        raise RuntimeError("No DSEC dataloaders created. Check datapath/split.")

    sum_c = np.zeros(3, dtype=np.float64)
    sum_sq_c = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    total_frames = 0

    bins_raw = np.linspace(0.0, 1.0, 256 + 1)
    bins_norm = np.linspace(-3.0, 3.0, 256 + 1)
    hist_raw = np.zeros((3, 256), dtype=np.float64)
    hist_norm = np.zeros((3, 256), dtype=np.float64)

    for seq_name, loader in dataloaders.items():
        seq_count = 0
        for sample in loader:
            if "rgb" not in sample:
                raise KeyError("Dataloader sample missing 'rgb'. Ensure load_images=True.")
            rgb = sample["rgb"][:, 0]  # (B,C,H,W), B=1
            rgb_np = rgb.squeeze(0).permute(1, 2, 0).numpy()  # H,W,C

            h, w = rgb_np.shape[:2]
            total_pixels += h * w
            total_frames += 1
            seq_count += 1

            flat = rgb_np.reshape(-1, 3)
            sum_c += flat.sum(axis=0, dtype=np.float64)
            sum_sq_c += (flat ** 2).sum(axis=0, dtype=np.float64)

            for c in range(3):
                hist_raw[c] += np.histogram(flat[:, c], bins=bins_raw)[0]

            norm = (flat - IMAGENET_MEAN) / IMAGENET_STD
            for c in range(3):
                hist_norm[c] += np.histogram(norm[:, c], bins=bins_norm)[0]

        print(f"[{seq_name}] {seq_count} frames")

    if total_pixels == 0:
        raise RuntimeError("No pixels processed. Check dataset path and split.")

    mean = sum_c / total_pixels
    var = sum_sq_c / total_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    print("\n=== DSEC RGB Stats ===")
    print(f"Split: {DATA_LOADER_CONFIG['split']}")
    if DATA_LOADER_CONFIG.get("preprocessing"):
        for prep in DATA_LOADER_CONFIG["preprocessing"]:
            if prep.get("preprocessing_type") == "CenterCrop":
                print(f"Center crop: {prep.get('height')}x{prep.get('width')}")
                break
        else:
            print("Center crop: None")
    else:
        print("Center crop: None")
    print(f"Frames: {total_frames}")
    print(f"Pixels: {total_pixels}")
    print(f"Mean (R,G,B): {mean.tolist()}")
    print(f"Std  (R,G,B): {std.tolist()}")
    print(f"Var  (R,G,B): {var.tolist()}")

    plot_rgb_hist(
        hist_counts=hist_raw,
        bins=bins_raw,
        title="DSEC RGB Distribution (Raw)",
        out_path=OUTPUT_DIR / "dsec_rgb_hist_raw.png",
        total_pixels=total_pixels,
    )
    plot_rgb_hist(
        hist_counts=hist_norm,
        bins=bins_norm,
        title="DSEC RGB Distribution (ImageNet Normalized)",
        out_path=OUTPUT_DIR / "dsec_rgb_hist_imagenet.png",
        total_pixels=total_pixels,
    )
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
