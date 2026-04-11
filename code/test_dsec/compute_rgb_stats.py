from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.dsec_dataset import fetch_dataloader as fetch_dsec_dataloader
from networks.fully_conv import FullyConv
from networks.unet_dav2 import SmallUNet, SmallUNet2, SmallUNet3


BASE_DIR = Path(__file__).resolve().parents[1]
UNET_CKPT_PATH = BASE_DIR / "train_output/train_dsec_unet_dav2_batch10/epoch_050.pt"
FULLY_CONV_CKPT_PATH = BASE_DIR / "train_output/train_dsec_fully_conv_dav2_batch10/epoch_050.pt"

OUTPUT_DIR = BASE_DIR / "test_dsec_output/rgb_stats"
SPLIT = "validation"


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


class RunningStats:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sum = np.zeros(3, dtype=np.float64)
        self.sum_sq = np.zeros(3, dtype=np.float64)
        self.min = np.full(3, np.inf, dtype=np.float64)
        self.max = np.full(3, -np.inf, dtype=np.float64)
        self.pixels = 0
        self.frames = 0

    def update(self, img: np.ndarray) -> None:
        flat = img.reshape(-1, 3)
        self.sum += flat.sum(axis=0, dtype=np.float64)
        self.sum_sq += (flat ** 2).sum(axis=0, dtype=np.float64)
        self.min = np.minimum(self.min, flat.min(axis=0))
        self.max = np.maximum(self.max, flat.max(axis=0))
        self.pixels += flat.shape[0]
        self.frames += 1

    def finalize(self) -> dict:
        mean = self.sum / max(self.pixels, 1)
        var = self.sum_sq / max(self.pixels, 1) - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))
        return {
            "mean": mean,
            "std": std,
            "var": var,
            "min": self.min,
            "max": self.max,
            "pixels": self.pixels,
            "frames": self.frames,
        }


def build_data_loader_config() -> dict:
    return {
        "dataset": "dsec",
        "datapath": "datasets/DSEC/data",
        "split": SPLIT,
        "concatenate_sequences": False,
        "event_representation": {
            "representation_type": "voxelgrid",
            "channels": 5,
            "height": DSEC_HEIGHT,
            "width": DSEC_WIDTH,
            "normalize": True,
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


def build_unet_rep() -> torch.nn.Module:
    return SmallUNet(
        in_channels=5,
        base_channels=32,
        out_channels=3,
    )


def load_unet_weights(unet: torch.nn.Module, ckpt_path: Path) -> None:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"UNet checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    unet_state = {k.replace("unet.", ""): v for k, v in state.items() if k.startswith("unet.")}
    if not unet_state:
        raise RuntimeError(f"No UNet weights found in checkpoint: {ckpt_path}")
    unet.load_state_dict(unet_state, strict=True)


def build_fully_conv_rep() -> torch.nn.Module:
    # Create fullyconv representation learner
    return FullyConv(
        in_channels=5,
        out_channels=3,
    )


def load_fully_conv_weights(fc: torch.nn.Module, ckpt_path: Path) -> None:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"FullyConv checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    fc_state = {k.replace("fully_conv.", ""): v for k, v in state.items() if k.startswith("fully_conv.")}
    if not fc_state:
        raise RuntimeError(f"No FullyConv weights found in checkpoint: {ckpt_path}")
    fc.load_state_dict(fc_state, strict=True)


def compute_histograms(
    dataloaders: dict,
    unet: torch.nn.Module,
    fc: torch.nn.Module,
    device: torch.device,
    stats_raw: RunningStats,
    stats_unet: RunningStats,
    stats_fc: RunningStats,
) -> dict:
    def bins_from_stats(stats: RunningStats) -> np.ndarray:
        min_val = float(np.min(stats.min))
        max_val = float(np.max(stats.max))
        if max_val - min_val < 1e-6:
            min_val -= 0.5
            max_val += 0.5
        return np.linspace(min_val, max_val, 256 + 1)

    bins_raw = bins_from_stats(stats_raw)
    bins_unet = bins_from_stats(stats_unet)
    bins_fc = bins_from_stats(stats_fc)

    hist_raw = np.zeros((3, 256), dtype=np.float64)
    hist_unet = np.zeros((3, 256), dtype=np.float64)
    hist_fc = np.zeros((3, 256), dtype=np.float64)

    with torch.no_grad():
        for _, loader in dataloaders.items():
            for sample in loader:
                rgb = sample["rgb"][:, 0]
                events = sample["depth_aligned_events"][:, 0]

                rgb_np = rgb.squeeze(0).permute(1, 2, 0).numpy()
                unet_out = unet(events.to(device)).cpu()
                fc_out = fc(events.to(device)).cpu()

                unet_np = unet_out.squeeze(0).permute(1, 2, 0).numpy()
                fc_np = fc_out.squeeze(0).permute(1, 2, 0).numpy()

                for c in range(3):
                    hist_raw[c] += np.histogram(rgb_np[..., c].ravel(), bins=bins_raw)[0]
                    hist_unet[c] += np.histogram(unet_np[..., c].ravel(), bins=bins_unet)[0]
                    hist_fc[c] += np.histogram(fc_np[..., c].ravel(), bins=bins_fc)[0]

    return {
        "raw": (hist_raw, bins_raw),
        "unet": (hist_unet, bins_unet),
        "fc": (hist_fc, bins_fc),
    }


def main() -> None:
    data_loader_config = build_data_loader_config()
    dataloaders = fetch_dsec_dataloader(data_loader_config, test=True)
    if not dataloaders:
        raise RuntimeError("No DSEC dataloaders created. Check datapath/split.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    unet = build_unet_rep().to(device).eval()
    fc = build_fully_conv_rep().to(device).eval()
    load_unet_weights(unet, UNET_CKPT_PATH)
    load_fully_conv_weights(fc, FULLY_CONV_CKPT_PATH)

    stats_raw = RunningStats("raw")
    stats_unet = RunningStats("unet")
    stats_fc = RunningStats("fc")

    for seq_name, loader in dataloaders.items():
        seq_count = 0
        with torch.no_grad():
            for sample in loader:
                if "rgb" not in sample:
                    raise KeyError("Dataloader sample missing 'rgb'. Ensure load_images=True.")
                rgb = sample["rgb"][:, 0]
                events = sample["depth_aligned_events"][:, 0]

                rgb_np = rgb.squeeze(0).permute(1, 2, 0).numpy()
                unet_out = unet(events.to(device)).cpu()
                fc_out = fc(events.to(device)).cpu()

                unet_np = unet_out.squeeze(0).permute(1, 2, 0).numpy()
                fc_np = fc_out.squeeze(0).permute(1, 2, 0).numpy()

                stats_raw.update(rgb_np)
                stats_unet.update(unet_np)
                stats_fc.update(fc_np)
                seq_count += 1

        print(f"[{seq_name}] {seq_count} frames")

    if stats_raw.pixels == 0:
        raise RuntimeError("No pixels processed. Check dataset path and split.")

    raw_stats = stats_raw.finalize()
    unet_stats = stats_unet.finalize()
    fc_stats = stats_fc.finalize()

    print("\n=== DSEC RGB Stats ===")
    print(f"Split: {SPLIT}")
    if data_loader_config.get("preprocessing"):
        for prep in data_loader_config["preprocessing"]:
            if prep.get("preprocessing_type") == "CenterCrop":
                print(f"Center crop: {prep.get('height')}x{prep.get('width')}")
                break
        else:
            print("Center crop: None")
    else:
        print("Center crop: None")
    print(f"Frames: {raw_stats['frames']}")
    print(f"Pixels: {raw_stats['pixels']}")

    def print_stats(label: str, stats: dict) -> None:
        print(f"\n[{label}]")
        print(f"Mean (R,G,B): {stats['mean'].tolist()}")
        print(f"Std  (R,G,B): {stats['std'].tolist()}")
        print(f"Var  (R,G,B): {stats['var'].tolist()}")
        print(f"Min  (R,G,B): {stats['min'].tolist()}")
        print(f"Max  (R,G,B): {stats['max'].tolist()}")

    print_stats("Raw RGB", raw_stats)
    print_stats("UNet Output", unet_stats)
    print_stats("FullyConv Output", fc_stats)

    hists = compute_histograms(
        dataloaders=dataloaders,
        unet=unet,
        fc=fc,
        device=device,
        stats_raw=stats_raw,
        stats_unet=stats_unet,
        stats_fc=stats_fc,
    )

    plot_rgb_hist(
        hist_counts=hists["raw"][0],
        bins=hists["raw"][1],
        title="DSEC RGB Distribution (Raw)",
        out_path=OUTPUT_DIR / "dsec_rgb_hist_raw.png",
        total_pixels=raw_stats["pixels"],
    )
    plot_rgb_hist(
        hist_counts=hists["unet"][0],
        bins=hists["unet"][1],
        title="DSEC RGB Distribution (UNet Output)",
        out_path=OUTPUT_DIR / "dsec_rgb_hist_unet.png",
        total_pixels=raw_stats["pixels"],
    )
    plot_rgb_hist(
        hist_counts=hists["fc"][0],
        bins=hists["fc"][1],
        title="DSEC RGB Distribution (FullyConv Output)",
        out_path=OUTPUT_DIR / "dsec_rgb_hist_fc.png",
        total_pixels=raw_stats["pixels"],
    )

    print(f"\nSaved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
