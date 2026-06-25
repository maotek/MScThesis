"""Visualize DSEC events, Tencode, the UNet reconstruction, and depth."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events.events_representations import Tencode, VoxelGrid
from datasets.utils import fetch_preprocessing
from evaluate import fetch_model
from evaluation import prepare_target_data_torch
from losses import normalized_depth_scale_and_shift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize blue/red DSEC events, Tencode, the UNet reconstruction, "
            "and depth."
        )
    )
    parser.add_argument(
        "sequence",
        nargs="?",
        default="datasets/DSEC/data/validation/interlaken_00_g",
        help="DSEC sequence directory (relative paths are resolved from code/).",
    )
    parser.add_argument("--index", type=int, default=350, help="Sample index.")
    parser.add_argument(
        "--config",
        default="configs/dsec/validation/unet_dav2_batch10.json",
        help="UNet-DAV2 validation config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint override for model.checkpoint_path.",
    )
    parser.add_argument(
        "--depth-source",
        choices=("predicted", "ground-truth"),
        default="predicted",
        help="Depth shown in the final panel.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        default="test_dsec_output/unet_sample_visualization",
        help="Output directory (relative paths are resolved from code/).",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else BASE_DIR / resolved


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_model_paths(model_config: dict, checkpoint: Optional[str]) -> dict:
    model_config = dict(model_config)
    for key in ("dav2_checkpoint", "checkpoint_path"):
        if key in model_config:
            model_config[key] = str(resolve_path(str(model_config[key])))
    if checkpoint is not None:
        model_config["checkpoint_path"] = str(resolve_path(checkpoint))
    return model_config


def build_dataset(sequence: Path, data_config: dict) -> DsecSequence:
    representation_config = data_config["event_representation"]
    representation_type = str(
        representation_config.get("representation_type", "")
    ).lower()
    if representation_type != "voxelgrid":
        raise ValueError(
            "This visualization requires a voxelgrid event representation, got "
            f"'{representation_type}'."
        )

    representation = VoxelGrid(
        channels=int(representation_config["channels"]),
        height=int(representation_config["height"]),
        width=int(representation_config["width"]),
        normalize=bool(representation_config.get("normalize", True)),
    )
    preprocessing = data_config.get("preprocessing", [])
    augmentator = fetch_preprocessing(preprocessing) if preprocessing else None
    return DsecSequence(
        sequence_path=str(sequence),
        event_representation=representation,
        time_window_ms=int(data_config.get("time_window_ms", 50)),
        augmentator=augmentator,
        load_images=False,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        self_supervised=False,
        postfix="",
    )


def voxelgrid_to_red_blue(voxel_grid: torch.Tensor) -> np.ndarray:
    """Collapse a signed voxel grid into RGB (positive blue, negative red)."""
    if voxel_grid.ndim != 3:
        raise ValueError(f"Expected voxel grid [C,H,W], got {tuple(voxel_grid.shape)}")

    event_sum = voxel_grid.detach().float().cpu().sum(dim=0).numpy()
    preview = np.zeros((*event_sum.shape, 3), dtype=np.float32)
    preview[event_sum > 0.0] = (0.0, 0.0, 1.0)
    preview[event_sum < 0.0] = (1.0, 0.0, 0.0)
    return preview


def load_tencode_sample(
    dataset: DsecSequence,
    index: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Render the same dataset window as Tencode without changing UNet input."""
    voxelgrid_representation = dataset.event_representation
    try:
        dataset.event_representation = Tencode(
            height=height,
            width=width,
            normalize=True,
            white_frame=False,
        )
        return dataset[index]["depth_aligned_events"][0]
    finally:
        dataset.event_representation = voxelgrid_representation


def tencode_to_rgb(tencode: torch.Tensor) -> np.ndarray:
    if tencode.ndim != 3 or tencode.shape[0] != 3:
        raise ValueError(f"Expected Tencode [3,H,W], got {tuple(tencode.shape)}")
    return tencode.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def reconstruction_to_rgb(reconstruction: torch.Tensor) -> np.ndarray:
    reconstruction = reconstruction.detach().float().cpu().squeeze(0)
    if reconstruction.shape[0] == 1:
        reconstruction = reconstruction.repeat(3, 1, 1)
    if reconstruction.shape[0] != 3:
        raise ValueError(
            "Expected the UNet reconstruction to have 1 or 3 channels, got "
            f"{reconstruction.shape[0]}."
        )
    return reconstruction.clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def predict_depth(
    model: torch.nn.Module,
    events: torch.Tensor,
    target: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    reconstruction = model.unet(events)
    dav2_input = reconstruction
    if dav2_input.shape[1] == 1:
        dav2_input = dav2_input.repeat(1, 3, 1, 1)

    prediction = model.dav2(dav2_input).squeeze(1)
    if bool(config.get("inv_prediction", True)):
        constant = float(config.get("inv_prediction_constant", 1.0))
        prediction = 1.0 / (prediction + constant)

    clip_distance = float(config.get("clip_distance", 80.0))
    target = prepare_target_data_torch(target, clip_distance)
    if bool(config.get("use_scaleshift", True)):
        scale, shift = normalized_depth_scale_and_shift(
            prediction, target, target > 0
        )
        prediction = scale[:, None, None] * prediction + shift[:, None, None]

    prediction_np = np.clip(
        prediction.detach().cpu().squeeze().numpy(), 0.0, clip_distance
    )
    target_np = target.detach().cpu().squeeze().numpy()
    return reconstruction, prediction_np, target_np


def save_outputs(
    output_dir: Path,
    stem: str,
    events_rgb: np.ndarray,
    tencode_rgb: np.ndarray,
    reconstruction_rgb: np.ndarray,
    depth: np.ndarray,
    depth_title: str,
    clip_distance: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_dir / f"{stem}_events.png", events_rgb)
    plt.imsave(output_dir / f"{stem}_tencode.png", tencode_rgb)
    plt.imsave(output_dir / f"{stem}_unet_reconstruction.png", reconstruction_rgb)
    plt.imsave(
        output_dir / f"{stem}_depth.png",
        depth,
        cmap="viridis",
        vmin=0.0,
        vmax=clip_distance,
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    axes[0].imshow(events_rgb)
    axes[0].set_title("Events (+ blue / - red)")
    axes[1].imshow(tencode_rgb)
    axes[1].set_title("Tencode")
    axes[2].imshow(reconstruction_rgb)
    axes[2].set_title("U-Net reconstruction")
    depth_image = axes[3].imshow(
        depth, cmap="viridis", vmin=0.0, vmax=clip_distance
    )
    axes[3].set_title(depth_title)
    colorbar = fig.colorbar(depth_image, ax=axes[3], fraction=0.046, pad=0.04)
    colorbar.set_label("Depth (m)")
    for axis in axes:
        axis.axis("off")

    output_path = output_dir / f"{stem}_events_tencode_unet_depth.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    data_config = config["data_loader"]
    model_config = resolve_model_paths(config["model"], args.checkpoint)
    if str(model_config.get("model_type", "")).lower() != "unet_dav2":
        raise ValueError("The selected config must define model_type='unet_dav2'.")

    sequence = resolve_path(args.sequence)
    dataset = build_dataset(sequence, data_config)
    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"Sample index {args.index} is outside [0, {len(dataset) - 1}]."
        )
    sample = dataset[args.index]
    representation_config = data_config["event_representation"]
    tencode = load_tencode_sample(
        dataset,
        args.index,
        height=int(representation_config["height"]),
        width=int(representation_config["width"]),
    )

    device = select_device(args.device)
    model = fetch_model(model_config, device=device)
    model.eval()
    events = sample["depth_aligned_events"][0]
    target = sample["depth"][0].to(device)
    reconstruction, predicted_depth, target_depth = predict_depth(
        model, events.unsqueeze(0).to(device), target, config
    )

    if args.depth_source == "predicted":
        depth = predicted_depth
        depth_title = "Predicted depth"
    else:
        depth = target_depth
        depth_title = "Ground-truth depth"

    output_path = save_outputs(
        output_dir=resolve_path(args.output_dir),
        stem=f"{sequence.name}_{args.index:05d}",
        events_rgb=voxelgrid_to_red_blue(events),
        tencode_rgb=tencode_to_rgb(tencode),
        reconstruction_rgb=reconstruction_to_rgb(reconstruction),
        depth=depth,
        depth_title=depth_title,
        clip_distance=float(config.get("clip_distance", 80.0)),
    )
    print(f"Device: {device}")
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
