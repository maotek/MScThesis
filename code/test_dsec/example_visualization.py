import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.MVSEC.mvsec_dataset import fetch_dataloader as fetch_mvsec_dataloader
from datasets.events.events_representations import VoxelGrid
from datasets.utils import fetch_preprocessing
from evaluate import fetch_model
from evaluation import prepare_target_data_torch
from losses import normalized_depth_scale_and_shift


# Edit these constants to change the shown samples.
DSEC_SEQUENCE = "datasets/DSEC/data/validation/interlaken_00_g"
DSEC_SAMPLE_INDEX = 350
MVSEC_SEQUENCE = "test/outdoor_day1"
MVSEC_SAMPLE_INDEX = 1650
#1650 2700

DSEC_UNET_CONFIG = "configs/dsec/validation/unet_dav2_batch10.json"
DSEC_FULLY_CONV_CONFIG = "configs/dsec/validation/fully_conv_dav2_batch10.json"
MVSEC_UNET_CONFIG = "configs/mvsec/validation/train_mvsec_unet_dav2_batch10.json"
MVSEC_FULLY_CONV_CONFIG = "configs/mvsec/validation/train_mvsec_fully_conv_dav2_batch10.json"

OUTPUT_DIR = "test_dsec_output/example_visualization"
OUTPUT_NAME = "example_visualization.png"

TIME_WINDOW_MS = 50
NUM_BINS = 5
CLIP_DISTANCE = 80.0
ERROR_PERCENTILE = 95.0


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(BASE_DIR / p)


def load_config(path: str) -> dict:
    with open(resolve_path(path), "r") as f:
        return json.load(f)


def resolve_model_paths(model_config: dict) -> dict:
    model_config = dict(model_config)
    for key in ("dav2_checkpoint", "checkpoint_path"):
        if key in model_config:
            model_config[key] = resolve_path(str(model_config[key]))
    return model_config


def build_model(config_path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    config = load_config(config_path)
    model_config = resolve_model_paths(config["model"])
    model = fetch_model(model_config, device=device)
    model.eval()
    return model, config


def load_dsec_sample() -> dict:
    rep = VoxelGrid(
        channels=NUM_BINS,
        height=DSEC_HEIGHT,
        width=DSEC_WIDTH,
        normalize=True,
    )
    dataset = DsecSequence(
        sequence_path=resolve_path(DSEC_SEQUENCE),
        event_representation=rep,
        time_window_ms=TIME_WINDOW_MS,
        augmentator=fetch_preprocessing(
            [{"preprocessing_type": "CenterCrop", "height": 320, "width": 640}]
        ),
        load_images=True,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        self_supervised=False,
        postfix="",
    )
    return dataset[DSEC_SAMPLE_INDEX]


def load_mvsec_sample() -> dict:
    config = load_config(MVSEC_UNET_CONFIG)
    data_config = dict(config["data_loader"])
    data_config["datapath"] = resolve_path(str(data_config["datapath"]))
    data_config["load_images"] = True
    data_config["batch_size"] = 1
    data_config["num_workers"] = 0
    data_config["shuffle"] = False

    dataloaders = fetch_mvsec_dataloader(data_config, test=True)
    if MVSEC_SEQUENCE not in dataloaders:
        available = ", ".join(dataloaders.keys())
        raise KeyError(f"MVSEC sequence '{MVSEC_SEQUENCE}' not found. Available: {available}")
    dataset = dataloaders[MVSEC_SEQUENCE].dataset
    return dataset[MVSEC_SAMPLE_INDEX]


def rgb_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().float()
    if arr.dim() == 4:
        arr = arr.squeeze(0)
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    elif arr.shape[0] > 3:
        arr = arr[:3]
    arr = torch.clamp(arr, 0.0, 1.0)
    return arr.permute(1, 2, 0).numpy()


def scale_shift_depth(
    depth: torch.Tensor,
    target: torch.Tensor,
    clip_distance: float,
    use_scaleshift: bool,
) -> np.ndarray:
    pred = depth.squeeze(1)
    if use_scaleshift:
        scale, shift = normalized_depth_scale_and_shift(pred, target, target > 0)
        pred = scale[:, None, None] * pred + shift[:, None, None]
    pred_np = pred.detach().cpu().squeeze().numpy()
    return np.clip(pred_np, 0.0, clip_distance)


def masked_error_and_rmse(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    mask = target > 0
    error = np.full_like(target, np.nan, dtype=np.float32)
    diff = pred[mask] - target[mask]
    error[mask] = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
    return error, rmse


def model_representation(model: torch.nn.Module, events: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "unet"):
        return model.unet(events)
    if hasattr(model, "fully_conv"):
        return model.fully_conv(events)
    raise TypeError(f"Unsupported model type for representation: {type(model).__name__}")


def dav2_depth_from_representation(
    model: torch.nn.Module,
    representation: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    dav2_input = representation
    if dav2_input.shape[1] == 1:
        dav2_input = dav2_input.repeat(1, 3, 1, 1)

    depth = model.dav2(dav2_input)
    if bool(config.get("inv_prediction", True)):
        constant = float(config.get("inv_prediction_constant", 1.0))
        depth = 1.0 / (depth + constant)
    return depth


def run_model(
    model: torch.nn.Module,
    config: dict,
    events: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    clip_distance = float(config.get("clip_distance", CLIP_DISTANCE))
    target_proc = prepare_target_data_torch(target, clip_distance)
    target_np = target_proc.detach().cpu().squeeze().numpy()

    representation = model_representation(model, events)
    depth = dav2_depth_from_representation(model, representation, config)
    pred_np = scale_shift_depth(
        depth=depth,
        target=target_proc,
        clip_distance=clip_distance,
        use_scaleshift=bool(config.get("use_scaleshift", True)),
    )
    error, rmse = masked_error_and_rmse(pred_np, target_np)

    return {
        "representation": representation.detach().cpu().squeeze(0),
        "prediction": pred_np,
        "error": error,
        "rmse": rmse,
        "target": target_np,
    }


def build_row(
    sample: dict,
    unet_model: torch.nn.Module,
    unet_config: dict,
    fully_conv_model: torch.nn.Module,
    fully_conv_config: dict,
    device: torch.device,
) -> dict:
    events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)
    target = sample["depth"][0].to(device)

    unet = run_model(unet_model, unet_config, events, target)
    fully_conv = run_model(fully_conv_model, fully_conv_config, events, target)

    return {
        "rgb": sample["rgb"][0],
        "target": unet["target"],
        "unet_representation": unet["representation"],
        "unet_prediction": unet["prediction"],
        "unet_error": unet["error"],
        "unet_rmse": unet["rmse"],
        "fully_conv_representation": fully_conv["representation"],
        "fully_conv_prediction": fully_conv["prediction"],
        "fully_conv_error": fully_conv["error"],
        "fully_conv_rmse": fully_conv["rmse"],
    }


def finite_for_plot(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def add_colorbar(fig, ax, im) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.03)
    fig.colorbar(im, cax=cax)
    cax.tick_params(labelsize=8)


def save_figure(dsec_row: dict, mvsec_row: dict, output_path: str) -> None:
    rows = [("DSEC", dsec_row), ("MVSEC", mvsec_row)]
    error_vmax = max(
        float(np.nanmax(row["unet_error"]))
        for _, row in rows
    )
    error_vmax = max(
        error_vmax,
        *[
            float(np.nanmax(row["fully_conv_error"]))
            for _, row in rows
        ],
        1e-6,
    )

    panel_rows = [
        (
            "DSEC / U-Net",
            [
                (rgb_image(dsec_row["rgb"]), "RGB", None, None, None),
                (rgb_image(dsec_row["unet_representation"]), "Representation", None, None, None),
                (dsec_row["unet_prediction"], "Prediction", "viridis", 0.0, CLIP_DISTANCE),
                (
                    dsec_row["unet_error"],
                    f"RMSE {dsec_row['unet_rmse']:.2f} m",
                    "turbo",
                    0.0,
                    error_vmax,
                ),
            ],
        ),
        (
            "DSEC / FullyConv",
            [
                (dsec_row["target"], "GT LiDAR", "viridis", 0.0, CLIP_DISTANCE),
                (rgb_image(dsec_row["fully_conv_representation"]), "Representation", None, None, None),
                (dsec_row["fully_conv_prediction"], "Prediction", "viridis", 0.0, CLIP_DISTANCE),
                (
                    dsec_row["fully_conv_error"],
                    f"RMSE {dsec_row['fully_conv_rmse']:.2f} m",
                    "turbo",
                    0.0,
                    error_vmax,
                ),
            ],
        ),
        (
            "MVSEC / U-Net",
            [
                (rgb_image(mvsec_row["rgb"]), "RGB", None, None, None),
                (rgb_image(mvsec_row["unet_representation"]), "Representation", None, None, None),
                (mvsec_row["unet_prediction"], "Prediction", "viridis", 0.0, CLIP_DISTANCE),
                (
                    mvsec_row["unet_error"],
                    f"RMSE {mvsec_row['unet_rmse']:.2f} m",
                    "turbo",
                    0.0,
                    error_vmax,
                ),
            ],
        ),
        (
            "MVSEC / FullyConv",
            [
                (mvsec_row["target"], "GT LiDAR", "viridis", 0.0, CLIP_DISTANCE),
                (rgb_image(mvsec_row["fully_conv_representation"]), "Representation", None, None, None),
                (mvsec_row["fully_conv_prediction"], "Prediction", "viridis", 0.0, CLIP_DISTANCE),
                (
                    mvsec_row["fully_conv_error"],
                    f"RMSE {mvsec_row['fully_conv_rmse']:.2f} m",
                    "turbo",
                    0.0,
                    error_vmax,
                ),
            ],
        ),
    ]

    fig_w = 15.0
    fig_h = 10.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    n_rows = len(panel_rows)
    n_cols = 4
    left = 0.06
    right = 0.975
    bottom = 0.045
    top = 0.99
    col_gap = 0.02
    row_gap = 0.005
    title_h = 0.038
    slot_w = (right - left - (n_cols - 1) * col_gap) / n_cols

    def image_aspect(image) -> float:
        height, width = np.asarray(image).shape[:2]
        return height / float(width)

    row_panel_heights = [
        slot_w * fig_w / fig_h * image_aspect(panels[0][0])
        for _, panels in panel_rows
    ]

    def panel_position(
        row_top: float,
        row_panel_h: float,
        col_idx: int,
    ) -> tuple[float, float, float, float]:
        x = left + col_idx * (slot_w + col_gap)
        y = row_top - title_h - row_panel_h
        return x, y, slot_w, row_panel_h

    row_top = top
    for row_idx, (row_label, panels) in enumerate(panel_rows):
        panel_h = row_panel_heights[row_idx]
        row_x, row_y, _, _ = panel_position(row_top, panel_h, 0)
        fig.text(
            row_x - 0.038,
            row_y + panel_h / 2.0,
            row_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        for col_idx, (image, title, cmap, vmin, vmax) in enumerate(panels):
            x, y, width, height = panel_position(row_top, panel_h, col_idx)
            ax = fig.add_axes([x, y, width, height])
            ax.axis("off")
            if cmap is None:
                ax.imshow(image)
            else:
                im = ax.imshow(finite_for_plot(image), cmap=cmap, vmin=vmin, vmax=vmax)
                add_colorbar(fig, ax, im)
            fig.text(
                x + width / 2.0,
                y + height + 0.008,
                title,
                ha="center",
                va="bottom",
                fontsize=10,
                linespacing=0.95,
            )
        row_top = row_y - row_gap

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    output_dir = Path(resolve_path(OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / OUTPUT_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dsec_unet, dsec_unet_config = build_model(DSEC_UNET_CONFIG, device)
    dsec_fully_conv, dsec_fully_conv_config = build_model(DSEC_FULLY_CONV_CONFIG, device)
    mvsec_unet, mvsec_unet_config = build_model(MVSEC_UNET_CONFIG, device)
    mvsec_fully_conv, mvsec_fully_conv_config = build_model(MVSEC_FULLY_CONV_CONFIG, device)

    dsec_sample = load_dsec_sample()
    mvsec_sample = load_mvsec_sample()

    dsec_row = build_row(
        sample=dsec_sample,
        unet_model=dsec_unet,
        unet_config=dsec_unet_config,
        fully_conv_model=dsec_fully_conv,
        fully_conv_config=dsec_fully_conv_config,
        device=device,
    )
    mvsec_row = build_row(
        sample=mvsec_sample,
        unet_model=mvsec_unet,
        unet_config=mvsec_unet_config,
        fully_conv_model=mvsec_fully_conv,
        fully_conv_config=mvsec_fully_conv_config,
        device=device,
    )

    save_figure(dsec_row, mvsec_row, output_path)
    print(f"Saved example visualization to {output_path}")


if __name__ == "__main__":
    main()
