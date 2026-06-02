import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events import Tencode
from datasets.events.events_representations import VoxelGrid
from datasets.utils import fetch_preprocessing
from evaluation import prepare_target_data_torch
from losses import normalized_depth_scale_and_shift
from networks.dae import DAE
from networks.dav2 import Dav2
from networks.fully_conv import FullyConv
from networks.unet_dav2 import UNetDav2


BASE_DIR = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(BASE_DIR / p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 2x5 teaser visualization for one DSEC sample."
    )
    parser.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default="datasets/DSEC/data/validation/interlaken_00_g",
        help="Path to a DSEC sequence root. Default matches test_dsec/dav2.py.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=350,
        help="Index within the sequence.",
    )
    parser.add_argument("--time-window-ms", type=int, default=50)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--clip-distance", type=float, default=80.0)
    parser.add_argument("--inv-prediction-constant", type=float, default=1.0)
    parser.add_argument(
        "--dav2-checkpoint",
        type=str,
        default="models/dav2/checkpoints/depth_anything_v2_vits.pth",
    )
    parser.add_argument(
        "--dae-checkpoint",
        type=str,
        default="models/depthanyevent/weights/dav2/finetuned_dsec/finetuned_dsec.pth",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=str,
        default="train_output/train_dsec_unet_dav2_batch10/epoch_050.pt",
    )
    parser.add_argument(
        "--fully-conv-checkpoint",
        type=str,
        default="train_output/train_dsec_fully_conv_dav2_batch10/epoch_050.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_dsec_output/teaser_visualization",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def tensor_bytes(state: dict) -> int:
    return sum(v.numel() * v.element_size() for v in state.values() if torch.is_tensor(v))


def extract_unet_state(state: dict, model: UNetDav2) -> dict:
    if any(k.startswith("unet.") for k in state):
        return {
            k.replace("unet.", "", 1): v
            for k, v in state.items()
            if k.startswith("unet.")
        }
    if any(k.startswith("concentrator.") for k in state):
        return {
            k.replace("concentrator.", "", 1): v
            for k, v in state.items()
            if k.startswith("concentrator.")
        }

    expected = set(model.unet.state_dict().keys())
    raw_unet_state = {k: v for k, v in state.items() if k in expected}
    if raw_unet_state:
        return raw_unet_state

    sample_keys = ", ".join(list(state.keys())[:10])
    raise RuntimeError(f"Could not find UNet weights in checkpoint. First keys: {sample_keys}")


def extract_fully_conv_state(state: dict, model: FullyConv) -> dict:
    if any(k.startswith("fully_conv.") for k in state):
        return {
            k.replace("fully_conv.", "", 1): v
            for k, v in state.items()
            if k.startswith("fully_conv.")
        }

    expected = set(model.state_dict().keys())
    raw_state = {k: v for k, v in state.items() if k in expected}
    if raw_state:
        return raw_state

    sample_keys = ", ".join(list(state.keys())[:10])
    raise RuntimeError(
        f"Could not find FullyConv weights in checkpoint. First keys: {sample_keys}"
    )


def load_unet(args: argparse.Namespace, device: torch.device) -> UNetDav2:
    model = UNetDav2(
        input_channels=args.num_bins,
        dav2_encoder="vits",
        dav2_checkpoint=args.dav2_checkpoint,
        device=device,
        normalize_imagenet=False,
    )
    ckpt = torch.load(args.unet_checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("dav2.") for k in state):
        model.load_state_dict(state, strict=True)
    else:
        model.unet.load_state_dict(extract_unet_state(state, model), strict=True)
    model.eval()
    print(f"Loaded UNet checkpoint: {args.unet_checkpoint}")
    print(f"UNet checkpoint tensor size: {tensor_bytes(state) / (1024 * 1024):.2f} MB")
    return model


def load_fully_conv(args: argparse.Namespace, device: torch.device) -> FullyConv:
    model = FullyConv(in_channels=args.num_bins).to(device)
    ckpt = torch.load(args.fully_conv_checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(extract_fully_conv_state(state, model), strict=True)
    model.eval()
    print(f"Loaded FullyConv checkpoint: {args.fully_conv_checkpoint}")
    print(f"FullyConv checkpoint tensor size: {tensor_bytes(state) / (1024 * 1024):.2f} MB")
    return model


def load_dav2(args: argparse.Namespace, device: torch.device) -> Dav2:
    model = Dav2(
        encoder="vits",
        checkpoint=args.dav2_checkpoint,
        device=device,
        input_size_width=350,
        input_size_height=266,
        normalize_imagenet=False,
    )
    model.eval()
    return model


def load_dae(args: argparse.Namespace, device: torch.device) -> DAE:
    model = DAE(
        encoder="vits",
        checkpoint=args.dae_checkpoint,
        device=device,
        input_size_width=350,
        input_size_height=266,
        activation="relu",
        scale_factor=1.0,
        inv_prediction=True,
        input_channels=3,
        nopretrain=False,
    )
    model.eval()
    return model


def build_sample(sequence: str, rep, args: argparse.Namespace, load_images: bool = False) -> dict:
    preprocess_config = [
        {"preprocessing_type": "CenterCrop", "height": 320, "width": 640}
    ]
    dataset = DsecSequence(
        sequence_path=sequence,
        event_representation=rep,
        time_window_ms=args.time_window_ms,
        augmentator=fetch_preprocessing(preprocess_config),
        load_images=load_images,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        self_supervised=False,
        postfix="",
    )
    assert 0 <= args.index < len(dataset), (
        f"Index {args.index} out of range for sequence of length {len(dataset)}"
    )
    return dataset[args.index]


def scale_shift_depth(depth: torch.Tensor, target: torch.Tensor, clip_distance: float) -> np.ndarray:
    depth_raw = depth.squeeze(1)
    scale, shift = normalized_depth_scale_and_shift(depth_raw, target, target > 0)
    scaled = scale * depth + shift
    return np.clip(scaled.detach().cpu().squeeze().numpy(), 0.0, clip_distance)


def masked_error_and_rmse(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    mask = target > 0
    error = np.full_like(target, np.nan, dtype=np.float32)
    diff = pred[mask] - target[mask]
    error[mask] = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
    return error, rmse


def rgb_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().float()
    arr = torch.clamp(arr, 0.0, 1.0).permute(1, 2, 0).numpy()
    return arr


def finite_for_plot(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)


def add_panel(
    fig,
    left: float,
    bottom: float,
    width: float,
    height: float,
    image,
    title: str,
    cmap=None,
    vmin=None,
    vmax=None,
    cbar: bool = False,
    cbar_pad: float = 0.0,
    cbar_width: float = 0.0,
):
    ax = fig.add_axes([left, bottom, width, height])
    if cmap is None:
        im = ax.imshow(image)
    else:
        im = ax.imshow(
            finite_for_plot(image, fill_value=vmin or 0.0),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    fig.text(
        left + width / 2.0,
        bottom + height + 0.012,
        title,
        ha="center",
        va="bottom",
        fontsize=14,
        linespacing=0.9,
    )
    ax.axis("off")
    if cbar:
        cax = fig.add_axes([left + width + cbar_pad, bottom, cbar_width, height])
        plt.colorbar(im, cax=cax)
        cax.tick_params(labelsize=7)


def save_teaser(
    out_path: str,
    rgb: torch.Tensor,
    dae_pred: np.ndarray,
    unet_recon: torch.Tensor,
    unet_pred: np.ndarray,
    fc_recon: torch.Tensor,
    fc_pred: np.ndarray,
    target: np.ndarray,
    clip_distance: float,
) -> None:
    dae_error, dae_rmse = masked_error_and_rmse(dae_pred, target)
    unet_error, unet_rmse = masked_error_and_rmse(unet_pred, target)
    fc_error, fc_rmse = masked_error_and_rmse(fc_pred, target)

    error_vmax = max(
        float(np.nanmax(dae_error)),
        float(np.nanmax(unet_error)),
        float(np.nanmax(fc_error)),
        1e-6,
    )

    image_h, image_w = target.shape
    n_cols = 5
    fig_w = 18.5
    margin_x = 0.08
    margin_y = 0.05
    title_h = 0.28
    col_gap = 0.06
    row_gap = 0.06
    cbar_w = 0.07
    cbar_pad = 0.03
    cbar_label_w = 0.25
    cbar_space_w = cbar_w + cbar_pad + cbar_label_w
    cbar_cols = n_cols - 1
    panel_w = (
        fig_w - 2 * margin_x - (n_cols - 1) * col_gap - cbar_cols * cbar_space_w
    ) / n_cols
    panel_h = panel_w * image_h / image_w
    fig_h = 2 * panel_h + 2 * title_h + row_gap + 2 * margin_y
    fig = plt.figure(figsize=(fig_w, fig_h))

    def panel_position(row: int, col: int) -> tuple[float, float, float, float]:
        top_row_bottom = margin_y + panel_h + title_h + row_gap
        bottom = top_row_bottom if row == 0 else margin_y
        left = margin_x + col * (panel_w + col_gap) + min(col, cbar_cols) * cbar_space_w
        return left / fig_w, bottom / fig_h, panel_w / fig_w, panel_h / fig_h

    add_panel(
        fig,
        *panel_position(0, 0),
        rgb_image(rgb),
        "RGB",
    )
    add_panel(
        fig,
        *panel_position(1, 0),
        target,
        "GT LiDAR",
        cmap="viridis",
        vmin=0.0,
        vmax=clip_distance,
        cbar=True,
        cbar_pad=cbar_pad / fig_w,
        cbar_width=cbar_w / fig_w,
    )
    add_panel(
        fig,
        *panel_position(0, 1),
        dae_pred,
        "DAE prediction",
        cmap="viridis",
        vmin=0.0,
        vmax=clip_distance,
        cbar=True,
        cbar_pad=cbar_pad / fig_w,
        cbar_width=cbar_w / fig_w,
    )
    add_panel(
        fig,
        *panel_position(0, 2),
        unet_pred,
        "U-Net prediction",
        cmap="viridis",
        vmin=0.0,
        vmax=clip_distance,
        cbar=True,
        cbar_pad=cbar_pad / fig_w,
        cbar_width=cbar_w / fig_w,
    )
    add_panel(
        fig,
        *panel_position(0, 3),
        fc_pred,
        "Fully Convolutional prediction",
        cmap="viridis",
        vmin=0.0,
        vmax=clip_distance,
        cbar=True,
        cbar_pad=cbar_pad / fig_w,
        cbar_width=cbar_w / fig_w,
    )

    add_panel(
        fig,
        *panel_position(1, 1),
        dae_error,
        f"DAE RMSE {dae_rmse:.2f} m",
        cmap="turbo",
        vmin=0.0,
        vmax=error_vmax,
        cbar=True,
        cbar_pad=cbar_pad / fig_w,
        cbar_width=cbar_w / fig_w,
    )
    add_panel(
        fig,
        *panel_position(1, 2),
        unet_error,
        f"U-Net RMSE {unet_rmse:.2f} m",
        cmap="turbo",
        vmin=0.0,
        vmax=error_vmax,
        cbar=True,
        cbar_pad=cbar_pad / fig_w,
        cbar_width=cbar_w / fig_w,
    )
    add_panel(
        fig,
        *panel_position(1, 3),
        fc_error,
        f"Fully Convolutional RMSE {fc_rmse:.2f} m",
        cmap="turbo",
        vmin=0.0,
        vmax=error_vmax,
        cbar=True,
        cbar_pad=cbar_pad / fig_w,
        cbar_width=cbar_w / fig_w,
    )
    add_panel(fig, *panel_position(0, 4), rgb_image(unet_recon), "U-Net reconstruction")
    add_panel(fig, *panel_position(1, 4), rgb_image(fc_recon), "Fully Convolutional reconstruction")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.sequence = resolve_path(args.sequence)
    args.dav2_checkpoint = resolve_path(args.dav2_checkpoint)
    args.dae_checkpoint = resolve_path(args.dae_checkpoint)
    args.unet_checkpoint = resolve_path(args.unet_checkpoint)
    args.fully_conv_checkpoint = resolve_path(args.fully_conv_checkpoint)
    args.output_dir = ensure_dir(resolve_path(args.output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tencode_rep = Tencode(height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True, white_frame=False)
    voxel_rep = VoxelGrid(
        channels=args.num_bins,
        height=DSEC_HEIGHT,
        width=DSEC_WIDTH,
        normalize=True,
    )
    sample_tencode = build_sample(args.sequence, tencode_rep, args)
    sample_voxel = build_sample(args.sequence, voxel_rep, args, load_images=True)

    target_depth_t = sample_voxel["depth"][0].to(device)
    target_proc_t = prepare_target_data_torch(target_depth_t, args.clip_distance)
    target_np = target_proc_t.detach().cpu().squeeze().numpy()

    dae_model = load_dae(args, device)
    unet_model = load_unet(args, device)
    fc_model = load_fully_conv(args, device)
    dav2_model = load_dav2(args, device)

    events_tencode = sample_tencode["depth_aligned_events"][0].unsqueeze(0).to(device)
    events_voxel = sample_voxel["depth_aligned_events"][0].unsqueeze(0).to(device)

    dae_depth = dae_model(events_tencode)
    dae_np = scale_shift_depth(dae_depth, target_proc_t, args.clip_distance)

    unet_recon = unet_model.unet(events_voxel)
    unet_depth = unet_model.dav2(unet_recon)
    unet_depth = 1.0 / (unet_depth + args.inv_prediction_constant)
    unet_np = scale_shift_depth(unet_depth, target_proc_t, args.clip_distance)

    fc_recon = fc_model(events_voxel)
    fc_depth = dav2_model(fc_recon)
    fc_depth = 1.0 / (fc_depth + args.inv_prediction_constant)
    fc_np = scale_shift_depth(fc_depth, target_proc_t, args.clip_distance)

    out_path = os.path.join(args.output_dir, f"{args.index:05d}_teaser.png")
    save_teaser(
        out_path=out_path,
        rgb=sample_voxel["rgb"][0],
        dae_pred=dae_np,
        unet_recon=unet_recon[0].detach().cpu(),
        unet_pred=unet_np,
        fc_recon=fc_recon[0].detach().cpu(),
        fc_pred=fc_np,
        target=target_np,
        clip_distance=args.clip_distance,
    )

    print(f"Saved teaser visualization to {out_path}")


if __name__ == "__main__":
    main()
