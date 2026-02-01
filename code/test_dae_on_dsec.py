import argparse
import os
from pathlib import Path
from pprint import pprint

import numpy as np
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events import Tencode
from networks.dae_wrapper import DAE
from evaluation import prepare_target_data_torch, prepare_target_data
from losses import normalized_depth_scale_and_shift
from util import depth_to_colormap, save_image, save_rgb
from evaluation import add_to_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Depth AnyEvent DAv2 on a DSEC sequence and visualize depth.")
    parser.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default="datasets/DSEC/data/validate/interlaken_00_f",
        help="Path to a DSEC sequence root",
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Index within the sequence to visualize (depth-aligned events)."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Which DAE encoder to use.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to models/depthanyevent/checkpoints/finetuned_dsec.pth",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "softplus"],
        help="Output activation for the depth head.",
    )
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Scale factor applied to the prediction.")
    parser.add_argument(
        "--inv-prediction",
        action="store_true",
        default=True,
        help="Invert depth prediction (matches Depth AnyEvent configs).",
    )
    parser.add_argument(
        "--time-window-ms",
        type=int,
        default=50,
        help="Event window size for building tencode representations.",
    )
    parser.add_argument("--output-dir", type=str, default="output/test_dae_on_dsec", help="Where to save visualizations.")
    return parser.parse_args()


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def visualize(sample: dict, pred_np: np.ndarray, pred_np_raw: np.ndarray, target_np: np.ndarray, out_dir: str, idx: int) -> None:
    events = sample["depth_aligned_events"][0]
    events_rgb = events

    pred_rgb = depth_to_colormap(pred_np)
    pred_raw_rgb = depth_to_colormap(pred_np_raw)
    gt_rgb = depth_to_colormap(target_np)

    # Overlay: blend scaled prediction and ground truth
    overlay_rgb = 0.5 * pred_rgb + 0.5 * gt_rgb

    save_rgb(os.path.join(out_dir, f"{idx:05d}_events.png"), events_rgb)
    save_image(os.path.join(out_dir, f"{idx:05d}_pred_raw_depth.png"), pred_raw_rgb)
    save_image(os.path.join(out_dir, f"{idx:05d}_pred_scaled_depth.png"), pred_rgb)
    save_image(os.path.join(out_dir, f"{idx:05d}_gt_depth.png"), gt_rgb)
    save_image(os.path.join(out_dir, f"{idx:05d}_overlay.png"), overlay_rgb.astype(np.uint8))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rep = Tencode(height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True, white_frame=False)
    dataset = DsecSequence(
        sequence_path=args.sequence,
        event_representation=rep,
        time_window_ms=args.time_window_ms, # 50 ms
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

    model = DAE(
        encoder=args.encoder,
        checkpoint=args.checkpoint,
        device=device,
        input_size=518,
        activation="relu",
        scale_factor=1.0,
        inv_prediction=True,
    )

    events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)
    depth_pred = model(events)

    # Apply scale-shift normalization to match ground truth
    target_depth_t = sample["depth"][0].to(device)
    target_proc_t = prepare_target_data_torch(target_depth_t, 80.0)
    scale, shift = normalized_depth_scale_and_shift(
        depth_pred.squeeze(1), target_proc_t, target_proc_t > 0
    )
    pred_depth_scaled = scale[:, None, None] * depth_pred.squeeze(1) + shift[:, None, None]
    pred_np = np.clip(pred_depth_scaled.detach().cpu().squeeze().numpy(), 0, 80.0)
    pred_np_raw = np.clip(depth_pred.squeeze().detach().cpu().numpy(), 0, 80.0)
    target_np = prepare_target_data(target_proc_t.detach().cpu().squeeze().numpy(), 80.0)

    mask = np.ones_like(target_np, dtype=bool)

    metrics_sum = add_to_metrics(
        0,
        {},
        target_np,
        pred_np,
        mask,
        event_frame=None,
        prefix="_",
        debug=False,
        output_folder=None,
    )

    metrics_filtered = {
        k: v
        for k, v in metrics_sum.items()
        if not k.startswith(("_10_", "_20_", "_30_"))
    }
    pprint(metrics_filtered)

    out_dir = ensure_dir(args.output_dir)
    visualize(sample, pred_np, pred_np_raw, target_np, out_dir, args.index)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
