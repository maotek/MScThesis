import argparse
import json
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import tqdm

from util import save_depth_colormap, save_rgb
from train_validation import validate_epoch
from wandb_logging import (
    finish_training_wandb,
    init_training_wandb,
    log_train_epoch,
    log_train_step,
    log_validation_epoch,
)

from datasets.DSEC.dsec_dataset import fetch_dataloader as fetch_dsec_dataloader
from datasets.MVSEC.mvsec_dataset import fetch_dataloader as fetch_mvsec_dataloader
from evaluation import prepare_target_data_torch
from losses import MultiScaleGradient, ScaleAndShiftInvariantLoss
from networks.unet_dav2 import UNetDav2
from networks.fully_conv_dav2 import FullyConvDav2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNetDav2 on DSEC.")
    parser.add_argument(
        "--config-path",
        required=True,
        type=str,
        help="JSON config with data_loader and model sections.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_config(config_path: str) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    with open(config_path, "r") as f:
        config = json.load(f)

    if "data_loader" not in config or "model" not in config or "training" not in config:
        raise KeyError("Config must contain top-level 'data_loader', 'model', and 'training'")

    data_loader_config = dict(config["data_loader"])
    model_config = dict(config["model"])
    training_config = dict(config["training"])
    return data_loader_config, model_config, training_config


def setup_device_and_seeds(seed: int) -> torch.device:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    return device


def build_model(model_config: Dict[str, object], device: torch.device) -> UNetDav2:
    if str(model_config.get("model_type", "")).lower() == "fully_conv_dav2":
        return FullyConvDav2(
            input_channels=int(model_config.get("input_channels", 5)),
            dav2_encoder=str(model_config.get("dav2_encoder", "vits")),
            dav2_checkpoint=model_config.get("dav2_checkpoint", os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vits.pth")),
            input_size_width=int(model_config.get("input_size_width", 350)),
            input_size_height=int(model_config.get("input_size_height", 266)),
            freeze_dav2=bool(model_config.get("freeze_dav2", True)),
            device=device,
        )
    elif str(model_config.get("model_type", "")).lower() == "unet_dav2_rgb":
        return UNetDav2(
            input_channels=int(model_config.get("input_channels", 3)),
            unet_base_channels=int(model_config.get("unet_base_channels", 32)),
            unet_type=str(model_config.get("unet_type", "small")),
            dav2_encoder=str(model_config.get("dav2_encoder", "vits")),
            dav2_checkpoint=model_config.get("dav2_checkpoint", os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vits.pth")),
            input_size_width=int(model_config.get("input_size_width", 350)),
            input_size_height=int(model_config.get("input_size_height", 266)),
            freeze_dav2=bool(model_config.get("freeze_dav2", True)),
            device=device,
        )
    elif str(model_config.get("model_type", "")).lower() == "unet_dav2":
        return UNetDav2(
            input_channels=int(model_config.get("input_channels", 5)),
            unet_base_channels=int(model_config.get("unet_base_channels", 32)),
            unet_type=str(model_config.get("unet_type", "small")),
            dav2_encoder=str(model_config.get("dav2_encoder", "vits")),
            dav2_checkpoint=model_config.get("dav2_checkpoint", os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vits.pth")),
            input_size_width=int(model_config.get("input_size_width", 350)),
            input_size_height=int(model_config.get("input_size_height", 266)),
            freeze_dav2=bool(model_config.get("freeze_dav2", True)),
            device=device,
        )


def save_checkpoint(save_dir: str, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def save_visualization(
    save_dir: str,
    seq_name: str,
    epoch: int,
    step: int,
    unet_rgb: torch.Tensor,
    depth: torch.Tensor,
) -> None:
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    rgb_path = os.path.join(
        vis_dir, f"{seq_name}_epoch_{epoch:03d}_step_{step:06d}_unet.png"
    )
    depth_path = os.path.join(
        vis_dir, f"{seq_name}_epoch_{epoch:03d}_step_{step:06d}_depth.png"
    )

    if unet_rgb.shape[0] > 1:
        # If batch dimension exists, take the first sample for visualization.
        save_rgb(rgb_path, unet_rgb[0].detach().cpu().squeeze(0))
        save_depth_colormap(depth_path, depth[0].detach().cpu().squeeze(0))
    else:
        save_rgb(rgb_path, unet_rgb.detach().cpu().squeeze(0))
        save_depth_colormap(depth_path, depth.detach().cpu().squeeze(0))


def train_epoch(
    epoch: int,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    model: torch.nn.Module,
    device: torch.device,
    clip_distance: float,
    ssi_loss: ScaleAndShiftInvariantLoss,
    grad_loss: MultiScaleGradient,
    optimizer: torch.optim.Optimizer,
    log_interval: int,
    save_dir: str,
    delta: float,
    model_type: str,
    inv_prediction: bool,
    inv_prediction_constant: float,
    samples_seen: int,
) -> Tuple[float, int]:
    model.train()
    running_loss = 0.0
    step = 0

    for seq_name, data_loader in dataloaders.items():
        for batch_idx, sample in enumerate(tqdm.tqdm(data_loader, desc=f"Epoch {epoch} {seq_name}")):
            target_depth_t = sample["depth"][:, 0, 0].to(device)  # (B,H,W)
            if model_type == "unet_dav2_rgb":
                events = sample["rgb"][:, 0].to(device)  # (B,C,H,W)
            else:
                events = sample["depth_aligned_events"][:, 0].to(device)  # (B,C,H,W)

            pred_depth = model(events)  # (B,1,H,W)
            if inv_prediction:
                pred_depth = 1.0 / (pred_depth + inv_prediction_constant)

            target_proc_t = prepare_target_data_torch(target_depth_t, clip_distance)
            valid_mask = (target_proc_t > 0) & (~torch.isnan(target_proc_t))

            if valid_mask.sum() == 0:
                continue

            loss_ssi = ssi_loss(pred_depth, target_proc_t, valid_mask)
            if delta != 0.0:
                loss_grad = grad_loss(pred_depth, target_proc_t.unsqueeze(1), valid_mask.unsqueeze(1))
                loss_grad_value = loss_grad.item()
                loss = loss_ssi + delta * loss_grad
            else:
                loss_grad_value = 0.0
                loss = loss_ssi

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            samples_seen += target_depth_t.shape[0]
            log_train_step(
                loss=loss.item(),
                loss_ssi=loss_ssi.item(),
                loss_grad=loss_grad_value,
                epoch=epoch,
                samples_seen=samples_seen,
            )

            if step % 500 == 0 and model.vis_temp is not None:
                save_visualization(
                    save_dir=save_dir,
                    seq_name=seq_name,
                    epoch=epoch,
                    step=step,
                    unet_rgb=model.vis_temp,
                    depth=pred_depth,
                )
            step += 1

            if log_interval > 0 and step % log_interval == 0:
                avg_loss = running_loss / float(step)
                print(
                    f"Epoch {epoch} | step {step} | loss {avg_loss:.6f} | "
                    f"ssi {loss_ssi.item():.6f} | grad {loss_grad_value:.6f}"
                )

    return running_loss / float(max(step, 1)), samples_seen


def main() -> None:
    args = parse_args()
    device = setup_device_and_seeds(args.seed)

    data_loader_config, model_config, training_config = load_config(args.config_path)
    model_type = str(model_config.get("model_type", "")).lower()

    dataset_name = str(data_loader_config.get("dataset", "")).lower()
    if dataset_name == "dsec":
        dataloaders = fetch_dsec_dataloader(data_loader_config, test=False)
    elif dataset_name == "mvsec":
        dataloaders = fetch_mvsec_dataloader(data_loader_config, test=False)
    else:
        raise ValueError(f"Unsupported dataset in config: {dataset_name}")
    
    model = build_model(model_config, device)

    ssi_loss = ScaleAndShiftInvariantLoss(
        alpha=float(training_config.get("ssi_alpha", 0.0)),
        scales=int(training_config.get("ssi_scales", 4)),
        reduction_type=str(training_config.get("ssi_reduction", "batch")),
        weight=1.0,
    ).to(device)
    grad_loss = MultiScaleGradient(
        start_scale=int(training_config.get("grad_start_scale", 1)),
        num_scales=int(training_config.get("grad_num_scales", 4)),
        weight=float(training_config.get("grad_weight", 1.0)),
    ).to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(training_config.get("lr", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    init_training_wandb(
        args=args,
        data_loader_config=data_loader_config,
        model_config=model_config,
        training_config=training_config,
        name=str(training_config.get("wandb_name")),
    )

    start_epoch = 1
    end_epoch = start_epoch + int(training_config.get("epochs", 50)) - 1
    samples_seen = 0

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            avg_loss, samples_seen = train_epoch(
                epoch=epoch,
                dataloaders=dataloaders,
                model=model,
                device=device,
                clip_distance=float(training_config.get("clip_distance", 80.0)),
                ssi_loss=ssi_loss,
                grad_loss=grad_loss,
                optimizer=optimizer,
                log_interval=int(training_config.get("log_interval", 100)),
                save_dir=str(training_config.get("save_dir", "output/train_unet_dav2")),
                delta=float(training_config.get("delta", 0.25)),
                model_type=model_type,
                inv_prediction=bool(training_config.get("inv_prediction", True)),
                inv_prediction_constant=float(training_config.get("inv_prediction_constant", 1.0)),
                samples_seen=samples_seen,
            )
            print(f"Epoch {epoch} complete | avg loss {avg_loss:.6f}")
            log_train_epoch(avg_loss=avg_loss, epoch=epoch)
            if bool(training_config.get("no_validation", False)):
                print(f"Epoch {epoch} validation skipped (--no-validation)")
            else:
                val_metrics = validate_epoch(
                    model=model,
                    dataset_path=str(data_loader_config["datapath"]),
                    data_loader_config=data_loader_config,
                    device=device,
                    clip_distance=float(training_config.get("clip_distance", 80.0)),
                    ssi_loss=ssi_loss,
                    grad_loss=grad_loss,
                    input_key="rgb" if model_type == "unet_dav2_rgb" else "depth_aligned_events",
                    delta=float(training_config.get("delta", 0.25)),
                    inv_prediction=bool(training_config.get("inv_prediction", True)),
                    inv_prediction_constant=float(training_config.get("inv_prediction_constant", 1.0)),
                )
                print(
                    f"Epoch {epoch} validation | loss {val_metrics['loss']:.6f} | "
                    f"abs_rel {val_metrics['_abs_rel_diff']:.6f}"
                )
                log_validation_epoch(metrics=val_metrics, epoch=epoch)

            save_every = int(training_config.get("save_every", 1))
            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(str(training_config.get("save_dir", "output/train_unet_dav2")), epoch, model, optimizer)
    finally:
        finish_training_wandb()


if __name__ == "__main__":
    main()
