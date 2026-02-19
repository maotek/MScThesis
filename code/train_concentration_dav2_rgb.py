import argparse
import json
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import tqdm

from util import save_depth_colormap, save_rgb

from datasets.DSEC.dsec_dataset import fetch_dataloader as fetch_dsec_dataloader
from evaluation import prepare_target_data_torch
from losses import MultiScaleGradient, ScaleAndShiftInvariantLoss
from networks.concentration_dav2 import ConcentrationDav2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConcentrationDav2 on DSEC.")
    parser.add_argument(
        "--config-path",
        required=True,
        type=str,
        help="JSON config with data_loader and model sections.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--clip-distance", type=float, default=80.0, help="Max depth value (meters).")
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between loss logs.")
    parser.add_argument("--save-dir", type=str, default="output/train_concentration_dav2_rgb", help="Checkpoint output dir.")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        # default=os.path.join("output", "train_concentration_dav2", "epoch_050.pt"),
        help="Checkpoint to resume from; set to empty string to disable.",
    )
    parser.add_argument("--ssi-alpha", type=float, default=0.0, help="Scale-and-shift loss alpha term.")
    parser.add_argument("--ssi-scales", type=int, default=4, help="Scales for scale-and-shift loss.")
    parser.add_argument(
        "--ssi-reduction",
        type=str,
        default="batch",
        choices=["batch", "image"],
        help="Reduction type for scale-and-shift loss.",
    )
    parser.add_argument("--grad-start-scale", type=int, default=1, help="MultiScaleGradient start scale.")
    parser.add_argument("--grad-num-scales", type=int, default=4, help="MultiScaleGradient number of scales.")
    parser.add_argument("--grad-weight", type=float, default=1.0, help="MultiScaleGradient weight.")
    return parser.parse_args()


def load_config(config_path: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    with open(config_path, "r") as f:
        config = json.load(f)

    if "data_loader" not in config or "model" not in config:
        raise KeyError("Config must contain top-level 'data_loader' and 'model'")

    data_loader_config = dict(config["data_loader"])
    model_config = dict(config["model"])
    return data_loader_config, model_config


def setup_device_and_seeds(seed: int) -> torch.device:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    return device


def build_model(model_config: Dict[str, object], device: torch.device) -> ConcentrationDav2:
    return ConcentrationDav2(
        input_channels=int(model_config.get("input_channels", 3)),
        concentrator_base_channels=int(model_config.get("concentrator_base_channels", 32)),
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
    concentrator_rgb: torch.Tensor,
    depth: torch.Tensor,
) -> None:
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    rgb_path = os.path.join(
        vis_dir, f"{seq_name}_epoch_{epoch:03d}_step_{step:06d}_concentrator.png"
    )
    depth_path = os.path.join(
        vis_dir, f"{seq_name}_epoch_{epoch:03d}_step_{step:06d}_depth.png"
    )

    save_rgb(rgb_path, concentrator_rgb.detach().cpu().squeeze(0))
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
) -> float:
    model.train()
    running_loss = 0.0
    step = 0

    for seq_name, data_loader in dataloaders.items():
        for batch_idx, sample in enumerate(tqdm.tqdm(data_loader, desc=f"Epoch {epoch} {seq_name}")):
            target_depth_t = sample["depth"][:, 0, 0].to(device)  # (B,H,W)
            events = sample["rgb"][:, 0].to(device)  # (B,C,H,W)

            pred_depth = model(events)  # (B,1,H,W)
            pred_depth = 1.0 / (pred_depth + 1.0)

            target_proc_t = prepare_target_data_torch(target_depth_t, clip_distance)
            valid_mask = (target_proc_t > 0) & (~torch.isnan(target_proc_t))

            if valid_mask.sum() == 0:
                continue

            loss_ssi = ssi_loss(pred_depth, target_proc_t, valid_mask)
            loss_grad = grad_loss(pred_depth, target_proc_t.unsqueeze(1), valid_mask.unsqueeze(1))
            loss = loss_ssi + loss_grad

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 500 == 0 and model.vis_temp is not None:
                save_visualization(
                    save_dir=save_dir,
                    seq_name=seq_name,
                    epoch=epoch,
                    step=step,
                    concentrator_rgb=model.vis_temp,
                    depth=pred_depth,
                )
            step += 1

            if log_interval > 0 and step % log_interval == 0:
                avg_loss = running_loss / float(step)
                print(
                    f"Epoch {epoch} | step {step} | loss {avg_loss:.6f} | "
                    f"ssi {loss_ssi.item():.6f} | grad {loss_grad.item():.6f}"
                )

    return running_loss / float(max(step, 1))


def main() -> None:
    args = parse_args()
    device = setup_device_and_seeds(args.seed)

    data_loader_config, model_config = load_config(args.config_path)

    dataloaders = fetch_dsec_dataloader(data_loader_config, test=False)
    model = build_model(model_config, device)

    ssi_loss = ScaleAndShiftInvariantLoss(
        alpha=args.ssi_alpha,
        scales=args.ssi_scales,
        reduction_type=args.ssi_reduction,
        weight=1.0,
    ).to(device)
    grad_loss = MultiScaleGradient(
        start_scale=args.grad_start_scale,
        num_scales=args.grad_num_scales,
        weight=args.grad_weight,
    ).to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    if args.resume_checkpoint:
        resume_path = args.resume_checkpoint
        if os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            concentrator_state = {
                k.replace("concentrator.", ""): v
                for k, v in state.items()
                if k.startswith("concentrator.")
            }
            model.concentrator.load_state_dict(concentrator_state, strict=True)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1
            print(f"Resumed from {resume_path} at epoch {start_epoch}")
        else:
            print(f"Resume checkpoint not found: {resume_path}. Starting from scratch.")

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        avg_loss = train_epoch(
            epoch=epoch,
            dataloaders=dataloaders,
            model=model,
            device=device,
            clip_distance=args.clip_distance,
            ssi_loss=ssi_loss,
            grad_loss=grad_loss,
            optimizer=optimizer,
            log_interval=args.log_interval,
            save_dir=args.save_dir,
        )
        print(f"Epoch {epoch} complete | avg loss {avg_loss:.6f}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(args.save_dir, epoch, model, optimizer)


if __name__ == "__main__":
    main()
