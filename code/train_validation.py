from typing import Dict

import numpy as np
import torch
import tqdm

from datasets.DSEC.dsec_dataset import fetch_dataloader as fetch_dsec_dataloader
from evaluation import add_to_metrics, prepare_target_data_torch


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    dataset_path: str,
    data_loader_config: Dict[str, object],
    device: torch.device,
    clip_distance: float,
    ssi_loss: torch.nn.Module,
    grad_loss: torch.nn.Module,
    input_key: str,
) -> Dict[str, float]:
    if str(data_loader_config.get("dataset", "")).lower() != "dsec":
        raise ValueError("validate_epoch currently supports only the DSEC dataset.")

    validation_config = dict(data_loader_config)
    validation_config["datapath"] = dataset_path
    validation_config["split"] = "validation"
    validation_config["shuffle"] = False

    dataloaders = fetch_dsec_dataloader(validation_config, test=True)

    model.eval()
    total_loss = 0.0
    total_loss_ssi = 0.0
    total_loss_grad = 0.0
    total_batches = 0
    metrics_sum: Dict[str, float] = {}
    total_frames = 0

    for seq_name, data_loader in dataloaders.items():
        for sample in tqdm.tqdm(data_loader, desc=f"Validation {seq_name}", leave=False):
            target_depth_t = sample["depth"][:, 0, 0].to(device)
            events = sample[input_key][:, 0].to(device)

            pred_depth = model(events)
            pred_depth = 1.0 / (pred_depth + 1.0)

            target_proc_t = prepare_target_data_torch(target_depth_t, clip_distance)
            valid_mask = (target_proc_t > 0) & (~torch.isnan(target_proc_t))
            if valid_mask.sum() == 0:
                continue

            loss_ssi = ssi_loss(pred_depth, target_proc_t, valid_mask)
            loss_grad = grad_loss(pred_depth, target_proc_t.unsqueeze(1), valid_mask.unsqueeze(1))
            loss = loss_ssi + loss_grad

            total_loss += loss.item()
            total_loss_ssi += loss_ssi.item()
            total_loss_grad += loss_grad.item()
            total_batches += 1
            total_frames += target_depth_t.shape[0]

            pred_np = np.clip(pred_depth.detach().cpu().squeeze(1).numpy(), 0, clip_distance)
            target_np = target_proc_t.detach().cpu().numpy()

            for batch_idx in range(target_np.shape[0]):
                metrics_sum = add_to_metrics(
                    batch_idx,
                    metrics_sum,
                    target_np[batch_idx],
                    pred_np[batch_idx],
                    np.ones_like(target_np[batch_idx], dtype=bool),
                    event_frame=None,
                    prefix="_",
                    debug=False,
                    output_folder=None,
                )

    if total_batches == 0:
        return {
            "loss": 0.0,
            "loss_ssi": 0.0,
            "loss_grad": 0.0,
            "abs_rel_diff": 0.0,
            "frames": 0.0,
        }

    return {
        "loss": total_loss / total_batches,
        "loss_ssi": total_loss_ssi / total_batches,
        "loss_grad": total_loss_grad / total_batches,
        "abs_rel_diff": metrics_sum["_abs_rel_diff"] / total_frames if total_frames > 0 else 0.0,
        "frames": float(total_frames),
    }
