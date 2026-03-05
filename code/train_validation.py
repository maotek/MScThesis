from typing import Dict

import numpy as np
import torch
import tqdm

from datasets.DSEC.dsec_dataset import fetch_dataloader as fetch_dsec_dataloader
from evaluation import add_to_metrics, prepare_target_data_torch
from losses import normalized_depth_scale_and_shift


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
    validation_config["batch_size"] = 1

    dataloaders = fetch_dsec_dataloader(validation_config, test=True)

    model.eval()
    total_loss = 0.0
    total_loss_ssi = 0.0
    total_loss_grad = 0.0
    total_batches = 0
    total_frames = 0
    metrics_sequence_dict: Dict[str, Dict[str, float]] = {}

    for seq_name, data_loader in dataloaders.items():
        metrics_sum: Dict[str, float] = {}
        seq_frames = 0
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

            pred_depth_for_metrics = pred_depth.squeeze(1)
            scale, shift = normalized_depth_scale_and_shift(
                pred_depth_for_metrics,
                target_proc_t,
                target_proc_t > 0,
            )
            pred_depth_for_metrics = scale[:, None, None] * pred_depth_for_metrics + shift[:, None, None]

            pred_np = np.clip(pred_depth_for_metrics.detach().cpu().numpy(), 0, clip_distance)
            target_np = target_proc_t.detach().cpu().numpy()

            for batch_idx in range(pred_np.shape[0]):
                mask = np.ones_like(target_np[batch_idx], dtype=bool)
                metrics_sum = add_to_metrics(
                    batch_idx,
                    metrics_sum,
                    target_np[batch_idx],
                    pred_np[batch_idx],
                    mask,
                    event_frame=None,
                    prefix="_",
                    debug=False,
                    output_folder=None,
                )
                seq_frames += 1
                total_frames += 1

        if seq_frames > 0:
            metrics_sequence_dict[seq_name] = {
                k: v / seq_frames
                for k, v in metrics_sum.items()
                if not (k.startswith("_10_") or k.startswith("_20_") or k.startswith("_30_"))
            }

    if total_batches == 0:
        return {
            "loss": 0.0,
            "loss_ssi": 0.0,
            "loss_grad": 0.0,
            "_abs_rel_diff": 0.0,
            "_RMS_linear": 0.0,
            "_threshold_delta_1.25": 0.0,
            "frames": 0.0,
        }

    metrics_mean: Dict[str, list] = {}
    for seq in metrics_sequence_dict:
        for key, value in metrics_sequence_dict[seq].items():
            if key not in metrics_mean:
                metrics_mean[key] = []
            metrics_mean[key].append(value)

    overall_avg = {
        key: float(np.nanmean(np.array(values)))
        for key, values in metrics_mean.items()
    } if len(metrics_mean) > 0 else {}

    return {
        "loss": total_loss / total_batches,
        "loss_ssi": total_loss_ssi / total_batches,
        "loss_grad": total_loss_grad / total_batches,
        "_abs_rel_diff": overall_avg.get("_abs_rel_diff", 0.0),
        "_RMS_linear": overall_avg.get("_RMS_linear", 0.0),
        "_threshold_delta_1.25": overall_avg.get("_threshold_delta_1.25", 0.0),
        "frames": float(total_frames),
    }
