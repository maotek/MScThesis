import os
from typing import Any, Dict, Optional

import wandb


def init_training_wandb(
    args: Any,
    data_loader_config: Dict[str, object],
    model_config: Dict[str, object],
    project: Optional[str] = None,
    entity: str = "maoshengj-tu-delft",
) -> None:
    config_name = os.path.splitext(os.path.basename(args.config_path))[0]

    wandb.init(
        entity=entity,
        project="MScThesis",
        name=config_name,
        config={
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "clip_distance": args.clip_distance,
            "ssi_alpha": args.ssi_alpha,
            "ssi_scales": args.ssi_scales,
            "ssi_reduction": args.ssi_reduction,
            "grad_start_scale": args.grad_start_scale,
            "grad_num_scales": args.grad_num_scales,
            "grad_weight": args.grad_weight,
            "config_path": args.config_path,
            "save_dir": args.save_dir,
            "model_config": model_config,
            "data_loader_config": data_loader_config,
        },
    )


def log_train_step(loss: float, loss_ssi: float, loss_grad: float, epoch: int) -> None:
    wandb.log(
        {
            "train/loss": loss,
            "train/loss_ssi": loss_ssi,
            "train/loss_grad": loss_grad,
            "train/epoch": epoch,
        }
    )


def log_train_epoch(avg_loss: float, epoch: int) -> None:
    wandb.log(
        {
            "train/epoch_avg_loss": avg_loss,
            "train/epoch": epoch,
        }
    )


def log_validation_epoch(metrics: Dict[str, float], epoch: int) -> None:
    payload = {
        "validation/loss": metrics["loss"],
        "validation/loss_ssi": metrics["loss_ssi"],
        "validation/loss_grad": metrics["loss_grad"],
        "validation/_abs_rel_diff": metrics.get("_abs_rel_diff", 0.0),
        "validation/_RMS_linear": metrics.get("_RMS_linear", 0.0),
        "validation/_threshold_delta_1.25": metrics.get("_threshold_delta_1.25", 0.0),
        "validation/frames": metrics["frames"],
        "validation/epoch": epoch,
    }
    wandb.log(payload)


def finish_training_wandb() -> None:
    wandb.finish()
