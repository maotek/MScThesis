import os
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from datasets.tencode_dataset import TencodeDataset
from models.dav2.depth_anything_v2.dpt import DepthAnythingV2
from losses import sparse_si_loss

MODEL_CONFIGS = {
    "vits": {
        "checkpoint": "models/dav2/checkpoints/depth_anything_v2_vits.pth",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "checkpoint": "models/dav2/checkpoints/depth_anything_v2_vitb.pth",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "checkpoint": "models/dav2/checkpoints/depth_anything_v2_vitl.pth",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}

CONFIG = {
    "data_root": "datasets/DSEC/data/train",
    "scenes": None,  # None => all scenes
    "batch_size": 4,
    "epochs": 100,
    "lr": 1e-5,
    "weight_decay": 1e-2,
    "num_workers": 0,

    "time_window_us": 50000,
    "height": 480,
    "width": 640,

    # Training resize (simple square resize). If you want aspect-ratio preserve + pad,
    # you can implement it, but keep it torch-only and apply to y/mask too.
    "input_size": 518,

    "encoder": "vitb",  # choices: vits, vitb, vitl
    "save_path": "output/dav2_tencode_finetuned.pth",

    # Sparse loss settings
    "eps": 1e-6,

    # Visualization
    "vis_dir": "output/train_vis_tencode",
    "vis_interval": 250,
}


def build_dataloader(
    data_root: str,
    batch_size: int,
    num_workers: int,
    time_window_us: int,
    shape: Tuple[int, int],
    scenes: Optional[Sequence[str]] = None,
) -> DataLoader:
    dataset = TencodeDataset(
        data_root=data_root,
        time_window_us=time_window_us,
        shape=shape,
        scenes=scenes,
    )
    print(dataset.__len__(), "samples in dataset")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def save_vis(
    x: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    out_path: str,
):
    """
    pred/target/mask are single-sample tensors:
      pred:   (1, H, W) or (H,W)
      target: (1, H, W) or (H,W)
      mask:   (1, H, W) or (H,W)
    """
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    if target.dim() == 3:
        target = target.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    m_np = mask.detach().cpu().numpy().astype(bool)

    x_np = x.detach().cpu().numpy()
    if x_np.ndim == 3:
        # CHW -> HWC for plotting
        x_np = np.transpose(x_np, (1, 2, 0))
    if x_np.ndim == 3 and x_np.shape[-1] == 1:
        x_np = x_np[..., 0]
    x_np = np.clip(x_np, 0.0, 1.0)

    # show target only where valid
    tgt_show = tgt_np.copy()
    tgt_show[~m_np] = 0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x_np, cmap=None if x_np.ndim == 3 else "gray")
    axes[0].set_title("tencode (x)")
    axes[0].axis("off")

    axes[1].imshow(tgt_show, cmap="magma")
    axes[1].set_title("target (sparse y)")
    axes[1].axis("off")

    axes[2].imshow(pred_np, cmap="magma")
    axes[2].set_title("pred (relative)")
    axes[2].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    input_size: int,
    eps: float,
    vis_dir: str,
    vis_interval: int,
    epoch_idx: int,
) -> float:
    model.train()
    total_loss = 0.0

    os.makedirs(vis_dir, exist_ok=True)

    for step, (x, y) in enumerate(loader):
        # x: (B,3,H,W) in [0,1]
        # y: (B,1,H,W) sparse (0 invalid)
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        mask = (y > 0)

        # Resize inputs to model train size
        x_in = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
        # print(x_in.shape)
        pred = model(x_in)
        # Normalize shape to (B,1,h,w)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        elif pred.dim() == 4 and pred.shape[1] != 1:
            pred = pred[:, :1]

        # Resize prediction back to target size
        pred = F.interpolate(pred, size=y.shape[-2:], mode="bilinear", align_corners=False)

        loss = sparse_si_loss(pred, y, mask, eps=eps)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

        if step % 1 == 0:
            # quick sanity: show target stats on valid pixels
            with torch.no_grad():
                if mask.any():
                    valid = y[mask]
                    print(
                        f"step {step:05d}/{len(loader)} "
                        f"loss={loss.item():.4f} "
                        f"target_valid[min={valid.min().item():.3f}, max={valid.max().item():.3f}] "
                        f"valid_px={mask.sum().item()}"
                    )
                else:
                    print(f"step {step:05d}/{len(loader)} loss={loss.item():.4f} valid_px=0")

        if vis_interval > 0 and (step + 1) % vis_interval == 0:
            # visualize first sample in batch
            out_path = os.path.join(vis_dir, f"epoch{epoch_idx+1:02d}_step_{step+1:06d}.png")
            save_vis(
                x=x[0],
                pred=pred[0, 0],
                target=y[0, 0],
                mask=mask[0, 0],
                out_path=out_path,
            )

    return total_loss / max(1, len(loader))


def run(config: Optional[dict] = None):
    cfg = {**CONFIG, **(config or {})}
    device = get_device()
    print("Device:", device)

    encoder = cfg["encoder"]
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"encoder must be one of {list(MODEL_CONFIGS.keys())}, got {encoder}")
    model_cfg = MODEL_CONFIGS[encoder]
    checkpoint = cfg.get("checkpoint") or model_cfg["checkpoint"]

    loader = build_dataloader(
        data_root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        time_window_us=cfg["time_window_us"],
        shape=(cfg["height"], cfg["width"]),
        scenes=cfg["scenes"],
    )

    model = DepthAnythingV2(
        encoder=encoder,
        features=model_cfg["features"],
        out_channels=model_cfg["out_channels"],
    )
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    save_dir = os.path.dirname(cfg["save_path"])
    os.makedirs(save_dir, exist_ok=True)
    base, ext = os.path.splitext(cfg["save_path"])

    for epoch in range(cfg["epochs"]):
        avg_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            input_size=cfg["input_size"],
            eps=cfg["eps"],
            vis_dir=cfg["vis_dir"],
            vis_interval=cfg["vis_interval"],
            epoch_idx=epoch,
        )
        print(f"Epoch {epoch+1}/{cfg['epochs']} done | loss={avg_loss:.4f}")

        epoch_path = f"{base}_epoch{epoch+1}{ext}"
        torch.save(model.state_dict(), epoch_path)
        print(f"Saved checkpoint: {epoch_path}")

    torch.save(model.state_dict(), cfg["save_path"])
    print(f"Saved finetuned checkpoint: {cfg['save_path']}")


if __name__ == "__main__":
    run()
