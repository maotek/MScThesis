import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from models.dav2.depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
    # vits/vitb/vitl checkpoints live in models/dav2/checkpoints/
}


class Dav2Wrapper(torch.nn.Module):
    """Thin wrapper around DepthAnythingV2 for easy loading/inference.

    Args:
        encoder: one of vits/vitb/vitl.
        checkpoint: optional checkpoint path; defaults to models/dav2/checkpoints/depth_anything_v2_<encoder>.pth
        device: torch device or None to auto-select cuda/mps/cpu.
        input_size: square resize applied before forwarding into the model.
    """

    def __init__(
        self,
        encoder: str = "vitb",
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
        input_size: int = 518,
    ) -> None:
        super().__init__()
        assert encoder in MODEL_CONFIGS, f"Unknown encoder {encoder}"
        self.encoder = encoder
        self.input_size = input_size
        self.device = self._select_device(device)

        ckpt = checkpoint or os.path.join(
            Path(__file__).resolve().parent.parent,
            "models",
            "dav2",
            "checkpoints",
            f"depth_anything_v2_{encoder}.pth",
        )
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        cfg = MODEL_CONFIGS[encoder]
        self.model = DepthAnythingV2(encoder=encoder, **cfg)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _select_device(device: Optional[torch.device]) -> torch.device:
        if device is not None:
            return device
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B,3,H,W) tensor in [0,1].
        Returns:
            depth: (B,1,H,W) resized back to input spatial size.
        """
        assert x.dim() == 4 and x.shape[1] == 3, "Expected x of shape (B,3,H,W)"
        orig_hw = x.shape[-2:]
        x_resized = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        depth = self.model(x_resized)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.shape[1] != 1:
            depth = depth[:, :1]
        depth = F.interpolate(depth, size=orig_hw, mode="bilinear", align_corners=False)
        return depth

    def to_device(self, device: torch.device) -> "Dav2Wrapper":
        self.device = device
        self.model.to(device)
        return self
