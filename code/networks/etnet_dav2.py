from typing import Optional

import torch

from networks.etnet import ETNet
from networks.dav2_wrapper import Dav2


class ETNetDav2(torch.nn.Module):
    """Pipeline: events -> ET-Net intensity -> DAV2 depth.

    Expects voxel-grid events (B,C,H,W) matching ET-Net input.
    ET-Net outputs (B,1,H,W) or (B,3,H,W); we ensure 3 channels and feed DAV2 for depth.
    """

    def __init__(
        self,
        etnet_checkpoint: Optional[str] = None,
        dav2_encoder: str = "vits",
        dav2_checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = self._select_device(device)
        self.etnet = ETNet(checkpoint_path=etnet_checkpoint, device=self.device)
        self.dav2 = Dav2(encoder=dav2_encoder, checkpoint=dav2_checkpoint, device=self.device)
        self.to(self.device)

    @staticmethod
    def _select_device(device: Optional[torch.device]) -> torch.device:
        if device is not None:
            return device
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def reset_state(self) -> None:
        self.etnet.reset_state()

    @torch.no_grad()
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        # events: (B,C,H,W)
        intensity = self.etnet(events)  # (B,1,H,W) or (B,3,H,W)
        if intensity.shape[1] == 1:
            intensity_3ch = intensity.repeat(1, 3, 1, 1)  # (B,3,H,W)
        else:
            intensity_3ch = intensity
        depth = self.dav2(intensity_3ch)
        return depth