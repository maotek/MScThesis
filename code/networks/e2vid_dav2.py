from typing import Optional

import torch

from networks.e2vid_wrapper import E2VIDWrapper
from networks.dav2_wrapper import Dav2Wrapper


class E2VIDDav2(torch.nn.Module):
    """Pipeline: events -> E2VID intensity -> DAV2 depth.

    Expects voxel-grid events (B,C,H,W) matching E2VID num_bins (default 5 for lightweight checkpoint).
    E2VID outputs (B,1,H,W); we replicate to 3 channels and feed DAV2 for depth.
    """

    def __init__(
        self,
        e2vid_weights: Optional[str] = None,
        dav2_encoder: str = "vitb",
        dav2_checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.e2vid = E2VIDWrapper(weights_path=e2vid_weights, device=self.device)
        self.dav2 = Dav2Wrapper(encoder=dav2_encoder, checkpoint=dav2_checkpoint, device=self.device)
        self.to(self.device)

    def reset_state(self) -> None:
        self.e2vid.reset_state()

    @torch.no_grad()
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        # events: (B,C,H,W)
        intensity = self.e2vid(events)  # (B,1,H,W)
        intensity_3ch = intensity.repeat(1, 3, 1, 1)  # (B,3,H,W)
        depth = self.dav2(intensity_3ch)
        return depth
