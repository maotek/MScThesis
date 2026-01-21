from typing import Optional, Tuple

import torch

from networks.e2vid_wrapper import E2VIDWrapper
from networks.dav2_wrapper import Dav2Wrapper


class E2VIDDav2Composite(torch.nn.Module):
    """Pipeline: voxel-grid events -> E2VID intensity + per-pixel recency/counts -> DAV2 depth.

    - Red channel: E2VID intensity (grayscale reconstruction).
    - Green channel: recency like Tencode (1 - normalized last-bin index).
    - Blue channel: per-pixel event counts normalized to 0..1 within the frame.
    """

    def __init__(
        self,
        e2vid_weights: Optional[str] = None,
        dav2_encoder: str = "vits",
        dav2_checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = self._select_device(device)
        self.e2vid = E2VIDWrapper(weights_path=e2vid_weights, device=self.device)
        self.dav2 = Dav2Wrapper(encoder=dav2_encoder, checkpoint=dav2_checkpoint, device=self.device)
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
        self.e2vid.reset_state()

    @torch.no_grad()
    def forward(self, events: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # events: (B,C,H,W) voxel grid
        intensity = self.e2vid(events)  # (B,1,H,W)
        # Normalize intensity per-frame to 0..1
        imin = intensity.amin(dim=(2, 3), keepdim=True)
        imax = intensity.amax(dim=(2, 3), keepdim=True)
        scale = (imax - imin).clamp(min=1e-8)
        intensity = (intensity - imin) / scale

        b, c, h, w = events.shape
        device = events.device

        # Green: recency proxy from last non-zero bin index per pixel
        bin_indices = torch.arange(c, device=device, dtype=torch.float).view(1, c, 1, 1)
        mask = (events.abs() > 0).float()
        last_bin = (mask * bin_indices).amax(dim=1)  # (B,H,W)
        denom = max(c - 1, 1)
        green = 1.0 - (last_bin / denom)
        green = green.clamp(0.0, 1.0).unsqueeze(1)

        # Blue: counts normalized per-frame to 0..1 over nonzero pixels
        counts = events.abs().sum(dim=1)  # (B,H,W)
        blue = torch.zeros_like(counts)
        nz_mask = counts > 0
        if nz_mask.any():
            nz = counts[nz_mask]
            cmin = nz.min()
            cmax = nz.max()
            if cmax > cmin:
                scaled = (nz - cmin) / (cmax - cmin)
            else:
                scaled = torch.ones_like(nz)
            blue[nz_mask] = scaled
        blue = blue.unsqueeze(1)

        # blue = 0 * blue  # Disable blue channel for now
        # intensity = 0 * intensity  # Disable red channel for now

        composite = torch.cat([intensity, green, intensity], dim=1)
        depth = self.dav2(composite)
        return depth, composite


class E2VIDDav2Composite2(torch.nn.Module):
    """Pipeline: voxel-grid events -> E2VID intensity -> Tencode-like RGB -> DAV2 depth.

    - Event pixels use a Tencode-style encoding (red for positive, blue for negative, green for recency).
    - Pixels without events are filled with the normalized E2VID intensity replicated to 3 channels (grayscale).
    """

    def __init__(
        self,
        e2vid_weights: Optional[str] = None,
        dav2_encoder: str = "vits",
        dav2_checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = self._select_device(device)
        self.e2vid = E2VIDWrapper(weights_path=e2vid_weights, device=self.device)
        self.dav2 = Dav2Wrapper(encoder=dav2_encoder, checkpoint=dav2_checkpoint, device=self.device)
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
        self.e2vid.reset_state()

    @torch.no_grad()
    def forward(self, events: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # events: (B,C,H,W) voxel grid ordered from oldest->newest along C
        intensity = self.e2vid(events)  # (B,1,H,W)

        # Normalize intensity per-frame to 0..1
        imin = intensity.amin(dim=(2, 3), keepdim=True)
        imax = intensity.amax(dim=(2, 3), keepdim=True)
        scale = (imax - imin).clamp(min=1e-8)
        intensity = (intensity - imin) / scale

        # Use E2VID intensity as base, but make any pixel with events pure black
        # (no relative normalization). Pixels without events keep the intensity.
        counts = events.abs().sum(dim=1)  # (B,H,W)
        overlay = torch.zeros_like(intensity)
        any_event = counts > 0  # (B,H,W)
        if any_event.any():
            mask = any_event.unsqueeze(1).expand_as(overlay)
            overlay = overlay.clone()
            overlay[mask] = 1.0

        # Blend only pixels that are black in `overlay`; for other pixels use `intensity`.
        blend = 0.2 * overlay + 0.8 * intensity
        mask_black = (overlay == 1.0)
        merged = torch.where(mask_black, blend, intensity)
        tencode = merged.repeat(1, 3, 1, 1)

        depth = self.dav2(tencode)
        return depth, tencode
