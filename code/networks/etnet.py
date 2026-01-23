import sys
from pathlib import Path
from typing import Optional

import torch

# Add ET-Net repo to path to allow absolute imports used by the original codebase.
_ETNET_ROOT = Path(__file__).resolve().parents[1] / "models" / "etnet"
if str(_ETNET_ROOT) not in sys.path:
    # Prepend so ET-Net's local modules (e.g., logger) shadow similarly named pip packages.
    sys.path.insert(0, str(_ETNET_ROOT))

from parse_config import ConfigParser  # type: ignore
from model import model as model_arch  # type: ignore
from utils.henri_compatible import make_henri_compatible  # type: ignore
from utils.util import minmax_normalization  # type: ignore

def _default_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_config(checkpoint: dict) -> dict:
    """Ensure checkpoints have a ConfigParser config."""
    if isinstance(checkpoint.get("config"), ConfigParser):
        return checkpoint
    return make_henri_compatible(checkpoint)


class ETNet(torch.nn.Module):
    """Thin wrapper around the ET-Net model for event-to-intensity reconstruction.

    Expects voxel-grid event tensors shaped (B, C, H, W). Returns intensity
    reconstructions in [0,1] shaped (B, 1, H, W). Maintains recurrent state
    internally; call `reset_state()` between sequences.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_minmax_norm: bool = False,
    ) -> None:
        super().__init__()
        self.device = _default_device(device)
        self.use_minmax_norm = use_minmax_norm

        default_ckpt = _ETNET_ROOT / "checkpoints" / "etnet.pth"
        self.checkpoint_path = str(Path(checkpoint_path) if checkpoint_path else default_ckpt)

        checkpoint = self._load_checkpoint(self.checkpoint_path)
        checkpoint = _ensure_config(checkpoint)
        config = checkpoint["config"]

        model = config.init_obj("arch", model_arch)
        model.load_state_dict(checkpoint["state_dict"])

        self.model = model.to(self.device)
        self.model.eval()
        self.reset_state()

        print(f"Loading ET-Net checkpoint from {self.checkpoint_path}")
        print("Device:", self.device)

    @staticmethod
    def _load_checkpoint(path: str) -> dict:
        """Load ET-Net checkpoint handling PyTorch 2.6 weights_only default."""
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # weights_only not supported (older torch); fallback to classic load
            return torch.load(path, map_location="cpu")

    def reset_state(self) -> None:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    @torch.no_grad()
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        if events.dim() == 3:
            events = events.unsqueeze(0)
        assert events.dim() == 4, "events must be (B,C,H,W)"

        events = events.to(self.device)
        output = self.model(events)
        image = output["image"] if isinstance(output, dict) else output

        if self.use_minmax_norm:
            image = minmax_normalization(image, image.device)

        if image.dim() == 3:
            image = image.unsqueeze(1)
        return image


def load_etnet(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_minmax_norm: bool = False,
) -> ETNet:
    return ETNet(checkpoint_path=checkpoint_path, device=device, use_minmax_norm=use_minmax_norm)
