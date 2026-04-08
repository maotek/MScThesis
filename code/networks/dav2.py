import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import cv2
import numpy as np

from models.dav2.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

from models.dav2.depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
    # vits/vitb/vitl checkpoints live in models/dav2/checkpoints/
}


class Dav2(torch.nn.Module):
    """Thin wrapper around DepthAnythingV2 for easy loading/inference.

    Args:
        encoder: one of vits/vitb/vitl.
        checkpoint: optional checkpoint path; defaults to models/dav2/checkpoints/depth_anything_v2_<encoder>.pth
        device: torch device or None to auto-select cuda/mps/cpu.
        input_size: square resize applied before forwarding into the model.
    """

    def __init__(
        self,
        encoder: str = "vits",
        checkpoint: str = None,
        device: torch.device = None,
        input_size_height: int = 266,
        input_size_width: int = 350,
        normalize_imagenet: bool = False,
        use_torch_preprocess: bool = True,
    ) -> None:
        super().__init__()
        assert encoder in MODEL_CONFIGS, f"Unknown encoder {encoder}"
        self.encoder = encoder
        self.input_size_height = input_size_height
        self.input_size_width = input_size_width
        self.device = device
        self.normalize_imagenet = normalize_imagenet
        self.use_torch_preprocess = bool(use_torch_preprocess)

        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print("[DAv2] Loading DepthAnythingV2 checkpoint from:", checkpoint)
        print("[DAv2] Using encoder:", encoder)
        print("[DAv2] Input size (H,W):", (self.input_size_height, self.input_size_width))
        print("[DAv2] Device:", self.device)
        print("[DAv2] Normalize ImageNet:", self.normalize_imagenet)

        cfg = MODEL_CONFIGS[encoder]
        self.model = DepthAnythingV2(encoder=encoder, **cfg)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B,3,H,W) tensor in [0,1].
        Returns:
            depth: (B,1,H,W) resized back to input spatial size.
        """
        assert x.dim() == 4 and x.shape[1] == 3, "Expected x of shape (B,3,H,W)"

        normalize_imagenet = bool(self.normalize_imagenet)
        if self.use_torch_preprocess:
            return self.infer_image_torch(x, normalize_imagenet=normalize_imagenet)
        # if x.requires_grad:
        #     return self.infer_image_torch(x, normalize_imagenet=normalize_imagenet)
        return self.infer_image(x, normalize_imagenet=normalize_imagenet)

    def infer_image(self, x: torch.Tensor, normalize_imagenet: bool = True) -> torch.Tensor:
        """Inference path matching DepthAnythingV2 infer_image.

        Uses aspect-ratio preserving resize and ImageNet normalization as implemented
        in the original DepthAnythingV2 repo.
        """
        assert x.dim() == 4 and x.shape[1] == 3, "Expected x of shape (B,3,H,W)"
        orig_hw = x.shape[-2:]

        transform = [
            Resize(
                width=self.input_size_width,
                height=self.input_size_height,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
        ]
        if normalize_imagenet:
            transform.append(NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform.append(PrepareForNet())

        images = []
        for img in x:
            img_np = img.detach().cpu().permute(1, 2, 0).numpy()
            # img_np needs to be in [0,1] float and in RGB, which it already is.
            sample = {"image": img_np}
            for t in transform:
                sample = t(sample)

            image_t = torch.from_numpy(sample["image"]).unsqueeze(0)
            # print(image_t.shape)
            images.append(image_t)

        image_batch = torch.cat(images, dim=0).to(self.device)

        depth = self.model(image_batch).unsqueeze(1) # (B,1,H',W') / (B,1,266,532)

        # print(depth.shape)

        depth = F.interpolate(depth, size=orig_hw, mode="bilinear", align_corners=True)

        return depth

    def infer_image_torch(self, x: torch.Tensor, normalize_imagenet: bool = True) -> torch.Tensor:
        """Torch-only inference path that preserves gradients."""
        assert x.dim() == 4 and x.shape[1] == 3, "Expected x of shape (B,3,H,W)"
        orig_hw = x.shape[-2:]

        h, w = orig_hw
        scale = max(self.input_size_height / float(h), self.input_size_width / float(w))
        resized_h = int(np.ceil((h * scale) / 14.0) * 14)
        resized_w = int(np.ceil((w * scale) / 14.0) * 14)

        x = F.interpolate(x, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
        if normalize_imagenet:
            x = (x - self.imagenet_mean) / self.imagenet_std

        depth = self.model(x).unsqueeze(1)
        depth = F.interpolate(depth, size=orig_hw, mode="bilinear", align_corners=True)
        return depth
