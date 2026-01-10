# Compare depth predictions from base and finetuned DAV2 models on Tencode dataset

import argparse
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

from datasets.tencode_dataset import TencodeDataset
from models.dav2.depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: str, encoder: str, device: torch.device, dtype: torch.dtype) -> DepthAnythingV2:
    cfg = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(encoder=encoder, features=cfg["features"], out_channels=cfg["out_channels"])
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    if dtype == torch.float16:
        model = model.half()
    return model


def predict_depth_from_ckpt(
    checkpoint: str,
    encoder: str,
    x: torch.Tensor,
    input_size: int,
    device: torch.device,
    use_half: bool,
) -> np.ndarray:
    dtype = torch.float16 if (use_half and device.type == "cuda") else torch.float32
    model = load_model(checkpoint, encoder, device, dtype)
    with torch.inference_mode():
        x_in = F.interpolate(x.unsqueeze(0).to(device, dtype=dtype), size=(input_size, input_size), mode="bilinear", align_corners=False)
        pred = model(x_in)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        elif pred.dim() == 4 and pred.shape[1] != 1:
            pred = pred[:, :1]
        pred = F.interpolate(pred, size=x.shape[-2:], mode="bilinear", align_corners=False)
        depth = pred[0, 0].detach().cpu().float().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def colorize(depth: np.ndarray) -> np.ndarray:
    cmap = matplotlib.colormaps.get_cmap("magma")
    depth_vis = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
    depth_vis = depth_vis[:, :, ::-1]  # RGB -> BGR for cv2
    return depth_vis


def tencode_to_bgr(x: torch.Tensor) -> np.ndarray:
    """Convert CHW float[0,1] tensor to uint8 BGR image for cv2."""
    x_np = (x.cpu().permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Compare base vs finetuned DAV2 depth on Tencode")
    parser.add_argument("--data-root", default="datasets/DSEC/data/validate")
    parser.add_argument("--scene", default="interlaken_00_g")
    parser.add_argument("--idx", type=int, default=99)
    parser.add_argument("--time-window-us", type=int, default=50000)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--encoder", choices=list(MODEL_CONFIGS.keys()), default="vitb")
    parser.add_argument("--base-ckpt", default=None, help="Base DAV2 checkpoint; defaults to encoder's pretrained")
    parser.add_argument("--finetuned-ckpt", default="output/dav2_tencode_finetuned.pth")
    parser.add_argument("--use-half", action="store_true", help="Inference in float16 (CUDA only)")
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    dataset = TencodeDataset(
        data_root=args.data_root,
        time_window_us=args.time_window_us,
        shape=(480, 640),
        scenes=[args.scene],
    )
    x, _ = dataset[args.idx]
    input_bgr = tencode_to_bgr(x)

    sample = dataset.samples[args.idx]
    disp_ts = int(sample["timestamp"])
    scene_dir = Path(args.data_root) / args.scene
    img_ts_path = scene_dir / "image_timestamps.txt"
    if img_ts_path.exists():
        img_timestamps = np.loadtxt(img_ts_path, dtype=np.int64)
        img_idx = int(np.argmin(np.abs(img_timestamps - disp_ts)))
        img_path = scene_dir / f"{args.scene}_images_rectified_left" / f"{img_idx:06d}.png"
        rgb_img = cv2.imread(str(img_path))
        if rgb_img is not None:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (input_bgr.shape[1], input_bgr.shape[0]))
        else:
            rgb_img = np.zeros_like(input_bgr)
    else:
        rgb_img = np.zeros_like(input_bgr)

    if args.base_ckpt:
        base_ckpt = args.base_ckpt
    else:
        base_ckpt = f"models/dav2/checkpoints/depth_anything_v2_{args.encoder}.pth"
    finetuned_ckpt = args.finetuned_ckpt

    base_depth = predict_depth_from_ckpt(base_ckpt, args.encoder, x, args.input_size, device, args.use_half)
    ft_depth = predict_depth_from_ckpt(finetuned_ckpt, args.encoder, x, args.input_size, device, args.use_half)

    base_vis = colorize(base_depth)
    ft_vis = colorize(ft_depth)

    title_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    def add_title(img: np.ndarray, text: str) -> np.ndarray:
        canvas = np.ones((title_height, img.shape[1], 3), dtype=np.uint8) * 255
        ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, thickness)
        x_off = (img.shape[1] - text_w) // 2
        y_off = (title_height + text_h) // 2
        cv2.putText(canvas, text, (x_off, y_off), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return np.vstack([canvas, img])

    rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    input_t = add_title(input_bgr, "Input (Tencode)")
    rgb_t = add_title(rgb_bgr, "RGB (nearest)")
    base_t = add_title(base_vis, f"Base {args.encoder.upper()}")
    ft_t = add_title(ft_vis, "Finetuned (DAV2)")

    max_h = max(input_t.shape[0], rgb_t.shape[0], base_t.shape[0], ft_t.shape[0])
    def pad_to_h(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == max_h:
            return img
        pad_h = max_h - img.shape[0]
        return np.vstack([img, np.ones((pad_h, img.shape[1], 3), dtype=np.uint8) * 255])

    input_t = pad_to_h(input_t)
    rgb_t = pad_to_h(rgb_t)
    base_t = pad_to_h(base_t)
    ft_t = pad_to_h(ft_t)


    spacer = np.ones((max_h, 20, 3), dtype=np.uint8) * 255

    # Build two rows (2x2): top = [Input | RGB], bottom = [Base | Finetuned]
    top = cv2.hconcat([input_t, spacer, rgb_t])
    bottom = cv2.hconcat([base_t, spacer, ft_t])

    # Pad rows to same width if needed
    max_w = max(top.shape[1], bottom.shape[1])
    def pad_to_w(img: np.ndarray) -> np.ndarray:
        if img.shape[1] == max_w:
            return img
        pad_w = max_w - img.shape[1]
        return np.hstack([img, np.ones((img.shape[0], pad_w, 3), dtype=np.uint8) * 255])

    top = pad_to_w(top)
    bottom = pad_to_w(bottom)

    # vertical spacer between rows
    vspacer = np.ones((20, max_w, 3), dtype=np.uint8) * 255
    combined = cv2.vconcat([top, vspacer, bottom])

    out_dir = "output/compare"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"compare_{args.scene}_{args.time_window_us}_{args.idx}.png")
    cv2.imwrite(out_path, combined)
    print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()
