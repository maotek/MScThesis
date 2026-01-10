
import random
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
from datasets.tencode_dataset import TencodeDataset
from datasets.tencode_count_dataset import TencodeCountDataset
from datasets.time_surface_dataset import TimeSurfaceDataset
from models.dav2.depth_anything_v2.dpt import DepthAnythingV2



def _load_rgb(scene: str, data_root: str, disp_ts: int, target_shape) -> np.ndarray:
    scene_dir = Path(data_root) / scene
    img_ts_path = scene_dir / "image_timestamps.txt"
    if not img_ts_path.exists():
        return np.zeros(target_shape, dtype=np.uint8)
    img_timestamps = np.loadtxt(img_ts_path, dtype=np.int64)
    img_idx = int(np.argmin(np.abs(img_timestamps - disp_ts)))
    img_path = scene_dir / f"{scene}_images_rectified_left" / f"{img_idx:06d}.png"
    rgb = cv2.imread(str(img_path))
    if rgb is None:
        return np.zeros(target_shape, dtype=np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (target_shape[1], target_shape[0]))
    return rgb


def run_vitb_on_random_tencode_sample(
    data_root,
    scene="interlaken_00_d",
    time_window_us=5000,
    shape=(480, 640),
    outdir="output/test",
    input_size=518,
    pred_only=False,
    grayscale=False,
    idx=None
):
    # Load TencodeDataset
    dataset = TencodeDataset(
        data_root=data_root,
        time_window_us=time_window_us,
        shape=shape,
        scenes=[scene],
    )
    os.makedirs(outdir, exist_ok=True)
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    x, _ = dataset[idx]
    sample = dataset.samples[idx]
    rgb_img = _load_rgb(scene, data_root, int(sample["timestamp"]), target_shape=x.permute(1, 2, 0).shape)
    x_np = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")

    # Model config for vitb
    model_configs = {
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
    }
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # Load model
    depth_anything = DepthAnythingV2(**model_configs['vitb'])
    # checkpoint = os.path.join(os.path.dirname(__file__), "../models/dav2/checkpoints/depth_anything_v2_vitb.pth")
    checkpoint = os.path.join(os.path.dirname(__file__), "../output/dav2_tencode_finetuned.pth")
    depth_anything.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()

    with torch.no_grad():
        x_in = F.interpolate(x.unsqueeze(0).to(device), size=(input_size, input_size), mode="bilinear", align_corners=False)
        pred = depth_anything(x_in)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=x.shape[-2:], mode="bilinear", align_corners=False)
        depth = pred[0, 0].detach().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
    depth = depth.astype(np.uint8)

    # Colorize or grayscale
    if grayscale:
        depth_vis = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        cmap = matplotlib.colormaps.get_cmap("magma")
        depth_vis = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    # Output path
    base = f"tencode_{scene}_{time_window_us}_{idx}"
    out_path = os.path.join(outdir, base + '_vitb_depth.png')

    if pred_only:
        cv2.imwrite(out_path, depth_vis)
    else:
        split_region = np.ones((x_np.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([
            cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR),
            split_region,
            cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR),
            split_region,
            depth_vis,
        ])
        cv2.imwrite(out_path, combined_result)
    print(f"Saved depth prediction to {out_path}")



def run_vitb_on_random_timesurface_sample(
    data_root,
    scene="interlaken_00_d",
    time_window_us=10000,
    shape=(480, 640),
    outdir="output/test",
    input_size=518,
    pred_only=False,
    grayscale=False,
    idx=None,
    tau=5000.0
):
    # Load TimeSurfaceDataset
    dataset = TimeSurfaceDataset(
        data_root=data_root,
        time_window_us=time_window_us,
        shape=shape,
        tau=tau,
        scenes=[scene],
    )
    os.makedirs(outdir, exist_ok=True)
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    x, _ = dataset[idx]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    sample = dataset.samples[idx]
    rgb_img = _load_rgb(scene, data_root, int(sample["timestamp"]), target_shape=x.permute(1, 2, 0).shape)
    x_np = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")

    # Model config for vitb
    model_configs = {
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
    }
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # Load model
    depth_anything = DepthAnythingV2(**model_configs['vitb'])
    # checkpoint = os.path.join(os.path.dirname(__file__), "../models/dav2/checkpoints/depth_anything_v2_vitb.pth")
    checkpoint = os.path.join(os.path.dirname(__file__), "../output/dav2_tencode_finetuned.pth")
    depth_anything.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()

    with torch.no_grad():
        x_in = F.interpolate(x.unsqueeze(0).to(device), size=(input_size, input_size), mode="bilinear", align_corners=False)
        x_in = x_in.repeat(2, 1, 1, 1)
        print(x_in.shape)
        pred = depth_anything(x_in)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=x.shape[-2:], mode="bilinear", align_corners=False)
        depth = pred[0, 0].detach().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
    depth = depth.astype(np.uint8)

    # Colorize or grayscale
    if grayscale:
        depth_vis = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        cmap = matplotlib.colormaps.get_cmap("magma")
        depth_vis = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    # Output path
    base = f"timesurface_{scene}_{time_window_us}_{tau}_{idx}"
    out_path = os.path.join(outdir, base + '_vitb_depth.png')

    if pred_only:
        cv2.imwrite(out_path, depth_vis)
    else:
        split_region = np.ones((x_np.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([
            cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR),
            split_region,
            cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR),
            split_region,
            depth_vis,
        ])
        cv2.imwrite(out_path, combined_result)
    print(f"Saved depth prediction to {out_path}")


def run_vitb_on_random_tencodecount_sample(
    data_root,
    scene="interlaken_00_d",
    time_window_us=10000,
    shape=(480, 640),
    outdir="output/test",
    input_size=518,
    pred_only=False,
    grayscale=False,
    idx=None,
):
    # Load TencodeDataset with event-count representation
    dataset = TencodeCountDataset(
        data_root=data_root,
        time_window_us=time_window_us,
        shape=shape,
        scenes=[scene],
    )
    os.makedirs(outdir, exist_ok=True)
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    x, _ = dataset[idx]
    sample = dataset.samples[idx]
    rgb_img = _load_rgb(scene, data_root, int(sample["timestamp"]), target_shape=x.permute(1, 2, 0).shape)
    x_np = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")

    model_configs = {
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]}
    }
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    depth_anything = DepthAnythingV2(**model_configs["vitb"])
    checkpoint = os.path.join(os.path.dirname(__file__), "../output/dav2_tencode_finetuned.pth")
    depth_anything.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    depth_anything = depth_anything.to(device).eval()

    with torch.no_grad():
        x_in = F.interpolate(x.unsqueeze(0).to(device), size=(input_size, input_size), mode="bilinear", align_corners=False)
        pred = depth_anything(x_in)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=x.shape[-2:], mode="bilinear", align_corners=False)
        depth = pred[0, 0].detach().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
    depth = depth.astype(np.uint8)

    if grayscale:
        depth_vis = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        cmap = matplotlib.colormaps.get_cmap("magma")
        depth_vis = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    base = f"tencodecount_{scene}_{time_window_us}_{idx}"
    out_path = os.path.join(outdir, base + "_vitb_depth.png")

    if pred_only:
        cv2.imwrite(out_path, depth_vis)
    else:
        split_region = np.ones((x_np.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat(
            [
                cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR),
                split_region,
                cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR),
                split_region,
                depth_vis,
            ]
        )
        cv2.imwrite(out_path, combined_result)
    print(f"Saved depth prediction to {out_path}")



def pick_random_index(data_root, scene="interlaken_00_d", time_window_us=10000, shape=(480, 640)):
    dataset = TencodeDataset(
        data_root=data_root,
        time_window_us=time_window_us,
        shape=shape,
        scenes=[scene],
    )
    return random.randint(0, len(dataset) - 1)



if __name__ == "__main__":
    data_root = "datasets/DSEC/data/train"
    scene = "interlaken_00_d"
    idx = pick_random_index(data_root, scene=scene, time_window_us=10000, shape=(480, 640))
    idx = 60
    run_vitb_on_random_tencode_sample(
        data_root=data_root,
        scene=scene,
        time_window_us=50000,
        shape=(480, 640),
        outdir="./output/test",
        idx=idx
    )
    run_vitb_on_random_timesurface_sample(
        data_root=data_root,
        scene=scene,
        time_window_us=50000,
        shape=(480, 640),
        outdir="./output/test",
        idx=idx,
        tau=5000.0
    )
    run_vitb_on_random_tencodecount_sample(
        data_root=data_root,
        scene=scene,
        time_window_us=50000,
        shape=(480, 640),
        outdir="./output/test",
        idx=idx,
    )