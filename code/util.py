import os
from typing import Union

import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def rgb_to_uint8(rgb: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    arr = _to_numpy(rgb)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError("rgb_to_uint8 expects shape (3,H,W)")
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    return (255 * arr).astype(np.uint8)


def voxelgrid_to_uint8(voxel: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    arr = _to_numpy(voxel)
    if arr.ndim != 3:
        raise ValueError("voxelgrid_to_uint8 expects shape (C,H,W)")
    arr_mean = arr.mean(axis=0)
    arr_mean = (arr_mean - arr_mean.min()) / (arr_mean.max() - arr_mean.min() + 1e-8)
    return (255 * arr_mean).astype(np.uint8)


def grayscale_to_uint8(img: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    arr = _to_numpy(img)
    arr = np.clip(arr, 0.0, 1.0)
    return (255 * arr).astype(np.uint8)


def depth_to_colormap(depth: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    arr = _to_numpy(depth).squeeze()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    depth_min, depth_max = arr.min(), arr.max()
    if depth_max - depth_min < 1e-8:
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    depth_norm = (arr - depth_min) / (depth_max - depth_min + 1e-8)
    cmap = plt.get_cmap("viridis")
    return (255 * cmap(depth_norm)[..., :3]).astype(np.uint8)


def save_image(path: str, img: np.ndarray, cmap: Union[str, None] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    arr = img
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]  # (H,W)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]  # (H,W)

    plt.imsave(path, arr, cmap=cmap)


def save_rgb(path: str, rgb: Union[torch.Tensor, np.ndarray]) -> None:
    img = rgb_to_uint8(rgb)
    save_image(path, img)


def save_voxelgrid(path: str, voxel: Union[torch.Tensor, np.ndarray]) -> None:
    img = voxelgrid_to_uint8(voxel)
    save_image(path, img, cmap="gray")


def save_grayscale(path: str, img: Union[torch.Tensor, np.ndarray]) -> None:
    gray = grayscale_to_uint8(img)
    save_image(path, gray, cmap="gray")


def save_depth_colormap(path: str, depth: Union[torch.Tensor, np.ndarray]) -> None:
    colored = depth_to_colormap(depth)
    save_image(path, colored)


def save_depth_colormap_with_cbar(
    path: str,
    depth: Union[torch.Tensor, np.ndarray],
    label: str = "Depth (m)",
) -> None:
    arr = _to_numpy(depth).squeeze()
    depth_min, depth_max = arr.min(), arr.max()
    fig, ax = plt.subplots()
    im = ax.imshow(arr, cmap="viridis", vmin=depth_min, vmax=depth_max)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(label)
    ax.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
