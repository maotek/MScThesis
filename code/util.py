import os
from typing import Union

import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def tencode_to_uint8(tencode: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    arr = _to_numpy(tencode)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError("tencode_to_uint8 expects shape (3,H,W)")
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


def save_tencode(path: str, tencode: Union[torch.Tensor, np.ndarray]) -> None:
    img = tencode_to_uint8(tencode)
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
