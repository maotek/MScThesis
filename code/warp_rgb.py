import glob
import os

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt


def find_calibration_path(sequence_path: str) -> str:
    patterns = [
        os.path.join(sequence_path, "*calibration*", "cam_to_cam.yaml"),
        os.path.join(sequence_path, "calibration", "cam_to_cam.yaml"),
        os.path.join(sequence_path, "*_calibration", "cam_to_cam.yaml"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Calibration file not found in {sequence_path}")


def load_calibration_yaml(calib_path: str) -> dict:
    with open(calib_path, "r") as f:
        return yaml.safe_load(f)


def get_intrinsics_extrinsics(calib: dict):
    intr = calib["intrinsics"]
    extr = calib["extrinsics"]

    # Prefer rectified intrinsics when available.
    event_cam = intr["camRect0"] if "camRect0" in intr else intr["cam0"]
    rgb_cam = intr["camRect1"] if "camRect1" in intr else intr["cam1"]

    ev_params = np.array(event_cam["camera_matrix"]).reshape(4)
    rgb_params = np.array(rgb_cam["camera_matrix"]).reshape(4)
    event_K = np.array([[ev_params[0], 0, ev_params[2]],
                        [0, ev_params[1], ev_params[3]],
                        [0, 0, 1]])
    rgb_K = np.array([[rgb_params[0], 0, rgb_params[2]],
                      [0, rgb_params[1], rgb_params[3]],
                      [0, 0, 1]])

    # Base transform: cam0 -> cam1.
    T_10 = np.array(extr["T_10"])
    if "R_rect0" in extr and "R_rect1" in extr:
        R_rect0 = np.array(extr["R_rect0"])
        R_rect1 = np.array(extr["R_rect1"])
        T_rect0 = np.eye(4, dtype=np.float64)
        T_rect1 = np.eye(4, dtype=np.float64)
        T_rect0[:3, :3] = np.linalg.inv(R_rect0)
        T_rect1[:3, :3] = R_rect1
        T_10 = T_rect1 @ T_10 @ T_rect0

    return event_K, rgb_K, T_10


def warp_rgb_to_event(rgb_img: np.ndarray, depth: np.ndarray, event_K: np.ndarray, rgb_K: np.ndarray, T_10: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # Event pixel grid in homogeneous image coordinates (3, N).
    event_pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3).T

    # Back-project event pixels into 3D using the event intrinsics and depth.
    depth = depth.flatten()
    xyz_event = np.linalg.inv(event_K) @ event_pixels * depth
    xyz_event_h = np.vstack([xyz_event, np.ones((1, xyz_event.shape[1]))])

    # Transform event-frame points into the RGB camera frame.
    xyz_rgb_h = T_10 @ xyz_event_h
    xyz_rgb = xyz_rgb_h[:3]

    # Project into the RGB image plane to build the remap.
    uv_rgb = rgb_K @ xyz_rgb
    uv_rgb = uv_rgb[:2] / uv_rgb[2]
    uv_rgb = uv_rgb.T.reshape(H, W, 2)
    map_x = uv_rgb[..., 0].astype(np.float32)
    map_y = uv_rgb[..., 1].astype(np.float32)

    return cv2.remap(
        rgb_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def visualize_depth(depth: np.ndarray, title: str = "Depth Map", cmap: str = "viridis", vmin=None, vmax=None):
    """
    Visualize a depth map using matplotlib.

    Args:
        depth: 2D numpy array of depth values
        title: Title for the plot
        cmap: Colormap to use
        vmin: Minimum value for colormap (optional)
        vmax: Maximum value for colormap (optional)
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(depth, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Depth")
    plt.title(title)
    plt.axis("off")
    plt.show()
