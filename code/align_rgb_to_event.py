import numpy as np
import cv2
import yaml
import os

def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = yaml.safe_load(f)
    # cam0: event left, cam1: frame left
    K_event = np.array(calib['intrinsics']['cam0']['camera_matrix']).reshape(3, 3)
    K_rgb = np.array(calib['intrinsics']['cam1']['camera_matrix']).reshape(3, 3)
    D_rgb = np.array(calib['intrinsics']['cam1']['distortion_coeffs'])
    # Extrinsics: T_10 (cam0 <- cam1)
    T_10 = np.array(calib['extrinsics']['T_10'])
    return K_event, K_rgb, D_rgb, T_10


def warp_rgb_to_event(rgb_img, K_event, K_rgb, D_rgb, T_10, event_shape):
    h_e, w_e = event_shape
    # Generate event pixel grid
    y_e, x_e = np.indices((h_e, w_e))
    pts_event = np.stack([x_e.ravel(), y_e.ravel(), np.ones_like(x_e).ravel()])
    # Project to cam0 (event) normalized
    pts_event_norm = np.linalg.inv(K_event) @ pts_event
    # Transform to cam1 (rgb) frame
    pts_rgb = T_10[:3, :3] @ pts_event_norm + T_10[:3, 3:4]
    pts_rgb = pts_rgb / pts_rgb[2:3]
    # Project to rgb pixel
    pts_rgb_pix = K_rgb @ pts_rgb
    x_rgb = pts_rgb_pix[0].reshape(h_e, w_e)
    y_rgb = pts_rgb_pix[1].reshape(h_e, w_e)
    # Remap
    map_x = x_rgb.astype(np.float32)
    map_y = y_rgb.astype(np.float32)
    warped = cv2.remap(rgb_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Warp RGB to event camera using calibration.")
    parser.add_argument('--rgb', required=True, help='Path to RGB image')
    parser.add_argument('--calib', required=True, help='Path to cam_to_cam.yaml')
    parser.add_argument('--event-shape', type=int, nargs=2, required=True, help='Event image shape (H W)')
    parser.add_argument('--out', required=True, help='Output path for warped RGB')
    args = parser.parse_args()

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"Could not read {args.rgb}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    K_event, K_rgb, D_rgb, T_10 = load_calibration(args.calib)
    warped = warp_rgb_to_event(rgb, K_event, K_rgb, D_rgb, T_10, tuple(args.event_shape))
    cv2.imwrite(args.out, cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    print(f"Saved warped RGB to {args.out}")

if __name__ == '__main__':
    main()
