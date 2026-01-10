import hdf5plugin
import h5py
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from datasets.DSEC.scripts.utils.eventslicer import EventSlicer
import time

OUTPUT_DIR = os.path.join("output", "visualize_events")

def load_event_and_disparity(h5_path, disp_path, t_start_us, t_end_us):
    with h5py.File(h5_path, 'r') as h5f:
        slicer = EventSlicer(h5f)
        print(type(t_start_us), type(t_end_us))
        t_start_us = np.int64(t_start_us)
        t_end_us = np.int64(t_end_us)
        events = slicer.get_events(int(t_start_us), int(t_end_us))
        event_img = np.zeros((480, 640), dtype=np.uint8)
        if events:
            xs, ys = events['x'], events['y']
            event_img[ys, xs] = 255
    disp_img = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
    if disp_img is not None:
        disp_img = disp_img.astype(np.float32)
    else:
        disp_img = np.zeros((480, 640), dtype=np.float32)
    return event_img, disp_img

def load_image_and_disparity(img_path, disp_path):
    rgb_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) if rgb_img is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    disp_img = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
    if disp_img is None:
        disp_img = np.zeros((480, 640), dtype=np.float32)
    return rgb_img, disp_img

def plot_event_and_disparity(event_img, disparity_event_img, title=None, out_name: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(event_img, cmap='gray')
    axes[0].set_title('Event')
    axes[0].axis('off')
    # Disparity event as colored
    disp_gray = disparity_event_img
    vmax = np.nanmax(disp_gray)
    vmin = np.nanmin(disp_gray)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('inferno')
    disp_rgb = cmap(norm(disp_gray))[..., :3]
    axes[1].imshow((disp_rgb * 255).astype(np.uint8))
    axes[1].set_title('Disparity Event')
    axes[1].axis('off')
    # Overlay
    event_overlay = cv2.cvtColor(event_img, cv2.COLOR_GRAY2RGB)
    disp_rgb_img = (disp_rgb * 255).astype(np.uint8)
    if event_overlay.shape != disp_rgb_img.shape:
        disp_rgb_img = cv2.resize(disp_rgb_img, (event_overlay.shape[1], event_overlay.shape[0]))
    overlay = cv2.addWeighted(event_overlay, 0.3, disp_rgb_img, 0.7, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Event + Disparity Overlay')
    axes[2].axis('off')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if out_name:
        safe_title = out_name
    else:
        safe_title = (title or "event_disparity").replace(' ', '_')
    safe_title = ''.join(c if (c.isalnum() or c in ('_', '-')) else '_' for c in safe_title)
    filename = f"{safe_title}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

def plot_image_and_disparity(rgb_img, disparity_image_img, title=None, out_name: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb_img)
    axes[0].set_title('Image')
    axes[0].axis('off')
    # Disparity image as colored
    disp_gray = disparity_image_img
    vmax = np.nanmax(disp_gray)
    vmin = np.nanmin(disp_gray)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('inferno')
    disp_rgb = cmap(norm(disp_gray))[..., :3]
    disp_rgb_img = (disp_rgb * 255).astype(np.uint8)
    axes[1].imshow(disp_rgb_img)
    axes[1].set_title('Disparity Image')
    axes[1].axis('off')
    # Overlay
    if rgb_img.shape != disp_rgb_img.shape:
        disp_rgb_img = cv2.resize(disp_rgb_img, (rgb_img.shape[1], rgb_img.shape[0]))
    overlay = cv2.addWeighted(rgb_img, 0.3, disp_rgb_img, 0.9, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Image + Disparity Overlay')
    axes[2].axis('off')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if out_name:
        safe_title = out_name
    else:
        safe_title = (title or "image_disparity").replace(' ', '_')
    safe_title = ''.join(c if (c.isalnum() or c in ('_', '-')) else '_' for c in safe_title)
    filename = f"{safe_title}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    # Paths
    base = os.path.join('datasets', 'DSEC', 'data', 'train', 'interlaken_00_d')
    h5_path = os.path.join(base, 'interlaken_00_d_events_left', 'events.h5')
    disp_event_dir = os.path.join(base, 'interlaken_00_d_disparity_event')
    disp_image_dir = os.path.join(base, 'interlaken_00_d_disparity_image')
    img_dir = os.path.join(base, 'interlaken_00_d_images_rectified_left')
    disp_ts_path = os.path.join(base, 'disparity_timestamps.txt')
    img_ts_path = os.path.join(base, 'image_timestamps.txt')

    # Load timestamps
    disp_timestamps = np.loadtxt(disp_ts_path, dtype=np.int64)
    img_timestamps = np.loadtxt(img_ts_path, dtype=np.int64)

    # Pick a random frame index
    idx = int(np.random.randint(0, len(disp_timestamps)))
    disp_ts = disp_timestamps[idx]
    # Find closest image index
    img_idx = np.argmin(np.abs(img_timestamps - disp_ts))
    img_ts = img_timestamps[img_idx]

    # Get sorted filenames
    disp_event_files = sorted([f for f in os.listdir(disp_event_dir) if f.endswith('.png')])
    disp_image_files = sorted([f for f in os.listdir(disp_image_dir) if f.endswith('.png')])
    disp_event_path = os.path.join(disp_event_dir, disp_event_files[idx])
    disp_image_path = os.path.join(disp_image_dir, disp_image_files[idx])
    img_path = os.path.join(img_dir, f"{img_idx:06d}.png")

    # Load event and its disparity
    t_start_us = disp_ts - 10000  # 10ms window before
    t_end_us = disp_ts
    event_img, disparity_event_img = load_event_and_disparity(h5_path, disp_event_path, t_start_us, t_end_us)
    print(event_img.shape, disparity_event_img.shape)
    print(f"Event disparity min: {np.nanmin(disparity_event_img):.4f}, max: {np.nanmax(disparity_event_img):.4f}")
    scene_name = os.path.basename(base)
    plot_event_and_disparity(event_img, disparity_event_img, title=f"Event + Disparity Event (idx {idx})", out_name=f"{scene_name}_event_{idx}")

    # Load image and its disparity
    rgb_img, disparity_image_img = load_image_and_disparity(img_path, disp_image_path)
    print(rgb_img.shape, disparity_image_img.shape)
    print(f"Image disparity min: {np.nanmin(disparity_image_img):.4f}, max: {np.nanmax(disparity_image_img):.4f}")
    plot_image_and_disparity(rgb_img, disparity_image_img, title=f"Image + Disparity Image (idx {img_idx})", out_name=f"{scene_name}_image_{img_idx}")

if __name__ == "__main__":
    main()
