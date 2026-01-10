
from datasets.time_surface_dataset import TimeSurfaceDataset
from representations.time_surface import TimeSurface
import numpy as np
import cv2
import matplotlib.pyplot as plt



def run_example(idx=70, time_window_us=10000, tau=5000.0, scene="interlaken_00_d"):
    data_root = "datasets/DSEC/data/train"

    dataset = TimeSurfaceDataset(
        data_root=data_root,
        time_window_us=time_window_us,
        tau=tau,
        event_representation=TimeSurface(tau),
        shape=(480, 640),
        scenes=[scene],
    )

    x, y = dataset[idx]
    # x: (1, H, W), y: (1, H, W)
    event_input = x.squeeze(0).numpy()
    disp_img = y.squeeze(0).numpy()
    print("Event input shape:", event_input.shape)
    print("Disparity min:", disp_img.min(), "max:", disp_img.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax_ts = event_input.max()
    vmin_ts = event_input.min()
    norm_ts = plt.Normalize(vmin=vmin_ts, vmax=vmax_ts)
    cmap_ts = plt.get_cmap('inferno')
    ts_rgb = cmap_ts(norm_ts(event_input))[..., :3]
    axes[0].imshow((ts_rgb * 255).astype('uint8'))
    axes[0].set_title('Time Surface (inferno)')
    axes[0].axis('off')

    vmax = disp_img.max()
    vmin = disp_img.min()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('inferno')
    disp_rgb = cmap(norm(disp_img))[..., :3]
    axes[1].imshow((disp_rgb * 255).astype('uint8'))
    axes[1].set_title('Disparity')
    axes[1].axis('off')

    event_rgb = np.stack([event_input * 255] * 3, axis=-1).astype('uint8')
    if event_rgb.shape != disp_rgb.shape:
        disp_rgb_img = (disp_rgb * 255).astype('uint8')
        disp_rgb_img = disp_rgb_img[: event_rgb.shape[0], : event_rgb.shape[1], :]
    else:
        disp_rgb_img = (disp_rgb * 255).astype('uint8')
    overlay = cv2.addWeighted(event_rgb, 0.3, disp_rgb_img, 0.7, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_example()
