from datasets.tencode_dataset import TencodeDataset
import cv2
import matplotlib.pyplot as plt



def run_tencode_example(idx=70, time_window_us=10000, scene="interlaken_00_d"):
    data_root = "datasets/DSEC/data/train"

    dataset = TencodeDataset(
        data_root=data_root,
        time_window_us=time_window_us,
        shape=(480, 640),
        scenes=[scene],
    )

    x, y = dataset[idx]
    # x: (3, H, W), y: (1, H, W)
    tencode_frame = (x.permute(1, 2, 0).numpy() * 255).astype('uint8')
    disp_img = y.squeeze(0).numpy()
    print("Tencode frame shape:", tencode_frame.shape)
    print("Disparity min:", disp_img.min(), "max:", disp_img.max())

    print("Red channel:   min =", tencode_frame[..., 0].min(), ", max =", tencode_frame[..., 0].max())
    print("Green channel: min =", tencode_frame[..., 1].min(), ", max =", tencode_frame[..., 1].max())
    print("Blue channel:  min =", tencode_frame[..., 2].min(), ", max =", tencode_frame[..., 2].max())


    fig, axes = plt.subplots(1, 5, figsize=(24, 6))

    axes[0].imshow(tencode_frame)
    axes[0].set_title('Tencode (RGB)')
    axes[0].axis('off')

    axes[1].imshow(tencode_frame[..., 0], cmap='Reds')
    axes[1].set_title('Red channel')
    axes[1].axis('off')

    axes[2].imshow(tencode_frame[..., 1], cmap='Greens')
    axes[2].set_title('Green channel')
    axes[2].axis('off')

    axes[3].imshow(tencode_frame[..., 2], cmap='Blues')
    axes[3].set_title('Blue channel')
    axes[3].axis('off')

    vmax = disp_img.max()
    vmin = disp_img.min()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('inferno')
    disp_rgb = cmap(norm(disp_img))[..., :3]
    tencode_rgb = tencode_frame
    if tencode_rgb.shape != disp_rgb.shape:
        disp_rgb_img = (disp_rgb * 255).astype('uint8')
        disp_rgb_img = disp_rgb_img[: tencode_rgb.shape[0], : tencode_rgb.shape[1], :]
    else:
        disp_rgb_img = (disp_rgb * 255).astype('uint8')
    overlay = cv2.addWeighted(tencode_rgb, 0.5, disp_rgb_img, 0.5, 0)
    axes[4].imshow(overlay)
    axes[4].set_title('Tencode + Disparity Overlay')
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()

    # Save the Tencode RGB image with timewindow and idx in filename
    out_path = f"tencode_{scene}_{time_window_us}_{idx}.png"
    cv2.imwrite(out_path, cv2.cvtColor(tencode_frame, cv2.COLOR_RGB2BGR))
    print(f"Saved Tencode RGB image to {out_path}")


if __name__ == "__main__":
    run_tencode_example()
