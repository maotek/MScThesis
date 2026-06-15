# Paper Table Config Mapping

`train` is the config used to produce the checkpoint. `eval` is the validation
config used to produce the CSV containing the reported metrics. Rows without a
`train` config use a pretrained model or do not require local training.

## Table 1: In-Domain Depth Estimation

### DSEC

| Row | Train config | Eval config |
| --- | --- | --- |
| Depth AnyEvent | - | `configs/dsec/validation/dae_tencode_DSEC_checkpoint.json` |
| Tencode DAv2 | - | `configs/dsec/validation/dav2_tencode.json` |
| RGB DAv2 | - | `configs/dsec/validation/dav2_rgb.json` |
| E2VID DAv2 | - | `configs/dsec/validation/e2vid_dav2_voxelgrid.json` |
| ETNet DAv2 | - | `configs/dsec/validation/etnet_dav2_voxelgrid.json` |
| U-Net DAv2 (Ours) | `configs/dsec/train/unet_dav2_batch10.json` | `configs/dsec/validation/unet_dav2_batch10.json` |
| FullyConv DAv2 (Ours) | `configs/dsec/train/fully_conv_dav2_batch10_RC.json` | `configs/dsec/validation/fully_conv_dav2_batch10_RC.json` |

### MVSEC

| Row | Train config | Eval config |
| --- | --- | --- |
| Depth AnyEvent | - | `configs/mvsec/validation/dae_tencode_MVSEC_checkpoint.json` |
| Tencode DAv2 | - | `configs/mvsec/validation/dav2_tencode.json` |
| RGB DAv2 | - | `configs/mvsec/validation/dav2_rgb.json` |
| E2VID DAv2 | - | `configs/mvsec/validation/e2vid_dav2_voxelgrid.json` |
| ETNet DAv2 | - | `configs/mvsec/validation/etnet_dav2_voxelgrid.json` |
| U-Net DAv2 (Ours) | `configs/mvsec/train/unet_dav2_batch10.json` | `configs/mvsec/validation/train_mvsec_unet_dav2_batch10.json` |
| FullyConv DAv2 (Ours) | `configs/mvsec/train/fully_conv_dav2_batch10.json` | `configs/mvsec/validation/train_mvsec_fully_conv_dav2_batch10.json` |

## Table 2: Cross-Dataset Generalization

| Row | Train config | Eval config |
| --- | --- | --- |
| Depth AnyEvent: DSEC -> MVSEC | - | `configs/mvsec/validation/dae_tencode_DSEC_checkpoint.json` |
| U-Net DAv2: DSEC -> MVSEC | `configs/dsec/train/unet_dav2_batch10.json` | `configs/mvsec/validation/unet_dav2_batch10.json` |
| FullyConv DAv2: DSEC -> MVSEC | `configs/dsec/train/fully_conv_dav2_batch10_RC.json` | `configs/mvsec/validation/fully_conv_dav2_batch10_RC.json` |
| Depth AnyEvent: MVSEC -> DSEC | - | `configs/dsec/validation/dae_tencode_MVSEC_checkpoint.json` |
| U-Net DAv2: MVSEC -> DSEC | `configs/mvsec/train/unet_dav2_batch10.json` | `configs/dsec/validation/train_mvsec_unet_dav2_batch10.json` |
| FullyConv DAv2: MVSEC -> DSEC | `configs/mvsec/train/fully_conv_dav2_batch10.json` | `configs/dsec/validation/train_mvsec_fully_conv_dav2_batch10.json` |

## Table 3: Ablation Study

### DSEC

| Row | Train config | Eval config |
| --- | --- | --- |
| U-Net DAv2 (baseline) | `configs/dsec/train/unet_dav2_batch10.json` | `configs/dsec/validation/unet_dav2_batch10.json` |
| U-Net DAv2 (ReLU output) | `configs/dsec/train/unet_dav2_batch10_relu.json` | `configs/dsec/validation/unet_dav2_batch10_relu.json` |
| U-Net DAv2 (grayscale) | `configs/dsec/train/unet_1c_dav2_batch10.json` | `configs/dsec/validation/unet_1c_dav2_batch10.json` |
| U-Net DAv2 (16 channels) | `configs/dsec/train/unet_dav2_batch10_ch16.json` | `configs/dsec/validation/unet_dav2_batch10_ch16.json` |
| U-Net DAv2 (1 enc / 1 dec) | `configs/dsec/train/unet_small3_dav2_batch10.json` | `configs/dsec/validation/unet_small3_dav2_batch10.json` |
| U-Net DAv2 (+ learnable const) | `configs/dsec/train/newunet_dav2_batch10.json` | `configs/dsec/validation/newunet_dav2_batch10.json` |
| FullyConv DAv2 (baseline) | `configs/dsec/train/fully_conv_dav2_batch10_RC.json` | `configs/dsec/validation/fully_conv_dav2_batch10_RC.json` |
| FullyConv DAv2 (grayscale) | `configs/dsec/train/fully_conv_1c_dav2_batch10.json` | `configs/dsec/validation/fully_conv_1c_dav2_batch10.json` |
| FullyConv DAv2 (+ learnable const) | `configs/dsec/train/new_fully_conv_dav2_batch10_seed3.json` | `configs/dsec/validation/new_fully_conv_dav2_batch10_seed3.json` |

### MVSEC

| Row | Train config | Eval config |
| --- | --- | --- |
| U-Net DAv2 (baseline) | `configs/mvsec/train/unet_dav2_batch10.json` | `configs/mvsec/validation/train_mvsec_unet_dav2_batch10.json` |
| U-Net DAv2 (ReLU output) | `configs/mvsec/train/unet_dav2_batch10_relu.json` | `configs/mvsec/validation/train_mvsec_unet_dav2_batch10_relu.json` |
| U-Net DAv2 (grayscale) | `configs/mvsec/train/unet_1c_dav2_batch10.json` | `configs/mvsec/validation/train_mvsec_unet_1c_dav2_batch10.json` |
| U-Net DAv2 (16 channels) | `configs/mvsec/train/unet_dav2_batch10_ch16.json` | `configs/mvsec/validation/train_mvsec_unet_dav2_batch10_ch16.json` |
| U-Net DAv2 (1 enc / 1 dec) | `configs/mvsec/train/unet_small3_dav2_batch10.json` | `configs/mvsec/validation/train_mvsec_unet_small3_dav2_batch10.json` |
| U-Net DAv2 (+ learnable const) | `configs/mvsec/train/newunet_dav2_batch10.json` | `configs/mvsec/validation/train_mvsec_newunet_dav2_batch10.json` |
| FullyConv DAv2 (baseline) | `configs/mvsec/train/fully_conv_dav2_batch10.json` | `configs/mvsec/validation/train_mvsec_fully_conv_dav2_batch10.json` |
| FullyConv DAv2 (grayscale) | `configs/mvsec/train/fully_conv_1c_dav2_batch10.json` | `configs/mvsec/validation/train_mvsec_fully_conv_1c_dav2_batch10.json` |
| FullyConv DAv2 (+ learnable const) | `configs/mvsec/train/new_fully_conv_dav2_batch10.json` | `configs/mvsec/validation/train_mvsec_new_fully_conv_dav2_batch10.json` |

## Table 4: Cross-Dataset Ablation Study

### DSEC -> MVSEC

| Row | Train config | Eval config |
| --- | --- | --- |
| U-Net DAv2 (baseline) | `configs/dsec/train/unet_dav2_batch10.json` | `configs/mvsec/validation/unet_dav2_batch10.json` |
| U-Net DAv2 (ReLU output) | `configs/dsec/train/unet_dav2_batch10_relu.json` | `configs/mvsec/validation/unet_dav2_batch10_relu.json` |
| U-Net DAv2 (grayscale) | `configs/dsec/train/unet_1c_dav2_batch10.json` | `configs/mvsec/validation/unet_1c_dav2_batch10.json` |
| U-Net DAv2 (16 channels) | `configs/dsec/train/unet_dav2_batch10_ch16.json` | `configs/mvsec/validation/unet_dav2_batch10_ch16.json` |
| U-Net DAv2 (1 enc / 1 dec) | `configs/dsec/train/unet_small3_dav2_batch10.json` | `configs/mvsec/validation/unet_small3_dav2_batch10.json` |
| U-Net DAv2 (+ learnable const) | `configs/dsec/train/newunet_dav2_batch10.json` | `configs/mvsec/validation/newunet_dav2_batch10.json` |
| FullyConv DAv2 (baseline) | `configs/dsec/train/fully_conv_dav2_batch10_RC.json` | `configs/mvsec/validation/fully_conv_dav2_batch10_RC.json` |
| FullyConv DAv2 (grayscale) | `configs/dsec/train/fully_conv_1c_dav2_batch10.json` | `configs/mvsec/validation/fully_conv_1c_dav2_batch10.json` |
| FullyConv DAv2 (+ learnable const) | `configs/dsec/train/new_fully_conv_dav2_batch10_seed3.json` | `configs/mvsec/validation/new_fully_conv_dav2_batch10_seed3.json` |

### MVSEC -> DSEC

| Row | Train config | Eval config |
| --- | --- | --- |
| U-Net DAv2 (baseline) | `configs/mvsec/train/unet_dav2_batch10.json` | `configs/dsec/validation/train_mvsec_unet_dav2_batch10.json` |
| U-Net DAv2 (ReLU output) | `configs/mvsec/train/unet_dav2_batch10_relu.json` | `configs/dsec/validation/train_mvsec_unet_dav2_batch10_relu.json` |
| U-Net DAv2 (grayscale) | `configs/mvsec/train/unet_1c_dav2_batch10.json` | `configs/dsec/validation/train_mvsec_unet_1c_dav2_batch10.json` |
| U-Net DAv2 (16 channels) | `configs/mvsec/train/unet_dav2_batch10_ch16.json` | `configs/dsec/validation/train_mvsec_unet_dav2_batch10_ch16.json` |
| U-Net DAv2 (1 enc / 1 dec) | `configs/mvsec/train/unet_small3_dav2_batch10.json` | `configs/dsec/validation/train_mvsec_unet_small3_dav2_batch10.json` |
| U-Net DAv2 (+ learnable const) | `configs/mvsec/train/newunet_dav2_batch10.json` | `configs/dsec/validation/train_mvsec_newunet_dav2_batch10.json` |
| FullyConv DAv2 (baseline) | `configs/mvsec/train/fully_conv_dav2_batch10.json` | `configs/dsec/validation/train_mvsec_fully_conv_dav2_batch10.json` |
| FullyConv DAv2 (grayscale) | `configs/mvsec/train/fully_conv_1c_dav2_batch10.json` | `configs/dsec/validation/train_mvsec_fully_conv_1c_dav2_batch10.json` |
| FullyConv DAv2 (+ learnable const) | `configs/mvsec/train/new_fully_conv_dav2_batch10.json` | `configs/dsec/validation/train_mvsec_new_fully_conv_dav2_batch10.json` |
