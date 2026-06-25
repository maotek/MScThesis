# Event Depth Evaluation

This folder contains training and evaluation pipelines for event-based depth models on DSEC and MVSEC.

## Quick Start

Run these commands from the `code/` directory.

### 1) Create environment
```bash
conda env create -f environment/environment.yml
```

### 2) Activate environment
```bash
conda activate apptainer
```

### 3) Download DSEC data
```bash
python -m scripts.download_dsec
```

## Commands By Config

## Run Training

All training settings live in the JSON configs under `configs/dsec/train/` and `configs/mvsec/train/`.

Examples:
```bash
python -m train_unet_dav2 --config-path configs/dsec/train/unet_dav2_batch10.json
python -m train_unet_dav2_rgb --config-path configs/dsec/train/unet_dav2_rgb.json
```

Training output is written to the `training.save_dir` defined in the config.

## Run Evaluation (Single Config)

All evaluation settings, including `csv_path`, live in the JSON configs under:
`configs/dsec/validation/` and `configs/mvsec/validation/`.

Example:
```bash
python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch10_ch16.json
```

Optional override to swap only the checkpoint filename:
```bash
python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch10_ch16.json   --checkpoint-file epoch_050.pt
```

## Run All Evaluations

```bash
bash run_all_validations.sh
```

`run_all_validations.sh` iterates all validation configs, skips ones whose `csv_path`
already exists, and continues on failures.

## Reproduce Paper Tables

Run the table-specific scripts from the `code/` directory after the environment,
datasets, and checkpoints are available.

### Table 1: In-domain DSEC and MVSEC

```bash
bash reproduce_table1.sh
```

To inspect the exact commands without running evaluation:

```bash
bash reproduce_table1.sh --dry-run
```

The script writes one CSV per row to `evaluate_results/`, using each config's
`csv_path`. The table values are the `MEAN` column for:
`_abs_rel_diff`, `_squ_rel_diff`, `_RMS_linear`, `_RMS_log`, `_SILog`,
`_threshold_delta_1.25`, `_threshold_delta_1.25^2`, and
`_threshold_delta_1.25^3`.

| Table row | Validation config | Output CSV |
| --- | --- | --- |
| DSEC / Depth AnyEvent (DAE) | `configs/dsec/validation/dae_tencode_DSEC_checkpoint.json` | `evaluate_results/dsec_validation_dae_tencode_DSEC_checkpoint.csv` |
| DSEC / Tencode DAv2 | `configs/dsec/validation/dav2_tencode.json` | `evaluate_results/dsec_validation_dav2_tencode.csv` |
| DSEC / RGB DAv2 | `configs/dsec/validation/dav2_rgb.json` | `evaluate_results/dsec_validation_dav2_rgb.csv` |
| DSEC / E2VID DAv2 | `configs/dsec/validation/e2vid_dav2_voxelgrid.json` | `evaluate_results/dsec_validation_e2vid_dav2_voxelgrid.csv` |
| DSEC / ETNet DAv2 | `configs/dsec/validation/etnet_dav2_voxelgrid.json` | `evaluate_results/dsec_validation_etnet_dav2_voxelgrid.csv` |
| DSEC / U-Net DAv2 (Ours) | `configs/dsec/validation/unet_dav2_batch10.json` | `evaluate_results/dsec_validation_unet_dav2_batch10.csv` |
| DSEC / FullyConv DAv2 (Ours) | `configs/dsec/validation/fully_conv_dav2_batch10_RC.json` | `evaluate_results/dsec_validation_fully_conv_dav2_batch10_RC.csv` |
| MVSEC / Depth AnyEvent (DAE) | `configs/mvsec/validation/dae_tencode_MVSEC_checkpoint.json` | `evaluate_results/mvsec_validation_dae_tencode_MVSEC_checkpoint.csv` |
| MVSEC / Tencode DAv2 | `configs/mvsec/validation/dav2_tencode.json` | `evaluate_results/mvsec_validation_dav2_tencode.csv` |
| MVSEC / RGB DAv2 | `configs/mvsec/validation/dav2_rgb.json` | `evaluate_results/mvsec_validation_dav2_rgb.csv` |
| MVSEC / E2VID DAv2 | `configs/mvsec/validation/e2vid_dav2_voxelgrid.json` | `evaluate_results/mvsec_validation_e2vid_dav2_voxelgrid.csv` |
| MVSEC / ETNet DAv2 | `configs/mvsec/validation/etnet_dav2_voxelgrid.json` | `evaluate_results/mvsec_validation_etnet_dav2_voxelgrid.csv` |
| MVSEC / U-Net DAv2 (Ours) | `configs/mvsec/validation/train_mvsec_unet_dav2_batch10.json` | `evaluate_results/mvsec_validation_train_mvsec_unet_dav2_batch10.csv` |
| MVSEC / FullyConv DAv2 (Ours) | `configs/mvsec/validation/train_mvsec_fully_conv_dav2_batch10.json` | `evaluate_results/mvsec_validation_train_mvsec_fully_conv_dav2_batch10.csv` |

### Table 2: Cross-dataset generalization

```bash
bash reproduce_table2.sh
```

To inspect the exact commands without running evaluation:

```bash
bash reproduce_table2.sh --dry-run
```

| Table row | Validation config | Output CSV |
| --- | --- | --- |
| DSEC -> MVSEC / Depth AnyEvent (DAE) | `configs/mvsec/validation/dae_tencode_DSEC_checkpoint.json` | `evaluate_results/mvsec_validation_dae_tencode_DSEC_checkpoint.csv` |
| DSEC -> MVSEC / U-Net DAv2 | `configs/mvsec/validation/unet_dav2_batch10.json` | `evaluate_results/mvsec_validation_unet_dav2_batch10.csv` |
| DSEC -> MVSEC / FullyConv DAv2 | `configs/mvsec/validation/fully_conv_dav2_batch10_RC.json` | `evaluate_results/mvsec_validation_fully_conv_dav2_batch10_RC.csv` |
| MVSEC -> DSEC / Depth AnyEvent (DAE) | `configs/dsec/validation/dae_tencode_MVSEC_checkpoint.json` | `evaluate_results/dsec_validation_dae_tencode_MVSEC_checkpoint.csv` |
| MVSEC -> DSEC / U-Net DAv2 | `configs/dsec/validation/train_mvsec_unet_dav2_batch10.json` | `evaluate_results/dsec_validation_train_mvsec_unet_dav2_batch10.csv` |
| MVSEC -> DSEC / FullyConv DAv2 | `configs/dsec/validation/train_mvsec_fully_conv_dav2_batch10.json` | `evaluate_results/dsec_validation_train_mvsec_fully_conv_dav2_batch10.csv` |

### Table 3: In-domain ablation study

```bash
bash reproduce_table3.sh
```

To inspect the exact commands without running evaluation:

```bash
bash reproduce_table3.sh --dry-run
```

| Table row | Validation config | Output CSV |
| --- | --- | --- |
| DSEC / U-Net DAv2 (baseline) | `configs/dsec/validation/unet_dav2_batch10.json` | `evaluate_results/dsec_validation_unet_dav2_batch10.csv` |
| DSEC / U-Net DAv2 (ReLU activation) | `configs/dsec/validation/unet_dav2_batch10_relu.json` | `evaluate_results/dsec_validation_unet_dav2_batch10_relu.csv` |
| DSEC / U-Net DAv2 (grayscale) | `configs/dsec/validation/unet_1c_dav2_batch10.json` | `evaluate_results/dsec_validation_unet_1c_dav2_batch10.csv` |
| DSEC / U-Net DAv2 (16 channels) | `configs/dsec/validation/unet_dav2_batch10_ch16.json` | `evaluate_results/dsec_validation_unet_dav2_batch10_ch16.csv` |
| DSEC / U-Net DAv2 (1 encoder, 1 decoder) | `configs/dsec/validation/unet_small3_dav2_batch10.json` | `evaluate_results/dsec_validation_unet_small3_dav2_batch10.csv` |
| DSEC / U-Net DAv2 (learnable constant) | `configs/dsec/validation/newunet_dav2_batch10.json` | `evaluate_results/dsec_validation_newunet_dav2_batch10.csv` |
| DSEC / FullyConv DAv2 (baseline) | `configs/dsec/validation/fully_conv_dav2_batch10_RC.json` | `evaluate_results/dsec_validation_fully_conv_dav2_batch10_RC.csv` |
| DSEC / FullyConv DAv2 (grayscale) | `configs/dsec/validation/fully_conv_1c_dav2_batch10.json` | `evaluate_results/dsec_validation_fully_conv_1c_dav2_batch10.csv` |
| DSEC / FullyConv DAv2 (learnable constant) | `configs/dsec/validation/new_fully_conv_dav2_batch10_seed3.json` | `evaluate_results/dsec_validation_new_fully_conv_dav2_batch10_seed3.csv` |
| MVSEC / U-Net DAv2 (baseline) | `configs/mvsec/validation/train_mvsec_unet_dav2_batch10.json` | `evaluate_results/mvsec_validation_train_mvsec_unet_dav2_batch10.csv` |
| MVSEC / U-Net DAv2 (ReLU activation) | `configs/mvsec/validation/train_mvsec_unet_dav2_batch10_relu.json` | `evaluate_results/mvsec_validation_train_mvsec_unet_dav2_batch10_relu.csv` |
| MVSEC / U-Net DAv2 (grayscale) | `configs/mvsec/validation/train_mvsec_unet_1c_dav2_batch10.json` | `evaluate_results/mvsec_validation_train_mvsec_unet_1c_dav2_batch10.csv` |
| MVSEC / U-Net DAv2 (16 channels) | `configs/mvsec/validation/train_mvsec_unet_dav2_batch10_ch16.json` | `evaluate_results/mvsec_validation_train_mvsec_unet_dav2_batch10_ch16.csv` |
| MVSEC / U-Net DAv2 (1 encoder, 1 decoder) | `configs/mvsec/validation/train_mvsec_unet_small3_dav2_batch10.json` | `evaluate_results/mvsec_validation_train_mvsec_unet_small3_dav2_batch10.csv` |
| MVSEC / U-Net DAv2 (learnable constant) | `configs/mvsec/validation/train_mvsec_newunet_dav2_batch10.json` | `evaluate_results/mvsec_validation_train_mvsec_newunet_dav2_batch10.csv` |
| MVSEC / FullyConv DAv2 (baseline) | `configs/mvsec/validation/train_mvsec_fully_conv_dav2_batch10.json` | `evaluate_results/mvsec_validation_train_mvsec_fully_conv_dav2_batch10.csv` |
| MVSEC / FullyConv DAv2 (grayscale) | `configs/mvsec/validation/train_mvsec_fully_conv_1c_dav2_batch10.json` | `evaluate_results/mvsec_validation_train_mvsec_fully_conv_1c_dav2_batch10.csv` |
| MVSEC / FullyConv DAv2 (learnable constant) | `configs/mvsec/validation/train_mvsec_new_fully_conv_dav2_batch10.json` | `evaluate_results/mvsec_validation_train_mvsec_new_fully_conv_dav2_batch10.csv` |

### Table 4: Cross-dataset ablation study

```bash
bash reproduce_table4.sh
```

To inspect the exact commands without running evaluation:

```bash
bash reproduce_table4.sh --dry-run
```

| Table row | Validation config | Output CSV |
| --- | --- | --- |
| DSEC -> MVSEC / U-Net DAv2 (baseline) | `configs/mvsec/validation/unet_dav2_batch10.json` | `evaluate_results/mvsec_validation_unet_dav2_batch10.csv` |
| DSEC -> MVSEC / U-Net DAv2 (ReLU activation) | `configs/mvsec/validation/unet_dav2_batch10_relu.json` | `evaluate_results/mvsec_validation_unet_dav2_batch10_relu.csv` |
| DSEC -> MVSEC / U-Net DAv2 (grayscale) | `configs/mvsec/validation/unet_1c_dav2_batch10.json` | `evaluate_results/mvsec_validation_unet_1c_dav2_batch10.csv` |
| DSEC -> MVSEC / U-Net DAv2 (16 channels) | `configs/mvsec/validation/unet_dav2_batch10_ch16.json` | `evaluate_results/mvsec_validation_unet_dav2_batch10_ch16.csv` |
| DSEC -> MVSEC / U-Net DAv2 (1 encoder, 1 decoder) | `configs/mvsec/validation/unet_small3_dav2_batch10.json` | `evaluate_results/mvsec_validation_unet_small3_dav2_batch10.csv` |
| DSEC -> MVSEC / U-Net DAv2 (learnable constant) | `configs/mvsec/validation/newunet_dav2_batch10.json` | `evaluate_results/mvsec_validation_newunet_dav2_batch10.csv` |
| DSEC -> MVSEC / FullyConv DAv2 (baseline) | `configs/mvsec/validation/fully_conv_dav2_batch10_RC.json` | `evaluate_results/mvsec_validation_fully_conv_dav2_batch10_RC.csv` |
| DSEC -> MVSEC / FullyConv DAv2 (grayscale) | `configs/mvsec/validation/fully_conv_1c_dav2_batch10.json` | `evaluate_results/mvsec_validation_fully_conv_1c_dav2_batch10.csv` |
| DSEC -> MVSEC / FullyConv DAv2 (learnable constant) | `configs/mvsec/validation/new_fully_conv_dav2_batch10_seed3.json` | `evaluate_results/mvsec_validation_new_fully_conv_dav2_batch10_seed3.csv` |
| MVSEC -> DSEC / U-Net DAv2 (baseline) | `configs/dsec/validation/train_mvsec_unet_dav2_batch10.json` | `evaluate_results/dsec_validation_train_mvsec_unet_dav2_batch10.csv` |
| MVSEC -> DSEC / U-Net DAv2 (ReLU activation) | `configs/dsec/validation/train_mvsec_unet_dav2_batch10_relu.json` | `evaluate_results/dsec_validation_train_mvsec_unet_dav2_batch10_relu.csv` |
| MVSEC -> DSEC / U-Net DAv2 (grayscale) | `configs/dsec/validation/train_mvsec_unet_1c_dav2_batch10.json` | `evaluate_results/dsec_validation_train_mvsec_unet_1c_dav2_batch10.csv` |
| MVSEC -> DSEC / U-Net DAv2 (16 channels) | `configs/dsec/validation/train_mvsec_unet_dav2_batch10_ch16.json` | `evaluate_results/dsec_validation_train_mvsec_unet_dav2_batch10_ch16.csv` |
| MVSEC -> DSEC / U-Net DAv2 (1 encoder, 1 decoder) | `configs/dsec/validation/train_mvsec_unet_small3_dav2_batch10.json` | `evaluate_results/dsec_validation_train_mvsec_unet_small3_dav2_batch10.csv` |
| MVSEC -> DSEC / U-Net DAv2 (learnable constant) | `configs/dsec/validation/train_mvsec_newunet_dav2_batch10.json` | `evaluate_results/dsec_validation_train_mvsec_newunet_dav2_batch10.csv` |
| MVSEC -> DSEC / FullyConv DAv2 (baseline) | `configs/dsec/validation/train_mvsec_fully_conv_dav2_batch10.json` | `evaluate_results/dsec_validation_train_mvsec_fully_conv_dav2_batch10.csv` |
| MVSEC -> DSEC / FullyConv DAv2 (grayscale) | `configs/dsec/validation/train_mvsec_fully_conv_1c_dav2_batch10.json` | `evaluate_results/dsec_validation_train_mvsec_fully_conv_1c_dav2_batch10.csv` |
| MVSEC -> DSEC / FullyConv DAv2 (learnable constant) | `configs/dsec/validation/train_mvsec_new_fully_conv_dav2_batch10.json` | `evaluate_results/dsec_validation_train_mvsec_new_fully_conv_dav2_batch10.csv` |

## Notes

- All paths for checkpoints, CSV output, and visualization are defined in the JSON configs.
- `evaluate.py` creates parent directories for `csv_path`.
