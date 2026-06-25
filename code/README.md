# Event Depth Evaluation

This folder contains training and evaluation pipelines for event-based depth models on DSEC and MVSEC.

## Setup

Run all project commands from the `code/` directory. The configs use relative
paths, so running from another directory will make datasets and checkpoints look
missing.

### 1) Create and activate the environment

From the repository root:

```bash
cd code
conda env create -f environment/environment.yml
conda activate apptainer
```

If the environment already exists, update it instead:

```bash
conda env update -f environment/environment.yml --prune
conda activate apptainer
```

The environment installs the PyTorch CUDA 12.3 wheels from
`environment/requirements.txt`. If your machine uses a different CUDA setup,
install the matching PyTorch build before running training or evaluation.

### 2) Put data in the expected locations

All dataset paths are relative to `code/`. The validation and training configs
expect this layout:

```text
code/
  datasets/
    DSEC/
      data/
        train/<sequence>/
        validation/<sequence>/
    MVSEC/
      data/
        train/<sequence>/
        test/<sequence>/
```

Create the top-level directories if they do not exist:

```bash
mkdir -p datasets/DSEC/data datasets/MVSEC/data
```

For full table reproduction, download all DSEC and MVSEC splits:

```bash
python -m scripts.download_dsec --split all --out datasets/DSEC/data
python -m scripts.download_mvsec --split all --stage all --out datasets/MVSEC/data
```

For evaluation-only runs, use the split needed by the config you are running:
DSEC configs read from `datasets/DSEC/data`, and MVSEC configs read from
`datasets/MVSEC/data`.

### 3) Put model weights and checkpoints in place

The Depth Anything V2 backbone checkpoint should be placed under:

```text
models/dav2/checkpoints/depth_anything_v2_vits.pth
```

The DAE validation configs expect the Depth AnyEvent checkpoints here:

```text
models/depthanyevent/weights/dav2/finetuned_dsec/finetuned_dsec.pth
models/depthanyevent/weights/dav2/finetuned_mvsec/finetuned_mvsec.pth
```

Trainable adapter checkpoints are read from `train_output/<run_name>/`. For
example:

```text
train_output/train_dsec_unet_dav2_batch10/epoch_050.pt
train_output/train_mvsec_fully_conv_dav2_batch10/epoch_050.pt
```

You can create these checkpoints by running the training configs, or download
the checkpoints used for reproducing Tables 1-4 from Google Drive:

https://drive.google.com/drive/folders/141tTRwiy9V2DUP21tcBa4q42HQ0XpqRl?usp=sharing

Place the downloaded run directories under `train_output/` so the config paths
continue to resolve. If you have DAIC access, you can also fetch checkpoints
from the remote:

```bash
mkdir -p train_output
bash scripts/download_weights.sh --password '<password>' --file epoch_050.pt
```

After the datasets and checkpoints are present, use the commands below for
training, single-config evaluation, or reproducing paper tables.

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
