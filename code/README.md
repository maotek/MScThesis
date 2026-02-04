# Run Instructions

## Set Up Conda Environment
```
conda env create -f ./environment/environment.yml
```

## Activate
```
conda activate apptainer
```

## Download the required DSEC data
```
python -m scripts/download_dsec
python -m run_vitb_on_tencode_func
```

## Run commands

python -m evaluate --config-path configs/validation/dav2_tencode.json --csv-path results/dav2_tencode.csv

python -m evaluate --config-path configs/validation/dav2_tencode_pixelcount.json --csv-path results/dav2_tencode_pixelcount.csv

python -m evaluate --config-path configs/validation/dae_tencode.json --csv-path results/dae_tencode.csv

python -m evaluate --config-path configs/validation/dav2_rgb.json --csv-path results/dav2_rgb.csv

python -m evaluate --config-path configs/validation/e2vid_dav2_voxelgrid.json --csv-path results/e2vid_dav2_voxelgrid.csv

python -m evaluate --config-path configs/validation/e2vid_dav2_composite_voxelgrid.json --csv-path results/e2vid_dav2_composite_voxelgrid.csv

python -m evaluate --config-path configs/validation/etnet_dav2_voxelgrid.json --csv-path results/etnet_dav2_voxelgrid.csv