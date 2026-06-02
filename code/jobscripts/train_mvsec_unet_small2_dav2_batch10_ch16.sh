#!/bin/bash
#SBATCH --job-name=train_mvsec_unet_small2_dav2_batch10_ch16
#SBATCH --output=train_mvsec_unet_small2_dav2_batch10_ch16_%j.out
#SBATCH --error=train_mvsec_unet_small2_dav2_batch10_ch16_%j.err
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=160000

module use /opt/insy/modulefiles # Use DAIC INSY software collection
module load cuda cudnn miniconda # Load certain versions of cuda and cudnn

cd MScThesis/code

apptainer exec --nv ../../apptainer/image.sif python -m train_unet_dav2 --config-path configs/mvsec/train/unet_small2_dav2_batch10_ch16.json
