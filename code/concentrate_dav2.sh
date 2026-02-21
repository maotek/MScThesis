#!/bin/bash
#SBATCH --job-name=concentrate_dav2
#SBATCH --output=concentrate_dav2%j.out
#SBATCH --error=concentrate_dav2%j.err
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

apptainer exec --nv ../../apptainer/image.sif python -m train_concentration_dav2 --config configs/train/concentration_dav2_voxelgrid.json --save-dir output/train_concentration_dav2