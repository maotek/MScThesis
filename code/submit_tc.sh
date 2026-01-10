#!/bin/bash
#SBATCH --job-name=tc
#SBATCH --output=tc%j.out
#SBATCH --error=tc%j.err
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=4096

module use /opt/insy/modulefiles # Use DAIC INSY software collection
module load cuda cudnn miniconda # Load certain versions of cuda and cudnn

apptainer exec --nv ../apptainer/image3.sif python -m train_dav2_on_tencode