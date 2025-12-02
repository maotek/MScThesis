#!/bin/bash
#SBATCH --job-name=check_cuda
#SBATCH --output=check_cuda_%j.out
#SBATCH --error=check_cuda_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --gres=gpu

module use /opt/insy/modulefiles # Use DAIC INSY software collection
module load cuda cudnn miniconda # Load certain versions of cuda and cudnn 
conda init bash
conda deactivate
conda activate ./envs/env

/usr/bin/nvidia-smi

conda 

srun python script.py