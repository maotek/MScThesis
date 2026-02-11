# Notes on DAIC HPC

## 1. Connecting to DAIC
Connect to TU Delft eduroam or use EduVPN.\
`ssh maoshengjiang@login.daic.tudelft.nl`

## 2. Change Directory to project folder
`cd /tudelft.net/staff-umbrella/StudentsCVlab/mjiang`

Default:\
`cd /home/nfs/maoshengjiang`

## 3. Load modules
`module use /opt/insy/modulefiles`\
`module load miniconda cuda cudnn`\
List available:\
`module avail`

## 4. Set CONDA environment
This is needed, otherwise conda will install in nfs/home, which has disk quota.\
`export CONDA_PKGS_DIRS="/tudelft.net/staff-umbrella/StudentsCVlab/mjiang/conda_pkgs"`

## 5. Make conda environment
`conda create -p ./envs/env python=3.9`

## 6. Start the env
`conda activate ./envs/env`

## 7. Interactive session
After logging in, we are in the Login node with limited resources, insufficient to install packages. So go into a compute node and install it there.
`sinteractive --cpus-per-task=1 --mem=8000 --time=00:30:00 --gres=gpu`

## 8. Install packages
Pip does not work, use conda install.\
`conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch -c nvidia`