## Commands
- `ssh maoshengjiang@login.daic.tudelft.nl`
- `module use /opt/insy/modulefiles`


## Paths
- `/tudelft.net/staff-umbrella/StudentsCVlab/mjiang`
- `/home/nfs/maoshengjiang`

## Make env
- `conda create --prefix /tudelft.net/staff-umbrella/StudentsCVlab/mjiang/envs/env python=3.10`
- `conda create -p ./envs/env python=3.9`

## Remove env
- `conda env remove`

## interactive sesh
- `sinteractive --cpus-per-task=1 --mem=8000 --time=00:30:00 --gres=gpu`

- `export CONDA_PKGS_DIRS="/tudelft.net/staff-umbrella/StudentsCVlab/mjiang/conda_pkgs"`

- `conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia`