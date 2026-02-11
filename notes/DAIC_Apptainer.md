# Building the apptainer image

## 1. Install apptainer on MacOS
Install lima
```bash
brew install lima
brew install lima-additional-guestagents
```
### Create an x86 version so it can build conda with CUDA on apple silicon
```bash
limactl create \
  --name apptainer-x86 \
  --arch x86_64 \
  --vm-type=qemu \
  template://ubuntu
```
Or use yaml file to increase RAM (apparantly crashes without)
```bash
limactl create --name apptainer apptainer.yaml
```

## 2. Start the VM
```bash
limactl start apptainer
limactl shell apptainer
```

## 3. Install apptainer in the VM
```bash
add-apt-repository -y ppa:apptainer/ppa
apt-get update
apt-get install -y apptainer
```

## 4. Build the image
```bash
apptainer build image3.sif Apptainer2.def
```

## 5. Stop the VM when done
```bash
exit
limactl stop apptainer
```

## 6. Optional: remove the VM
```bash
limactl delete apptainer
```

# Test it out on DAIC
## 1. Connecting to DAIC
Connect to TU Delft eduroam or use EduVPN.
```bash
ssh maoshengjiang@login.daic.tudelft.nl
```

## 2. Load modules
```bash
module use /opt/insy/modulefiles
module load miniconda cuda cudnn
```

## 3. Change Directory to project folder
```bash
cd /tudelft.net/staff-umbrella/ThesisMaosheng
```

### Important: Cache and filesystem limits
By default, Apptainer images are saved to ~/.apptainer. To avoid quota issues, set the environment variable APPTAINER_CACHEDIR to a different location.

```bash
export APPTAINER_CACHEDIR=/tudelft.net/staff-umbrella/ThesisMaosheng/apptainer/cache
```

Pulling directly to bulk or umbrella is not supported, so pull large images locally, then copy the *.sif file to DAIC.

## 4. Load modules
```bash
module use /opt/insy/modulefiles
module load miniconda cuda cudnn
```

## 5. Start interactive shell for testing with GPU
```bash
sinteractive --cpus-per-task=1 --mem=8000 --time=00:30:00 --gres=gpu
```

Run script directly:
```bash
apptainer exec --nv apptainer/image3.sif python train.py
```

Or shell into it:
```bash
apptainer shell --nv apptainer/image.sif
```

# SCP data to DAIC

## 1. Copy image to DAIC from local machine
```bash
scp image3.sif maoshengjiang@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/ThesisMaosheng/apptainer/
```

## 2. Copy DSEC data to DAIC (same project folder)
```bash
scp -r datasets/DSEC/data maoshengjiang@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/ThesisMaosheng/MScThesis/code/datasets/DSEC/
```

## 3. (Optional) Copy results back to local
```bash
scp -r maoshengjiang@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/ThesisMaosheng/output ./output
```
