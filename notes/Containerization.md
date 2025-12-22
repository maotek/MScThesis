## Connecting to DAIC
Connect to TU Delft eduroam or use EduVPN.\
```
ssh maoshengjiang@login.daic.tudelft.nl
```

## Load modules
```
module use /opt/insy/modulefiles
module load miniconda cuda cudnn
```

## Change Directory to project folder
```
cd /tudelft.net/staff-umbrella/ThesisMaosheng
```

## Important: Cache and filesystem limits
By default, Apptainer images are saved to ~/.apptainer. To avoid quota issues, set the environment variable APPTAINER_CACHEDIR to a different location.

```
export APPTAINER_CACHEDIR=/tudelft.net/staff-umbrella/ThesisMaosheng/apptainer/cache
```

Pulling directly to bulk or umbrella is not supported, so pull large images locally, then copy the *.sif file to DAIC.

## Install apptainer on MacOS
Install lima
```
brew install lima
brew install lima-additional-guestagents
```
### Create an x86 version so it can build conda with CUDA on apple silicon
```
limactl stop apptainer-x86 2>/dev/null || true
limactl delete apptainer-x86 2>/dev/null || true
```
```
limactl create \
  --name apptainer-x86 \
  --arch x86_64 \
  --vm-type=qemu \
  template://ubuntu


limactl create --name apptainer apptainer.yaml
```

## Install apptainer
```
add-apt-repository -y ppa:apptainer/ppa
apt-get update
apt-get install -y apptainer
```

## Build the image
```
apptainer build image.sif Apptainer.def
```

## Test it out on DAIC
```
apptainer exec demo-env-image.sif which python
```