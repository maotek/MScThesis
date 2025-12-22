## Connecting to DAIC
Connect to TU Delft eduroam or use EduVPN.\
`ssh maoshengjiang@login.daic.tudelft.nl`

## Change Directory to project folder
`cd /tudelft.net/staff-umbrella/ThesisMaosheng`

## Important: Cache and filesystem limits
By default, Apptainer images are saved to ~/.apptainer. To avoid quota issues, set the environment variable APPTAINER_CACHEDIR to a different location.

```export APPTAINER_CACHEDIR=/tudelft.net/staff-umbrella/ThesisMaosheng/apptainer/cache```

Pulling directly to bulk or umbrella is not supported, so pull large images locally, then copy the *.sif file to DAIC.