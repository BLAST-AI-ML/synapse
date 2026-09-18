# Deployment

Synapse is deployed using Docker images and NERSC services.

## Build the dashboard image

From the repository root, build the image defined in {repo}`dashboard.Dockerfile`:

```bash
docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-gui -f dashboard.Dockerfile .
```

## Build the ML image

From the repository root, build the image defined in {repo}`ml.Dockerfile`:

```bash
docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-ml -f ml.Dockerfile .
```

The two build commands differ only by image tag and Dockerfile.

## Publish both images

{repo}`publish_container.py` builds and pushes both images:

```bash
python publish_container.py --gui --ml
```

## NERSC deployment assumptions

- Dashboard runs on Spin.
- Training and simulations run on Perlmutter through Superfacility API.
- Images are pushed to `registry.nersc.gov/m558/superfacility`.
- Before publishing, validate the images locally.
