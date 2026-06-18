# Deployment

Synapse is deployed with Docker images and NERSC services.

## Dashboard Image

From the repository root:

```bash
docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-gui -f dashboard.Dockerfile .
```

## ML Image

From the repository root:

```bash
docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-ml -f ml.Dockerfile .
```

## Publish Helper

```bash
python publish_container.py --gui --ml
```

## NERSC Assumptions

- Dashboard runs on Spin.
- Training and simulations run on Perlmutter through Superfacility API.
- Images are pushed to `registry.nersc.gov/m558/superfacility`.
- Production changes should be tested carefully before publishing images.
