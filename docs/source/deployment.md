# Deployment

Synapse is typically deployed using Docker images, e.g., on Kubernetes.

Below, we document our public deployment workflow (recipes currently in a `private repo <https://github.com/BLAST-AI-ML/synapse-kubernetes-nersc>`__) using NERSC services like `Spin <https://docs.nersc.gov/services/spin/>`__.

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
- Training and simulations run on Perlmutter through Genesis AmSC IRI API or NERSC Superfacility API.
- Images are pushed to the registry of the deployment's NERSC project:

  ::::{tab-set}
  :sync-group: deployment

  :::{tab-item} General
  :sync: general

  ```text
  registry.nersc.gov/<nersc_project>/[<namespace>/]
  ```
  :::

  :::{tab-item} Project Example: BELLA @ NERSC
  :sync: bella-nersc

  ```text
  registry.nersc.gov/m558/superfacility/
  ```
  :::
  ::::

  {repo}`publish_container.py` hardcodes the BELLA path.
  For other projects, tag and push the images manually, as described in [Dashboard](dashboard.md#push-the-docker-image) and [ML training](ml-training.md#push-the-docker-image).
- Before publishing, validate the images locally.
