#!/bin/bash -l

#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -J trainsf
#SBATCH -A m558
#SBATCH -q realtime
#SBATCH --constraint=gpu
# ideally single:1, but NERSC cgroups issue
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -o /global/cfs/cdirs/m558/superfacility/model_training/logs/sf.o%j
#SBATCH -e /global/cfs/cdirs/m558/superfacility/model_training/logs/sf.e%j

model=${1}  # e.g., "NN", "GP", etc.

# login to the registry, update if needed
# Note: If you encounter issues, note that we compare image
#       Ids, but should compare image Digests.
#       We did not find a way to compare digests between the docker registry
#       and the local podman-hpc images (they change, conversion?)
SECONDS=0                      # built-in bash timer: reset
source $HOME/registry.profile  # credential variables: REGISTRY_USER and REGISTRY_PASSWORD
REGISTRY_NAME="registry.nersc.gov"
IMAGE_NAME="m558/superfacility/synapse-ml"
IMAGE_VERSION="latest"

podman-hpc login --username "${REGISTRY_USER}" --password "${REGISTRY_PASSWORD}" ${REGISTRY_NAME}
# As the local image, we use the digest id.
#   podman-hpc lists to entries with the same digest id,
#   one local and one migrated (read-write and read-only).
LOCAL_SHA="sha256:"$(podman-hpc images --digests --format "{{.Id}}" ${REGISTRY_NAME}/${IMAGE_NAME}:${IMAGE_VERSION} | head -n 1)
# OCI File Types defines the expected format
# See https://specs.opencontainers.org/image-spec/manifest/#oci-image-manifest-specification
OCI_INDEX_TYPE="application/vnd.oci.image.index.v1+json"
OCI_MANIFEST_TYPE="application/vnd.oci.image.manifest.v1+json"
#   first we unwarp the provenance information to pick the right image manifest
MANIFEST_INDEX=$(curl -s \
    -H "Accept: ${OCI_INDEX_TYPE}" \
    -u "${REGISTRY_USER}:${REGISTRY_PASSWORD}" \
    "https://${REGISTRY_NAME}/v2/${IMAGE_NAME}/manifests/${IMAGE_VERSION}" | \
    jq -r '.manifests[] | select(.annotations["vnd.docker.reference.type"] != "attestation-manifest") | .digest' || echo "NOOCI")
# Fallback for old Docker builders without OCI information
if [ "${MANIFEST_INDEX}" == "NOOCI" ]; then
    MANIFEST_INDEX=${IMAGE_VERSION}
fi
#   then we download the manifest of the image and parse out the config digest SHA
REMOTE_SHA=$(curl -s \
    -H "Accept: ${OCI_MANIFEST_TYPE}" \
    -u "${REGISTRY_USER}:${REGISTRY_PASSWORD}" \
    "https://${REGISTRY_NAME}/v2/${IMAGE_NAME}/manifests/${MANIFEST_INDEX}" | \
    jq -r '.config.digest')

if [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
    echo "Ids are different."
    echo "  Local: ${LOCAL_SHA}"
    echo "  Remote: ${REMOTE_SHA}"
    echo "Pulling the latest image..."
    podman-hpc pull ${REGISTRY_NAME}/${IMAGE_NAME}:${IMAGE_VERSION}
else
    echo "Local image is up to date."
fi
echo "Container check/pull took ${SECONDS} seconds."

# CUDA visible devices are ordered inverse to local task IDs
#   Reference: nvidia-smi topo -m
srun podman-hpc run --gpu \
    -v /etc/localtime:/etc/localtime \
    -v $HOME/db.profile:/root/db.profile \
    -v /global/cfs/cdirs/m558/superfacility/model_training/config.yaml:/app/ml/config.yaml \
    --rm -it ${REGISTRY_NAME}/${IMAGE_NAME}:${IMAGE_VERSION} \
    python -u /app/ml/train_model.py --config_file /app/ml/config.yaml --model ${model}
