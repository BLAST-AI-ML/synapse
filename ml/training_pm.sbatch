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

experiment=${1}  # e.g., "bella-ip2"
model=${2}  # e.g., "NN"

if [[ -z "${experiment}" ]]; then
    echo "Must pass the experiment name/tag as a command line argument"
fi

# login to the registry, update if needed
# Note: If you encounter issues, note that we compare image
#       Ids, but should compare image Digests.
#       We did not find a way to compare digests between the docker registry
#       and the local podman-hpc images (they change, conversion?)
SECONDS=0                      # built-in bash timer: reset
source $HOME/registry.profile  # credential variables: REGISTRY_USER and REGISTRY_PASSWORD
REGISTRY_NAME="registry.nersc.gov"
IMAGE_NAME="m558/superfacility/ml-training"
IMAGE_VERSION="latest"

podman-hpc login --username "${REGISTRY_USER}" --password "${REGISTRY_PASSWORD}" ${REGISTRY_NAME}
LOCAL_SHA="sha256:"$(podman-hpc images --digests --format "{{.Id}}" ${REGISTRY_NAME}/${IMAGE_NAME}:${IMAGE_VERSION})
# OCI_FILE_TYPES defines the expected format, for the content returned by the `curl` command below
# See https://specs.opencontainers.org/image-spec/manifest/#oci-image-manifest-specification
OCI_FILE_TYPES="application/vnd.oci.image.manifest.v1+json"  # official spec for the modern OCI format
REMOTE_SHA=$(curl -s \
    -H "Accept: ${OCI_FILE_TYPES}" \
    -u "${REGISTRY_USER}:${REGISTRY_PASSWORD}" \
    "https://${REGISTRY_NAME}/v2/${IMAGE_NAME}/manifests/${IMAGE_VERSION}" | \
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
srun podman-hpc run --gpu -v /etc/localtime:/etc/localtime -v $HOME/db.profile:/root/db.profile --rm -it ${REGISTRY_NAME}/${IMAGE_NAME}:${IMAGE_VERSION} python -u /app/ml/train_model.py --experiment ${experiment} --model ${model}
