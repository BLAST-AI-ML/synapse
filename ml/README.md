## ML Training how-to guide for users and developers


### Prerequisites
- Ensure you have [Conda](https://conda-forge.org/download/) installed.
- Ensure you have Docker installed (for deployment)


### How to set up the conda environment

For local development, create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate ml-training
```


### How to build and run the Docker container

1. Move to the root directory of the repository.

2. Build the Docker image based on `Dockerfile`:
    ```console
    docker build --platform linux/amd64 -t ml-training -f ml/Dockerfile .
    ```

3. Optional: From time to time, as you develop the container, you might want to prune old, unused images to get back GBytes of storage on your development machine:
    ```console
    docker system prune -a
    ```

4. Publish the container privately to NERSC registry (https://registry.nersc.gov):
    ```console
    docker login registry.nersc.gov
    # Username: your NERSC username
    # Password: your NERSC password without 2FA
    ```

    ```console
    docker tag ml-training:latest registry.nersc.gov/m558/superfacility/ml-training:latest
    docker tag ml-training:latest registry.nersc.gov/m558/superfacility/ml-training:$(date "+%y.%m")
    docker push -a registry.nersc.gov/m558/superfacility/ml-training
    ```

5. Cache & Share the Docker container on Perlmutter:
    ```console
    ssh perlmutter-p1.nersc.gov

    podman-hpc login --username $USER registry.nersc.gov
    # Password: your NERSC password without 2FA

    podman-hpc pull registry.nersc.gov/m558/superfacility/ml-training:latest
    ```

6. Optional test: Run the Docker container manually on Perlmutter:
    Ensure the file `$HOME/db.profile` contains a line `SF_DB_ADMIN_PASSWORD=...` with the write password to the database.
    ```console
    salloc -N 1 --ntasks-per-node=1 -t 1:00:00 -q interactive -C gpu --gpu-bind=single:1 -c 32 -G 1 -A m558

    podman-hpc run --gpu -v $HOME/db.profile:/root/db.profile --rm -it ml-training:latest python -u /app/ml/train_model.py --experiment ip2 --model NN
    ```
    Note that `-v /etc/localtime:/etc/localtime` is necessary to synchronize the time zone in the container with the host machine.


## References

* https://docs.nersc.gov/development/containers/podman-hpc/overview/
* https://docs.nersc.gov/development/containers/registry/
