# ML Training

The ML training (implemented in ``train_model.py``) can be run in two ways:

- In your local Python environment, for testing/debugging: ``python train_model.py ...``

- Through the GUI, by clicking the ``Train`` button, or through SLURM by running ``sbatch training_pm.sbatch``.
In both cases, the training runs in a Docker container at NERSC. This Docker container
is pulled from the NERSC registry (https://registry.nersc.gov) and does not reflect any local changes
you may have made to ``train_model.py``, unless you re-build and re-deploy the container.

Both methods are described in more detail below.

## Training in a local Python environment (testing/debugging)

### On your local computer

For local development, ensure you have [Conda](https://conda-forge.org/download/) installed. Then:

1. Create the conda environment (this only needs to be done once):
   ```bash
   conda env create -f environment.yml
   ```

2. Open a separate terminal and keep it open:
   ```bash
   ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
   ```

3. Activate the conda environment and setup database read-write access:
   ```bash
   conda activate ml-training
   export SF_DB_ADMIN_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   ```

4. Run the training script in test mode:
   ```console
   python train_model.py --test --model <NN/GP> --config_file_path <your_test_yaml_file>
   ```

### At NERSC

1. Create the conda environment (this only needs to be done once):
   ```bash
   module load python
   conda env create --prefix /global/cfs/cdirs/m558/$(whoami)/sw/perlmutter/ml-training -f environment.yml
   ```

2. Activate the environment and setup database read-write access:
   ```bash
   module load python
   conda activate /global/cfs/cdirs/m558/$(whoami)/sw/perlmutter/ml-training
   export SF_DB_ADMIN_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   ```

3. Run the training script in test mode:
   ```console
   python train_model.py --test --model <NN/GP> --config_file_path <your_test_yaml_file>
   ```

## Training through the GUI or through SLURM

> **Warning:**
>
> Pushing a new Docker container affects training jobs launched from your locally-deployed GUI,
> but also from the production GUI (deployed on NERSC Spin), since in both cases, the training
> runs in a Docker container at NERSC, which is pulled from the NERSC registry (https://registry.nersc.gov).
>
> Yet, currently, this is the only way to test the end-to-end integration of the GUI with the training workflow.

1. Move to the root directory of the repository.

2. Build the Docker image based on `Dockerfile`:
   ```console
   docker build --platform linux/amd64 -t ml-training -f ml.Dockerfile .
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
    This has been also automated through the Python script [publish_container.py](https://github.com/BLAST-AI-ML/synapse/blob/main/publish_container.py), which can be executed via
    ```console
    python publish_container.py --ml
    ```

5. Optional test: Run the Docker container manually on Perlmutter:
   ```console
   ssh perlmutter-p1.nersc.gov

   podman-hpc login --username $USER registry.nersc.gov
   # Password: your NERSC password without 2FA

   podman-hpc pull registry.nersc.gov/m558/superfacility/ml-training:latest
   ```

   Ensure the file `$HOME/db.profile` contains a line `export SF_DB_ADMIN_PASSWORD=...` with the write password to the database.

   ```console
   salloc -N 1 --ntasks-per-node=1 -t 1:00:00 -q interactive -C gpu --gpu-bind=single:1 -c 32 -G 1 -A m558

   podman-hpc run --gpu -v /etc/localtime:/etc/localtime -v $HOME/db.profile:/root/db.profile -v /path/to/config.yaml:/app/ml/config.yaml --rm -it registry.nersc.gov/m558/superfacility/ml-training:latest python -u /app/ml/train_model.py --test --config_file_path /app/ml/config.yaml --model NN
   ```
   Note that `-v /etc/localtime:/etc/localtime` is necessary to synchronize the time zone in the container with the host machine.


> **Note:**
>
> When we run ML training jobs through the GUI, we use NERSC's Superfacility API with the collaboration account `sf558`.
> Since this is a non-interactive, non-user account, we also use a custom user to pull the image from https://registry.nersc.gov to Perlmutter.
> The registry login credentials need to be prepared (once) in the `$HOME` of `sf558` (`/global/homes/s/sf558/`) in a file named `registry.profile` with the following content:
> ```bash
> export REGISTRY_USER="robot\$m558+perlmutter-nersc-gov"
> export REGISTRY_PASSWORD="..."
> ```

## References

* https://docs.nersc.gov/development/containers/podman-hpc/overview/
* https://docs.nersc.gov/development/containers/registry/
