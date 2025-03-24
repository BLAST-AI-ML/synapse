## How to Run the GUI

### Prerequisites
- Ensure you have Conda installed.
- Ensure you have Docker installed (if you plan to use Docker).

### Setting Up the Conda Environment

1. Activate the `base` conda environment:
    ```console
    conda activate base
    ```

2. Install `conda-lock` (if not already installed):
    ```console
    conda install -c conda-forge conda-lock
    ```

3. Create the `gui` conda environment from the lock file:
    ```console
    conda-lock install --name gui gui-lock.yml
    ```

### Running the GUI

1. Activate the `gui` conda environment:
    ```console
    conda activate gui
    ```

2. Set the database settings (read+write):
    ```console
    export SF_DB_HOST='127.0.0.1'
    export SF_DB_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
    ```

3. For local development, open a separate terminal and keep it open while SSH forwarding the database connection:
    ```console
    ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
    ```

4. Run the GUI from the `dashboard/` folder:
    - Via the web browser interface:
    ```console
    python app.py --port 8080 --model $PWD/../ml/NN_training/saved_models/ip2.yml
    ```
    - As a desktop application:
    ```console
    python app.py --app --model $PWD/../ml/NN_training/saved_models/ip2.yml
    ```
    If you run the GUI as a desktop application, make sure to set the following environment variable first:
    ```console
    export PYWEBVIEW_GUI=qt
    ```

5. Terminate the GUI via `Ctrl` + `C`.

## How to Build and Run the Docker Container

1. Build the Docker image based on `Dockerfile`:
    ```console
    docker build -t gui .
    ```

2. Run the Docker container from the `dashboard/` folder:
    ```console
    docker run --network=host -p 27017:27017 -v $PWD/../ml:/app/ml -e SF_DB_HOST='127.0.0.1' -e SF_DB_PASSWORD='your_password_here' gui
    ```
    For debugging, you can also enter the container without starting the app:
    ```console
    docker run --network=host -p 27017:27017 -v $PWD/../ml:/app/ml -e SF_DB_HOST='127.0.0.1' -e SF_DB_PASSWORD='your_password_here' -it gui bash
    ```

3. Optional: Publish the container privately to NERSC registry (https://registry.nersc.gov):
    ```console
    docker login registry.nersc.gov
    # Username: your NERSC username
    # Password: your NERSC password without 2FA
    ```
    ```console
    docker tag gui:latest registry.nersc.gov/m558/superfacility/gui:latest
    docker tag gui:latest registry.nersc.gov/m558/superfacility/gui:$(date "+%y.%m")
    docker push -a registry.nersc.gov/m558/superfacility/gui
    ```

4. Optional: From time to time, as you develop the container, you might want to prune old, unused images to get back GBytes of storage on your development machine:
    ```console
    docker system prune -a
    ```

## How to Create the Conda Environment Lock File (For Maintainers)

1. Activate the `base` conda environment:
    ```console
    conda activate base
    ```

2. Install `conda-lock` (if not already installed):
    ```console
    conda install -c conda-forge conda-lock
    ```

3. Create the lock file starting from the existing minimal environment file:
    ```console
    conda-lock --file gui-base.yml --lockfile gui-lock.yml
    ```
