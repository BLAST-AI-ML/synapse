## GUI how-to guide for users and developers

### Prerequisites
- Ensure you have Conda installed.
- Ensure you have Docker installed (if you plan to use Docker).

### How to create a new conda environment lock file

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

### How to set up the conda environment

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

### How to run the GUI

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
    python app.py --port 8080
    ```
    - As a desktop application:
    ```console
    python app.py --app
    ```
    If you run the GUI as a desktop application, make sure to set the following environment variable first:
    ```console
    export PYWEBVIEW_GUI=qt
    ```

5. Terminate the GUI via `Ctrl` + `C`.

### How to build and run the Docker container

1. Build the Docker image based on `Dockerfile`:
    ```console
    docker build -t gui .
    ```

2. Run the Docker container from the `dashboard/` folder:
    ```console
    docker run --network=host -p 27017:27017 -v /etc/localtime:/etc/localtime -v $PWD/../ml:/app/ml -e SF_DB_HOST='127.0.0.1' -e SF_DB_PASSWORD='your_password_here' gui
    ```
    For debugging, you can also enter the container without starting the app:
    ```console
    docker run --network=host -p 27017:27017 -v /etc/localtime:/etc/localtime -v $PWD/../ml:/app/ml -e SF_DB_HOST='127.0.0.1' -e SF_DB_PASSWORD='your_password_here' -it gui bash
    ```
    Note that `-v /etc/localtime:/etc/localtime` is necessary to synchronize the time zone in the container with the host machine.

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

### How to get the Superfacility API credentials

Following the instructions at [docs.nersc.gov/services/sfapi/authentication/#client](https://docs.nersc.gov/services/sfapi/authentication/#client):

1. Log in to your profile page at [iris.nersc.gov/profile](https://iris.nersc.gov/profile).

2. Click the icon with your username in the upper right of the profile page.

3. Scroll down to the section "Superfacility API Clients" and click "New Client".

4. Enter a client name (e.g., "BELLA Superfacility"), choose "Red" security level, and select "Your IP" from the "IP Presets" menu.

5. Download your private key file (in pem format).

6. Copy your client ID and add it on the first line of your private key file as described in the instructions at [nersc.github.io/sfapi_client/quickstart/#storing-keys-in-files](https://nersc.github.io/sfapi_client/quickstart/#storing-keys-in-files):
    ```
    randmstrgz
    -----BEGIN RSA PRIVATE KEY-----
    ...
    -----END RSA PRIVATE KEY-----
    ```

7. Run `chmod 600 priv_key.pem` to change the permissions of your private key file to read/write only.
