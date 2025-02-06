## How to set up and run the GUI

1. Create a conda environment from the available YAML file (only once):
```console
conda env create -f gui.yml
```

2. Activate the `gui` conda environment:
```console
conda activate gui
```

3. Set the database settings:
```console
export SF_DB_HOST="127.0.0.1"
export SF_DB_READONLY_PASSWORD='...'  # mind the SINGLE quotes around the PW!
```

4. For local development, open an extra terminal and keep it open while we SSH forward the database connection:
```console
ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
```

5. Run the GUI from the `dashboard/` folder.

Via the web browser interface:
```console
python app.py --port 1234 --model ../ml/NN_training/base_simulation_model_with_transformers_calibration.yml
```
Alternatively, you can also run the GUI as a desktop application:
```console
python app.py --app --model ../ml/NN_training/base_simulation_model_with_transformers_calibration.yml
```
If you run the GUI as a desktop application, make sure to set the following environment variable first:
```console
export PYWEBVIEW_GUI=qt
```

4. Terminate the GUI via `Ctrl` + `C`.

## How to create the GUI conda environment from scratch

The GUI conda environment can be created from scratch (e.g., for debugging purposes) as follows:
```console
conda create -n gui
conda activate gui
conda install python==3.12
conda install -c conda-forge lume-model matplotlib pandas plotly scipy trame trame-plotly trame-vuetify
conda install -c pytorch pytorch torchvision torchaudio cpuonly
conda install -c conda-forge botorch==0.10.0
pip install pywebview[qt]
```

## How to build the Docker image and run the Docker container

1. Build the Docker image based on `Dockerfile`:
```console
docker build -t gui .
```

2. Run the Docker container:
```console
docker run -p 8080:8080 -v /path/to/ml/NN_training:/app/ml/NN_training gui
```
e.g., for development inside this `dashboard/` directory:
```console
docker run -p 8080:8080 -v $PWD/../ml:/app/ml gui
```
You can also enter the container for debugging, without starting the app, via:
```console
docker run -p 8080:8080 -v $PWD/../ml:/app/ml -it gui bash
```

3. Optional: Publish container privately to https://registry.nersc.gov (or publicly to https://hub.docker.com/):
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
docker image prune
docker container prune

docker system prune
```
