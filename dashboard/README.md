## How to set up and run the GUI

1. Create a conda environment from the available YAML file (only once):
```console
conda env create -f gui.yml
```

2. Activate the `gui` conda environment:
```console
conda activate gui
```

3. Run the GUI via web browser interface:
```console
python app.py --port 1234
```
Alternatively, you can also run the GUI as a desktop application:
```console
python app.py --app
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
docker run -p 8080:8080 -v /path/to/experimental_data:/app/experimental_data -v /path/to/simulation_data:/app/simulation_data -v /path/to/ml/NN_training:/app/ml/NN_training gui
```

3. Optional: Publish container privately to https://registry.nersc.gov (or publicly to https://hub.docker.com/):
```console
docker login registry.nersc.gov
# Username: your NERSC username
# Password: your NERSC password without 2FA
```
```console
docker tag gui:latest registry.nersc.gov/m558/superfacility_ldrd/gui:latest
docker push registry.nersc.gov/m558/superfacility_ldrd/gui:latest
```
