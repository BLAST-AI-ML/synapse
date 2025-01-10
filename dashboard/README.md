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
conda install -c conda-forge lume-model matplotlib pandas plotly scipy trame trame-vuetify trame-plotly
conda install -c pytorch pytorch torchvision torchaudio cpuonly
conda install -c conda-forge botorch==0.10.0
pip install pywebview[qt]
```

## How to build the Docker image and run the Docker container

1. Build the Docker image based on `Dockerfile`:
```console
sudo docker build -t gui-docker-image .
```

2. Run the Docker container:
```console
sudo docker run -p 8080:8080 gui-docker-image
```
