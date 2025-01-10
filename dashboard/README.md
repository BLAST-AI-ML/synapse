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
If you run the GUI as a desktop application, make sure to set the following environment variable, too:
```console
export PYWEBVIEW_GUI=qt
```

4. Terminate the GUI via `Ctrl` + `C`.
