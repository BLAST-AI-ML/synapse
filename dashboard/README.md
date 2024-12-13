## How to set up and run the GUI

1. Create a conda environment from the available YAML file (only once):
```console
conda env create -n gui -f gui.yml
```

2. Activate the `gui` conda environment:
```console
conda activate gui
```

3. Run the GUI via web browser interface:
```console
python app.py --port 1234
```

4. Terminate the GUI via `Ctrl` + `C`.
