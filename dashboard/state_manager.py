from pathlib import Path
from trame.app import get_server
from trame.widgets import vuetify3 as vuetify


EXPERIMENTS_PATH = Path.cwd().parent / "experiments/"


server = get_server(client_type="vue3")
state = server.state
ctrl = server.controller
vuetify.enable_lab()  # Enable Labs components


def initialize_state():
    """
    Helper function to initialize state variabes needed at startup.
    """
    print("Initializing state variables at startup...")
    # Experiment
    default_experiment = [
        d.name.removeprefix("synapse-")
        for d in EXPERIMENTS_PATH.iterdir()
        if d.is_dir()
    ][0]
    print(f"Setting default experiment to {default_experiment}...")
    state.experiment = default_experiment
    state.experiment_date_range = []
    # ML model
    state.model_type_verbose = "Neural Network (single)"
    state.model_training = False
    state.model_training_mode = "local"
    state.model_training_status = None
    state.model_training_time = None
    # Optimization
    state.optimization_type = "Maximize"
    state.optimization_status = None
    # Opacity
    state.opacity = 0.05
    # HPC connection
    state.hpc_connection = "local"
    # IRI API
    state.iriapi_key = None
    state.iriapi_key_dict = None
    state.iriapi_key_expiration = "Unavailable"
    state.iriapi_perlmutter_description = "Unavailable"
    state.iriapi_perlmutter_status = "unavailable"
    # Superfacility API
    state.sfapi_client_id = None
    state.sfapi_key = None
    state.sfapi_key_dict = None
    state.sfapi_key_expiration = "Unavailable"
    state.sfapi_perlmutter_description = "Unavailable"
    state.sfapi_perlmutter_status = "unavailable"
    # Simulation plots in interactive dialog
    state.simulation_url = None
    state.simulation_dialog = False
    state.simulation_video = False
    # Simulation jobs
    state.simulation_running = False
    # state.simulation_running_mode = "local"  # TODO
    state.simulation_running_status = None
    state.simulation_running_time = None
    state.simulatable = False
    # Errors management
    state.errors = []
    state.error_counter = 0
    # Calibration toggles
    state.use_inferred_calibration = False
