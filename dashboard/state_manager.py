import copy
import pandas as pd
from trame.app import get_server

server = get_server(client_type="vue2")
state = server.state
ctrl = server.controller

def initialize_state():
    """
    Helper function to initialize state variabes needed at startup.
    """
    print(f"Initializing state variables at startup...")
    # experiment and simulation data (pandas dataframes serialized)
    state.exp_data_serialized = pd.DataFrame().to_json(default_handler=str)
    state.sim_data_serialized = pd.DataFrame().to_json(default_handler=str)
    # experiment and model type
    state.experiment = "qed_ip2"
    state.model_type = "Neural Network"
    # opacity
    state.opacity = 0.05
    # calibration
    state.calibrate = True
    # Superfacility API
    state.sfapi_client_id = None
    state.sfapi_key = None
    state.sfapi_key_dict = None
    state.sfapi_key_expiration = "Unavailable"
    state.perlmutter_status = "Unavailable"
    # simulation plots in interactive dialog
    state.image_url = None
    state.image_dialog = False
