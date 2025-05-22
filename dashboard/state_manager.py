#from datetime import datetime, timedelta
import copy
import os
import pandas as pd
from trame.app import get_server

server = get_server(client_type="vue2")
state = server.state
ctrl = server.controller

def init_startup():
    """
    Helper function to initialize state variabes needed at startup.
    """
    print(f"Initializing state variables at startup...")
    state.nersc_route_built = False
    state.ui_layout_built = False
    # need separate variables to track changes in state.experiment and
    # state.model_type, which trigger re-initialization, to avoid multiple
    # reactive functions listening to changes in those variables (the order
    # of execution of such reactive functions cannot be prescribed)
    state.experiment = "qed_ip2"
    state.experiment_old = copy.deepcopy(state.experiment)
    state.experiment_changed = False
    state.model_type = "Neural Network"
    state.model_type_changed = False
    state.model_type_old = copy.deepcopy(state.model_type)
    # opacity
    state.opacity = 0.05
    # Superfacility API
    state.sfapi_key = None
    state.sfapi_key_dict = None
    state.sfapi_key_expiration = "Unavailable"
    state.sfapi_client_id = None
    state.perlmutter_status = "Unavailable"

def init_runtime():
    """
    Helper function to (re-)initialize state variables at runtime.
    """
    print(f"Initializing state variables at runtime...")
    # serialized data
    state.exp_data = pd.DataFrame().to_json(default_handler=str)
    state.sim_data = pd.DataFrame().to_json(default_handler=str)
    # calibration
    state.is_calibrated = False
    # parameters and objectives state variables are initialized
    # in the respective manager class constructors
