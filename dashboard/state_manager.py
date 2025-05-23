#from datetime import datetime, timedelta
import copy
import os
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
    state.is_initialized = False
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
    # calibration
    state.calibrate = True
    # Superfacility API
    state.sfapi_key = None
    state.sfapi_key_dict = None
    state.sfapi_key_expiration = "Unavailable"
    state.sfapi_client_id = None
    state.perlmutter_status = "Unavailable"
