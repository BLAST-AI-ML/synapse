#from datetime import datetime, timedelta
import copy
import inspect
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
    # need a separate variable to track changes in state.experiment,
    # which trigger re-initialization, to avoid multiple reactive functions
    # listening to changes in state.experiment, as the order of execution of
    # such reactive functions cannot be prescribed
    state.experiment = "qed_ip2"
    state.experiment_old = copy.deepcopy(state.experiment)
    state.experiment_changed = False
    state.model_type = "Neural Network"
    state.model_type_changed = False
    state.model_type_old = copy.deepcopy(state.model_type)
    # opacity
    state.opacity = 0.05

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
    # NOTE parameters state variables initialized in parameters manager constructor
    # NOTE objectives state variables initialized in objectives manager constructor
    # NERSC
    state.sfapi_output = ""
    # TODO
    #state.client_id = None
    #state.private_key = None
    #state.sfapi_expiration_days = 33
    #state.sfapi_expiration = str(
    #    datetime.now() +
    #    timedelta(days=int(state.sfapi_expiration_days))
    #)
    #state.perlmutter_status = None
