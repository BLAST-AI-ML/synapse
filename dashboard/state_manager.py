from datetime import datetime, timedelta
import pandas as pd
from trame.app import get_server

server = get_server(client_type="vue2")
state = server.state
ctrl = server.controller

def init_state():
    """
    Helper function to collect and define all state variabes.
    """
    state.trame_title = "IFE Superfacility"
    # experiment
    state.experiment = None
    # serialized data
    state.exp_data = pd.DataFrame().to_json(default_handler=str)
    state.sim_data = pd.DataFrame().to_json(default_handler=str)
    # opacity
    state.opacity = 0.1
    # calibration
    state.is_calibrated = False
    # parameters
    state.parameters = dict()
    state.parameters_min = dict()
    state.parameters_max = dict()
    # objectives
    state.objectives = dict()
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
