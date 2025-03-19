from datetime import datetime, timedelta
from trame.app import get_server

server = get_server(client_type="vue2")
state = server.state
ctrl = server.controller

def init_state():
    """
    Helper function to collect and define all state variabes.
    """
    state.trame_title = "IFE Superfacility"
    # opacity
    state.opacity = 0.1
    # calibration
    state.is_calibrated = False
    # experiment
    state.experiment = None
    # parameters
    state.parameters = dict()
    state.parameters_min = dict()
    state.parameters_max = dict()
    # TODO
    ## objectives
    #state.objectives = dict()
    ## NERSC
    #state.client_id = None
    #state.private_key = None
    #state.sfapi_expiration_days = 33
    #state.sfapi_expiration = str(
    #    datetime.now() +
    #    timedelta(days=int(state.sfapi_expiration_days))
    #)
    #state.sfapi_output = ""
    #state.perlmutter_status = None
