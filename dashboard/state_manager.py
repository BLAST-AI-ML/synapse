from trame.app import get_server

server = get_server(client_type="vue2")
state = server.state
ctrl = server.controller


def initialize_state():
    """
    Helper function to initialize state variabes needed at startup.
    """
    print("Initializing state variables at startup...")
    # experiment and model type
    state.experiment = "staging_injector"
    state.model_type = "Neural Network"
    state.model_training = False
    state.model_training_status = "Completed"
    state.model_training_time = None
    # opacity
    state.opacity = 0.05
    # Superfacility API
    state.sfapi_client_id = None
    state.sfapi_key = None
    state.sfapi_key_dict = None
    state.sfapi_key_expiration = "Unavailable"
    state.perlmutter_description = "Unavailable"
    state.perlmutter_status = "unavailable"
    # simulation plots in interactive dialog
    state.simulation_url = None
    state.simulation_dialog = False
    state.simulation_video = False
    # Errors management
    state.errors = []
    state.error_counter = 0
