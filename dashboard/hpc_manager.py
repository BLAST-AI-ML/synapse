from trame.widgets import vuetify3 as vuetify

from iriapi_manager import load_iriapi_card
from sfapi_manager import load_sfapi_card
from state_manager import state

# Mapping from internal execution mode values to display labels
EXECUTION_MODE_LABELS = {
    "local": "Local",
    "iriapi": "AmSC IRI API",
    "sfapi": "NERSC Superfacility API",
}
EXECUTION_MODE_ITEMS = [
    {"title": label, "value": mode} for mode, label in EXECUTION_MODE_LABELS.items()
]


def _log_mode_change(state_key, description):
    if len(state.modified_keys) == 1:
        value = state[state_key]
        label = EXECUTION_MODE_LABELS.get(value) or value
        print(f"{description} execution mode changed to {label}")


@state.change("simulation_running_mode")
def log_simulation_running_mode(**kwargs):
    _log_mode_change("simulation_running_mode", "Simulation")


@state.change("model_training_mode")
def log_model_training_mode(**kwargs):
    _log_mode_change("model_training_mode", "ML training")


def load_hpc_card():
    print("Setting HPC card...")
    with vuetify.VRow():
        with vuetify.VCol(cols=12):
            with vuetify.VCard():
                vuetify.VCardTitle("Execution Mode")
                with vuetify.VCardText():
                    with vuetify.VRow():
                        with vuetify.VCol(cols=12, md=6):
                            vuetify.VSelect(
                                v_model=("simulation_running_mode",),
                                label="Simulations",
                                items=(EXECUTION_MODE_ITEMS,),
                                dense=True,
                                hide_details=True,
                            )
                        with vuetify.VCol(cols=12, md=6):
                            vuetify.VSelect(
                                v_model=("model_training_mode",),
                                label="ML Training",
                                items=(EXECUTION_MODE_ITEMS,),
                                dense=True,
                                hide_details=True,
                            )
    with vuetify.VRow():
        with vuetify.VCol(cols=12, md=6):
            with vuetify.VCard():
                vuetify.VCardTitle("NERSC Superfacility API")
                with vuetify.VCardText():
                    load_sfapi_card()
        with vuetify.VCol(cols=12, md=6):
            with vuetify.VCard():
                vuetify.VCardTitle("AmSC IRI API")
                with vuetify.VCardText():
                    load_iriapi_card()
