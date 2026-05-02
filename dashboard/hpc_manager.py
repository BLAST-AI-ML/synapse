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


def execution_mode_items():
    return [
        {"title": label, "value": mode} for mode, label in EXECUTION_MODE_LABELS.items()
    ]


@state.change("simulation_running_mode")
def log_simulation_running_mode(**kwargs):
    # skip if triggered on server ready (all state variables marked as modified)
    if len(state.modified_keys) == 1:
        label = EXECUTION_MODE_LABELS.get(
            state.simulation_running_mode, state.simulation_running_mode
        )
        print(f"Simulation execution mode changed to {label}")


@state.change("model_training_mode")
def log_model_training_mode(**kwargs):
    # skip if triggered on server ready (all state variables marked as modified)
    if len(state.modified_keys) == 1:
        label = EXECUTION_MODE_LABELS.get(
            state.model_training_mode, state.model_training_mode
        )
        print(f"ML training execution mode changed to {label}")


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
                                items=(execution_mode_items(),),
                                dense=True,
                                hide_details=True,
                            )
                        with vuetify.VCol(cols=12, md=6):
                            vuetify.VSelect(
                                v_model=("model_training_mode",),
                                label="ML Training",
                                items=(execution_mode_items(),),
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
