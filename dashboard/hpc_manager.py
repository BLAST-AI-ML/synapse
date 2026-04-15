from trame.widgets import vuetify3 as vuetify

from sfapi_manager import load_sfapi_card
from state_manager import state

# Mapping from internal execution mode values to display labels
HPC_CONNECTION_LABELS = {
    "local": "Local",
    "iriapi": "AmSC IRI API",
    "sfapi": "NERSC Superfacility API",
}


@state.change("hpc_connection")
def sync_hpc_connection(**kwargs):
    # skip if triggered on server ready (all state variables marked as modified)
    if len(state.modified_keys) == 1:
        print(
            f"HPC connection changed to {HPC_CONNECTION_LABELS[state.hpc_connection]}"
        )
        # Overwrite model training mode and simulation running mode
        state.model_training_mode = state.hpc_connection
        # state.simulation_running_mode = state.hpc_connection  # TODO


def load_hpc_card():
    print("Setting HPC card...")
    with vuetify.VCard():
        with vuetify.VCardTitle("HPC Connection"):
            with vuetify.VCardText():
                with vuetify.VRow():
                    with vuetify.VCol():
                        vuetify.VSelect(
                            v_model=("hpc_connection",),
                            label="Execution Mode",
                            items=(
                                [
                                    {"title": label, "value": mode}
                                    for mode, label in HPC_CONNECTION_LABELS.items()
                                ],
                            ),
                            dense=True,
                            hide_details=True,
                        )
                with vuetify.VRow(v_if=("hpc_connection == 'sfapi'",)):
                    with vuetify.VCol():
                        load_sfapi_card()
                with vuetify.VRow(v_if=("hpc_connection == 'iriapi'",)):
                    with vuetify.VCol():
                        vuetify.VAlert(
                            "Coming soon",
                            type="info",
                            variant="tonal",
                        )
