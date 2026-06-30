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


def remote_backend_unavailable_expr(mode_state):
    return (
        f"(({mode_state} === 'sfapi' && sfapi_perlmutter_status !== 'active') || "
        f"({mode_state} === 'iriapi' && iriapi_perlmutter_status !== 'up'))"
    )


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
