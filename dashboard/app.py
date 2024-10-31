from trame.app import get_server
from trame.ui.vuetify2 import SinglePageLayout
from trame.widgets import plotly, vuetify2 as v2

import torch
from lume_model.models import TorchModel

from utils import read_variables, plot

# -----------------------------------------------------------------------------
# Trame initialization 
# -----------------------------------------------------------------------------

server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

state.trame__title = "IFE Superfacility"

# -----------------------------------------------------------------------------
# Initialize state
# -----------------------------------------------------------------------------

# read input and output variables
input_variables, output_variables = read_variables("variables.yml")
# FIXME generalize for multiple objectives
assert len(output_variables) == 1, "number of objectives > 1 not supported"

# load model
model = TorchModel("bella_saved_model.yml")

# initialize parameters
state.parameters_phys = dict()
state.parameters_phys_min = dict()
state.parameters_phys_max = dict()
for _, parameter_dict in input_variables.items():
    key = parameter_dict["name"]
    pmin = float(parameter_dict["value_range"][0])
    pmax = float(parameter_dict["value_range"][1])
    pval = float(parameter_dict["default"])
    state.parameters_phys[key] = pval
    state.parameters_phys_min[key] = pmin
    state.parameters_phys_max[key] = pmax

# initialize parameters for ML model
state.parameters_model = state.parameters_phys.copy()
# workaround to match keys:
# - model labels do not carry units (e.g., "TOD" instead of "TOD (fs^3)")
# - model inputs do not include GVD
for key_old in state.parameters_model.keys():
    key_new, _ = key_old.split(maxsplit=1)
    state.parameters_model[key_new] = state.parameters_model.pop(key_old)
gvd_key = [key_tmp for key_tmp in state.parameters_model.keys() if key_tmp == "GVD"][0]
state.parameters_model.pop(gvd_key)

# initialize objectives
state.objectives_phys = dict()
for _, objective_dict in output_variables.items():
    key = objective_dict["name"]
    state.objectives_phys[key] = float(model.evaluate(state.parameters_model)[key.split(maxsplit=1)[0]])
state.dirty("objectives_phys")  # pushed again at flush time

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@state.change("parameters_phys")
def update_state(**kwargs):
    for key in state.parameters_phys.keys():
        state.parameters_phys[key] = float(state.parameters_phys[key])
    # update model parameters
    state.parameters_model = state.parameters_phys.copy()
    # workaround to match keys:
    # - model labels do not carry units (e.g., "TOD" instead of "TOD (fs^3)")
    # - model inputs do not include GVD
    for key_old in state.parameters_model.keys():
        key_new, _ = key_old.split(maxsplit=1)
        state.parameters_model[key_new] = state.parameters_model.pop(key_old)
    gvd_key = [key_tmp for key_tmp in state.parameters_model.keys() if key_tmp == "GVD"][0]
    state.parameters_model.pop(gvd_key)
    # update objectives
    for key in state.objectives_phys.keys():
        state.objectives_phys[key] = float(model.evaluate(state.parameters_model)[key.split(maxsplit=1)[0]])
    # push again at flush time
    state.dirty("objectives_phys")
    # update plots
    fig = plot(
        state.parameters_phys,
        state.parameters_phys_min,
        state.parameters_phys_max,
        state.objectives_phys,
        model,
        **kwargs,
    )
    ctrl.plotly_figure_update = plotly_figure.update(fig)

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.title.set_text("IFE Superfacility")

    with layout.toolbar:
        # toolbar components
        pass

    with layout.content:
        # content components
        with v2.VContainer():
            with v2.VRow():
                with v2.VCol():
                    with v2.VRow():
                        with v2.VCol():
                            with v2.VCard(style="width: 500px"):
                                with v2.VCardTitle("Parameters"):
                                    with v2.VCardText():
                                        for key in state.parameters_phys.keys():
                                            pmin = state.parameters_phys_min[key]
                                            pmax = state.parameters_phys_max[key]
                                            step = (pmax - pmin) / 100.
                                            # create slider for each parameter
                                            with v2.VSlider(
                                                v_model_number=(f"parameters_phys['{key}']",),
                                                change="flushState('parameters_phys')",
                                                label=key,
                                                min=pmin,
                                                max=pmax,
                                                step=step,
                                                classes="align-center",
                                                hide_details=True,
                                                type="number",
                                            ):
                                                # append text field
                                                with v2.Template(
                                                    v_slot_append=True,
                                                ):
                                                    v2.VTextField(
                                                        v_model_number=(f"parameters_phys['{key}']",),
                                                        #change="flushState('parameters_phys')",
                                                        label=key,
                                                        density="compact",
                                                        hide_details=True,
                                                        readonly=True,
                                                        single_line=True,
                                                        style="width: 100px",
                                                        type="number",
                                                    )
                    with v2.VRow():
                        with v2.VCol():
                            with v2.VCard(style="width: 500px"):
                                with v2.VCardTitle("Objectives"):
                                    with v2.VCardText():
                                        for key in state.objectives_phys.keys():
                                            v2.VTextField(
                                                v_model_number=(f"objectives_phys['{key}']",),
                                                label=key,
                                                readonly=True,
                                                type="number",
                                            )
                with v2.VCol():
                    with v2.VCard():
                        with v2.VCardTitle("Plots"):
                            with v2.VContainer(style=f"height: {25*len(state.parameters_phys)}vh"):
                                plotly_figure = plotly.Figure(
                                        display_mode_bar="true", config={"responsive": True}
                                )
                                ctrl.plotly_figure_update = plotly_figure.update

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
