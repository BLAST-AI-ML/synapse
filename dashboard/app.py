import numpy as np
from scipy.optimize import minimize

from trame.app import get_server
from trame.ui.vuetify2 import SinglePageLayout
from trame.widgets import plotly, vuetify2 as v2

import torch
from lume_model.models import TorchModel

from utils import read_variables, convert_parameters, plot

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
state.parameters = dict()
state.parameters_min = dict()
state.parameters_max = dict()
for _, parameter_dict in input_variables.items():
    key = parameter_dict["name"]
    pmin = float(parameter_dict["value_range"][0])
    pmax = float(parameter_dict["value_range"][1])
    pval = float(parameter_dict["default"])
    state.parameters[key] = pval
    state.parameters_min[key] = pmin
    state.parameters_max[key] = pmax

# initialize parameters for ML model
state.parameters_model = convert_parameters(state.parameters)

# initialize objectives
state.objectives = dict()
for _, objective_dict in output_variables.items():
    key = objective_dict["name"]
    state.objectives[key] = float(model.evaluate(state.parameters_model)[key.split(maxsplit=1)[0]])
state.dirty("objectives")  # pushed again at flush time

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@state.change("parameters")
def update_state(**kwargs):
    for key in state.parameters.keys():
        state.parameters[key] = float(state.parameters[key])
    # update model parameters
    state.parameters_model = convert_parameters(state.parameters)
    # update objectives
    for key in state.objectives.keys():
        state.objectives[key] = float(model.evaluate(state.parameters_model)[key.split(maxsplit=1)[0]])
    # push again at flush time
    state.dirty("objectives")
    # update plots
    fig = plot(
        state.parameters,
        state.parameters_min,
        state.parameters_max,
        state.objectives,
        model,
    )
    ctrl.plotly_figure_update = plotly_figure.update(fig)

@ctrl.add("recenter")
def recenter():
    for key in state.parameters.keys():
        state.parameters[key] = (state.parameters_min[key] + state.parameters_max[key]) / 2.
    state.dirty("parameters")

def model_wrapper(parameters_array):
    parameters_dict = dict(zip(state.parameters.keys(), parameters_array))
    parameters_model = convert_parameters(parameters_dict)
    for key in state.objectives.keys():
        res = -float(model.evaluate(parameters_model)[key.split(maxsplit=1)[0]])
    return res

@ctrl.add("optimize")
def optimize():
    x0 = np.array(list(state.parameters.values()))
    bounds = []
    for key in state.parameters.keys():
        bounds.append((state.parameters_min[key], state.parameters_max[key]))
    res = minimize(
        fun=model_wrapper,
        x0=x0,
        bounds=bounds,
    )
    state.parameters = dict(zip(state.parameters.keys(), res.x))
    state.dirty("parameters")

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
                                        for key in state.parameters.keys():
                                            pmin = state.parameters_min[key]
                                            pmax = state.parameters_max[key]
                                            step = (pmax - pmin) / 100.
                                            # create slider for each parameter
                                            with v2.VSlider(
                                                v_model_number=(f"parameters['{key}']",),
                                                change="flushState('parameters')",
                                                label=key,
                                                min=pmin,
                                                max=pmax,
                                                step=step,
                                                classes="align-center",
                                                hide_details=True,
                                                type="number",
                                            ):
                                                # append text field
                                                with v2.Template(v_slot_append=True):
                                                    v2.VTextField(
                                                        v_model_number=(f"parameters['{key}']",),
                                                        #change="flushState('parameters')",
                                                        label=key,
                                                        density="compact",
                                                        hide_details=True,
                                                        readonly=True,
                                                        single_line=True,
                                                        style="width: 100px",
                                                        type="number",
                                                    )
                                        v2.VSpacer(style="height: 50px")
                                        with v2.VRow():
                                            with v2.VCol():
                                                v2.VBtn(
                                                    "recenter",
                                                    click=recenter,
                                                )
                                            with v2.VCol():
                                                v2.VBtn(
                                                    "optimize",
                                                    click=optimize,
                                                )
                                            with v2.VCol():
                                                pass
                    with v2.VRow():
                        with v2.VCol():
                            with v2.VCard(style="width: 500px"):
                                with v2.VCardTitle("Objectives"):
                                    with v2.VCardText():
                                        for key in state.objectives.keys():
                                            v2.VTextField(
                                                v_model_number=(f"objectives['{key}']",),
                                                label=key,
                                                readonly=True,
                                                type="number",
                                            )
                with v2.VCol():
                    with v2.VCard():
                        with v2.VCardTitle("Plots"):
                            with v2.VContainer(style=f"height: {25*len(state.parameters)}vh"):
                                plotly_figure = plotly.Figure(
                                        display_mode_bar="true", config={"responsive": True}
                                )
                                ctrl.plotly_figure_update = plotly_figure.update

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
