import pandas as pd
import torch
from trame.app import get_server
from trame.ui.vuetify2 import SinglePageLayout
from trame.widgets import plotly, vuetify2 as v2

from model import Model
from parameters import Parameters
from objectives import Objectives
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

# set file paths
model_data = "../ml/NN_training/base_simulation_model_with_transformers_calibration.yml"
experimental_file = "../experimental_data/experimental_data.csv"
simulation_file = "../simulation_data/simulation_data.csv"

# initialize data
experimental_data = pd.read_csv(experimental_file)
simulation_data = pd.read_csv(simulation_file)

# initialize model
model = Model(server, model_data)

# initialize parameters
parameters = Parameters(server, input_variables)

# initialize objectives
objectives = Objectives(server, model, output_variables)

# initialize opacity controller
state.opacity = 0.1

# calibration of simulation data
state.is_calibrated = False

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@state.change("opacity")
def update_plots(**kwargs):
    fig = plot(
        model,
        parameters,
        objectives,
        experimental_data,
        simulation_data,
        state.opacity,
    )
    ctrl.plotly_figure_update = plotly_figure.update(fig)

@state.change("parameters")
def update_state(**kwargs):
    # update parameters
    parameters.update()
    # update objectives
    objectives.update()
    # update plots (TODO plots.update())
    update_plots()

def pre_calibration():
    # get calibration and normalization transformers
    output_transformers = model.get_output_transformers()
    output_calibration = output_transformers[0]
    output_normalization = output_transformers[1]
    # de-normalize simulation data
    n_protons_tensor = torch.from_numpy(simulation_data["n_protons"].values)
    n_protons_tensor = output_normalization.untransform(n_protons_tensor)
    return (output_calibration, output_normalization, n_protons_tensor)

# TODO encapsulate in simulation class?
@ctrl.add("apply_calibration")
def apply_calibration():
    if not state.is_calibrated:
        # prepare
        output_calibration, output_normalization, n_protons_tensor = pre_calibration()
        # calibrate, and re-normalize simulation data
        n_protons_tensor = output_calibration.transform(n_protons_tensor)
        n_protons_tensor = output_normalization.transform(n_protons_tensor)
        simulation_data["n_protons"] = n_protons_tensor.numpy()[0]
        # update plots (TODO plots.update())
        update_plots()
        # update state
        state.is_calibrated = True

# TODO encapsulate in simulation class?
@ctrl.add("undo_calibration")
def undo_calibration():
    if state.is_calibrated:
        # prepare
        output_calibration, output_normalization, n_protons_tensor = pre_calibration()
        # calibrate, and re-normalize simulation data
        n_protons_tensor = output_calibration.untransform(n_protons_tensor)
        n_protons_tensor = output_normalization.transform(n_protons_tensor)
        simulation_data["n_protons"] = n_protons_tensor.numpy()[0]
        # update plots (TODO plots.update())
        update_plots()
        # update state
        state.is_calibrated = False

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
                                        for key in parameters.get().keys():
                                            pmin = parameters.get_min()[key]
                                            pmax = parameters.get_max()[key]
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
                                        for key in objectives.get().keys():
                                            v2.VTextField(
                                                v_model_number=(f"objectives['{key}']",),
                                                label=key,
                                                readonly=True,
                                                type="number",
                                            )
                    with v2.VRow():
                        with v2.VCol():
                            with v2.VCard(style="width: 500px"):
                                with v2.VCardTitle("Control"):
                                    with v2.VCardText():
                                        with v2.VRow():
                                            with v2.VCol():
                                                v2.VBtn(
                                                    "recenter",
                                                    click=parameters.recenter,
                                                    style="width: 200px",
                                                )
                                            with v2.VCol():
                                                v2.VBtn(
                                                    "optimize",
                                                    click=model.optimize,
                                                    style="width: 200px",
                                                )
                                        with v2.VRow():
                                            with v2.VCol():
                                                v2.VBtn(
                                                    "apply calibration",
                                                    click=apply_calibration,
                                                    style="width: 200px",
                                                )
                                            with v2.VCol():
                                                v2.VBtn(
                                                    "undo calibration",
                                                    click=undo_calibration,
                                                    style="width: 200px",
                                                )
                with v2.VCol():
                    with v2.VCard():
                        with v2.VCardTitle("Plots"):
                            with v2.VContainer(style=f"height: {25*len(parameters.get())}vh"):
                                plotly_figure = plotly.Figure(
                                        display_mode_bar="true", config={"responsive": True}
                                )
                                ctrl.plotly_figure_update = plotly_figure.update
                            # opacity slider
                            with v2.VCardText():
                                v2.VSlider(
                                    v_model_number=("opacity",),
                                    change="flushState('opacity')",
                                    label="OPACITY",
                                    min=0.0,
                                    max=1.0,
                                    step=0.1,
                                    classes="align-center",
                                    hide_details=True,
                                    style="width: 200px",
                                    thumb_label="always",
                                    thumb_size=25,
                                    type="number",
                                )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
