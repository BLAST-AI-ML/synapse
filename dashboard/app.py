import argparse
import pandas as pd
import torch
from trame.app import get_server
from trame.ui.router import RouterViewLayout
from trame.ui.vuetify2 import SinglePageWithDrawerLayout
from trame.widgets import plotly, router, vuetify2 as v2

from model import Model
from parameters import Parameters
from objectives import Objectives
from utils import read_variables, load_database, plot

# -----------------------------------------------------------------------------
# Command line parser
# -----------------------------------------------------------------------------

# define parser
parser = argparse.ArgumentParser()
# add arguments: path to model data
parser.add_argument(
    "--model",
    help="path to model data file (YAML)",
    type=str,
)
# parse arguments (ignore unknown arguments for internal Trame parser)
args, _ = parser.parse_known_args()

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
model_data = args.model

# load database
db_defaults = {
    "host": "mongodb05.nersc.gov",
    "port": 27017,
    "name": "bella_sf",
    "auth": "bella_sf",
    "user": "bella_sf_ro",
    "collection": "ip2",
}
experimental_docs, simulation_docs = load_database(db_defaults)
# convert database documents into pandas DataFrames
experimental_data = pd.DataFrame(experimental_docs)
simulation_data = pd.DataFrame(simulation_docs)

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

# TODO Implement
@ctrl.add("upload_credentials")
def upload_credentials():
    pass

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

# home route
with RouterViewLayout(server, "/"):
    with v2.VRow():
        with v2.VCol():
            with v2.VRow():
                with v2.VCol():
                    parameters.card()
            with v2.VRow():
                with v2.VCol():
                    objectives.card()
            with v2.VRow():
                with v2.VCol():
                    with v2.VCard(style="width: 500px"):
                        with v2.VCardTitle("Control"):
                            with v2.VCardText():
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

# NERSC route
with RouterViewLayout(server, "/nersc"):
    with v2.VRow():
        with v2.VCol():
            with v2.VCard(style="width: 500px"):
                with v2.VCardTitle("Control: NERSC"):
                    with v2.VCardText():
                        with v2.VRow():
                            with v2.VCol():
                                v2.VBtn(
                                    "upload NERSC credentials",
                                    click=upload_credentials,
                                    style="width: 250px",
                                )

# main page content
with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("IFE Superfacility")

    # add toolbar components
    with layout.toolbar:
        pass

    with layout.content:
        with v2.VContainer():
            router.RouterView()

    # add router components to the drawer
    with layout.drawer:
        with v2.VList(shaped=True, v_model=("selectedRoute", 0)):
            v2.VSubheader("")

            with v2.VListItem(to="/"):
                with v2.VListItemIcon():
                    v2.VIcon("mdi-home")
                with v2.VListItemContent():
                    v2.VListItemTitle("Home")

            with v2.VListItem(to="/nersc"):
                with v2.VListItemIcon():
                    v2.VIcon("mdi-lan-connect")
                with v2.VListItemContent():
                    v2.VListItemTitle("NERSC")

            with v2.VListItem(click="window.open('https://github.com/ECP-WarpX/2024_IFE-superfacility/tree/main/dashboard', '_blank')"):
                with v2.VListItemIcon():
                    v2.VIcon("mdi-github")
                with v2.VListItemContent():
                    v2.VListItemTitle("GitHub")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
