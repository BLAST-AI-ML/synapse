import argparse
from io import StringIO
import os
import pandas as pd
import torch
from trame.app import get_server
from trame.ui.router import RouterViewLayout
from trame.ui.vuetify2 import SinglePageWithDrawerLayout
from trame.widgets import plotly, router, vuetify2 as v2

from model_manager import ModelManager
from parameters_manager import ParametersManager
from objectives_manager import ObjectivesManager
from nersc import get_sfapi_client, build_sfapi_status, build_sfapi_auth
from state_manager import server, state, ctrl, init_state
from utils import read_variables, metadata_match, load_database, plot


# -----------------------------------------------------------------------------
# Command line parser
# -----------------------------------------------------------------------------

# define parser
parser = argparse.ArgumentParser()
# add arguments: path to model file
parser.add_argument(
    "--model",
    help="path to model data file (.yml)",
    default=None,
    type=str,
)
# parse arguments (ignore unknown arguments for internal Trame parser)
args, _ = parser.parse_known_args()

# -----------------------------------------------------------------------------
# Initialize experiment
# -----------------------------------------------------------------------------

state.trame_title = "IFE Superfacility"
state.experiment = "ip2"

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@state.change("experiment")
def reload(**kwargs):
    global mod_manager
    global par_manager
    global obj_manager
    # initialize state after experiment selection
    init_state()
    # initialize database
    config, exp_docs, sim_docs = load_database()
    # convert database documents into pandas DataFrames
    state.exp_data = pd.DataFrame(exp_docs).to_json(default_handler=str)
    state.sim_data = pd.DataFrame(sim_docs).to_json(default_handler=str)
    # read input and output variables
    current_dir = os.getcwd()
    config_file = os.path.join(current_dir, "..", "config", "variables.yml")
    input_variables, output_variables = read_variables(config_file)
    # initialize model
    model_file = args.model
    if not metadata_match(config_file, model_file):
        model_file = None
    mod_manager = ModelManager(model_file)
    # initialize parameters
    par_manager = ParametersManager(input_variables)
    # initialize objectives
    obj_manager = ObjectivesManager(mod_manager, output_variables)
    # reload home route
    home_route()
    # reload NERSC route
    nersc_route()
    # update app
    update()

@state.change(
    "exp_data",
    "sim_data",
    "parameters",
    "opacity",
)
def update(**kwargs):
    # update parameters
    par_manager.update()
    # update objectives
    obj_manager.update()
    # update plots
    fig = plot(mod_manager)
    ctrl.figure_update(fig)

def pre_calibration():
    # get calibration and normalization transformers
    output_transformers = mod_manager.get_output_transformers()
    output_calibration = output_transformers[0]
    output_normalization = output_transformers[1]
    # normalize simulation data
    sim_data = pd.read_json(StringIO(state.sim_data))
    n_protons_tensor = torch.from_numpy(sim_data["n_protons"].values)
    n_protons_tensor = output_normalization.transform(n_protons_tensor)
    return (output_calibration, output_normalization, n_protons_tensor)

# TODO encapsulate in simulation class?
@ctrl.add("apply_calibration")
def apply_calibration():
    if mod_manager.avail():
        if not state.is_calibrated:
            # prepare
            output_calibration, output_normalization, n_protons_tensor = pre_calibration()
            # calibrate, and denormalize simulation data
            n_protons_tensor = output_calibration.untransform(n_protons_tensor)
            n_protons_tensor = output_normalization.untransform(n_protons_tensor)
            sim_data = pd.read_json(StringIO(state.sim_data))
            sim_data["n_protons"] = n_protons_tensor.numpy()[0]
            # update state
            state.sim_data = sim_data.to_json(default_handler=str)
            state.dirty("sim_data")
            state.is_calibrated = True

# TODO encapsulate in simulation class?
@ctrl.add("undo_calibration")
def undo_calibration():
    if mod_manager.avail():
        if state.is_calibrated:
            # prepare
            output_calibration, output_normalization, n_protons_tensor = pre_calibration()
            # calibrate, and denormalize simulation data
            n_protons_tensor = output_calibration.transform(n_protons_tensor)
            n_protons_tensor = output_normalization.untransform(n_protons_tensor)
            sim_data = pd.read_json(StringIO(state.sim_data))
            sim_data["n_protons"] = n_protons_tensor.numpy()[0]
            # update state
            state.sim_data = sim_data.to_json(default_handler=str)
            state.dirty("sim_data")
            state.is_calibrated = False

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

# home route
def home_route():
    with RouterViewLayout(server, "/"):
        with v2.VRow():
            with v2.VCol(cols=4):
                with v2.VRow():
                    with v2.VCol():
                        par_manager.card()
                with v2.VRow():
                    with v2.VCol():
                        obj_manager.card()
                with v2.VRow():
                    with v2.VCol():
                        with v2.VCard():
                            with v2.VCardTitle("Control"):
                                with v2.VCardText():
                                    with v2.VRow():
                                        with v2.VCol():
                                            v2.VBtn(
                                                "Apply Calibration",
                                                click=apply_calibration,
                                                style="width: 100%; text-transform: none;",
                                            )
                                        with v2.VCol():
                                            v2.VBtn(
                                                "Undo Calibration",
                                                click=undo_calibration,
                                                style="width: 100%; text-transform: none;",
                                            )
                                    with v2.VRow():
                                        with v2.VCol():
                                            v2.VBtn(
                                                "Recenter",
                                                click=par_manager.recenter,
                                                style="width: 100%; text-transform: none;",
                                            )
                                        with v2.VCol():
                                            v2.VBtn(
                                                "Optimize",
                                                click=mod_manager.optimize,
                                                style="width: 100%; text-transform: none;",
                                            )
            with v2.VCol(cols=8):
                with v2.VCard():
                    with v2.VCardTitle("Plots"):
                        with v2.VContainer(style=f"height: {40*len(state.parameters)}vh"):
                            figure = plotly.Figure(
                                display_mode_bar="true",
                                config={"responsive": True},
                            )
                            ctrl.figure_update = figure.update
                        # opacity slider
                        with v2.VCardText():
                            v2.VSlider(
                                v_model_number=("opacity",),
                                change="flushState('opacity')",
                                label="Opacity",
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
def nersc_route():
    if get_sfapi_client() is not None:
        build_sfapi_status()
    else:
        build_sfapi_auth()

# main page content
with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text(state.trame_title)

    # add toolbar components
    with layout.toolbar:
        for _ in range(5):
            v2.VSpacer()
        v2.VSelect(
            v_model=("experiment",),
            items=("experiments", ["ip2", "acave"]),
        )

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
