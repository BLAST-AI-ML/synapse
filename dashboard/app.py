from io import StringIO
import os
import pandas as pd
import torch
from trame.app import get_server
from trame.ui.router import RouterViewLayout
from trame.ui.vuetify2 import SinglePageWithDrawerLayout
from trame.widgets import plotly, router, vuetify2 as vuetify

from model_manager import ModelManager
from parameters_manager import ParametersManager
from objectives_manager import ObjectivesManager
from nersc import get_sfapi_client, build_sfapi_status, build_sfapi_auth
from state_manager import server, state, ctrl, init_state
from utils import read_variables, metadata_match, load_database, plot


# -----------------------------------------------------------------------------
# Initialize experiment
# -----------------------------------------------------------------------------

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
    config_dir  = os.path.join(os.getcwd(), "config")
    config_file = os.path.join(config_dir, "variables.yml")
    if not os.path.isfile(config_file):
        raise ValueError(f"Configuration file {config_file} not found")
    input_variables, output_variables = read_variables(config_file)
    # initialize model
    model_dir_local  = os.path.join(os.getcwd(), "..", "ml", "NN_training", "saved_models")
    model_dir_docker = os.path.join("/", "app", "ml", "NN_training", "saved_models")
    model_dir = model_dir_local if os.path.exists(model_dir_local) else model_dir_docker
    model_file = os.path.join(model_dir, f"{state.experiment}.yml")
    if not os.path.isfile(model_file):
        raise ValueError(f"Model file {model_file} not found")
    if not metadata_match(config_file, model_file):
        model_file = None
    mod_manager = ModelManager(model_file)
    # initialize parameters
    par_manager = ParametersManager(mod_manager, input_variables)
    # initialize objectives
    obj_manager = ObjectivesManager(mod_manager, output_variables)
    # reload home route
    home_route()
    # reload NERSC route
    nersc_route()

@state.change(
    "exp_data",
    "sim_data",
    "parameters",
    "opacity",
)
def update(**kwargs):
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
        with vuetify.VRow():
            with vuetify.VCol(cols=4):
                with vuetify.VRow():
                    with vuetify.VCol():
                        par_manager.card()
                with vuetify.VRow():
                    with vuetify.VCol():
                        with vuetify.VCard():
                            with vuetify.VCardTitle("Control: Parameters"):
                                with vuetify.VCardText():
                                    with vuetify.VRow():
                                        with vuetify.VCol():
                                            with vuetify.VBtn(
                                                "Recenter",
                                                click=par_manager.recenter,
                                                style="width: 100%; text-transform: none;",
                                            ):
                                                vuetify.VSpacer()
                                                vuetify.VIcon("mdi-restart")
                                    with vuetify.VRow():
                                        with vuetify.VCol():
                                            with vuetify.VBtn(
                                                "Optimize",
                                                click=par_manager.optimize,
                                                style="width: 100%; text-transform: none;",
                                            ):
                                                vuetify.VSpacer()
                                                vuetify.VIcon("mdi-laptop")
                with vuetify.VRow():
                    with vuetify.VCol():
                        with vuetify.VCard():
                            with vuetify.VCardTitle("Control: Plots"):
                                with vuetify.VCardText():
                                    with vuetify.VRow():
                                        with vuetify.VCol():
                                            pass
                                    with vuetify.VRow():
                                        with vuetify.VCol():
                                            vuetify.VSlider(
                                                v_model_number=("opacity",),
                                                change="flushState('opacity')",
                                                label="Opacity",
                                                min=0.0,
                                                max=1.0,
                                                step=0.1,
                                                classes="align-center",
                                                hide_details=True,
                                                style="width: 100%;",
                                                thumb_label="always",
                                                thumb_size=25,
                                                type="number",
                                            )
                                    with vuetify.VRow():
                                        with vuetify.VCol():
                                            with vuetify.VBtn(
                                                "Apply Calibration",
                                                click=apply_calibration,
                                                style="width: 100%; text-transform: none;",
                                            ):
                                                vuetify.VSpacer()
                                                vuetify.VIcon("mdi-redo")
                                    with vuetify.VRow():
                                        with vuetify.VCol():
                                            with vuetify.VBtn(
                                                "Undo Calibration",
                                                click=undo_calibration,
                                                style="width: 100%; text-transform: none;",
                                            ):
                                                vuetify.VSpacer()
                                                vuetify.VIcon("mdi-undo")
            with vuetify.VCol(cols=8):
                with vuetify.VCard():
                    with vuetify.VCardTitle("Plots"):
                        with vuetify.VContainer(style=f"height: {400*len(state.parameters)}px;"):
                            figure = plotly.Figure(
                                display_mode_bar="true",
                                config={"responsive": True},
                            )
                            ctrl.figure_update = figure.update

# NERSC route
def nersc_route():
    if get_sfapi_client() is not None:
        build_sfapi_status()
    else:
        build_sfapi_auth()

# trigger first reload manually (FIXME fix reload to wait for server response?)
reload()

# main page content
with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("BELLA Superfacility")

    # add toolbar components
    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VSelect(
            v_model=("experiment",),
            items=("experiments", ["ip2", "acave"]),
            dense=True,
            prepend_icon="mdi-atom",
            style="max-width: 200px;",
        )

    with layout.content:
        with vuetify.VContainer():
            router.RouterView()

    # add router components to the drawer
    with layout.drawer:
        with vuetify.VList(shaped=True, v_model=("selectedRoute", 0)):
            vuetify.VSubheader("")

            with vuetify.VListItem(to="/"):
                with vuetify.VListItemIcon():
                    vuetify.VIcon("mdi-home")
                with vuetify.VListItemContent():
                    vuetify.VListItemTitle("Home")

            with vuetify.VListItem(to="/nersc"):
                with vuetify.VListItemIcon():
                    vuetify.VIcon("mdi-lan-connect")
                with vuetify.VListItemContent():
                    vuetify.VListItemTitle("NERSC")

            with vuetify.VListItem(click="window.open('https://github.com/ECP-WarpX/2024_IFE-superfacility/tree/main/dashboard', '_blank')"):
                with vuetify.VListItemIcon():
                    vuetify.VIcon("mdi-github")
                with vuetify.VListItemContent():
                    vuetify.VListItemTitle("GitHub")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
