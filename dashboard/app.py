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
# Callbacks
# -----------------------------------------------------------------------------

@state.change("experiment")
def reload(**kwargs):
    print(f"Executing reload...")
    global mod_manager
    global par_manager
    global obj_manager
    # initialize state after experiment selection
    print("Calling init_state...")
    init_state()
    # initialize database
    print("Calling load_database...")
    config, exp_docs, sim_docs = load_database()
    # convert database documents into pandas DataFrames
    print("Setting exp_data, sim_data...")
    state.exp_data = pd.DataFrame(exp_docs).to_json(default_handler=str)
    state.sim_data = pd.DataFrame(sim_docs).to_json(default_handler=str)
    # read input and output variables
    print("Setting config_dir, config_file...")
    config_dir  = os.path.join(os.getcwd(), "config")
    config_file = os.path.join(config_dir, "variables.yml")
    if not os.path.isfile(config_file):
        raise ValueError(f"Configuration file {config_file} not found")
    print("Calling read_variables...")
    input_variables, output_variables = read_variables(config_file)
    # initialize model
    print("Setting model_dir, model_file...")
    model_dir_local  = os.path.join(os.getcwd(), "..", "ml", "NN_training", "saved_models")
    model_dir_docker = os.path.join("/", "app", "ml", "NN_training", "saved_models")
    model_dir = model_dir_local if os.path.exists(model_dir_local) else model_dir_docker
    model_file = os.path.join(model_dir, f"{state.experiment}.yml")
    if not os.path.isfile(model_file):
        raise ValueError(f"Model file {model_file} not found")
    if not metadata_match(config_file, model_file):
        model_file = None
    print("Initializing mod_manager...")
    mod_manager = ModelManager(model_file)
    # initialize parameters
    print("Initializing par_manager...")
    par_manager = ParametersManager(mod_manager, input_variables)
    # initialize objectives
    print("Initializing obj_manager...")
    obj_manager = ObjectivesManager(mod_manager, output_variables)
    # reload home route
    print("Calling home_route...")
    home_route()
    # reload NERSC route
    print("Calling nersc_route...")
    nersc_route()
    # setup GUI components
    print("Calling gui_setup...")
    gui_setup()
    print("Exiting reload...")

@state.change(
    "exp_data",
    "sim_data",
    "parameters",
    "opacity",
)
def update(**kwargs):
    print(f"Executing update...")
    # update objectives
    print("Calling update... (ObjectivesManager)")
    obj_manager.update()
    # update plots
    print("Calling plot...")
    fig = plot(mod_manager)
    ctrl.figure_update(fig)
    print("Exiting update...")

@ctrl.add("on_server_ready")
def initialize(**kwargs):
    print("Executing initialize...")
    state.experiment = "ip2"
    # Flush state so that listener 'reload' runs right now
    state.flush()
    init_state()
    print("Exiting initialize...")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def pre_calibration(objective_name):
    # get calibration and normalization transformers
    output_transformers = mod_manager.get_output_transformers()
    output_calibration = output_transformers[0]
    output_normalization = output_transformers[1]
    # normalize simulation data
    sim_data = pd.read_json(StringIO(state.sim_data))
    objective_tensor = torch.from_numpy(sim_data[objective_name].values)
    objective_tensor = output_normalization.transform(objective_tensor)
    return (output_calibration, output_normalization, objective_tensor)

# TODO encapsulate in simulation class?
@ctrl.add("apply_calibration")
def apply_calibration():
    if mod_manager.avail():
        if not state.is_calibrated:
            #FIXME generalize for multiple objectives
            objective_name = list(state.objectives.keys())[0]
            # prepare
            output_calibration, output_normalization, objective_tensor = pre_calibration(objective_name)
            # calibrate, and denormalize simulation data
            objective_tensor = output_calibration.untransform(objective_tensor)
            objective_tensor = output_normalization.untransform(objective_tensor)
            sim_data = pd.read_json(StringIO(state.sim_data))
            sim_data[objective_name] = objective_tensor.numpy()[0]
            # update state
            state.sim_data = sim_data.to_json(default_handler=str)
            state.dirty("sim_data")
            state.is_calibrated = True

# TODO encapsulate in simulation class?
@ctrl.add("undo_calibration")
def undo_calibration():
    if mod_manager.avail():
        if state.is_calibrated:
            #FIXME generalize for multiple objectives
            objective_name = list(state.objectives.keys())[0]
            # prepare
            output_calibration, output_normalization, objective_tensor = pre_calibration(objective_name)
            # calibrate, and denormalize simulation data
            objective_tensor = output_calibration.transform(objective_tensor)
            objective_tensor = output_normalization.untransform(objective_tensor)
            sim_data = pd.read_json(StringIO(state.sim_data))
            sim_data[objective_name] = objective_tensor.numpy()[0]
            # update state
            state.sim_data = sim_data.to_json(default_handler=str)
            state.dirty("sim_data")
            state.is_calibrated = False

# -----------------------------------------------------------------------------
# GUI components
# -----------------------------------------------------------------------------

# home route
def home_route():
    print("Executing home_route...")
    with RouterViewLayout(server, "/"):
        with vuetify.VRow():
            with vuetify.VCol(cols=4):
                with vuetify.VRow():
                    with vuetify.VCol():
                        par_manager.card()
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
    print("Executing nersc_route...")
    print("Calling get_sfapi_client...")
    if get_sfapi_client() is not None:
        build_sfapi_status()
    else:
        build_sfapi_auth()
    print("Exiting nersc_route...")

# GUI layout
def gui_setup():
    print("Executing gui_setup...")
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("BELLA Superfacility")
        # add toolbar components
        with layout.toolbar:
            vuetify.VSpacer()
            vuetify.VSelect(
                v_model=("experiment",state.experiment),
                items=("experiments", ["ip2", "acave"]),
                dense=True,
                prepend_icon="mdi-atom",
                style="max-width: 200px;",
            )
        # set up router view
        with layout.content:
            with vuetify.VContainer():
                router.RouterView()
        # add router components to the drawer
        with layout.drawer:
            with vuetify.VList(shaped=True, v_model=("selectedRoute", 0)):
                vuetify.VSubheader("")
                # Home route    
                with vuetify.VListItem(to="/"):
                    with vuetify.VListItemIcon():
                        vuetify.VIcon("mdi-home")
                    with vuetify.VListItemContent():
                        vuetify.VListItemTitle("Home")
                # NERSC route
                with vuetify.VListItem(to="/nersc"):
                    with vuetify.VListItemIcon():
                        vuetify.VIcon("mdi-lan-connect")
                    with vuetify.VListItemContent():
                        vuetify.VListItemTitle("NERSC")
                # GitHub route    
                with vuetify.VListItem(click="window.open('https://github.com/ECP-WarpX/2024_IFE-superfacility/tree/main/dashboard', '_blank')"):
                    with vuetify.VListItemIcon():
                        vuetify.VIcon("mdi-github")
                    with vuetify.VListItemContent():
                        vuetify.VListItemTitle("GitHub")
    print("Exiting gui_setup...")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting server...")
    server.start()
