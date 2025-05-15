import copy
import inspect
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
from state_manager import server, state, ctrl, init_startup, init_runtime
from utils import read_variables, metadata_match, load_database, plot

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

mod_manager = None
par_manager = None
obj_manager = None

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

# Triggered automatically also on server ready,
# internal checks avoid redundant function calls.
@state.change(
    "experiment",
    "exp_data",
    "sim_data",
    "parameters",
    "opacity",
)
def update(initialize=False, **kwargs):
    print("Updating...")
    global mod_manager
    global par_manager
    global obj_manager
    state.experiment_changed = not (state.experiment == state.experiment_old)
    if state.experiment_changed:
        print("Loading new experiment...")
        initialize = True
        # reset state variables
        state.experiment_old = copy.deepcopy(state.experiment)
        state.experiment_changed = False
    if initialize:
        # initialize state after experiment selection
        init_runtime()
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
        # set up home route (reload components, e.g., parameters card)
        home_route()
        if not state.nersc_route_built:
            # set up NERSC route (only once at startup)
            nersc_route()
            state.nersc_route_built = True
        if not state.ui_layout_built:
            # set up GUI components (only once at startup)
            gui_setup()
            state.ui_layout_built = True
    # update objectives
    obj_manager.update()
    # update plots
    fig = plot(mod_manager)
    ctrl.figure_update(fig)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def pre_calibration(objective_name):
    print("Preparing calibration...")
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
    print("Applying calibration...")
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
    print("Undoing calibration...")
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
    print("Setting GUI home route...")
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
                                    # create a row for the slider label
                                    with vuetify.VRow():
                                        vuetify.VSubheader("Projected Data Depth")
                                    # create a row for the slider and text field
                                    with vuetify.VRow(no_gutters=True):
                                        with vuetify.VSlider(
                                            v_model_number=("opacity",),
                                            change="flushState('opacity')",
                                            classes="align-center",
                                            hide_details=True,
                                            max=1.0,
                                            min=0.0,
                                            step=0.025,
                                        ):
                                            with vuetify.Template(v_slot_append=True):
                                                vuetify.VTextField(
                                                    v_model_number=("opacity",),
                                                    classes="mt-0 pt-0",
                                                    density="compact",
                                                    hide_details=True,
                                                    readonly=True,
                                                    single_line=True,
                                                    style="width: 80px;",
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
    print("Setting GUI NERSC route...")
    if get_sfapi_client() is not None:
        build_sfapi_status()
    else:
        build_sfapi_auth()

# GUI layout
def gui_setup():
    print("Setting GUI layout...")
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("BELLA Superfacility")
        # add toolbar components
        with layout.toolbar:
            vuetify.VSpacer()
            vuetify.VSelect(
                v_model=("experiment",),
                items=("experiments", ["qed_ip2", "ip2", "acave"]),
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

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # initialize state variables needed at startup
    init_startup()
    # initialize all other variables and components
    update(initialize=True)
    print("Starting server...")
    server.start()
