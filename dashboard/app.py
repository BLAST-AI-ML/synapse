import copy
from io import StringIO
import os
import pandas as pd
import torch
from trame.app import get_server
from trame.ui.router import RouterViewLayout
from trame.ui.vuetify2 import SinglePageWithDrawerLayout
from trame.widgets import plotly, router, vuetify2 as vuetify

from model_manager import ModelManager
from objectives_manager import ObjectivesManager
from parameters_manager import ParametersManager
from sfapi_manager import initialize_sfapi, load_sfapi_card
from state_manager import server, state, ctrl, initialize_state
from utils import read_variables, metadata_match, load_database, plot

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

mod_manager = None
par_manager = None
obj_manager = None

# list of all available model types
model_type_list = [
    "Gaussian Process",
    "Neural Network",
]
# dict of auxiliary model types tags
model_type_tag_dict = {
    "Gaussian Process": "GP",
    "Neural Network": "NN",
}
# list of all available experiments (TODO parse automatically)
experiment_list = [
    "acave",
    "ip2",
    "qed_ip2",
]

# -----------------------------------------------------------------------------
# Functions and callbacks
# -----------------------------------------------------------------------------

def pre_calibration(sim_data, objective_name):
    print("Preparing calibration...")
    # get calibration and normalization transformers
    output_transformers = mod_manager.get_output_transformers()
    output_calibration = output_transformers[0]
    output_normalization = output_transformers[1]
    # normalize simulation data
    sim_data = pd.read_json(StringIO(sim_data))
    objective_tensor = torch.from_numpy(sim_data[objective_name].values)
    objective_tensor = output_normalization.transform(objective_tensor)
    return (output_calibration, output_normalization, objective_tensor)

def load_data():
    print("Loading data from database...")
    config, exp_docs, sim_docs = load_database()
    exp_data = pd.DataFrame(exp_docs).to_json(default_handler=str)
    sim_data = pd.DataFrame(sim_docs).to_json(default_handler=str)
    return (exp_data, sim_data)

def load_config_file():
    config_dir = os.path.join(os.getcwd(), "config")
    config_file = os.path.join(config_dir, "variables.yml")
    if not os.path.isfile(config_file):
        raise ValueError(f"Configuration file {config_file} not found")
    return config_file

def load_model_file():
    config_file = load_config_file()
    model_type_tag = model_type_tag_dict[state.model_type]
    model_dir_local = os.path.join(os.getcwd(), "..", "ml", f"{model_type_tag}_training", "saved_models")
    model_dir_docker = os.path.join("/", "app", "ml", f"{model_type_tag}_training", "saved_models")
    model_dir = model_dir_local if os.path.exists(model_dir_local) else model_dir_docker
    model_file = os.path.join(model_dir, f"{state.experiment}.yml")
    if not os.path.isfile(model_file):
        raise ValueError(f"Model file {model_file} not found")
    if not metadata_match(config_file, model_file):
        model_file = None
    return model_file

def load_variables():
    config_file = load_config_file()
    input_variables, output_variables = read_variables(config_file)
    return (input_variables, output_variables)

def initialize(
    reset_gui_route_home=True,
    reset_gui_route_nersc=True,
    reset_gui_layout=True,
    reset_parameters=True,
    reset_model_attr=False,
):
    print("Initializing...")
    global mod_manager
    global par_manager
    global obj_manager
    # load data
    exp_data, sim_data = load_data()
    # initialize model
    model_file = load_model_file()
    mod_manager = ModelManager(model_file)
    # load input and output variables
    input_variables, output_variables = load_variables()
    if reset_parameters:
        # initialize parameters
        par_manager = ParametersManager(mod_manager, input_variables)
    elif reset_model_attr:
        # reset model attribute in the parameters class
        par_manager.model = mod_manager
    # initialize objectives
    obj_manager = ObjectivesManager(mod_manager, output_variables)
    if reset_gui_route_home:
        # set up home route
        home_route()
    if reset_gui_route_nersc:
        # set up NERSC route
        nersc_route()
    if reset_gui_layout:
        # set up GUI layout
        gui_setup()
    # initialize plots
    fig = plot(exp_data, sim_data, mod_manager)
    ctrl.figure_update(fig)

@state.change(
    "parameters",
    "opacity",
    "calibrate",
)
def update_objectives_and_plots(**kwargs):
    print(f"Updating objectives and plots...")
    global obj_manager
    # load data
    exp_data, sim_data = load_data()
    # initialize model
    model_file = load_model_file()
    mod_manager = ModelManager(model_file)
    # calibration
    if mod_manager.avail() and not mod_manager.is_gaussian_process:
        # FIXME generalize for multiple objectives
        objective_name = list(state.objectives.keys())[0]
        # prepare
        output_calibration, output_normalization, objective_tensor = pre_calibration(sim_data, objective_name)
        if state.calibrate:
            objective_tensor = output_calibration.untransform(objective_tensor)
            objective_tensor = output_normalization.untransform(objective_tensor)
        else:
            objective_tensor = output_calibration.transform(objective_tensor)
            objective_tensor = output_normalization.untransform(objective_tensor)
        sim_data = pd.read_json(StringIO(sim_data))
        sim_data[objective_name] = objective_tensor.numpy()[0]
        # update state
        sim_data = sim_data.to_json(default_handler=str)
    # update objectives
    obj_manager.update()
    # update plots
    fig = plot(exp_data, sim_data, mod_manager)
    ctrl.figure_update(fig)

@state.change("experiment")
def update_on_experiment_change(**kwargs):
    print("Experiment changed...")
    print("Initializing again...")
    initialize(
        reset_gui_route_home=True,
        reset_gui_route_nersc=False,
        reset_gui_layout=False,
        reset_parameters=True,
        reset_model_attr=False,
    )

@state.change("model_type")
def update_on_model_change(**kwargs):
    print("Model type changed...")
    print("Initializing again...")
    initialize(
        reset_gui_route_home=True,
        reset_gui_route_nersc=False,
        reset_gui_layout=False,
        reset_parameters=False,
        reset_model_attr=True,
    )

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
                            with vuetify.VCardTitle("Control: Models"):
                                with vuetify.VCardText():
                                    vuetify.VSelect(
                                        v_model=("model_type",),
                                        items=("Models", model_type_list),
                                        dense=True,
                                        prepend_icon="mdi-brain",
                                        style="max-width: 210px;",
                                    )
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
                                    # create a row for the calibration switch
                                    with vuetify.VRow():
                                        vuetify.VSwitch(
                                            v_model=("calibrate",),
                                            label="Calibration",
                                            color="primary",
                                        )
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
    with RouterViewLayout(server, "/nersc"):
        with vuetify.VRow():
            with vuetify.VCol(cols=4):
                with vuetify.VRow():
                    with vuetify.VCol():
                        load_sfapi_card()

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
                items=("experiments", experiment_list),
                dense=True,
                prepend_icon="mdi-atom",
                style="max-width: 210px;",
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
    initialize_state()
    # initialize Superfacility API
    initialize_sfapi()
    # initialize
    initialize()
    # start server
    print("Starting server...")
    server.start()
