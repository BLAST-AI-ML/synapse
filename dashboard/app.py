from bson.objectid import ObjectId
import os
import re
from trame.assets.local import LocalFileManager
from trame.ui.router import RouterViewLayout
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import plotly, router, vuetify3 as vuetify, html

from model_manager import ModelManager
from outputs_manager import OutputsManager
from optimization_manager import OptimizationManager
from inputs_manager import InputsManager
from calibration_manager import SimulationCalibrationManager
from sfapi_manager import initialize_sfapi, load_sfapi_card
from state_manager import server, state, ctrl, initialize_state
from error_manager import error_panel, add_error
from utils import (
    data_depth_panel,
    load_experiments,
    load_database,
    load_data,
    load_variables,
    plot,
)

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

inputs_manager = None
outputs_manager = None
mod_manager = None
opt_manager = None
cal_manager = None

# list of available experiments
experiments = load_experiments()

# -----------------------------------------------------------------------------
# Functions and callbacks
# -----------------------------------------------------------------------------


def update(
    reset_model=True,
    reset_inputs=True,
    reset_outputs=True,
    reset_calibration=True,
    reset_plots=True,
    reset_gui_route_home=True,
    reset_gui_route_nersc=True,
    reset_gui_route_chat=True,
    reset_gui_layout=True,
    **kwargs,
):
    print("Updating...")
    global mod_manager
    global inputs_manager
    global outputs_manager
    global opt_manager
    global cal_manager
    # load input and output variables
    inputs, outputs, simulation_calibration = load_variables(state.experiment)
    # load data
    db = load_database(state.experiment)
    exp_data, sim_data = load_data(db)
    # reset outputs
    if reset_outputs:
        outputs_manager = OutputsManager(outputs)
    # reset model
    if reset_model:
        mod_manager = ModelManager(db)
        opt_manager = OptimizationManager(mod_manager)
    # reset inputs
    if reset_inputs:
        inputs_manager = InputsManager(mod_manager, inputs)
    elif reset_model:
        # if resetting only model, model attribute must be updated
        inputs_manager.model = mod_manager
    # reset calibration
    if reset_calibration:
        cal_manager = SimulationCalibrationManager(simulation_calibration)
    # reset GUI home route
    if reset_gui_route_home:
        home_route()
    # reset GUI NERSC route
    if reset_gui_route_nersc:
        nersc_route()
    # reset GUI chat route
    if reset_gui_route_chat:
        chat_route()
    # reset GUI layout
    if reset_gui_layout:
        gui_setup()
    # reset plots
    if reset_plots:
        fig = plot(
            exp_data=exp_data,
            sim_data=sim_data,
            model_manager=mod_manager,
            cal_manager=cal_manager,
        )
        ctrl.figure_update(fig)


@state.change("experiment")
def update_on_change_experiment(**kwargs):
    # skip if triggered on server ready (all state variables marked as modified)
    if len(state.modified_keys) == 1:
        print("Experiment changed...")
        update(
            reset_model=True,
            reset_inputs=True,
            reset_outputs=True,
            reset_calibration=True,
            reset_plots=True,
            reset_gui_route_home=True,
            reset_gui_route_nersc=False,
            reset_gui_route_chat=False,
            reset_gui_layout=False,
        )


@state.change("model_type", "model_training_time")
def update_on_change_model(**kwargs):
    # skip if triggered on server ready (all state variables marked as modified)
    if len(state.modified_keys) == 1:
        print("Model type changed...")
        update(
            reset_model=True,
            reset_inputs=False,
            reset_outputs=False,
            reset_calibration=False,
            reset_plots=True,
            reset_gui_route_home=True,
            reset_gui_route_nersc=False,
            reset_gui_route_chat=False,
            reset_gui_layout=False,
        )


@state.change(
    "displayed_output",
    "inputs",
    "opacity",
    "inputs_min",
    "inputs_max",
    "inputs_show_all",
    "simulation_calibration",
    "use_inferred_calibration",
)
def update_on_change_others(**kwargs):
    # skip if triggered on server ready (all state variables marked as modified)
    if len(state.modified_keys) == 1:
        print("Inputs, opacity changed...")
        update(
            reset_model=False,
            reset_inputs=False,
            reset_outputs=False,
            reset_calibration=False,
            reset_plots=True,
            reset_gui_route_home=False,
            reset_gui_route_nersc=False,
            reset_gui_route_chat=False,
            reset_gui_layout=False,
        )


def find_simulation(event, db):
    try:
        # extract the ID of the point that the user clicked on
        this_point_id = event["points"][0]["customdata"][0]
        # find the document with matching ID from the experiment collection
        documents = list(db[state.experiment].find({"_id": ObjectId(this_point_id)}))
        if len(documents) == 1:
            this_point_inputs = {
                input: documents[0][input]
                for input in state.inputs.keys()
                if input in documents[0]
            }
            print(f"Clicked on data point ({this_point_inputs})")
        else:
            title = "Unable to find database document"
            msg = f"Error occurred when searching for database document that matches ID {this_point_id}"
            add_error(title, msg)
            print(msg)
            return
        # get data directory from the document
        data_directory = documents[0]["data_directory"]
        # replace the absolute path preceding "simulation_data"
        # with the work directory "/app" set in the Dockerfile:
        # - "^(.*)" captures everything before "simulation_data"
        # - "(.*)$" captures everything after "simulation_data"
        # - "\g<2>" inserts the captured part after "simulation_data"
        data_directory = re.sub(
            pattern=r"^(.*)simulation_data/(.*)$",
            repl=r"/app/simulation_data/\g<2>",
            string=data_directory,
        )
        if not os.path.isdir(data_directory):
            print(f"Could not find data directory {data_directory}")
            return
        # get file directory
        file_directory = os.path.join(data_directory, "plots")
        if not os.path.isdir(file_directory):
            print(f"Could not find file directory {file_directory}")
            return
        # find plot file(s) to display
        file_list = os.listdir(file_directory)
        file_list.sort()
        file_video = [file for file in file_list if file.endswith(".mp4")]
        file_png = [
            file for file in file_list if file.endswith(".png") and "iteration" in file
        ]
        if len(file_video) == 1:
            # select video file
            file_name = file_video[0]
        elif len(file_png) > 0:
            # select image file from last iteration
            file_name = file_png[-1]
        else:
            print("Could not find valid plot files to display")
            return
        # set file path and verify that it exists
        file_path = os.path.join(file_directory, file_name)
        if os.path.isfile(file_path):
            print(f"Found file {file_path}")
        else:
            print(f"Could not find file {file_path}")
            return
        # store a URL encoded file content under a given key name
        return data_directory, file_path
    except Exception as e:
        title = "Unable to find simulation"
        msg = f"Error occured when searching for simulation: {e}"
        add_error(title, msg)
        print(msg)


def open_simulation_dialog(event):
    db = load_database(state.experiment)
    try:
        data_directory, file_path = find_simulation(event, db)
        state.simulation_video = file_path.endswith(".mp4")
        assets = LocalFileManager(data_directory)
        assets.url(
            key="simulation_key",
            file_path=file_path,
        )
        state.simulation_url = assets["simulation_key"]
        state.simulation_dialog = True
    except Exception as e:
        title = "Unable to open simulation dialog"
        msg = f"Error occurred when opening simulation dialog: {e}"
        add_error(title, msg)
        print(msg)


def close_simulation_dialog(**kwargs):
    state.simulation_url = None
    state.simulation_dialog = False
    state.simulation_video = False


# -----------------------------------------------------------------------------
# GUI components
# -----------------------------------------------------------------------------


# home route
def home_route():
    print("Setting GUI home route...")
    with RouterViewLayout(server, "/"):
        with vuetify.VRow():
            with vuetify.VCol(cols=4):
                with vuetify.VCard():
                    with vuetify.VTabs(
                        v_model=("active_tab", "inputs_tab"),
                        color="primary",
                        mandatory=True,
                    ):
                        vuetify.VTab("Inputs", value="inputs_tab")
                        vuetify.VTab("Optimization", value="optimization_tab")
                        vuetify.VTab("ML", value="ml_tab")
                    with vuetify.VWindow(v_model=("active_tab",), mandatory=True):
                        with vuetify.VWindowItem(value="inputs_tab"):
                            # outputs control panel
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    outputs_manager.panel()
                            # inputs control panel
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    inputs_manager.panel()
                            # plots control panel
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    data_depth_panel()
                        with vuetify.VWindowItem(value="optimization_tab"):
                            # optimization control panel
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    opt_manager.panel()
                        with vuetify.VWindowItem(value="ml_tab"):
                            # model control panel
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    mod_manager.panel()
                            # calibration control panel
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    cal_manager.panel()
            # plots card
            with vuetify.VCol(cols=8):
                with vuetify.VCard():
                    with vuetify.VCardTitle("Plots"):
                        with vuetify.VContainer(
                            style=f"height: {400 * len(state.inputs)}px;"
                        ):
                            figure = plotly.Figure(
                                display_mode_bar="true",
                                config={"responsive": True},
                                click=(
                                    open_simulation_dialog,
                                    "[utils.safe($event)]",
                                ),
                            )
                            ctrl.figure_update = figure.update


# NERSC route
def nersc_route():
    print("Setting GUI NERSC route...")
    with RouterViewLayout(server, "/nersc"):
        with vuetify.VRow():
            with vuetify.VCol(cols=4):
                # Superfacility API card
                with vuetify.VRow():
                    with vuetify.VCol():
                        load_sfapi_card()


# Chat route
def chat_route():
    print("Setting GUI Chat route...")
    with RouterViewLayout(server, "/chat"):
        with vuetify.VContainer(fluid=True, style="height: 80vh; width: 100%;"):
            html.Iframe(
                src="https://synapse-chat.lbl.gov/",
                style="width: 100%; height: 100%; border: none;",
            )


# GUI layout
def gui_setup():
    print("Setting GUI layout...")
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("Synapse")
        # add toolbar components
        with layout.toolbar:
            vuetify.VSpacer()
            vuetify.VSelect(
                v_model=("experiment",),
                label="Experiments",
                items=(experiments,),
                dense=True,
                hide_details=True,
                prepend_icon="mdi-atom",
                style="max-width: 250px",
            )
        # set up router view
        with layout.content:
            error_panel()
            with vuetify.VContainer(
                fluid=True, style="height: 100vh; overflow-y: auto"
            ):
                router.RouterView()
        # add router components to the drawer
        with layout.drawer:
            with vuetify.VList(shaped=True, v_model=("selectedRoute", 0)):
                vuetify.VListSubheader("")
                # Home route
                vuetify.VListItem(
                    to="/",
                    prepend_icon="mdi-home",
                    title="Home",
                )
                # NERSC route
                vuetify.VListItem(
                    to="/nersc",
                    prepend_icon="mdi-lan-connect",
                    title="NERSC",
                )
                # Chat route
                vuetify.VListItem(
                    to="/chat",
                    prepend_icon="mdi-chat",
                    title="AI Chat",
                )
        # interactive dialog for simulation plots
        with vuetify.VDialog(
            v_model=("simulation_dialog",),
            content_class="d-flex align-center justify-center",
        ):
            with vuetify.VCard(style="width: 80vw; height: 80vh;"):
                with vuetify.VCardTitle(
                    "Simulation Plots",
                    classes="d-flex align-center",
                ):
                    vuetify.VSpacer()
                    vuetify.VBtn(
                        click=close_simulation_dialog,
                        icon="mdi-close",
                        variant="plain",
                    )
                with vuetify.VRow(
                    align="center",
                    justify="center",
                    style="width: 80vw; height: 60vh;",
                ):
                    html.Video(
                        v_if=("simulation_video",),
                        controls=True,
                        src=("simulation_url",),
                        style="width: 100%; height: 100%",
                    )
                    vuetify.VImg(
                        v_if=("!simulation_video",),
                        src=("simulation_url",),
                        contain=True,
                        style="width: 100%; height: 100%",
                    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # initialize state variables needed at startup
    initialize_state()
    # initialize Superfacility API
    initialize_sfapi()
    # update for the first time
    update()
    # start server
    print("Starting server...")
    server.start()
