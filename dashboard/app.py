from bson.objectid import ObjectId
import os
import re
from trame.assets.local import LocalFileManager
from trame.ui.router import RouterViewLayout
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import plotly, router, vuetify3 as vuetify, html

from model_manager import ModelManager
from objectives_manager import ObjectivesManager
from parameters_manager import ParametersManager
from calibration_manager import SimulationCalibrationManager
from sfapi_manager import initialize_sfapi, load_sfapi_card
from state_manager import server, state, ctrl, initialize_state
from error_manager import error_panel, add_error
from utils import (
    load_experiments,
    load_database,
    load_data,
    load_variables,
    plot,
)

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

mod_manager = None
par_manager = None
obj_manager = None
cal_manager = None

# load database
db = load_database()
# list of available experiments
experiment_list = load_experiments()

# -----------------------------------------------------------------------------
# Functions and callbacks
# -----------------------------------------------------------------------------


def update(
    reset_model=True,
    reset_parameters=True,
    reset_objectives=True,
    reset_calibration=True,
    reset_plots=True,
    reset_gui_route_home=True,
    reset_gui_route_nersc=True,
    reset_gui_layout=True,
    **kwargs,
):
    print("Updating...")
    global mod_manager
    global par_manager
    global obj_manager
    global cal_manager
    # load data
    exp_data, sim_data = load_data(db)
    # reset model
    if reset_model:
        mod_manager = ModelManager(db)
    # load input and output variables
    input_variables, output_variables, simulation_calibration = load_variables()
    # reset parameters
    if reset_parameters:
        par_manager = ParametersManager(mod_manager, input_variables)
    elif reset_model:
        # if resetting only model, model attribute must be updated
        par_manager.model = mod_manager
    # reset objectives
    if reset_objectives:
        obj_manager = ObjectivesManager(mod_manager, output_variables)
    # reset calibration
    if reset_calibration:
        cal_manager = SimulationCalibrationManager(simulation_calibration)
    # reset GUI home route
    if reset_gui_route_home:
        home_route()
    # reset GUI NERSC route
    if reset_gui_route_nersc:
        nersc_route()
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
            reset_parameters=True,
            reset_objectives=True,
            reset_calibration=True,
            reset_plots=True,
            reset_gui_route_home=True,
            reset_gui_route_nersc=False,
            reset_gui_layout=False,
        )


@state.change("model_type", "model_training_time")
def update_on_change_model(**kwargs):
    # skip if triggered on server ready (all state variables marked as modified)
    if len(state.modified_keys) == 1:
        print("Model type changed...")
        update(
            reset_model=True,
            reset_parameters=False,
            reset_objectives=False,
            reset_calibration=False,
            reset_plots=True,
            reset_gui_route_home=True,
            reset_gui_route_nersc=False,
            reset_gui_layout=False,
        )


@state.change(
    "parameters",
    "opacity",
    "parameters_min",
    "parameters_max",
    "parameters_show_all",
)
def update_on_change_others(**kwargs):
    # skip if triggered on server ready (all state variables marked as modified)
    if len(state.modified_keys) == 1:
        print("Parameters, opacity changed...")
        update(
            reset_model=False,
            reset_parameters=False,
            reset_objectives=False,
            reset_calibration=False,
            reset_plots=True,
            reset_gui_route_home=False,
            reset_gui_route_nersc=False,
            reset_gui_layout=False,
        )


def find_simulation(event, db):
    try:
        # extract the ID of the point that the user clicked on
        this_point_id = event["points"][0]["customdata"][0]
        # find the document with matching ID from the experiment collection
        documents = list(db[state.experiment].find({"_id": ObjectId(this_point_id)}))
        if len(documents) == 1:
            this_point_parameters = {
                parameter: documents[0][parameter]
                for parameter in state.parameters.keys()
                if parameter in documents[0]
            }
            print(f"Clicked on data point ({this_point_parameters})")
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
                # parameters control panel
                with vuetify.VRow():
                    with vuetify.VCol():
                        par_manager.panel()
                # model control panel
                with vuetify.VRow():
                    with vuetify.VCol():
                        mod_manager.panel()
                # plots control panel
                with vuetify.VRow():
                    with vuetify.VCol():
                        with vuetify.VExpansionPanels(
                            v_model=("expand_panel_control_plots", 0)
                        ):
                            with vuetify.VExpansionPanel(
                                title="Control: Plots",
                                style="font-size: 20px; font-weight: 500;",
                            ):
                                with vuetify.VExpansionPanelText():
                                    # create a row for the slider label
                                    with vuetify.VRow():
                                        vuetify.VListSubheader(
                                            "Projected Data Depth",
                                            style="margin-top: 16px;",
                                        )
                                    # create a row for the slider and text field
                                    with vuetify.VRow(no_gutters=True):
                                        with vuetify.VSlider(
                                            v_model_number=("opacity",),
                                            change="flushState('opacity')",
                                            hide_details=True,
                                            max=1.0,
                                            min=0.0,
                                            step=0.025,
                                            style="align-items: center;",
                                        ):
                                            with vuetify.Template(v_slot_append=True):
                                                vuetify.VTextField(
                                                    v_model_number=("opacity",),
                                                    density="compact",
                                                    hide_details=True,
                                                    readonly=True,
                                                    single_line=True,
                                                    style="margin-top: 0px; padding-top: 0px; width: 80px;",
                                                    type="number",
                                                )
            # plots card
            with vuetify.VCol(cols=8):
                with vuetify.VCard():
                    with vuetify.VCardTitle("Plots"):
                        with vuetify.VContainer(
                            style=f"height: {400 * len(state.parameters)}px;"
                        ):
                            figure = plotly.Figure(
                                display_mode_bar="true",
                                config={"responsive": True},
                                click=(open_simulation_dialog, "[utils.safe($event)]"),
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
            error_panel()
            with vuetify.VContainer(style="height: 100vh; overflow-y: auto"):
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
                    style="width: 100%; height: 100%;",
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
