from bson.objectid import ObjectId
import os
import re
from trame.assets.local import LocalFileManager
from trame.ui.router import RouterViewLayout
from trame.ui.vuetify2 import SinglePageWithDrawerLayout
from trame.widgets import plotly, router, vuetify2 as vuetify, html

from model_manager import ModelManager
from outputs_manager import OutputManager
from objectives_manager import ObjectivesManager
from parameters_manager import ParametersManager
from calibration_manager import SimulationCalibrationManager
from sfapi_manager import initialize_sfapi, load_sfapi_card
from state_manager import server, state, ctrl, initialize_state
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

out_manager = None
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
    reset_output=True,
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
    global out_manager
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
    # reset output
    if reset_output:
        out_manager = OutputManager(output_variables)
    # reset parameters
    if reset_parameters:
        par_manager = ParametersManager(mod_manager, input_variables)
    elif reset_model:
        # if resetting only model, model attribute must be updated
        par_manager.model = mod_manager
    # reset objectives
    if reset_objectives:
        # Select the current objective
        obj_manager = ObjectivesManager(
            mod_manager,
            {k: v for k, v in output_variables.items() if output_variables[k]['name'] == state.displayed_output},
            # This creates a dictionary with only one key, to adapt to the expected interface
        )
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
            reset_output=True,
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
            reset_output=False,
            reset_parameters=False,
            reset_objectives=False,
            reset_calibration=False,
            reset_plots=True,
            reset_gui_route_home=True,
            reset_gui_route_nersc=False,
            reset_gui_layout=False,
        )


@state.change(
    "displayed_output",
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
            reset_output=False,
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
            print(f"Could not find database document that matches ID {this_point_id}")
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
        print(f"An unexpected error occurred: {e}")


def open_simulation_dialog(event):
    data_directory, file_path = find_simulation(event, db)
    state.simulation_video = file_path.endswith(".mp4")
    assets = LocalFileManager(data_directory)
    assets.url(
        key="simulation_key",
        file_path=file_path,
    )
    state.simulation_url = assets["simulation_key"]
    state.simulation_dialog = True


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
                # objectives control panel
                with vuetify.VRow():
                    with vuetify.VCol():
                        out_manager.panel()
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
                            with vuetify.VExpansionPanel():
                                vuetify.VExpansionPanelHeader(
                                    "Control: Plots",
                                    style="font-size: 20px; font-weight: 500;",
                                )
                                with vuetify.VExpansionPanelContent():
                                    # create a row for the slider label
                                    with vuetify.VRow():
                                        vuetify.VSubheader(
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
        # interactive dialog for simulation plots
        with vuetify.VDialog(v_model=("simulation_dialog",), max_width="600"):
            with vuetify.VCard(style="overflow: hidden;"):
                with vuetify.VCardTitle("Simulation Plots"):
                    vuetify.VSpacer()
                    with vuetify.VBtn(icon=True, click=close_simulation_dialog):
                        vuetify.VIcon("mdi-close")
                with vuetify.VRow(align="center", justify="center"):
                    html.Video(
                        v_if=("simulation_video",),
                        controls=True,
                        src=("simulation_url",),
                        height="480",
                    )
                    vuetify.VImg(
                        v_if=("!simulation_video",),
                        src=("simulation_url",),
                        contain=True,
                        height="480",
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
