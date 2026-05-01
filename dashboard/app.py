import asyncio
from bson.objectid import ObjectId
import os
import re
from trame.assets.local import LocalFileManager
from trame.ui.router import RouterViewLayout
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import plotly, router, vuetify3 as vuetify, html

from model_manager import (
    GENESIS_MODEL_TYPE,
    GENESIS_LOGO_URL,
    ModelManager,
    clear_model_load_errors,
    is_missing_mlflow_model,
    load_model_from_mlflow_with_progress,
    model_type_dict,
)
from outputs_manager import OutputManager
from optimization_manager import OptimizationManager
from parameters_manager import ParametersManager
from calibration_manager import SimulationCalibrationManager
from sfapi_manager import load_sfapi_card
from state_manager import server, state, initialize_state
from error_manager import error_panel, add_error
from utils import (
    data_depth_panel,
    load_config_dict,
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
opt_manager = None
cal_manager = None
PLOTS_FIGURE_STATE = "plots_figure"

# list of available experiments
experiments = load_experiments()

# -----------------------------------------------------------------------------
# Functions and callbacks
# -----------------------------------------------------------------------------


def update_plot_figure(fig):
    """Replace the figure shown in the Plots card."""
    state[PLOTS_FIGURE_STATE] = plotly.Figure.to_data(fig)
    state.dirty(PLOTS_FIGURE_STATE)
    state.flush()


def update(
    reset_model=True,
    reset_output=True,
    reset_parameters=True,
    reset_calibration=True,
    reset_plots=True,
    reset_gui_route_home=True,
    reset_gui_route_nersc=True,
    reset_gui_route_chat=True,
    reset_gui_layout=True,
    preloaded_model_manager=None,
    **kwargs,
):
    print("Updating...")
    global mod_manager
    global out_manager
    global par_manager
    global opt_manager
    global cal_manager
    # load input and output variables
    input_variables, output_variables, simulation_calibration = load_variables(
        state.experiment
    )
    # load data
    config_dict = load_config_dict(state.experiment)
    # derive execution mode from execution_mode in the experiment configuration file
    execution_mode = config_dict.get("execution_mode") or {}
    state.model_training_mode = execution_mode.get("ml_training", "local")
    db = load_database(config_dict)
    exp_data, sim_data = load_data(db, state.experiment, state.experiment_date_range)
    # reset output
    if reset_output:
        out_manager = OutputManager(output_variables)
    # reset calibration
    if reset_calibration:
        cal_manager = SimulationCalibrationManager(simulation_calibration)
    # reset model
    if reset_model:
        state.model_available = False
        if preloaded_model_manager is None:
            mod_manager = ModelManager(
                config_dict=config_dict,
                model_type=model_type_dict[state.model_type_verbose],
            )
        else:
            mod_manager = preloaded_model_manager
        state.model_available = mod_manager.avail()
        opt_manager = OptimizationManager(mod_manager)
    # reset parameters
    if reset_parameters:
        par_manager = ParametersManager(mod_manager, input_variables)
    elif reset_model:
        # if resetting only model, model attribute must be updated
        par_manager.model = mod_manager
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
    if reset_plots:
        fig = plot(
            exp_data=exp_data,
            sim_data=sim_data,
            model_manager=mod_manager,
            cal_manager=cal_manager,
        )
        update_plot_figure(fig)


async def update_with_model_download_indicator(**update_kwargs):
    """Run a dashboard update with visible download feedback for NN models."""
    show_model_download = (
        update_kwargs.get("reset_model", True)
        and state.model_type_verbose == GENESIS_MODEL_TYPE
    )
    load_error = None
    if show_model_download:
        experiment = state.experiment
        model_type_verbose = state.model_type_verbose
        config_dict = load_config_dict(experiment)
        model_type = model_type_dict[model_type_verbose]
        state.model_available = False
        state.model_downloading = True
        state.model_download_status = "Downloading from American Science Cloud..."
        state.model_download_progress = None
        clear_model_load_errors()
        state.flush()
        await asyncio.sleep(0.05)
        try:
            loaded_model = await asyncio.to_thread(
                load_model_from_mlflow_with_progress,
                config_dict,
                model_type,
                asyncio.get_running_loop(),
            )
        except Exception as e:
            loaded_model = None
            model_name = f"synapse-{config_dict['experiment']}_{model_type}"
            if is_missing_mlflow_model(e):
                print(f"Model {model_name} not found in MLflow; continuing without it.")
            else:
                load_error = e
        if (
            state.experiment != experiment
            or state.model_type_verbose != model_type_verbose
        ):
            state.model_downloading = False
            state.model_download_status = None
            state.model_download_progress = None
            state.flush()
            return
        update_kwargs["preloaded_model_manager"] = ModelManager(
            config_dict=config_dict,
            model_type=model_type,
            loaded_model=loaded_model,
        )
    try:
        update(**update_kwargs)
        if load_error is not None:
            title = f"Unable to load model {model_type}"
            msg = f"Error occurred when loading model from MLflow: {load_error}"
            add_error(title, msg)
            state.flush()
    finally:
        if show_model_download:
            state.model_downloading = False
            state.model_download_status = None
            state.model_download_progress = None
            state.flush()


@state.change(
    "experiment",
    "experiment_date_range",
    "model_type_verbose",
    "model_training_time",
    "displayed_output",
    "parameters",
    "opacity",
    "parameters_min",
    "parameters_max",
    "parameters_show_all",
    "simulation_calibration",
    "use_inferred_calibration",
)
async def reset(**kwargs):
    experiment_keys = {
        "experiment",
        "experiment_date_range",
    }
    model_keys = {
        "model_type_verbose",
        "model_training_time",
    }
    plot_keys = {
        "displayed_output",
        "parameters",
        "opacity",
        "parameters_min",
        "parameters_max",
        "parameters_show_all",
        "simulation_calibration",
        "use_inferred_calibration",
    }
    watched_keys = experiment_keys | model_keys | plot_keys
    modified_keys = set(state.modified_keys) & watched_keys

    if not modified_keys or modified_keys == watched_keys:
        return

    print(f"Reacting to state change in {modified_keys}...")
    if modified_keys & experiment_keys:
        await update_with_model_download_indicator(
            reset_model=True,
            reset_output=True,
            reset_parameters=True,
            reset_calibration=True,
            reset_plots=True,
            reset_gui_route_home=True,
            reset_gui_route_nersc=False,
            reset_gui_route_chat=False,
            reset_gui_layout=False,
        )
    elif modified_keys & model_keys:
        await update_with_model_download_indicator(
            reset_model=True,
            reset_output=False,
            reset_parameters=False,
            reset_calibration=False,
            reset_plots=True,
            reset_gui_route_home=True,
            reset_gui_route_nersc=False,
            reset_gui_route_chat=False,
            reset_gui_layout=False,
        )
    elif modified_keys & plot_keys:
        update(
            reset_model=False,
            reset_output=False,
            reset_parameters=False,
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
    config_dict = load_config_dict(state.experiment)
    db = load_database(config_dict)
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
                        v_model=("active_tab", "parameters_tab"),
                        color="primary",
                        mandatory=True,
                    ):
                        vuetify.VTab("Parameters", value="parameters_tab")
                        vuetify.VTab("Optimization", value="optimization_tab")
                        vuetify.VTab("ML", value="ml_tab")
                    with vuetify.VWindow(v_model=("active_tab",), mandatory=True):
                        with vuetify.VWindowItem(value="parameters_tab"):
                            # output control panel
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    out_manager.panel()
                            # parameters control panel
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    par_manager.panel()
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
                            style=f"height: {400 * len(state.parameters)}px;"
                        ):
                            plotly.Figure(
                                state_variable_name=PLOTS_FIGURE_STATE,
                                display_mode_bar="true",
                                config={"responsive": True},
                                click=(
                                    open_simulation_dialog,
                                    "[utils.safe($event)]",
                                ),
                            )


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
    if GENESIS_LOGO_URL:
        state.trame__favicon = GENESIS_LOGO_URL
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("Synapse")
        # add toolbar components
        with layout.toolbar:
            vuetify.VSpacer()
            # experiment selector
            vuetify.VSelect(
                v_model=("experiment",),
                label="Experiments",
                items=(experiments,),
                dense=True,
                hide_details=True,
                prepend_icon="mdi-atom",
                style="max-width: 250px; margin-right: 14px;",
            )
            # date range selector for experiment filtering
            vuetify.VDateInput(
                v_model=("experiment_date_range",),
                label="Date range",
                multiple="range",
                dense=True,
                hide_details=True,
                style="max-width: 250px; margin-right: 14px;",
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
                # Dashboard route
                vuetify.VListItem(
                    to="/",
                    prepend_icon="mdi-monitor-dashboard",
                    title="Digital Twin Prototype",
                )
                # Chat route
                vuetify.VListItem(
                    to="/chat",
                    prepend_icon="mdi-chat",
                    title="AI Assistant",
                )
                # NERSC route
                vuetify.VListItem(
                    to="/nersc",
                    prepend_icon="mdi-lan-connect",
                    title="NERSC API key",
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
    # update for the first time
    update()
    # start server
    print("Starting server...")
    server.start()
