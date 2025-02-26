import argparse
import pandas as pd
import requests
import torch
from datetime import datetime, timedelta
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
    "user": "bella_sf_admin",
    "collection": "ip2",
}
config, experimental_docs, simulation_docs = load_database(db_defaults)
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

# terminal output for NERSC control
state.sfapi_output = ""

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

# TODO encapsulate in sfapi class?
def check_status():
    # Check status
    from sfapi_client import Client
    from sfapi_client.compute import Machine

    # restore private key from DB
    sfapi = config.find_one({"name": "sfapi"})
    sfapi_client_id = sfapi["client_id"]
    sfapi_key_pem = sfapi["key"]
    sfapi_expiration = sfapi["expiration"]

    # create an authenticated client
    output = []
    with Client(client_id=sfapi_client_id, secret=sfapi_key_pem) as client:
        # does not need authentication
        status = client.compute(Machine.perlmutter)
        output += [str(status)]

        # needs authentication
        perlmutter = client.compute(Machine.perlmutter)
        ls_results = perlmutter.ls("/global/cfs/cdirs/m558")
        output += ["ls in CFS:"]
        for x in ls_results:
            output += [x.name]

    return output

# TODO encapsulate in sfapi class?
@ctrl.add("exchange_credentials")
def exchange_credentials(state):
    """Read a PEM file and store it in the database"""

    sfapi_client_id = state.client_id
    private_key = state.private_key
    sfapi_expiration =  datetime.now() + timedelta(days=int(state.expiration_days))

    # Read private key file
    output = []
    output.append("\nReading Private Key File...")
    try:
        if not private_key:
            raise ValueError("No Private Key File Uploaded")

        sfapi_key_pem = private_key["content"].decode("utf-8")
        #output.append(f"sfapi_key_pem: {sfapi_key_pem}")
        output.append(f"Client ID: {sfapi_client_id}")
        output.append(f"Expiration: {sfapi_expiration}")

        # store in DB
        update_data = {"$set": {
            "client_id": sfapi_client_id,
            "key": sfapi_key_pem,
            "expiration": sfapi_expiration,
        }}
        config.update_one({"name": "sfapi"}, update_data, upsert=True)

    except ValueError as e:
        # Record exception
        output.append(f"ValueError: {e}")
        # Update state terminal output
        state.sfapi_output = "\n".join(output)
        return

    # Create session
    output.append("\nCreating Session...")

    output += check_status()
    print(output)

    # Update state terminal output
    state.sfapi_output = "\n".join(output)

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

# home route
with RouterViewLayout(server, "/"):
    with v2.VRow():
        with v2.VCol(cols=4):
            with v2.VRow():
                with v2.VCol():
                    parameters.card()
            with v2.VRow():
                with v2.VCol():
                    objectives.card()
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
                                            click=parameters.recenter,
                                            style="width: 100%; text-transform: none;",
                                        )
                                    with v2.VCol():
                                        v2.VBtn(
                                            "Optimize",
                                            click=model.optimize,
                                            style="width: 100%; text-transform: none;",
                                        )
        with v2.VCol(cols=8):
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
with RouterViewLayout(server, "/nersc"):
    with v2.VRow():
        with v2.VCol(cols=4):
            with v2.VCard():
                with v2.VCardTitle("Superfacility API"):
                    with v2.VCardText():
                        with v2.VRow():
                            with v2.VCol():
                                v2.VSlider(
                                    v_model_number=("expiration_days", 33),
                                    label="Expiration (days)",
                                    min=0,
                                    max=63,
                                    step=1,
                                    classes="align-center",
                                    hide_details=True,
                                    #style="width: 200px",
                                    thumb_label="always",
                                    thumb_size=25,
                                    type="number",
                                )
                        with v2.VRow():
                            with v2.VCol():
                                v2.VTextField(
                                    label="Client Id",
                                    v_model=("client_id", None),
                                    single_line=True,
                                )
                        with v2.VRow():
                            with v2.VCol():
                                v2.VFileInput(
                                    label="Select Private Key File (PEM)",
                                    v_model=("private_key", None),
                                    accept=".pem",
                                    __properties=["accept"],
                                )
                        with v2.VRow():
                            with v2.VCol():
                                v2.VBtn(
                                    "Access Superfacility API",
                                    click=lambda:exchange_credentials(state),
                                    style="width: 100%; text-transform: none;",
                                )
                        with v2.VRow():
                            with v2.VCol():
                                v2.VTextarea(
                                    v_model=("sfapi_output", ""),
                                    readonly=True,
                                    rows=10,
                                    style="width: 100%",
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
