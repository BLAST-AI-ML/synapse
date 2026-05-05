import asyncio
import os

from amsc_client import Client
from trame.widgets import vuetify3 as vuetify

from error_manager import add_error
from state_manager import state

IRI_ACCESS_TOKEN_ENV = "IRI_ACCESS_TOKEN"


def create_iriapi_client():
    # TODO Alternative authentication
    # iriapi_key = (state.iriapi_key or "").strip()
    # if not iriapi_key:
    #    raise ValueError("Missing AmSC IRI API token")
    # client = Client(token=iriapi_key)
    # client.register_facility(
    #     "nersc",
    #     auth_method="token",
    #     token=iriapi_key,
    # )
    # Use Globus auth while token-based facility registration is disabled.
    client = Client(auth_method="globus")
    return client


async def monitor_iriapi_job(iriapi_job, state_variable):
    while not iriapi_job.is_terminal:
        await asyncio.sleep(5)
        # Refresh in a worker thread because the IRI client call is synchronous.
        await asyncio.to_thread(iriapi_job.refresh)
        # Make the status more readable by putting in spaces and capitalizing the words.
        job_status = iriapi_job.state.replace("_", " ").title()
        if state[state_variable] != job_status:
            state[state_variable] = job_status
            state.flush()
            print("Job status: ", state[state_variable])
    return iriapi_job.state == "completed"


def update_iriapi_info():
    print("Updating AmSC IRI API info...")
    try:
        # Create an authenticated client
        client = create_iriapi_client()
        # Ping Perlmutter so the UI reflects the current IRI resource status.
        nersc = client.facility("nersc")
        perlmutter = nersc.resource("compute")
        state.iriapi_perlmutter_description = f"{perlmutter.description}"
        state.iriapi_perlmutter_status = f"{perlmutter.status}"
        print(
            f"Perlmutter status is {state.iriapi_perlmutter_status} with description '{state.iriapi_perlmutter_description}'"
        )
    except Exception as e:
        print(f"An unexpected error occurred when connecting to AmSC IRI API:\n{e}")
        # Reset key expiration date
        state.iriapi_key_expiration = "Unavailable"
        # Reset Perlmutter status
        state.iriapi_perlmutter_description = "Unavailable"
        state.iriapi_perlmutter_status = "unavailable"
        title = "Unable to connect to NERSC"
        msg = f"Error occurred when connecting to NERSC through the AmSC IRI API: {e}"
        add_error(title, msg)
        print(msg)


@state.change("iriapi_key_dict")
def load_iriapi_credentials(**kwargs):
    # Decode the uploaded token file and immediately validate it against IRI.
    if state.iriapi_key_dict is not None:
        print("Loading AmSC IRI API credentials from file...")
        state.iriapi_key = state.iriapi_key_dict["content"].decode("utf-8").strip()
        if state.iriapi_key:
            update_iriapi_info()
        else:
            print("Uploaded AmSC IRI API token file is empty.")


def load_iriapi_card():
    print("Setting AmSC IRI API card...")
    # Prefer an environment-provided token when running in deployed contexts.
    iri_access_token = os.environ.get(IRI_ACCESS_TOKEN_ENV, "").strip()
    if iri_access_token:
        print("Loading AmSC IRI API credentials from environment...")
        state.iriapi_key = iri_access_token
        update_iriapi_info()
    elif IRI_ACCESS_TOKEN_ENV in os.environ:
        print(f"{IRI_ACCESS_TOKEN_ENV} is set but empty; waiting for token upload.")
    # Row with component to upload input file with top padding
    with vuetify.VRow(style="padding-top: 20px;"):
        with vuetify.VCol():
            vuetify.VFileInput(
                v_model=("iriapi_key_dict",),
                label="Token File",
                accept=".txt",
                prepend_icon="",
                prepend_inner_icon="mdi-paperclip",
                __properties=["accept"],
            )
    # Row with text field to display key expiration date
    with vuetify.VRow():
        with vuetify.VCol():
            vuetify.VTextField(
                v_model=("iriapi_key_expiration",),
                label="Token Expiration (if expired or unavailable, please upload a valid token)",
                readonly=True,
            )
    # Row with text field to display Perlmutter status
    with vuetify.VRow():
        with vuetify.VCol():
            vuetify.VTextField(
                v_model=("iriapi_perlmutter_description",),
                label="Perlmutter Status",
                readonly=True,
            )
