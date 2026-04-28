import os

from amsc_client import Client
from trame.widgets import vuetify3 as vuetify

from error_manager import add_error
from state_manager import state

IRI_ACCESS_TOKEN_ENV = "IRI_ACCESS_TOKEN"


def update_iriapi_info():
    print("Updating AmSC IRI API info...")
    try:
        # Create an authenticated client
        client = Client(token=state.iriapi_key)
        # Update Perlmutter info
        nersc = client.facility("nersc")
        perlmutter = nersc.resource("compute")
        state.perlmutter_description = f"{perlmutter.description}"
        state.perlmutter_status = f"{perlmutter.status}"
        print(
            f"Perlmutter status is {state.perlmutter_status} with description '{state.perlmutter_description}'"
        )
    except Exception as e:
        print(f"An unexpected error occurred when connecting to AmSC IRI API:\n{e}")
        # Reset key expiration date
        state.iriapi_key_expiration = "Unavailable"
        # Reset Perlmutter status
        state.perlmutter_description = "Unavailable"
        state.perlmutter_status = "unavailable"
        title = "Unable to connect to NERSC"
        msg = f"Error occurred when connecting to NERSC through the AmSC IRI API: {e}"
        add_error(title, msg)
        print(msg)


@state.change("iriapi_key_dict")
def load_iriapi_credentials(**kwargs):
    # Load credentials from the uploaded token file
    if state.iriapi_key_dict is not None:
        print("Loading AmSC IRI API credentials from file...")
        state.iriapi_key = state.iriapi_key_dict["content"].decode("utf-8")
        update_iriapi_info()


def load_iriapi_card():
    print("Setting AmSC IRI API card...")
    # Prefer an environment-provided token when running in deployed contexts
    if IRI_ACCESS_TOKEN_ENV in os.environ:
        print("Loading AmSC IRI API credentials from environment...")
        state.iriapi_key = os.environ[IRI_ACCESS_TOKEN_ENV]
        update_iriapi_info()
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
                v_model=("perlmutter_description",),
                label="Perlmutter Status",
                readonly=True,
            )
