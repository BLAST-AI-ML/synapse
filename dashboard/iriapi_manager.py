import asyncio
import json
from datetime import datetime, timezone

from amsc_client import Client
from trame.widgets import vuetify3 as vuetify

from error_manager import add_error
from state_manager import state


def parse_iriapi_credentials():
    content = state.iriapi_key_dict["content"]
    if isinstance(content, dict):
        credentials = content
    else:
        if isinstance(content, (bytes, bytearray)):
            content = bytes(content).decode("utf-8")
        credentials = json.loads(content)

    if not isinstance(credentials, dict) or len(credentials) != 1:
        raise ValueError(
            "Uploaded AmSC IRI API credentials must contain one Globus token entry"
        )

    globus_key, credentials = next(iter(credentials.items()))
    if not globus_key.startswith("globus_") or not isinstance(credentials, dict):
        raise ValueError(
            "Uploaded AmSC IRI API credentials must contain one globus_* token entry"
        )

    required_keys = {"access_token", "refresh_token", "expires_at"}
    if not required_keys <= credentials.keys():
        raise ValueError(
            "Uploaded AmSC IRI API credentials are missing required token fields"
        )

    access_token = credentials["access_token"]
    if isinstance(access_token, (bytes, bytearray)):
        access_token = bytes(access_token).decode("utf-8")

    return str(access_token).strip(), float(credentials["expires_at"])


async def monitor_iriapi_job(iriapi_job, state_variable):
    while not iriapi_job.is_terminal:
        await asyncio.sleep(5)
        # Refresh in a worker thread because the IRI client call is synchronous
        await asyncio.to_thread(iriapi_job.refresh)
        # Make the status more readable by putting in spaces and capitalizing the words
        job_status = iriapi_job.state.replace("_", " ").title()
        if state[state_variable] != job_status:
            state[state_variable] = job_status
            state.flush()
            print("Job status: ", state[state_variable])
    return iriapi_job.state == "completed"


def update_iriapi_info():
    print("Updating AmSC IRI API info...")
    try:
        # Store the access token from the uploaded file in the corresponding state variable
        state.iriapi_key, expires_at = parse_iriapi_credentials()
        # (see https://docs.python.org/3/library/datetime.html#format-codes
        # for all format codes accepted by the methods strftime and strptime)
        user_format = "%B %d, %Y, %H:%M %Z"
        # Parse token expiration date from Unix timestamp
        expiration_utc = datetime.fromtimestamp(
            expires_at,
            tz=timezone.utc,
        )
        expiration = expiration_utc.astimezone(
            timezone(expiration_utc.astimezone().utcoffset())
        )
        # If token is not expired, update info, else set to expired/unavailable
        if expiration_utc > datetime.now(timezone.utc):
            # Update token expiration date
            state.iriapi_key_expiration = (
                f"Valid Until {expiration.strftime(user_format)}"
            )
            # Create an authenticated client
            client = Client(auth_method="globus")
            # Ping Perlmutter so the UI reflects the current IRI resource status
            nersc = client.facility("nersc")
            perlmutter = nersc.resource("compute")
            state.iriapi_perlmutter_description = f"{perlmutter.description}"
            state.iriapi_perlmutter_status = f"{perlmutter.status}"
            print(
                f"Perlmutter status is {state.iriapi_perlmutter_status} with description '{state.iriapi_perlmutter_description}'"
            )
        else:
            # Reset token expiration date
            state.iriapi_key_expiration = (
                f"Expired On {expiration.strftime(user_format)}"
            )
            # Reset Perlmutter status
            state.iriapi_perlmutter_description = "Unavailable"
            state.iriapi_perlmutter_status = "unavailable"
            title = "Unable to find a valid AmSC IRI API token"
            msg = f"AmSC IRI API token expired on {expiration.strftime(user_format)}"
            add_error(title, msg)
            print(msg)
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
    # Decode the uploaded token file and immediately validate it against IRI
    if state.iriapi_key_dict is not None:
        print("Loading AmSC IRI API credentials...")
        try:
            update_iriapi_info()
        except Exception as e:
            print(f"An error occurred while loading AmSC IRI API credentials: {e}")


def load_iriapi_card():
    print("Setting AmSC IRI API card...")
    # Row with component to upload input file with top padding
    with vuetify.VRow(style="padding-top: 20px;"):
        with vuetify.VCol():
            vuetify.VFileInput(
                v_model=("iriapi_key_dict",),
                label="Token File",
                accept=".json",
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
