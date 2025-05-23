from datetime import datetime
from sfapi_client import Client
from sfapi_client.compute import Machine
from trame.widgets import vuetify2 as vuetify

from state_manager import state
from utils import load_database


@state.change(
    "sfapi_client_id",
    "sfapi_key",
)
def update_sfapi_info(**kwargs):
    # return if no client ID or key have been set
    if state.sfapi_client_id is None or state.sfapi_key is None:
        return
    print("Updating Superfacility API info...")
    try:
        # create an authenticated client and update info
        with Client(client_id=state.sfapi_client_id, secret=state.sfapi_key) as client:
            # get the user object
            user = client.user()
            # get client associated with the user and the client ID stored in the key file
            credential_client = [this_client for this_client in user.clients() if this_client.clientId == state.sfapi_client_id][0]
            # (see https://docs.python.org/3/library/datetime.html#format-codes
            # for all format codes accepted by the methods strftime and strptime)
            sfapi_format = "%Y-%m-%dT%H:%M:%S.%f%z"
            user_format = "%B %d, %Y, %H:%M %Z"
            # parse key expiration date from string
            expiration = datetime.strptime(credential_client.expiresAt, sfapi_format)
            # if key is not expired, update info, else set to expired/unavailable
            if expiration.replace(tzinfo=None) > datetime.now():
                # update key expiration date
                state.sfapi_key_expiration = f"Valid Until {expiration.strftime(user_format)}"
                # update Perlmutter status
                status = client.compute(Machine.perlmutter)
                state.perlmutter_status = f"{status.description}"
            else:
                # reset key expiration date
                state.sfapi_key_expiration = f"Expired On {expiration.strftime(user_format)}"
                # reset Perlmutter status
                state.perlmutter_status = "Unavailable"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # reset key expiration date
        state.sfapi_key_expiration = "Unavailable"
        # reset Perlmutter status
        state.perlmutter_status = "Unavailable"


def initialize_sfapi():
    print("Initializing Superfacility API...")
    config, _, _ = load_database()
    # get existing configuration from the database, if any
    sfapi_config = config.find_one({"name": "sfapi"})
    if sfapi_config is not None:
        state.sfapi_client_id = sfapi_config["client_id"]
        state.sfapi_key = sfapi_config["key"]


@state.change("sfapi_key_dict")
def load_sfapi_credentials(**kwargs):
    # return if no key file has been uploaded
    if state.sfapi_key_dict is not None:
        print("Loading Superfacility API credentials...")
        # extract key file lines from file input dictionary
        key_lines = state.sfapi_key_dict["content"].decode("utf-8").splitlines(keepends=True)
        # the first line must contain the client ID,
        # check that the RSA key begins on the second line
        if key_lines[0].rstrip() == "-----BEGIN RSA PRIVATE KEY-----":
            raise ValueError("Key file must include client ID in the first line")
        # get the client ID from the first line, remove it from the file lines
        state.sfapi_client_id = key_lines.pop(0).rstrip()
        # store remaining file lines
        state.sfapi_key = "".join(key_lines)
        # update configuration in the database
        config, _, _ = load_database()
        sfapi_config = {
            "$set": {
                "client_id": state.sfapi_client_id,
                "key": state.sfapi_key,
            }
        }
        config.update_one({"name": "sfapi"}, sfapi_config, upsert=True)


def load_sfapi_card():
    print("Setting Superfacility API card...")
    with vuetify.VCard():
        with vuetify.VCardTitle("Superfacility API"):
            with vuetify.VCardText():
                # row with component to upload input file
                with vuetify.VRow():
                    vuetify.VFileInput(
                        v_model=("sfapi_key_dict",),
                        label="Key File (pem format, must include client ID in first line)",
                        accept=".pem",
                        __properties=["accept"],
                    )
                # row with text field to display key expiration date
                with vuetify.VRow():
                    with vuetify.VCol():
                        vuetify.VTextField(
                            v_model=("sfapi_key_expiration",),
                            label="Key Expiration (if expired or unavailable, please upload a valid key)",
                            readonly=True,
                        )
                # row with text field to display Perlmutter status
                with vuetify.VRow():
                    with vuetify.VCol():
                        vuetify.VTextField(
                            v_model=("perlmutter_status",),
                            label="Perlmutter Status",
                            readonly=True,
                        )
