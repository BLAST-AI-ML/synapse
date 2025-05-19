from datetime import datetime
from sfapi_client import Client
from sfapi_client.compute import Machine
from trame.widgets import vuetify2 as vuetify

from state_manager import state
from utils import load_database

def sfapi_info(client):
    print("Updating Superfacility API info...")
    status = client.compute(Machine.perlmutter)
    # see https://docs.python.org/3/library/datetime.html#format-codes
    # for all format codes accepted by the methods strftime and strptime
    sfapi_format = "%Y-%m-%dT%H:%M:%S.%f%z"
    user_format = "%A, %B %d, %Y, %H:%M %Z"
    # update Perlmutter status
    state.perlmutter_status = f"{status.description}"
    # get the user object
    user = client.user()
    # get API clients associated with the user
    credential_clients = user.clients()
    # loop over API clients
    # FIXME filter based on client name (e.g., "bella") if more than one client
    for credential_client in credential_clients:
        # update key expiration date
        state.sfapi_key_expiration = datetime.strptime(
            credential_client.expiresAt, sfapi_format).strftime(user_format)

def sfapi_init():
    print("Initializing Superfacility API...")
    config, _, _ = load_database()
    # get existing configuration from the database, if any
    sfapi_config = config.find_one({"name": "sfapi"})
    if sfapi_config is not None:
        state.sfapi_client_id = sfapi_config["client_id"]
        state.sfapi_key = sfapi_config["key"]
        try:
            # create an authenticated client and update info
            with Client(client_id=state.sfapi_client_id, secret=state.sfapi_key) as client:
                sfapi_info(client)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

@state.change("sfapi_key_dict")
def load_credentials(sfapi_key_dict, **kwargs):
    # return if no key file has been uploaded
    if state.sfapi_key_dict is None:
        return
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
    try:
        # create an authenticated client and update info
        with Client(client_id=state.sfapi_client_id, secret=state.sfapi_key) as client:
            sfapi_info(client)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def sfapi_card():
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
                # row with text field to display Perlmutter status
                with vuetify.VRow():
                    with vuetify.VCol():
                        vuetify.VTextField(
                            v_model=("perlmutter_status",),
                            label="Perlmutter Status",
                            readonly=True,
                        )
                # row with text field to display key expiration date
                with vuetify.VRow():
                    with vuetify.VCol():
                        vuetify.VTextField(
                            v_model=("sfapi_key_expiration",),
                            label="Key Expiration Date",
                            readonly=True,
                        )
