from datetime import datetime
import os
from sfapi_client import Client
from sfapi_client.compute import Machine
from trame.widgets import vuetify2 as vuetify

from state_manager import state


def parse_sfapi_key(key_str):
    # extract lines from the key string
    key_lines = key_str.splitlines(keepends=True)
    # the first line must contain the client ID,
    # check that the RSA key begins on the second line
    if key_lines[0].rstrip() == "-----BEGIN RSA PRIVATE KEY-----":
        raise ValueError("Key file must include client ID in the first line")
    # set the client ID from the first line and remove the line
    state.sfapi_client_id = key_lines.pop(0).rstrip()
    # set the key from the remaining lines in the file
    state.sfapi_key = "".join(key_lines)


def initialize_sfapi():
    print("Initializing Superfacility API...")
    # look for a key file in the current directory
    key_path = os.path.join(os.getcwd(), "priv_key.pem")
    if os.path.isfile(key_path):
        try:
            with Client(key=key_path) as client:
                # store the whole content of the key file in a string
                key_str = client._secret
                # store the client ID and key in the respective state variables
                parse_sfapi_key(key_str)
                # update Superfacility API info
                update_sfapi_info()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def update_sfapi_info():
    print("Updating Superfacility API info...")
    try:
        # create an authenticated client and update info
        with Client(client_id=state.sfapi_client_id, secret=state.sfapi_key) as client:
            # get the user object
            user = client.user()
            # get client associated with the user and the client ID stored in the key file
            credential_client = [
                this_client
                for this_client in user.clients()
                if this_client.clientId == state.sfapi_client_id
            ][0]
            # (see https://docs.python.org/3/library/datetime.html#format-codes
            # for all format codes accepted by the methods strftime and strptime)
            sfapi_format = "%Y-%m-%dT%H:%M:%S.%f%z"
            user_format = "%B %d, %Y, %H:%M %Z"
            # parse key expiration date from string
            expiration = datetime.strptime(credential_client.expiresAt, sfapi_format)
            # if key is not expired, update info, else set to expired/unavailable
            if expiration.replace(tzinfo=None) > datetime.now():
                # update key expiration date
                state.sfapi_key_expiration = (
                    f"Valid Until {expiration.strftime(user_format)}"
                )
                # update Perlmutter status
                status = client.compute(Machine.perlmutter)
                state.perlmutter_description = f"{status.description}"
                state.perlmutter_status = f"{status.status.value}"
                print(f"Perlmutter status is {state.perlmutter_status} with description {state.perlmutter_description}")
            else:
                # reset key expiration date
                state.sfapi_key_expiration = (
                    f"Expired On {expiration.strftime(user_format)}"
                )
                # reset Perlmutter status
                state.perlmutter_description = "Unavailable"
                state.perlmutter_status = "unavailable"
                print("Key is expired, setting perlmutter status to unavailable")
    except Exception as e:
        print(f"An unexpected error occurred when connecting to superfacility:\n{e}")
        # reset key expiration date
        state.sfapi_key_expiration = "Unavailable"
        # reset Perlmutter status
        state.perlmutter_description = "Unavailable"
        state.perlmutter_status = "unavailable"
        print("Setting perlmutter status to unavailable")


@state.change("sfapi_key_dict")
def load_sfapi_credentials(**kwargs):
    # skip if triggered on server ready (all state variables marked as modified)
    if len(state.modified_keys) == 1:
        # return if no key file has been uploaded (redundant)
        if state.sfapi_key_dict is not None:
            print("Loading Superfacility API credentials...")
            # store the whole content of the key file in a string
            key_str = state.sfapi_key_dict["content"].decode("utf-8")
            # store the client ID and key in the respective state variables
            parse_sfapi_key(key_str)
            # update Superfacility API info
            update_sfapi_info()


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
                            v_model=("perlmutter_description",),
                            label="Perlmutter Status",
                            readonly=True,
                        )
