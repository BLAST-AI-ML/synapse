from datetime import datetime, timedelta
import inspect
import os
from trame.app import get_server
from trame.ui.router import RouterViewLayout
from trame.widgets import vuetify2 as vuetify
from sfapi_client import Client
from sfapi_client.compute import Machine

from state_manager import server, state, ctrl
from utils import load_database, load_collection


def get_sfapi_config():
    db = load_database()
    collection_config = load_collection(db, "config")

    # restore private key from DB
    sfapi = collection_config.find_one({"name": "sfapi"})
    sfapi_client_id = sfapi["client_id"]
    sfapi_key_pem = sfapi["key"]
    sfapi_expiration = sfapi["expiration"]

    return sfapi_client_id, sfapi_key_pem, sfapi_expiration


def get_sfapi_client():
    sfapi_client_id, sfapi_key_pem, sfapi_expiration = get_sfapi_config()

    if (sfapi_client_id is None or
        sfapi_key_pem is None or
        sfapi_expiration < datetime.now() is None):
        return None
    else:
        # create an authenticated client
        return Client(client_id=sfapi_client_id, secret=sfapi_key_pem)


def check_status():
    output = []
    with get_sfapi_client() as client:
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


@ctrl.add("exchange_credentials")
def exchange_credentials(state):
    """Read a PEM file and store it in the database"""
    db = load_database()
    collection_config = load_collection(db, "config")

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
        collection_config.update_one({"name": "sfapi"}, update_data, upsert=True)

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

    build_sfapi_status()


def build_sfapi_status():
    # inspect current function and module names
    cfunct = inspect.currentframe().f_code.co_name
    cmodul = os.path.basename(inspect.currentframe().f_code.co_filename)
    # get SFAPI configuration parameters (client ID, private key, expiration)
    sfapi_client_id, sfapi_key_pem, sfapi_expiration = get_sfapi_config()
    print(sfapi_expiration)
    # route
    with RouterViewLayout(server, "/nersc"):
        with vuetify.VRow():
            with vuetify.VCol(cols=4):
                with vuetify.VCard():
                    with vuetify.VCardTitle("Superfacility API"):
                        with get_sfapi_client() as client:
                            # does not need authentication
                            try:
                                pm_status = client.compute(Machine.perlmutter)
                                pm_status_description = pm_status.description
                            except Exception as e:
                                print(f"{cmodul}:{cfunct}: {e}")
                                pm_status = None
                                pm_status_description = "N/A"
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    with vuetify.VCardText():
                                        vuetify.VTextField(
                                            v_model=("perlmutter_status", pm_status_description),
                                            label="Perlmutter Status",
                                            readonly=True,
                                        )

                            with vuetify.VRow():
                                with vuetify.VCol():
                                    with vuetify.VCardText():
                                        vuetify.VTextField(
                                            v_model=("sfapi_expiration", str(sfapi_expiration)),
                                            label="API Key Expiration",
                                            readonly=True,
                                        )
                        # test = check_status() != []
                        with vuetify.VRow():
                            with vuetify.VCol():
                                vuetify.VBtn(
                                    "Refresh SFAPI Credentials",
                                    click=lambda: build_sfapi_auth(),
                                    style="width: 100%; text-transform: none;",
                                )

def build_sfapi_auth():
    with RouterViewLayout(server, "/nersc"):
        with vuetify.VRow():
            with vuetify.VCol(cols=4):
                with vuetify.VCard():
                    with vuetify.VCardTitle("Superfacility API"):
                        with vuetify.VCardText():
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    vuetify.VSlider(
                                        v_model_number=("expiration_days", 33),
                                        label="Expiration (days)",
                                        min=0,
                                        max=63,
                                        step=1,
                                        classes="align-center",
                                        hide_details=True,
                                        #style="width: 200px;",
                                        thumb_label="always",
                                        thumb_size=25,
                                        type="number",
                                    )
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    vuetify.VTextField(
                                        label="Client Id",
                                        v_model=("client_id", None),
                                        single_line=True,
                                    )
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    vuetify.VFileInput(
                                        label="Select Private Key File (PEM)",
                                        v_model=("private_key", None),
                                        accept=".pem",
                                        __properties=["accept"],
                                    )
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    vuetify.VBtn(
                                        "Connect Superfacility API",
                                        click=lambda: exchange_credentials(state),
                                        style="width: 100%; text-transform: none;",
                                    )
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    vuetify.VTextarea(
                                        v_model=("sfapi_output",),
                                        readonly=True,
                                        rows=10,
                                        style="width: 100%;",
                                    )
