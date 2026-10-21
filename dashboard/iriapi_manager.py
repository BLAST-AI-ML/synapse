import asyncio
import json
import os
import tempfile
from datetime import UTC, datetime, timezone
from pathlib import Path

from amsc_client import Client

from error_manager import add_error
from nersc_api_common import (
    USER_DATE_FORMAT,
    load_credentials_card,
    monitor_job,
)
from state_manager import state

IRIAPI_CREDENTIALS_PATH = Path.home() / ".amsc" / "credentials.json"


def get_uploaded_iriapi_credentials():
    content = state.iriapi_key_dict["content"]
    if isinstance(content, dict):
        return content

    if isinstance(content, (bytes, bytearray)):
        content = bytes(content).decode("utf-8")

    return json.loads(content)


def write_iriapi_credentials(credentials):
    IRIAPI_CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write the new credentials to an owner-only temporary file in the same
    # directory. Once the write finishes, replace the destination in one step:
    # readers see either the old complete file or the new complete file, never
    # a partially written new file. Existing permissions on the old file are
    # not changed before it is replaced.
    fd, tmp_path = tempfile.mkstemp(
        dir=IRIAPI_CREDENTIALS_PATH.parent,
        prefix=f".{IRIAPI_CREDENTIALS_PATH.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(credentials, indent=2) + "\n")
        os.replace(tmp_path, IRIAPI_CREDENTIALS_PATH)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
    print(f"Wrote AmSC IRI API credentials to {IRIAPI_CREDENTIALS_PATH}")


def parse_iriapi_credentials(credentials):
    if not isinstance(credentials, dict) or len(credentials) != 1:
        raise ValueError(
            "Uploaded AmSC IRI API credentials must contain one Globus token entry"
        )

    globus_key, credentials = next(iter(credentials.items()))
    if not globus_key.startswith("globus_") or not isinstance(credentials, dict):
        raise ValueError(
            "Uploaded AmSC IRI API credentials must contain one globus_* token entry"
        )

    required_keys = {"access_token", "expires_at"}
    if not required_keys <= credentials.keys():
        raise ValueError(
            "Uploaded AmSC IRI API credentials are missing required token fields"
        )

    access_token = credentials["access_token"]
    if isinstance(access_token, (bytes, bytearray)):
        access_token = bytes(access_token).decode("utf-8")

    return str(access_token).strip(), float(credentials["expires_at"])


async def refresh_iriapi_job(iriapi_job):
    # Refresh in a worker thread because the IRI client call is synchronous
    await asyncio.to_thread(iriapi_job.refresh)


async def monitor_iriapi_job(iriapi_job, state_variable):
    return await monitor_job(
        iriapi_job,
        state_variable,
        refresh=refresh_iriapi_job,
        is_terminal=lambda job: job.is_terminal,
        state_text=lambda job: job.state,
        is_completed=lambda job: job.state == "completed",
    )


def update_iriapi_info():
    print("Updating AmSC IRI API info...")
    try:
        # Store the access token from the uploaded file in the corresponding state variable
        credentials = get_uploaded_iriapi_credentials()
        state.iriapi_key, expires_at = parse_iriapi_credentials(credentials)
        write_iriapi_credentials(credentials)
        # Parse token expiration date from Unix timestamp
        expiration_utc = datetime.fromtimestamp(
            expires_at,
            tz=UTC,
        )
        expiration = expiration_utc.astimezone(
            timezone(expiration_utc.astimezone().utcoffset())
        )
        # If token is not expired, update info, else set to expired/unavailable
        if expiration_utc > datetime.now(UTC):
            # Update token expiration date
            state.iriapi_key_expiration = (
                f"Valid Until {expiration.strftime(USER_DATE_FORMAT)}"
            )
            # Create an authenticated client
            client = Client(auth_method="globus")
            # Ping Perlmutter so the UI reflects the current IRI resource status
            nersc = client.facility("nersc")
            perlmutter = nersc.resource("compute")
            state.iriapi_perlmutter_description = f"{perlmutter.description}"
            state.iriapi_perlmutter_status = f"{perlmutter.status}"
            print(
                f"Perlmutter status is '{state.iriapi_perlmutter_status}' with description '{state.iriapi_perlmutter_description}'"
            )
        else:
            # Reset token expiration date
            state.iriapi_key_expiration = (
                f"Expired On {expiration.strftime(USER_DATE_FORMAT)}"
            )
            # Reset Perlmutter status
            state.iriapi_perlmutter_description = "Unavailable"
            state.iriapi_perlmutter_status = "unavailable"
            title = "Unable to find a valid AmSC IRI API token"
            msg = (
                f"AmSC IRI API token expired on {expiration.strftime(USER_DATE_FORMAT)}"
            )
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
    # Skip if triggered on server ready (all state variables marked as modified)
    if len(state.modified_keys) == 1 and state.iriapi_key_dict is not None:
        print("Loading AmSC IRI API credentials...")
        # Decode, persist, and immediately validate the uploaded token file
        update_iriapi_info()


def load_iriapi_card():
    print("Setting AmSC IRI API card...")
    load_credentials_card(
        key_dict_variable="iriapi_key_dict",
        key_file_label="Token File",
        key_file_accept=".json",
        key_expiration_variable="iriapi_key_expiration",
        key_expiration_label="Token Expiration",
        perlmutter_description_variable="iriapi_perlmutter_description",
    )
