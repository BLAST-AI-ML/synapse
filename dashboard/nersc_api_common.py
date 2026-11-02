"""Helpers shared by the Superfacility API and AmSC IRI API managers."""

import asyncio

from trame.widgets import vuetify3 as vuetify

from state_manager import state

# Format used to display expiration dates to the user
# (see https://docs.python.org/3/library/datetime.html#format-codes
# for all format codes accepted by the methods strftime and strptime)
USER_DATE_FORMAT = "%B %d, %Y, %H:%M %Z"

# Interval in seconds between two polls of a running job
JOB_POLL_INTERVAL = 5


async def monitor_job(
    job, state_variable, refresh, is_terminal, state_text, is_completed
):
    """Poll a job until it reaches a terminal state and mirror its status in the state.

    ``refresh`` is an awaitable callback that updates ``job`` in place; the other
    callbacks are synchronous and inspect ``job``.
    """
    while not is_terminal(job):
        await asyncio.sleep(JOB_POLL_INTERVAL)
        await refresh(job)
        # Make the status more readable by putting in spaces and capitalizing the words
        job_status = state_text(job).replace("_", " ").title()
        if state[state_variable] != job_status:
            state[state_variable] = job_status
            state.flush()
            print("Job status: ", state[state_variable])
    return is_completed(job)


def load_credentials_card(
    key_dict_variable,
    key_file_label,
    key_file_accept,
    key_expiration_variable,
    key_expiration_label,
    perlmutter_description_variable,
):
    # row with component to upload input file with top padding
    with vuetify.VRow(style="padding-top: 20px;"):
        with vuetify.VCol():
            vuetify.VFileInput(
                v_model=(key_dict_variable,),
                label=key_file_label,
                accept=key_file_accept,
                prepend_icon="",
                prepend_inner_icon="mdi-paperclip",
                __properties=["accept"],
            )
    # row with text field to display key expiration date
    with vuetify.VRow():
        with vuetify.VCol():
            vuetify.VTextField(
                v_model=(key_expiration_variable,),
                label=key_expiration_label,
                readonly=True,
            )
    # row with text field to display Perlmutter status
    with vuetify.VRow():
        with vuetify.VCol():
            vuetify.VTextField(
                v_model=(perlmutter_description_variable,),
                label="Perlmutter Status",
                readonly=True,
            )
