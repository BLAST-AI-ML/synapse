from trame.widgets import vuetify3 as vuetify

from iriapi_manager import load_iriapi_card
from sfapi_manager import load_sfapi_card


def load_hpc_card():
    print("Setting HPC card...")
    with vuetify.VRow():
        with vuetify.VCol(cols=12, md=6):
            with vuetify.VCard():
                vuetify.VCardTitle("NERSC Superfacility API")
                with vuetify.VCardText():
                    load_sfapi_card()
        with vuetify.VCol(cols=12, md=6):
            with vuetify.VCard():
                vuetify.VCardTitle("AmSC IRI API")
                with vuetify.VCardText():
                    load_iriapi_card()
