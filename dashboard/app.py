import numpy as np

from trame.app import get_server
from trame.ui.router import RouterViewLayout
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import router, vuetify, xterm

# Get a server to work with
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

@state.change("tod", "gvd", "z_target")
def update_model(tod, gvd, z_target, **kwargs):
    tod = np.float64(tod)
    gvd = np.float64(gvd)
    z_target = np.float64(z_target)
    model(tod, gvd, z_target, **kwargs)

def model(tod, gvd, z_target, **kwargs):
    state.nop = tod + gvd + z_target

# parameters
state.tod = 0.0
state.gvd = 0.0
state.z_target = 0.0

# objectives
model(state.tod, state.gvd, state.z_target)

# GUI
with SinglePageLayout(server) as layout:
    layout.title.set_text("IFE Superfacility")

    with layout.toolbar:
        # toolbar components
        pass

    with layout.content:
        # content components
        with vuetify.VContainer():
            with vuetify.VRow(no_gutters=True):
                with vuetify.VCol(cols="auto"):
                    with vuetify.VCard(style="width: 300px"):
                        with vuetify.VCardTitle("Parameters"):
                            with vuetify.VCardText():
                                vuetify.VTextField(
                                    clearable=True,
                                    hide_details=True,
                                    label="TOD",
                                    v_model=("tod",),
                                )
                                vuetify.VTextField(
                                    clearable=True,
                                    hide_details=True,
                                    label="GVD",
                                    v_model=("gvd",),
                                )
                                vuetify.VTextField(
                                    clearable=True,
                                    hide_details=True,
                                    label="Z (target)",
                                    v_model=("z_target",),
                                )
            with vuetify.VRow(no_gutters=True):
                with vuetify.VCol(cols="auto"):
                    with vuetify.VCard(style="width: 300px"):
                        with vuetify.VCardTitle("Objectives"):
                            with vuetify.VCardText():
                                vuetify.VTextField(
                                    label="number of protons",
                                    readonly=True,
                                    v_model=("nop",),
                                )

# Main
if __name__ == "__main__":
    server.start()
