import matplotlib.pyplot as plt
import numpy as np

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import matplotlib, vuetify

# Get a server to work with
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

def model(tod, gvd, z_target, **kwargs):
    nop = tod + gvd + z_target
    return nop

def plot(tod, gvd, z_target, **kwargs):
    # tod array
    tod_min = tod - 0.5
    tod_max = tod + 0.5
    tod_arr = np.linspace(tod_min, tod_max, 100)
    # gvd array
    gvd_min = gvd - 0.5
    gvd_max = gvd + 0.5
    gvd_arr = np.linspace(gvd_min, gvd_max, 100)
    # z_target array
    z_target_min = z_target - 0.5
    z_target_max = z_target + 0.5
    z_target_arr = np.linspace(z_target_min, z_target_max, 100)
    # plot
    fig, ax = plt.subplots(nrows=3, ncols=1)
    # objective vs tod
    ax[0].plot(tod_arr, model(tod_arr, gvd, z_target))
    ax[0].set_xlabel("TOD")
    ax[0].set_ylabel("Number of protons")
    # objective vs gvd
    ax[1].plot(gvd_arr, model(tod, gvd_arr, z_target))
    ax[1].set_xlabel("GVD")
    ax[1].set_ylabel("Number of protons")
    # objective vs z_target
    ax[2].plot(z_target_arr, model(tod, gvd, z_target_arr))
    ax[2].set_xlabel("Z (target)")
    ax[2].set_ylabel("Number of protons")
    fig.tight_layout()
    return fig

@state.change("tod", "gvd", "z_target")
def update_objective(tod, gvd, z_target, **kwargs):
    tod = np.float64(tod)
    gvd = np.float64(gvd)
    z_target = np.float64(z_target)
    state.nop = model(tod, gvd, z_target, **kwargs)

@state.change("tod", "gvd", "z_target")
def update_plots(tod, gvd, z_target, **kwargs):
    tod = np.float64(tod)
    gvd = np.float64(gvd)
    z_target = np.float64(z_target)
    fig  = plot(tod, gvd, z_target, **kwargs)
    ctrl.matplotlib_figure_update = matplotlib_figure.update(fig)

# parameters
state.tod = 0.0
state.gvd = 0.0
state.z_target = 0.0

# min and max values
state.tod_min = -1.0
state.tod_max =  1.0
state.gvd_min = -1.0
state.gvd_max =  1.0
state.z_target_min = -1.0
state.z_target_max =  1.0

# objectives (state.nop)
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
            with vuetify.VRow():
                with vuetify.VCol():
                    with vuetify.VRow():
                        with vuetify.VCol():
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
                    with vuetify.VRow():
                        with vuetify.VCol():
                            with vuetify.VCard(style="width: 300px"):
                                with vuetify.VCardTitle("Objectives"):
                                    with vuetify.VCardText():
                                        vuetify.VTextField(
                                            label="Number of protons",
                                            readonly=True,
                                            v_model=("nop",),
                                        )
                with vuetify.VCol():
                    with vuetify.VCard():
                        with vuetify.VCardTitle("Plots"):
                            with vuetify.VContainer():
                                matplotlib_figure = matplotlib.Figure()
                                ctrl.matplotlib_figure_update = matplotlib_figure.update

# Main
if __name__ == "__main__":
    server.start()
