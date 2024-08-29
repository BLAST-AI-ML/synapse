import matplotlib.pyplot as plt
import numpy as np

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import matplotlib, vuetify

from variables import read_variables

# Get a server to work with
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# TODO generalize for different objectives
def model(*_args, **kwargs):
    return sum(_args)

# read state variables
yaml_file = "variables.yml"
input_variables, output_variables = read_variables(yaml_file)

# initialize input variables (parameters)
parameters_name = []
parameters_value = []
for _, parameter_dict in input_variables.items():
    parameter_name = parameter_dict["name"]
    parameter_default = parameter_dict["default"]
    parameter_min = parameter_dict["value_range"][0]
    parameter_max = parameter_dict["value_range"][1]
    exec(f"state.parameter_{parameter_name} = {parameter_default}")
    exec(f"state.parameter_{parameter_name}_min = {parameter_min}")
    exec(f"state.parameter_{parameter_name}_max = {parameter_max}")
    parameters_name.append(parameter_name)
    parameters_value.append(parameter_default)

# initialize output variables (objectives)
objectives_name = []
for _, objective_dict in output_variables.items():
    objective_name = objective_dict["name"]
    exec(f"state.objective_{objective_name} = {model(*parameters_value)}")
    objectives_name.append(objective_name)

@state.change(*[f"parameter_{name}" for name in parameters_name])
def update_objectives(**kwargs):
    parameters = []
    for parameter in [f"state.parameter_{name}" for name in parameters_name]:
        exec(f"parameters.append(np.float64({parameter}))")
    for name in objectives_name:
        exec(f"state.objective_{name} = model(*parameters, **kwargs)")

#def plot(tod, gvd, z_target, **kwargs):
#    # tod array
#    tod_min = tod - 0.5
#    tod_max = tod + 0.5
#    tod_arr = np.linspace(tod_min, tod_max, 100)
#    # gvd array
#    gvd_min = gvd - 0.5
#    gvd_max = gvd + 0.5
#    gvd_arr = np.linspace(gvd_min, gvd_max, 100)
#    # z_target array
#    z_target_min = z_target - 0.5
#    z_target_max = z_target + 0.5
#    z_target_arr = np.linspace(z_target_min, z_target_max, 100)
#    # plot
#    fig, ax = plt.subplots(nrows=3, ncols=1)
#    # objective vs tod
#    ax[0].plot(tod_arr, model(tod_arr, gvd, z_target))
#    ax[0].set_xlabel("TOD")
#    ax[0].set_ylabel("Number of protons")
#    # objective vs gvd
#    ax[1].plot(gvd_arr, model(tod, gvd_arr, z_target))
#    ax[1].set_xlabel("GVD")
#    ax[1].set_ylabel("Number of protons")
#    # objective vs z_target
#    ax[2].plot(z_target_arr, model(tod, gvd, z_target_arr))
#    ax[2].set_xlabel("Z (target)")
#    ax[2].set_ylabel("Number of protons")
#    fig.tight_layout()
#    return fig

#@state.change("tod", "gvd", "z_target")
#def update_plots(tod, gvd, z_target, **kwargs):
#    tod = np.float64(tod)
#    gvd = np.float64(gvd)
#    z_target = np.float64(z_target)
#    fig  = plot(tod, gvd, z_target, **kwargs)
#    ctrl.matplotlib_figure_update = matplotlib_figure.update(fig)

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
                                        for name in parameters_name:
                                            vuetify.VTextField(
                                                clearable=True,
                                                hide_details=True,
                                                label=f"{name}",
                                                v_model=(f"parameter_{name}",),
                                            )
                    with vuetify.VRow():
                        with vuetify.VCol():
                            with vuetify.VCard(style="width: 300px"):
                                with vuetify.VCardTitle("Objectives"):
                                    with vuetify.VCardText():
                                        for name in objectives_name:
                                            vuetify.VTextField(
                                                label=f"{name}",
                                                readonly=True,
                                                v_model=(f"objective_{name}",),
                                            )
                with vuetify.VCol():
                    with vuetify.VCard():
                        with vuetify.VCardTitle("Plots"):
                            with vuetify.VContainer():
                                pass
                                #matplotlib_figure = matplotlib.Figure()
                                #ctrl.matplotlib_figure_update = matplotlib_figure.update

# Main
if __name__ == "__main__":
    server.start()
