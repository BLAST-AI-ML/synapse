import numpy as np
import plotly.graph_objects as go

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import plotly, vuetify

from variables import read_variables

# Get a server to work with
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# TODO generalize for different objectives
def model(*_args, **kwargs):
    inputs = np.array(_args)
    result = np.sum(inputs)
    return result

def plot(*_args, **kwargs):
    inputs = np.array(_args)
    result = np.sum(inputs)
    x = np.arange(0.0, 2.0, 0.01)
    y = np.sin(2 * np.pi * x)
    y *= result
    fig = go.Figure(data=go.Scatter(x=x, y=y))
    fig.update_xaxes(exponentformat="e")
    fig.update_yaxes(exponentformat="e")
    return fig

# read state variables
yaml_file = "variables.yml"
input_variables, output_variables = read_variables(yaml_file)

# initialize input variables (parameters)
parameters_name = []  # FIXME global variable?
parameters_value = [] # FIXME global variable?
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
objectives_name = [] # FIXME global variable?
for _, objective_dict in output_variables.items():
    objective_name = objective_dict["name"]
    exec(f"state.objective_{objective_name} = {model(*parameters_value)}")
    objectives_name.append(objective_name)

def get_state_parameters():
    parameters = []
    for parameter in [f"state.parameter_{name}" for name in parameters_name]:
        exec(f"parameters.append(np.float64({parameter}))")
    return parameters

@state.change(*[f"parameter_{name}" for name in parameters_name])
def update_objectives(**kwargs):
    parameters = get_state_parameters()
    for name in objectives_name:
        exec(f"state.objective_{name} = model(*parameters, **kwargs)")

@state.change(*[f"parameter_{name}" for name in parameters_name])
def update_plots(**kwargs):
    parameters = get_state_parameters()
    fig = plot(*parameters, **kwargs)
    ctrl.plotly_figure_update = plotly_figure.update(fig)

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
                            with vuetify.VContainer(style="height: 50vh"):
                                plotly_figure = plotly.Figure(
                                        display_mode_bar="true", config={"responsive": True}
                                )
                                ctrl.plotly_figure_update = plotly_figure.update

# Main
if __name__ == "__main__":
    server.start()
