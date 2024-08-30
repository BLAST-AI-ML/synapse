import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import plotly, vuetify

from variables import read_variables

# Get a server to work with
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# TODO generalize for different objectives
def model(inputs_list, **kwargs):
    inputs = np.array(inputs_list)
    result = np.sum(inputs)
    return result

def plot(inputs_list, **kwargs):
    # number of inputs
    inputs_num = len(inputs_list)
    # NumPy array of inputs for math operations
    inputs = np.array(inputs_list)
    result = np.sum(inputs)
    x = np.arange(0.0, 2.0, 0.01)
    y = np.sin(2 * np.pi * x)
    y *= result
    fig = make_subplots(rows=inputs_num, cols=1)
    for i, input in enumerate(inputs_list):
        # NOTE row count starts from 1, enumerate count starts from 0
        this_row = i+1
        this_col = 1
        fig.add_trace(
            go.Scatter(x=x, y=y),
            row=this_row, col=this_col)
        fig.add_vline(x=input, line_dash="dash", row=this_row, col=this_col)
        fig.update_xaxes(exponentformat="e", row=this_row, col=this_col)
        fig.update_yaxes(exponentformat="e", row=this_row, col=this_col)
    return fig

# read state variables
yaml_file = "variables.yml"
input_variables, output_variables = read_variables(yaml_file)

# initialize input variables (parameters)
# FIXME global variables?
parameters_name = []
parameters_value = []
parameters_min = []
parameters_max = []
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
    parameters_min.append(parameter_min)
    parameters_max.append(parameter_max)

# initialize output variables (objectives)
# FIXME global variables?
objectives_name = []
for _, objective_dict in output_variables.items():
    objective_name = objective_dict["name"]
    exec(f"state.objective_{objective_name} = {model(parameters_value)}")
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
        exec(f"state.objective_{name} = model(parameters, **kwargs)")

@state.change(*[f"parameter_{name}" for name in parameters_name])
def update_plots(**kwargs):
    parameters = get_state_parameters()
    fig = plot(parameters, **kwargs)
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
                            with vuetify.VContainer(style=f"height: {25*len(parameters_name)}vh"):
                                plotly_figure = plotly.Figure(
                                        display_mode_bar="true", config={"responsive": True}
                                )
                                ctrl.plotly_figure_update = plotly_figure.update

# Main
if __name__ == "__main__":
    server.start()
