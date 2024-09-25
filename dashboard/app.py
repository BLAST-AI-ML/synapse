import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import plotly, vuetify

from variables import read_variables

# Get a server to work with
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# TODO generalize for different objectives
def model(parameters_list, **kwargs):
    parameters = np.array(parameters_list)
    result = np.sum(parameters)
    return result

def plot(
        parameters_list,
        parameters_name,
        parameters_min,
        parameters_max,
        objectives_name,
        **kwargs
    ):
    # FIXME generalize for multiple objectives
    objective_name = objectives_name[0]
    # number of parameters
    parameters_num = len(parameters_list)
    # load experimental data
    df = pd.read_csv("experimental_data.csv")
    # plot
    fig = make_subplots(rows=parameters_num, cols=1)
    for i in range(parameters_num):
        # NOTE row count starts from 1, enumerate count starts from 0
        this_row = i+1
        this_col = 1
        #----------------------------------------------------------------------
        # figure trace from CSV data
        #----------------------------------------------------------------------
        # set opacity map based on distance from current inputs
        # compute Euclidean distance
        df_copy = df.copy()
        df_copy["distance"] = 0.
        # loop over all inputs except the current one
        for j in [j for j in range(parameters_num) if j != i]:
            pname_loc = parameters_name[j]
            pmin_loc = parameters_min[j]
            pmax_loc = parameters_max[j]
            pval_loc = parameters_list[j]
            df_copy["distance"] += ((df_copy[f"{pname_loc}"] - pval_loc) / (pmax_loc - pmin_loc))**2
        df_copy["distance"] = np.sqrt(df_copy["distance"])
        # normalize distance in [0,1] and compute opacity
        df_copy["distance"] = df_copy["distance"] / df_copy["distance"].max()
        df_copy["opacity"] = 1. - df_copy["distance"]
        # scatter plot with opacity
        pname = parameters_name[i]
        pmin = parameters_min[i]
        pmax = parameters_max[i]
        pval = parameters_list[i]
        exp_fig = px.scatter(
            df_copy,
            x=f"{pname}",
            y=f"{objective_name}",
            opacity=df_copy["opacity"],
        )
        exp_trace = exp_fig["data"][0]
        fig.add_trace(exp_trace, row=this_row, col=this_col)
        #----------------------------------------------------------------------
        # figure trace from model data
        #----------------------------------------------------------------------
        #x = np.linspace(start=pmin, stop=pmax, num=100)
        #y = model(x)
        #mod_trace = go.Scatter(x=x, y=y)
        #fig.add_trace(mod_trace, row=this_row, col=this_col)
        #----------------------------------------------------------------------
        # add reference input line
        #----------------------------------------------------------------------
        fig.add_vline(x=pval, line_dash="dash", row=this_row, col=this_col)
        #----------------------------------------------------------------------
        # figures style
        #----------------------------------------------------------------------
        fig.update_xaxes(
            exponentformat="e",
            title_text=f"{pname}",
            row=this_row,
            col=this_col,
        )
        fig.update_yaxes(
            exponentformat="e",
            title_text=f"{objective_name}",
            row=this_row,
            col=this_col,
        )
    fig.update_layout(showlegend=False)
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
    pname = parameter_dict["name"]
    pmin = np.float64(parameter_dict["value_range"][0])
    pmax = np.float64(parameter_dict["value_range"][1])
    pval = np.float64(parameter_dict["default"])
    exec(f"state.parameter_{pname} = {pval}")
    parameters_name.append(pname)
    parameters_min.append(pmin)
    parameters_max.append(pmax)
    parameters_value.append(pval)
parameters_num = len(parameters_name)

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
    fig = plot(
        parameters,
        parameters_name,
        parameters_min,
        parameters_max,
        objectives_name,
        **kwargs,
    )
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
                            with vuetify.VCard(style="width: 500px"):
                                with vuetify.VCardTitle("Parameters"):
                                    with vuetify.VCardText():
                                        for i in range(parameters_num):
                                            pname = parameters_name[i]
                                            pmin = parameters_min[i]
                                            pmax = parameters_max[i]
                                            pstep = (pmax - pmin) / 20.
                                            # create slider for each parameter
                                            with vuetify.VSlider(
                                                v_model=(f"parameter_{pname}",),
                                                label=f"{pname}",
                                                min=np.float64(pmin),
                                                max=np.float64(pmax),
                                                step=np.float64(pstep),
                                                classes="align-center",
                                                hide_details=True,
                                            ):
                                                # append text field
                                                with vuetify.Template(
                                                    v_slot_append=True,
                                                    __properties=[("v_slot_append", "v-slot:append")],
                                                ):
                                                    # TODO type="number"
                                                    vuetify.VTextField(
                                                        v_model=(f"parameter_{pname}",),
                                                        label=f"{pname}",
                                                        clearable=True,
                                                        density="compact",
                                                        hide_details=True,
                                                        single_line=True,
                                                        style="width: 100px",
                                                    )
                    with vuetify.VRow():
                        with vuetify.VCol():
                            with vuetify.VCard(style="width: 500px"):
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
