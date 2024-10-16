import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from trame.app import get_server
from trame.ui.vuetify2 import SinglePageLayout
from trame.widgets import plotly, vuetify2 as v2

from variables import read_variables

# Get a server to work with
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# normalize data in [0,1]
def normalize(x, xmin, xmax):
    y = (x - xmin) / (xmax - xmin)
    return y

# rescale data to physical range
def denormalize(y, xmin, xmax):
    x = xmin + (xmax - xmin) * y
    return x

# TODO generalize for different objectives
def model(parameters, **kwargs):
    pvals = np.array(list(parameters))
    result = float(np.sum(pvals))
    return result

def plot(
        parameters,
        parameters_min,
        parameters_max,
        objectives,
        **kwargs,
    ):
    # FIXME generalize for multiple objectives
    objective_name = list(objectives.keys())[0]
    # load experimental data
    df_exp = pd.read_csv("experimental_data.csv")
    df_sim = pd.read_csv("simulation_data.csv")
    df_cds = ["blue", "red"]
    # plot
    fig = make_subplots(rows=len(parameters), cols=1)
    for i, key in enumerate(parameters.keys()):
        # NOTE row count starts from 1, enumerate count starts from 0
        this_row = i+1
        this_col = 1
        #----------------------------------------------------------------------
        # figure trace from CSV data
        # set opacity map based on distance from current inputs
        # compute Euclidean distance
        for df_count, df in enumerate([df_exp]):
        #for df_count, df in enumerate([df_exp, df_sim]):
            df_copy = df.copy()
            df_copy["distance"] = 0.
            # loop over all inputs except the current one
            for subkey in [subkey for subkey in parameters.keys() if subkey != key]:
                pname_loc = subkey
                pval_loc = parameters[subkey]
                pmin_loc = parameters_min[subkey]
                pmax_loc = parameters_max[subkey]
                df_copy["distance"] += ((df_copy[f"{pname_loc}"] - pval_loc) / (pmax_loc - pmin_loc))**2
            df_copy["distance"] = np.sqrt(df_copy["distance"])
            # normalize distance in [0,1] and compute opacity
            df_copy["distance"] = df_copy["distance"] / df_copy["distance"].max()
            df_copy["opacity"] = 1. - df_copy["distance"]
            # scatter plot with opacity
            exp_fig = px.scatter(
                df_copy,
                x=key,
                y=f"{objective_name}",
                opacity=df_copy["opacity"],
                color_discrete_sequence=[df_cds[df_count]],
            )
            exp_trace = exp_fig["data"][0]
            fig.add_trace(exp_trace, row=this_row, col=this_col)
        #----------------------------------------------------------------------
        # figure trace from model data
        #x = np.linspace(start=pmin, stop=pmax, num=100)
        #y = model(x)
        #mod_trace = go.Scatter(x=x, y=y)
        #fig.add_trace(mod_trace, row=this_row, col=this_col)
        #----------------------------------------------------------------------
        # add reference input line
        fig.add_vline(x=parameters[key], line_dash="dash", row=this_row, col=this_col)
        #----------------------------------------------------------------------
        # figures style
        fig.update_xaxes(
            exponentformat="e",
            title_text=key,
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
# FIXME generalize for multiple objectives
assert len(output_variables) == 1, "number of objectives > 1 not supported"

# initialize parameters
state.parameters_norm = dict()
state.parameters_phys = dict()
state.parameters_phys_min = dict()
state.parameters_phys_max = dict()
for _, parameter_dict in input_variables.items():
    key = parameter_dict["name"]
    pmin = float(parameter_dict["value_range"][0])
    pmax = float(parameter_dict["value_range"][1])
    pval = float(parameter_dict["default"])
    state.parameters_phys[key] = pval
    state.parameters_phys_min[key] = pmin
    state.parameters_phys_max[key] = pmax
    state.parameters_norm[key] = normalize(pval, pmin, pmax)
parameters_num = len(state.parameters_phys)
# push again at flush time
state.dirty("parameters_norm")

# initialize objectives
state.objectives_phys = dict()
for _, objective_dict in output_variables.items():
    key = objective_dict["name"]
    state.objectives_phys[key] = model(state.parameters_phys.values())
state.dirty("objectives_phys")  # pushed again at flush time

@state.change("parameters_norm")
def update_state(parameters_norm, parameters_phys_min, parameters_phys_max, **kwargs):
    # update parameters in physical units
    for key, value in parameters_norm.items():
        state.parameters_phys[key] = denormalize(
            parameters_norm[key],
            parameters_phys_min[key],
            parameters_phys_max[key],
        )
    # push again at flush time
    state.dirty("parameters_phys")
    # update objectives
    for key in state.objectives_phys.keys():
        state.objectives_phys[key] = model(state.parameters_phys.values(), **kwargs)
    # push again at flush time
    state.dirty("objectives_phys")
    # update plots
    fig = plot(
        state.parameters_phys,
        state.parameters_phys_min,
        state.parameters_phys_max,
        state.objectives_phys,
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
        with v2.VContainer():
            with v2.VRow():
                with v2.VCol():
                    with v2.VRow():
                        with v2.VCol():
                            with v2.VCard(style="width: 500px"):
                                with v2.VCardTitle("Parameters"):
                                    with v2.VCardText():
                                        for key in state.parameters_norm.keys():
                                            # create slider for each parameter
                                            with v2.VSlider(
                                                v_model=(f"parameters_norm['{key}']",),
                                                change="flushState('parameters_norm')",
                                                label=key,
                                                min=0.,
                                                max=1.,
                                                step=0.01,
                                                classes="align-center",
                                                hide_details=True,
                                            ):
                                                # append text field
                                                with v2.Template(
                                                    v_slot_append=True,
                                                ):
                                                    v2.VTextField(
                                                        v_model=(f"parameters_phys['{key}']",),
                                                        #change="flushState('parameters_norm')",
                                                        label=key,
                                                        density="compact",
                                                        hide_details=True,
                                                        readonly=True,
                                                        single_line=True,
                                                        style="width: 100px",
                                                    )
                    with v2.VRow():
                        with v2.VCol():
                            with v2.VCard(style="width: 500px"):
                                with v2.VCardTitle("Objectives"):
                                    with v2.VCardText():
                                        for key in state.objectives_phys.keys():
                                            v2.VTextField(
                                                v_model=(f"objectives_phys['{key}']",),
                                                label=key,
                                                readonly=True,
                                            )
                with v2.VCol():
                    with v2.VCard():
                        with v2.VCardTitle("Experimental data"):
                            with v2.VContainer(style=f"height: {25*len(state.parameters_norm)}vh"):
                                plotly_figure = plotly.Figure(
                                        display_mode_bar="true", config={"responsive": True}
                                )
                                ctrl.plotly_figure_update = plotly_figure.update

# Main
if __name__ == "__main__":
    server.start()
