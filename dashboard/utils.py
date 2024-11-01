import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import yaml

def read_variables(yaml_file):
    # read YAML file
    with open(yaml_file) as f:
        yaml_str = f.read()
    # load YAML dictionary
    yaml_dict = yaml.safe_load(yaml_str)
    # dictionary of input variables (parameters)
    input_variables = yaml_dict["input_variables"]
    # dictionary of output variables (objectives)
    output_variables = yaml_dict["output_variables"]
    return (input_variables, output_variables)

# plot experimental, simulation, and ML data
def plot(
        parameters,
        parameters_min,
        parameters_max,
        objectives,
        model,
    ):
    # FIXME generalize for multiple objectives
    objective_name = list(objectives.keys())[0]
    # load experimental data
    df_exp = pd.read_csv("experimental_data.csv")
    df_sim = pd.read_csv("simulation_data.csv")
    df_cds = ["blue", "red"]
    df_leg = ["experiment", "simulation"]
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
        for df_count, df in enumerate([df_exp, df_sim]):
            df_copy = df.copy()
            # some data sets do not include all parameters
            # (e.g., simulation data set does not include GVD)
            if key not in df_copy.columns:
                continue
            df_copy["distance"] = 0.
            # loop over all inputs except the current one
            for subkey in [subkey for subkey in parameters.keys() if (subkey != key and subkey in df_copy.columns)]:
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
                y=objective_name,
                opacity=df_copy["opacity"],
                color_discrete_sequence=[df_cds[df_count]],
            )
            exp_fig["data"][0]["showlegend"] = (True if i==0 else False)  # do not repeat legend
            exp_fig["data"][0]["name"] = df_leg[df_count]
            exp_trace = exp_fig["data"][0]
            # add trace
            fig.add_trace(
                exp_trace,
                row=this_row,
                col=this_col,
            )
        #----------------------------------------------------------------------
        # figure trace from model data
        if "GVD" not in key:
            input_dict_loc = dict()
            steps = 1000
            input_dict_loc[key.split(maxsplit=1)[0]] = torch.linspace(
                start=parameters_min[key],
                end=parameters_max[key],
                steps=steps,
            )
            for subkey in [subkey for subkey in parameters.keys() if (subkey != key and "GVD" not in subkey)]:
                input_dict_loc[subkey.split(maxsplit=1)[0]] = parameters[subkey] * torch.ones(steps)
            y = model.evaluate(input_dict_loc)
            # scatter plot
            mod_trace = go.Scatter(
                x=input_dict_loc[key.split(maxsplit=1)[0]],
                y=y,
                line=dict(color="orange"),
                name="ML model",
                showlegend=(True if i==0 else False),
            )
            # add trace
            fig.add_trace(
                mod_trace,
                row=this_row,
                col=this_col,
            )
        #----------------------------------------------------------------------
        # add reference input line
        fig.add_vline(
            x=parameters[key],
            line_dash="dash",
            row=this_row,
            col=this_col,
        )
        #----------------------------------------------------------------------
        # figures style
        xlabel = key
        fig.update_xaxes(
            exponentformat="e",
            title_text=xlabel,
            row=this_row,
            col=this_col,
        )
        fig.update_yaxes(
            exponentformat="e",
            title_text=objective_name,
            row=this_row,
            col=this_col,
        )
    fig.update_layout()
    return fig
