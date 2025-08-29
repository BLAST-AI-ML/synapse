import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymongo
import torch
import yaml
from state_manager import state
from error_manager import add_error


def load_config_file():
    print("Reading configuration file...")
    # find configuration file in the local file system
    config_dir = os.path.join(os.getcwd(), "config")
    config_file = os.path.join(config_dir, "variables.yml")
    if not os.path.isfile(config_file):
        raise ValueError(f"Configuration file {config_file} not found")
    return config_file


def load_config_dict():
    print("Loading configuration dictionary...")
    config_file = load_config_file()
    # read configuration file
    with open(config_file) as f:
        config_str = f.read()
    # load configuration dictionary
    config_dict = yaml.safe_load(config_str)
    return config_dict


def load_experiments():
    print("Reading experiments from configuration file...")
    # load configuration dictionary
    config_dict = load_config_dict()
    # read list of available experiments from higher-level keys
    experiment_list = list(config_dict.keys())
    return experiment_list


def load_variables():
    print("Reading input/output variables from configuration file...")
    # load configuration dictionary
    config_dict = load_config_dict()
    config_spec = config_dict[state.experiment]
    # dictionary of input variables (parameters)
    input_variables = config_spec["input_variables"]
    # dictionary of output variables (objectives)
    output_variables = config_spec["output_variables"]
    # dictionary of calibration variables
    if "simulation_calibration" in config_spec:
        simulation_calibration = config_spec["simulation_calibration"]
    else:
        simulation_calibration = {}
    return (input_variables, output_variables, simulation_calibration)


def load_data(db):
    print("Loading data from database...")
    # load experiment and simulation data points in dataframes
    exp_data = pd.DataFrame(db[state.experiment].find({"experiment_flag": 1}))
    sim_data = pd.DataFrame(db[state.experiment].find({"experiment_flag": 0}))
    # Make sure that the _id is stored as a string (important for interactivity in plotly)
    if "_id" in exp_data.columns:
        exp_data["_id"] = exp_data["_id"].astype(str)
    if "_id" in sim_data.columns:
        sim_data["_id"] = sim_data["_id"].astype(str)
    return (exp_data, sim_data)


def metadata_match(config_file, model_file):
    print("Checking model consistency...")
    match = False
    # read configuration file
    with open(config_file) as f:
        config_str = f.read()
    # load configuration dictionary
    config_dict = yaml.safe_load(config_str)
    # load configuration input variables list
    config_vars = [
        value["name"]
        for value in config_dict[state.experiment]["input_variables"].values()
    ]
    config_vars.sort()
    # read model file
    with open(model_file) as f:
        model_str = f.read()
    # load model dictionary
    model_dict = yaml.safe_load(model_str)
    # load model input variables list
    model_vars = list(model_dict["input_variables"].keys())
    model_vars.sort()
    # check if configuration list and model list match
    match = config_vars == model_vars
    if not match:
        print("Input variables in configuration file and model file do not match")
    return match


def load_database():
    print("Loading database...")
    # load database
    db_defaults = {
        "host": "mongodb05.nersc.gov",
        "port": 27017,
        "name": "bella_sf",
        "auth": "bella_sf",
        "user": "bella_sf_ro",
    }
    # read database information from environment variables (if unset, use defaults)
    db_host = os.getenv("SF_DB_HOST", db_defaults["host"])
    db_port = int(os.getenv("SF_DB_PORT", db_defaults["port"]))
    db_name = os.getenv("SF_DB_NAME", db_defaults["name"])
    db_auth = os.getenv("SF_DB_AUTH_SOURCE", db_defaults["auth"])
    db_user = os.getenv("SF_DB_USER", db_defaults["user"])
    # read database password from environment variable (no default provided)
    db_password = os.getenv("SF_DB_READONLY_PASSWORD")
    if db_password is None:
        raise RuntimeError("Environment variable SF_DB_READONLY_PASSWORD must be set!")
    # SSH forward?
    if db_host == "localhost" or db_host == "127.0.0.1":
        direct_connection = True
    else:
        direct_connection = False
    # get database instance
    print(f"Connecting to database {db_name}@{db_host}:{db_port}...")
    db = pymongo.MongoClient(
        host=db_host,
        port=db_port,
        username=db_user,
        password=db_password,
        authSource=db_auth,
        directConnection=direct_connection,
    )[db_name]
    return db


# plot experimental, simulation, and ML data
def plot(exp_data, sim_data, model_manager, cal_manager):
    print("Plotting...")
    # convert simulation data to experimental data
    cal_manager.convert_sim_to_exp(sim_data)
    # local aliases
    parameters = state.parameters
    parameters_min = state.parameters_min
    parameters_max = state.parameters_max
    parameters_show_all = state.parameters_show_all
    try:
        objective_name = state.displayed_output
    except Exception as e:
        title = "Unable to find objective to plot"
        msg = f"Error occurred when searching for objective to plot: {e}"
        add_error(title, msg)
        print(msg)
        objective_name = ""
    # set auxiliary properties
    df_cds = ["blue", "red"]
    df_leg = ["Experiment", "Simulation"]
    # plot
    fig = make_subplots(rows=len(parameters), cols=1)
    global_ymin = float("inf")
    global_ymax = float("-inf")
    for i, key in enumerate(parameters.keys()):
        # NOTE row count starts from 1, enumerate count starts from 0
        this_row = i + 1
        this_col = 1
        # ----------------------------------------------------------------------
        # figure trace from CSV data
        # set opacity map based on distance from current inputs
        # compute Euclidean distance
        for df_count, df in enumerate([exp_data, sim_data]):
            df_copy = df.copy()
            # some data sets do not include all parameters
            # (e.g., simulation data set does not include GVD)
            if key not in df_copy.columns:
                continue
            df_copy["distance"] = 0.0
            # loop over all inputs except the current one
            for subkey in [
                subkey
                for subkey in parameters.keys()
                if (subkey != key and subkey in df_copy.columns)
            ]:
                pname_loc = subkey
                pval_loc = parameters[subkey]
                pmin_loc = parameters_min[subkey]
                pmax_loc = parameters_max[subkey]
                df_copy["distance"] += (
                    (df_copy[f"{pname_loc}"] - pval_loc) / (pmax_loc - pmin_loc)
                ) ** 2
            df_copy["distance"] = np.sqrt(df_copy["distance"])
            # normalize distance in [0,1] and compute opacity
            df_copy["distance"] = df_copy["distance"]
            df_copy["opacity"] = np.where(
                df_copy["distance"] > state.opacity,
                0.0,
                1.0 - df_copy["distance"] / state.opacity,
            )
            # filter out data with zero opacity
            df_copy_filtered = df_copy[df_copy["opacity"] != 0.0]

            if not df_copy_filtered.empty:
                y_vals = df_copy_filtered[objective_name].values
                global_ymin = min(global_ymin, y_vals.min())
                global_ymax = max(global_ymax, y_vals.max())

            # Determine which data is shown when hovering over the plot
            hover_data = list(state.parameters.keys()) + state.output_variables
            if df_leg[df_count] == "Experiment":
                hover_data += [ name for name in ["date", "scan_number", "shot_number"] if name in df_copy_filtered.columns ]
            elif df_leg[df_count] == "Simulation":
                hover_data += [v["name"] for v in cal_manager.simulation_calibration.values()]
            hover_data.sort()


            # scatter plot with opacity
            exp_fig = px.scatter(
                df_copy_filtered,
                x=key,
                y=objective_name,
                opacity=df_copy_filtered["opacity"],
                color_discrete_sequence=[df_cds[df_count]],
                hover_data=hover_data,
                custom_data="_id",
            )
            # do now show default legend affected by opacity map
            exp_fig["data"][0]["showlegend"] = False
            # create custom legend empty trace (i==0 only, avoid repetition)
            if i == 0:
                legend = go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=df_cds[df_count], opacity=1),
                    showlegend=True,
                    name=df_leg[df_count],
                )
                # add custom legend trace to display custom legend
                fig.add_trace(legend)
            # add original trace (with correct opacity)
            exp_trace = exp_fig["data"][0]
            fig.add_trace(
                exp_trace,
                row=this_row,
                col=this_col,
            )
        # ----------------------------------------------------------------------
        # figure trace from model data
        if model_manager.avail():
            input_dict_loc = dict()
            steps = 1000
            input_dict_loc[key] = torch.linspace(
                start=parameters_min[key],
                end=parameters_max[key],
                steps=steps,
            )
            for subkey in [subkey for subkey in parameters.keys() if subkey != key]:
                input_dict_loc[subkey] = parameters[subkey] * torch.ones(steps)
            # get mean and lower/upper bounds for uncertainty prediction
            # (when lower/upper bounds are not predicted by the model,
            # their values are set to zero to collapse the error range)
            mean, lower, upper = model_manager.evaluate(
                input_dict_loc, state.displayed_output
            )

            global_ymin = min(global_ymin, lower.numpy().min())
            global_ymax = max(global_ymax, upper.numpy().max())

            # upper bound
            upper_bound = go.Scatter(
                x=input_dict_loc[key],
                y=upper,
                line=dict(color="orange", width=0.3),
                showlegend=False,
                hoverinfo="skip",
            )
            fig.add_trace(
                upper_bound,
                row=this_row,
                col=this_col,
            )
            # lower bound
            lower_bound = go.Scatter(
                x=input_dict_loc[key],
                y=lower,
                fill="tonexty",  # fill area between this trace and the next one
                fillcolor="rgba(255,165,0,0.25)",  # orange with alpha
                line=dict(color="orange", width=0.3),
                showlegend=False,
                hoverinfo="skip",
            )
            fig.add_trace(
                lower_bound,
                row=this_row,
                col=this_col,
            )
            # scatter plot
            mod_trace = go.Scatter(
                x=input_dict_loc[key],
                y=mean,
                line=dict(color="orange"),
                name="ML Model",
                showlegend=(True if i == 0 else False),
            )
            # add trace
            fig.add_trace(
                mod_trace,
                row=this_row,
                col=this_col,
            )
        # ----------------------------------------------------------------------
        # add reference input line
        fig.add_vline(
            x=parameters[key],
            line_dash="dash",
            row=this_row,
            col=this_col,
        )
        # ----------------------------------------------------------------------
        # figures style
        if parameters_show_all[key]:
            fig.update_xaxes(
                exponentformat="e",
                title_text=key,
                row=this_row,
                col=this_col,
            )
        else:
            fig.update_xaxes(
                range=(parameters_min[key], parameters_max[key]),
                exponentformat="e",
                title_text=key,
                row=this_row,
                col=this_col,
            )

    # A bit of padding on either end of the y range so we can see all the data.
    padding = 0.05 * (global_ymax - global_ymin)
    for i, key in enumerate(parameters.keys()):
        this_row = i + 1
        this_col = 1
        fig.update_yaxes(
            range=(global_ymin - padding, global_ymax + padding),
            exponentformat="e",
            title_text=objective_name,
            row=this_row,
            col=this_col,
        )

    fig.update_layout(clickmode="event")
    return fig
