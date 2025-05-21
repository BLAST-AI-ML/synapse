import inspect
from io import StringIO
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

# global database variable
db = None

def read_variables(config_file):
    print("Reading configuration file...")
    # read configuration file
    with open(config_file) as f:
        config_str = f.read()
    # load configuration dictionary
    config_dict = yaml.safe_load(config_str)
    config_spec = config_dict[state.experiment]
    # dictionary of input variables (parameters)
    input_variables = config_spec["input_variables"]
    # dictionary of output variables (objectives)
    output_variables = config_spec["output_variables"]
    return (input_variables, output_variables)

def metadata_match(config_file, model_file):
    print("Checking model consistency...")
    match = False
    # read configuration file
    with open(config_file) as f:
        config_str = f.read()
    # load configuration dictionary
    config_dict = yaml.safe_load(config_str)
    # load configuration input variables list
    config_vars = [value["name"] for value in config_dict[state.experiment]["input_variables"].values()]
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
    match = (config_vars == model_vars)
    if not match:
        print(f"Input variables in configuration file and model file do not match")
    return match

def load_database():
    print("Loading database...")
    global db
    # load database
    db_defaults = {
        "host": "mongodb05.nersc.gov",
        "port": 27017,
        "name": "bella_sf",
        "auth": "bella_sf",
        "user": "bella_sf_admin",
    }
    # read database information from environment variables (if unset, use defaults)
    db_host = os.getenv("SF_DB_HOST", db_defaults["host"])
    db_port = int(os.getenv("SF_DB_PORT", db_defaults["port"]))
    db_name = os.getenv("SF_DB_NAME", db_defaults["name"])
    db_auth = os.getenv("SF_DB_AUTH_SOURCE", db_defaults["auth"])
    db_user = os.getenv("SF_DB_USER", db_defaults["user"])
    # read database experiment from environment variable (no default provided)
    db_collection = state.experiment
    # read database password from environment variable (no default provided)
    db_password = os.getenv("SF_DB_PASSWORD")
    if db_password is None:
        raise RuntimeError("Environment variable SF_DB_PASSWORD must be set!")
    # SSH forward?
    if db_host == "localhost" or db_host == "127.0.0.1":
        direct_connection = True
    else:
        direct_connection = False
    # get database instance
    if db is None:
        print(f"Connecting to database {db_name}@{db_host}:{db_port}...")
        db = pymongo.MongoClient(
            host=db_host,
            port=db_port,
            username=db_user,
            password=db_password,
            authSource=db_auth,
            directConnection=direct_connection,
        )[db_name]
    # get collection: ip2, acave, config, ...
    collection = db[db_collection]
    if "config" not in db.list_collection_names():
        db.create_collection("config")
    config = db["config"]
    # retrieve all documents
    documents = list(collection.find())
    # separate documents: experimental and simulation
    exp_docs = [doc for doc in documents if doc["experiment_flag"] == 1]
    sim_docs = [doc for doc in documents if doc["experiment_flag"] == 0]
    return (config, exp_docs, sim_docs)

# plot experimental, simulation, and ML data
def plot(model):
    print("Plotting...")
    # local aliases
    parameters = state.parameters
    parameters_min = state.parameters_min
    parameters_max = state.parameters_max
    objectives = state.objectives
    try:
        # FIXME generalize for multiple objectives
        objective_name = list(objectives.keys())[0]
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        objective_name = ""
    # load experimental data
    df_exp = pd.read_json(StringIO(state.exp_data))
    df_sim = pd.read_json(StringIO(state.sim_data))
    df_cds = ["blue", "red"]
    df_leg = ["Experiment", "Simulation"]
    # plot
    fig = make_subplots(rows=len(parameters), cols=1)
    parameters_key_order_list = list(parameters.keys())
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
            df_copy["opacity"] = np.where(df_copy["distance"] > state.opacity, 0., 1. - df_copy["distance"]/state.opacity)
            # filter out data with zero opacity
            df_copy_filtered = df_copy[df_copy["opacity"] != 0.0]
            # scatter plot with opacity
            exp_fig = px.scatter(
                df_copy_filtered,
                x=key,
                y=objective_name,
                opacity=df_copy_filtered["opacity"],
                color_discrete_sequence=[df_cds[df_count]],
            )
            # do now show default legend affected by opacity map
            exp_fig["data"][0]["showlegend"] = False
            # create custom legend empty trace (i==0 only, avoid repetition)
            if i==0:
                legend = go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
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
        #----------------------------------------------------------------------
        # figure trace from model data
        if model.avail():
            input_dict_loc = dict()
            steps = 1000
            input_dict_loc[key] = torch.linspace(
                start=parameters_min[key],
                end=parameters_max[key],
                steps=steps,
            )
            for subkey in [subkey for subkey in parameters.keys() if subkey != key]:
                input_dict_loc[subkey] = parameters[subkey] * torch.ones(steps)
            # reorder the dictionary with respect to the parameters_key_order_list
            # TODO currently needed for GP model, see if this can be worked around
            ordered_input_dict_loc = {k: input_dict_loc[k] for k in parameters_key_order_list if k in input_dict_loc}
            # get mean and lower/upper bounds for uncertainty prediction
            # (when lower/upper bounds are not predicted by the model,
            # their values are set to zero to collapse the error range)
            mean, lower, upper = model.evaluate(ordered_input_dict_loc)
            # upper bound
            upper_bound = go.Scatter(
                x=input_dict_loc[key],
                y=upper,
                line=dict(color='orange', width=0.3),
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
                fill='tonexty',  # fill area between this trace and the next one
                fillcolor='rgba(255,165,0,0.25)',  # orange with alpha
                line=dict(color='orange', width=0.3),
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
        fig.update_xaxes(
            exponentformat="e",
            title_text=key,
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
