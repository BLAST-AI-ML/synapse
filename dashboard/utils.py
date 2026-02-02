import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymongo
import time
import torch
import yaml
from trame.widgets import vuetify3 as vuetify
from state_manager import state, EXPERIMENTS_PATH
from error_manager import add_error


def timer(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Executed '{function.__qualname__}' from module '{function.__module__}' in {elapsed_time:.4f} seconds"
        )
        return result

    return wrapper


def load_config_file(experiment):
    print("Reading configuration file...")
    # find configuration file in the local file system
    config_file = EXPERIMENTS_PATH / f"synapse-{experiment}/config.yaml"
    if not config_file.is_file():
        raise ValueError(f"Configuration file {config_file} not found")
    return config_file


def load_config_dict(experiment):
    print("Loading configuration dictionary...")
    config_file = load_config_file(experiment)
    with open(config_file) as f:
        config_str = f.read()
    # load configuration dictionary
    config_dict = yaml.safe_load(config_str)
    return config_dict


def load_experiments():
    print("Reading experiments from experiments directory")
    return [
        d.name.removeprefix("synapse-")
        for d in EXPERIMENTS_PATH.iterdir()
        if d.is_dir()
    ]


def load_variables(experiment):
    print("Reading input/output variables from configuration file...")
    # load configuration dictionary
    config_dict = load_config_dict(experiment)
    # dictionary of input variables (parameters)
    input_variables = config_dict["inputs"]
    # dictionary of output variables (objectives)
    output_variables = config_dict["outputs"]
    # dictionary of calibration variables
    if "simulation_calibration" in config_dict:
        simulation_calibration = config_dict["simulation_calibration"]
    else:
        simulation_calibration = {}
    return (input_variables, output_variables, simulation_calibration)


@timer
def load_data(db):
    print("Loading data from database...")
    # build date filter if date range is set
    date_filter = {}
    if state.experiment_date_range:
        start_date = pd.to_datetime(state.experiment_date_range[0].to_datetime())
        start_date = start_date.to_pydatetime().replace(hour=0, minute=0, second=0)
        # VDateInput returns exclusive end date for date ranges, so we need to subtract 1 day
        end_date = pd.to_datetime(state.experiment_date_range[-1].to_datetime())
        end_date_correction = (
            pd.Timedelta(days=0)
            if len(state.experiment_date_range) == 1
            else pd.Timedelta(days=1)
        )
        end_date = end_date - end_date_correction
        end_date = end_date.to_pydatetime().replace(hour=23, minute=59, second=59)
        # remove timezone info to match naive datetime in database
        start_date = (
            start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
        )
        end_date = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
        date_filter = {
            "date": {
                "$gte": start_date,
                "$lte": end_date,
            }
        }
        print(f"Filtering data between {start_date.date()} and {end_date.date()}...")
    # load experiment and simulation data points in dataframes
    exp_data = pd.DataFrame(
        db[state.experiment].find({**{"experiment_flag": 1}, **date_filter})
    )
    sim_data = pd.DataFrame(db[state.experiment].find({"experiment_flag": 0}))
    # Store '_id', 'date' as string
    for key in ["_id", "date"]:
        if key in exp_data.columns:
            exp_data[key] = exp_data[key].astype(str)
        if key in sim_data.columns:
            sim_data[key] = sim_data[key].astype(str)
    return (exp_data, sim_data)


def verify_input_variables(model_file, experiment):
    print("Checking model consistency...")
    # read configuration file
    input_vars, _, _ = load_variables(experiment)
    config_vars = [input_var["name"] for input_var in input_vars.values()]
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


@timer
def load_database(experiment):
    print("Loading database...")
    # load configuration dictionary
    config_dict = load_config_dict(experiment)
    # read database information from configuration dictionary
    db_host = config_dict["database"]["host"]
    db_port = config_dict["database"]["port"]
    db_name = config_dict["database"]["name"]
    db_auth = config_dict["database"]["auth"]
    db_username = config_dict["database"]["username_ro"]
    db_password_env = config_dict["database"]["password_ro_env"]
    db_password = os.getenv(db_password_env)
    if db_password is None:
        raise RuntimeError(f"Environment variable {db_password_env} must be set!")
    # get database instance
    print(f"Connecting to database {db_name}@{db_host}:{db_port}...")
    db = pymongo.MongoClient(
        host=db_host,
        port=db_port,
        authSource=db_auth,
        username=db_username,
        password=db_password,
        directConnection=(db_host in ["localhost", "127.0.0.1"]),  # SSH forwarding
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

            # Helper to build a section of the hover tooltip
            def hover_section(title, cols, hover_data):
                if not cols:
                    return []
                section = [f"<br><b>{title}</b>"]
                for col in cols:
                    # For string/date columns, use no format specifier (displays as-is)
                    format = "" if col == "date" else ":.4g"
                    section.append(
                        f"{col}=%{{customdata[{hover_data.index(col)}]{format}}}"
                    )
                return section

            # Determine which data is shown when hovering over the plot
            hover_parameters = list(state.parameters.keys())
            hover_output_variables = state.output_variables
            hover_customdata = ["_id"] + hover_parameters + hover_output_variables

            hover_template_lines = hover_section(
                "Input variables", hover_parameters, hover_customdata
            )
            hover_template_lines += hover_section(
                "Output variables", hover_output_variables, hover_customdata
            )
            if df_leg[df_count] == "Experiment":
                hover_experiment = [
                    name
                    for name in ["date", "scan_number", "shot_number"]
                    if name in df_copy_filtered.columns
                ]
                hover_customdata += hover_experiment
                hover_template_lines += hover_section(
                    "Experiment", hover_experiment, hover_customdata
                )

            elif df_leg[df_count] == "Simulation":
                hover_simulation = [
                    v["name"] for v in state.simulation_calibration.values()
                ]
                hover_customdata += hover_simulation
                hover_template_lines += hover_section(
                    "Simulation", hover_simulation, hover_customdata
                )

            exp_fig = go.Figure(
                data=[
                    go.Scatter(
                        x=df_copy_filtered[key],
                        y=df_copy_filtered[objective_name],
                        mode="markers",
                        marker=dict(
                            color=df_cds[df_count], opacity=df_copy_filtered["opacity"]
                        ),
                    )
                ]
            )

            # Attach customdata:
            exp_fig.update_traces(customdata=df_copy_filtered[hover_customdata].values)
            hovertemplate = "<br>".join(hover_template_lines) + "<extra></extra>"
            # Apply hovertemplate
            exp_fig.update_traces(hovertemplate=hovertemplate)
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
        custom_range = (
            [None, None]
            if parameters_show_all[key]
            else [
                parameters_min[key],
                parameters_max[key],
            ]
        )
        fig.update_xaxes(
            range=custom_range,
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


def data_depth_panel():
    with vuetify.VExpansionPanels(v_model=("expand_panel_control_plots", 0)):
        with vuetify.VExpansionPanel(
            title="Control: Plots",
            style="font-size: 20px; font-weight: 500;",
        ):
            with vuetify.VExpansionPanelText():
                # create a row for the slider label
                with vuetify.VRow():
                    vuetify.VListSubheader(
                        "Projected Data Depth",
                        style="margin-top: 16px;",
                    )
                # create a row for the slider and text field
                with vuetify.VRow(no_gutters=True):
                    with vuetify.VSlider(
                        v_model_number=("opacity",),
                        change="flushState('opacity')",
                        hide_details=True,
                        max=1.0,
                        min=0.0,
                        step=0.025,
                        style="align-items: center;",
                    ):
                        with vuetify.Template(v_slot_append=True):
                            vuetify.VTextField(
                                v_model_number=("opacity",),
                                density="compact",
                                hide_details=True,
                                readonly=True,
                                single_line=True,
                                style="margin-top: 0px; padding-top: 0px; width: 80px;",
                                type="number",
                            )
