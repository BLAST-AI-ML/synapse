import asyncio
from datetime import datetime
import os
import re
import shlex
from pathlib import Path
import tempfile

import mlflow
import yaml
from sfapi_client import AsyncClient
from sfapi_client.compute import Machine
from trame.widgets import vuetify3 as vuetify
from utils import timer, load_config_dict, create_date_filter
from error_manager import add_error
from sfapi_manager import monitor_sfapi_job
from iriapi_manager import create_iriapi_client, monitor_iriapi_job
from state_manager import state

model_type_dict = {
    "Gaussian Process": "GP",
    "Neural Network (single)": "NN",
    "Neural Network (ensemble)": "ensemble_NN",
}


SBATCH_SUBMIT_OPTION_MAP = {
    "-J": "name",
    "--job-name": "name",
    "-q": "queue",
    "--qos": "queue",
    "-A": "account",
    "--account": "account",
    "-t": "duration",
    "--time": "duration",
    "-N": "nodes",
    "--nodes": "nodes",
    "-o": "stdout_path",
    "--output": "stdout_path",
    "-e": "stderr_path",
    "--error": "stderr_path",
    "-C": "constraint",
    "--constraint": "constraint",
    "--gpus-per-node": "gpus-per-node",
    "--ntasks-per-node": "ntasks-per-node",
}
SBATCH_REQUIRED_SUBMIT_OPTIONS = set(SBATCH_SUBMIT_OPTION_MAP.values())
IRI_TRAINING_LAUNCH_PREFIX = ("srun", "podman-hpc", "run")  # Container launch marker.
# Match model=$1 or model=${1}, with optional comment.
SCRIPT_MODEL_ARGUMENT_RE = re.compile(r"^model=\$\{?1\}?\s*(#.*)?$")
# Capture shell assignments like REGISTRY_NAME=value, ignoring trailing comments.
SHELL_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+?)(?:\s+#.*)?$")


def find_training_script_path():
    # Multiple locations supported, to make development easier:
    #   container (production): script is in cwd
    #   development, starting the gui app from dashboard/: script is in ../ml/
    #   development, starting the gui app from the repo root dir: script is in ml/
    script_locations = [Path.cwd(), Path.cwd() / "../ml", Path.cwd() / "ml"]
    for script_dir in script_locations:
        script_path = script_dir / "training_pm.sbatch"
        if script_path.exists():
            return script_path
    raise RuntimeError("Could not find training_pm.sbatch")


def parse_slurm_duration(duration):
    # Normalize Slurm duration variants to the seconds expected by IRI.
    days = 0
    time_part = duration.strip()
    has_days = "-" in time_part

    if has_days:
        day_part, time_part = time_part.split("-", 1)
        days = int(day_part)

    time_values = [int(v) for v in time_part.split(":")]
    n = len(time_values)

    # Slurm treats ambiguous two-part durations differently with a day prefix.
    if n == 3:
        hours, minutes, secs = time_values
    elif n == 2:
        hours, minutes, secs = (
            (time_values[0], time_values[1], 0)
            if has_days
            else (0, time_values[0], time_values[1])
        )
    elif n == 1:
        hours, minutes, secs = (
            (time_values[0], 0, 0) if has_days else (0, time_values[0], 0)
        )
    else:
        raise ValueError(f"Unsupported Slurm duration format: {duration}")

    return days * 86400 + hours * 3600 + minutes * 60 + secs


def parse_sbatch_submit_options(script_path):
    # Extract only SBATCH fields that map cleanly onto IRI submit options.
    submit_options = {}
    with open(script_path) as script_file:
        for line in script_file:
            line = line.strip()
            if not line.startswith("#SBATCH"):
                continue

            directive = line.removeprefix("#SBATCH").strip()
            if not directive:
                continue

            # Use shell parsing so quoted SBATCH values are handled correctly.
            tokens = shlex.split(directive, comments=True)
            if not tokens:
                continue

            option = tokens[0]
            value = None
            if "=" in option:
                option, value = option.split("=", 1)
            elif len(tokens) > 1:
                value = tokens[1]

            submit_option = SBATCH_SUBMIT_OPTION_MAP.get(option)
            if submit_option and value is not None:
                submit_options[submit_option] = value

    missing_options = [
        option
        for option in SBATCH_REQUIRED_SUBMIT_OPTIONS
        if option not in submit_options
    ]
    if missing_options:
        missing_list = ", ".join(sorted(missing_options))
        raise ValueError(f"Missing required SBATCH option(s): {missing_list}")

    # Convert values to the types expected by the IRI submit endpoint.
    submit_options["duration"] = parse_slurm_duration(submit_options["duration"])
    submit_options["nodes"] = int(submit_options["nodes"])
    return submit_options


def collapse_shell_command(lines):
    # Preserve trailing-backslash continuations before tokenizing with shlex.
    command_parts = []
    current_command = ""
    for line in lines:
        stripped = line.strip()
        if stripped.endswith("\\"):
            current_command += f"{stripped[:-1]} "
        else:
            command_parts.append(f"{current_command}{stripped}".strip())
            current_command = ""

    if current_command:
        command_parts.append(current_command.strip())

    return " ".join(command_parts)


def parse_shell_variable_assignments(lines):
    # Keep static shell assignments that can be expanded without execution.
    variables = {}
    for line in lines:
        match = SHELL_ASSIGNMENT_RE.match(line.strip())
        if not match:
            continue

        name, raw_value = match.groups()
        if "$(" in raw_value or "`" in raw_value:
            continue

        values = shlex.split(raw_value, comments=True)
        if len(values) == 1:
            variables[name] = values[0]

    return variables


def expand_iri_launch_argument(argument, model_type, variables):
    # Expand the simple $name and ${name} forms used by training_pm.sbatch.
    replacements = {**variables, "model": model_type}
    for name, value in replacements.items():
        argument = argument.replace(f"${{{name}}}", value)
        argument = argument.replace(f"${name}", value)
    return argument


def parse_iri_training_launch_spec(script_path, model_type):
    # Split the batch script into setup lines and the podman-hpc launch command.
    pre_launch_lines = []
    launch_lines = []
    found_launch = False

    with open(script_path) as script_file:
        for raw_line in script_file:
            line = raw_line.rstrip()
            stripped = line.strip()

            if found_launch:
                launch_lines.append(line)
            elif stripped.startswith(" ".join(IRI_TRAINING_LAUNCH_PREFIX)):
                found_launch = True
                launch_lines.append(line)
            elif SCRIPT_MODEL_ARGUMENT_RE.match(stripped):
                pre_launch_lines.append(f"model={shlex.quote(model_type)}")
            elif (
                stripped
                and not stripped.startswith("#!")
                and not stripped.startswith("#SBATCH")
            ):
                pre_launch_lines.append(line)

    if not launch_lines:
        raise ValueError("Could not find the IRI training launch command")

    launch_command = collapse_shell_command(launch_lines)
    launch_tokens = shlex.split(launch_command)
    variables = parse_shell_variable_assignments(pre_launch_lines)
    # IRI submit expects launcher, executable, and arguments separately.
    launch_prefix_length = len(IRI_TRAINING_LAUNCH_PREFIX)
    launch_prefix = launch_tokens[:launch_prefix_length]
    launcher, *executable_tokens = launch_prefix
    executable = " ".join(executable_tokens)
    arguments = [
        expand_iri_launch_argument(argument, model_type, variables)
        for argument in launch_tokens[launch_prefix_length:]
    ]

    return {
        "arguments": arguments,
        "executable": executable,
        "launcher": launcher,
        "pre_launch": "\n".join(pre_launch_lines),
    }


def enable_amsc_x_api_key(config_dict):
    """
    MLflow authentication helper for the AmSC MLflow server.
    Standard MLflow does not automatically inject custom headers like 'X-Api-Key'.
    This patches the http_request function to ensure every request to the server
    includes the AmSC API key.

    See https://gitlab.com/amsc2/ai-services/model-services/intro-to-mlflow-pytorch for more details.
    """
    import mlflow.utils.rest_utils as rest_utils

    mlflow_cfg = config_dict.get("mlflow") or {}
    api_key_env = mlflow_cfg.get("api_key_env")
    if not api_key_env:
        title = "Unable to enable AmSC X-Api-Key authentication"
        msg = "MLFlow configuration is missing 'mlflow.api_key_env'"
        add_error(title, msg)
        print(msg)
        return

    api_key = os.environ.get(api_key_env)
    if not api_key:
        title = "Unable to enable AmSC X-Api-Key authentication"
        msg = f"Environment variable '{api_key_env}' in 'mlflow.api_key_env' is not set"
        add_error(title, msg)
        print(msg)
        return
    _orig = rest_utils.http_request

    def patched(host_creds, endpoint, method, *args, **kwargs):
        if "headers" in kwargs and kwargs["headers"] is not None:
            h = dict(kwargs["headers"])
            h["X-Api-Key"] = api_key
            kwargs["headers"] = h
        else:
            h = dict(kwargs.get("extra_headers") or {})
            h["X-Api-Key"] = api_key
            kwargs["extra_headers"] = h
        return _orig(host_creds, endpoint, method, *args, **kwargs)

    rest_utils.http_request = patched


class ModelManager:
    def __init__(self, config_dict, model_type):
        print("Initializing model manager...")
        self.__model = None
        self.__model_type = model_type

        if "mlflow" not in config_dict or not config_dict["mlflow"].get("tracking_uri"):
            print(
                f"No mlflow.tracking_uri in configuration file for {config_dict['experiment']}; cannot load model from MLflow."
            )
            return

        mlflow.set_tracking_uri(config_dict["mlflow"]["tracking_uri"])
        # When using the AmSC MLflow: inject the X-Api-Key into the requests to authenticate with the MLflow server
        # (See https://gitlab.com/amsc2/ai-services/model-services/intro-to-mlflow-pytorch)
        if (
            config_dict["mlflow"]["tracking_uri"]
            == "https://mlflow.american-science-cloud.org"
        ):
            enable_amsc_x_api_key(config_dict)

        experiment = config_dict["experiment"]
        model_name = f"synapse-{experiment}_{model_type}"

        try:
            # Download model from MLflow server
            self.__model = (
                mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
                .unwrap_python_model()
                .model
            )
            if model_type not in ("NN", "ensemble_NN", "GP"):
                raise ValueError(f"Unsupported model type: {model_type}")
            # Populate inferred calibration in physics units for GUI
            # (only meaningful inside the dashboard where state.simulation_calibration is set)
            if state.simulation_calibration is not None:
                self.populate_inferred_calibration(
                    config_dict["inputs"], config_dict["outputs"]
                )
        except Exception as e:
            title = f"Unable to load model {model_type}"
            msg = f"Error occurred when loading model from MLflow: {e}"
            add_error(title, msg)
            print(msg)

    def avail(self):
        print("Checking model availability...")
        model_avail = True if self.__model is not None else False
        return model_avail

    @timer
    def evaluate(self, parameters, output):
        print("Evaluating model...")
        if self.__model is not None:
            # evaluate model
            output_dict = self.__model.evaluate(parameters)
            if self.__model_type == "NN":
                # compute mean and mean error
                mean = output_dict[output]
                mean_error = 0.0  # trick to collapse error range when lower/upper bounds are not predicted
            elif self.__model_type in ("GP", "ensemble_NN"):
                # compute mean, standard deviation and mean error
                # (call detach method to detach gradients from tensors)
                mean = output_dict[output].mean.detach()
                std_dev = output_dict[output].variance.sqrt().detach()
                mean_error = 2.0 * std_dev
            else:
                raise ValueError(f"Unsupported model type: {self.__model_type}")
            # compute lower/upper bounds for error range
            lower = mean - mean_error
            upper = mean + mean_error
            # convert to Python float if tensor has only one element
            # because Trame state variables must be serializable
            if mean.numel() == 1:
                mean = float(mean)
            return (mean, lower, upper)

    def populate_inferred_calibration(self, input_variables, output_variables):
        """
        Populate alpha_inferred/beta_inferred in state.simulation_calibration for
        both input and output calibration entries.
        """
        # Clear stale inferred values
        for value in state.simulation_calibration.values():
            value.pop("alpha_inferred", None)
            value.pop("beta_inferred", None)

        # Input calibration
        # For ensemble_NN, transformers live on each inner TorchModel (not on NNEnsemble itself)
        if self.__model_type == "ensemble_NN":
            input_transformers = self.__model.models[0].input_transformers
        else:
            input_transformers = self.__model.input_transformers
        assert len(input_transformers) == 2, (
            f"Expected exactly 2 input transformers (calibration + normalization), "
            f"but got {len(input_transformers)}."
        )
        input_inferred_calibration = input_transformers[0]
        alpha_inferred = 1.0 / input_inferred_calibration.coefficient
        beta_inferred = input_inferred_calibration.offset
        for i, key in enumerate(input_variables.keys()):
            state.simulation_calibration[key]["alpha_inferred"] = float(
                alpha_inferred[i]
            )
            state.simulation_calibration[key]["beta_inferred"] = float(beta_inferred[i])

        # Output calibration
        # For ensemble_NN, transformers live on each inner TorchModel (not on NNEnsemble itself)
        if self.__model_type == "ensemble_NN":
            output_transformers = self.__model.models[0].output_transformers
        else:
            output_transformers = self.__model.output_transformers
        assert len(output_transformers) == 2, (
            f"Expected exactly 2 output transformers (normalization + calibration), "
            f"but got {len(output_transformers)}."
        )
        output_inferred_calibration = output_transformers[-1]
        alpha_inferred = 1.0 / output_inferred_calibration.coefficient
        beta_inferred = output_inferred_calibration.offset
        for i, key in enumerate(output_variables.keys()):
            state.simulation_calibration[key]["alpha_inferred"] = float(
                alpha_inferred[i]
            )
            state.simulation_calibration[key]["beta_inferred"] = float(beta_inferred[i])
        # Notify Trame that the dict was modified in-place, so the UI updates
        state.dirty("simulation_calibration")

    def _prepare_training_config(self, temp_dir):
        """Prepare a training configuration file in the given temporary directory,
        updated with information from the dashboard.

        Returns the path to the written configuration file.
        """
        config_dict = load_config_dict(state.experiment)
        config_dict["simulation_calibration"] = state.simulation_calibration
        date_filter = create_date_filter(state.experiment_date_range)
        config_dict["date_filter"] = date_filter
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)
        return config_path

    async def _training_kernel_sfapi(self):
        try:
            # create an authenticated client
            async with AsyncClient(
                client_id=state.sfapi_client_id, secret=state.sfapi_key
            ) as client:
                perlmutter = await client.compute(Machine.perlmutter)
                # upload the configuration file to NERSC
                with tempfile.TemporaryDirectory() as temp_dir:
                    config_path = self._prepare_training_config(temp_dir)
                    # define the target path on NERSC
                    target_path = "/global/cfs/cdirs/m558/superfacility/model_training"
                    [target_path] = await perlmutter.ls(target_path, directory=True)
                    with open(config_path, "rb") as temp_file:
                        print("Uploading configuration file to NERSC...")
                        temp_file.filename = "config.yaml"
                        await target_path.upload(temp_file)

                # set the path of the script used to submit the training job on NERSC
                training_script_path = find_training_script_path()
                with open(training_script_path) as file:
                    training_script = file.read()

                # replace the --model argument in the python command with the current model type from the state
                training_script = re.sub(
                    pattern=r"--model \$\{model\}",
                    repl=rf"--model {model_type_dict[state.model_type_verbose]}",
                    string=training_script,
                )

                # submit the training job through the Superfacility API
                sfapi_job = await perlmutter.submit_job(training_script)
                state.model_training_status = "Submitted"
                state.flush()
                # print some logs
                print(f"Training job submitted (job ID: {sfapi_job.jobid})")
                return await monitor_sfapi_job(sfapi_job, "model_training_status")
        except Exception as e:
            title = "Unable to complete remote training"
            msg = f"Error occurred when executing remote training: {e}"
            add_error(title, msg)
            print(msg)
            state.model_training_status = "Failed"
            state.flush()
            return False

    async def _training_kernel_iriapi(self):
        try:
            # Create an authenticated client
            client = create_iriapi_client()
            # Connect to NERSC
            nersc = await asyncio.to_thread(client.facility, "nersc")
            # Get the compute resource (Perlmutter)
            perlmutter = await asyncio.to_thread(nersc.resource, "compute")
            # Get the CFS resource for uploading files shared with compute jobs
            cfs = await asyncio.to_thread(nersc.resource, "cfs")
            target_dir = "/global/cfs/cdirs/m558/superfacility/model_training"
            training_script_path = find_training_script_path()
            model_type = model_type_dict[state.model_type_verbose]
            # Reuse SBATCH directives so IRI submissions match the batch script.
            submit_options = parse_sbatch_submit_options(training_script_path)
            launch_spec = parse_iri_training_launch_spec(
                training_script_path, model_type
            )

            # Training script is parsed locally; only config.yaml is uploaded.
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = self._prepare_training_config(temp_dir)
                ls_task = await asyncio.to_thread(cfs.fs.ls, target_dir)
                await asyncio.to_thread(ls_task.wait)
                [target_path] = ls_task.result
                with open(config_path, "rb") as temp_file:
                    print("Uploading configuration file to NERSC...")
                    temp_file.filename = "config.yaml"
                    # FIXME: Verify the IRI upload API keeps this file-object contract
                    await target_path.upload(temp_file)

            # Submit the training job through the AmSC IRI API
            iriapi_job = await asyncio.to_thread(
                perlmutter.submit,
                directory=target_dir,
                **submit_options,
                **launch_spec,
            )
            state.model_training_status = "Submitted"
            state.flush()
            # Print some logs
            print(f"Training job submitted (job ID: {iriapi_job.id})")
            return await monitor_iriapi_job(iriapi_job, "model_training_status")

        except Exception as e:
            title = "Unable to complete remote training"
            msg = f"Error occurred when executing remote training: {e}"
            add_error(title, msg)
            print(msg)
            state.model_training_status = "Failed"
            state.flush()
            return False

    async def _training_kernel_local(self):
        try:
            ml_dir = (Path(__file__).parent / "../ml").resolve()
            train_model_path = ml_dir / "train_model.py"
            model_type = model_type_dict[state.model_type_verbose]

            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = self._prepare_training_config(temp_dir)
                state.model_training_status = "Running"
                state.flush()
                print(
                    f"Starting local training: {train_model_path} --model {model_type}"
                )
                proc = await asyncio.create_subprocess_exec(
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "synapse-ml",
                    "python",
                    str(train_model_path),
                    "--config_file",
                    str(config_path),
                    "--model",
                    model_type,
                    cwd=str(ml_dir),  # working directory of the subprocess
                    stdout=asyncio.subprocess.PIPE,  # capture the standard output into a pipe
                    stderr=asyncio.subprocess.STDOUT,  # redirect the standard error into the same pipe
                )
                # stream subprocess output to console
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    print(line.decode(), end="")
                await proc.wait()

            if proc.returncode == 0:
                state.model_training_status = "Completed"
                state.flush()
                return True
            else:
                state.model_training_status = "Failed"
                state.flush()
                return False
        except Exception as e:
            title = "Unable to complete local training"
            msg = f"Error occurred when executing local training: {e}"
            add_error(title, msg)
            print(msg)
            state.model_training_status = "Failed"
            state.flush()
            return False

    async def training_async(self):
        try:
            print("Training model...")
            state.model_training = True
            state.model_training_status = "Submitting"
            state.flush()
            if state.model_training_mode == "local":
                result = await self._training_kernel_local()
            elif state.model_training_mode == "sfapi":
                result = await self._training_kernel_sfapi()
            elif state.model_training_mode == "iriapi":
                result = await self._training_kernel_iriapi()
            else:
                raise ValueError(
                    f"Unsupported training mode: {state.model_training_mode}"
                )
            if result:
                state.model_training_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                state.flush()
                print(f"Finished training model at {state.model_training_time}")
            else:
                print("Unable to complete training job")
            # flush state and enable button
            state.model_training = False
            state.flush()
        except Exception as e:
            title = "Unable to train model"
            msg = f"Error occurred when training model: {e}"
            add_error(title, msg)
            print(msg)

    def training_trigger(self):
        try:
            # schedule asynchronous job
            asyncio.create_task(self.training_async())
        except Exception as e:
            title = "Unable to train model"
            msg = f"Error occurred when training model: {e}"
            add_error(title, msg)
            print(msg)

    def panel(self):
        print("Setting model card...")
        # list of available model types
        model_type_list = [
            "Gaussian Process",
            "Neural Network (single)",
            "Neural Network (ensemble)",
        ]
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_model", 0)):
            with vuetify.VExpansionPanel(
                title="Control: Models",
                style="font-size: 20px; font-weight: 500;",
            ):
                with vuetify.VExpansionPanelText():
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VSelect(
                                v_model=("model_type_verbose",),
                                label="Model type",
                                items=(model_type_list,),
                                dense=True,
                            )
                        with vuetify.VCol():
                            vuetify.VTextField(
                                v_model_number=("model_training_status",),
                                label="Training status",
                                readonly=True,
                            )
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VBtn(
                                "Train",
                                click=self.training_trigger,
                                disabled=(
                                    "model_training || (model_training_mode === 'sfapi' && sfapi_perlmutter_status !== 'active')",
                                ),
                                style="text-transform: none",
                            )
