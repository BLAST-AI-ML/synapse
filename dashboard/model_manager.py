import asyncio
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
import os
import re
import shlex
from pathlib import Path, PurePosixPath
import tempfile

import mlflow
import yaml
from amsc_client import Client
import mlflow.store.artifact.artifact_repo as mlflow_artifact_repo
import mlflow.store.artifact.cloud_artifact_repo as mlflow_cloud_artifact_repo
import mlflow.utils.file_utils as mlflow_file_utils
from mlflow.exceptions import MlflowException
from trame.assets.local import LocalFileManager
from iri_api_autogen.models import (
    JobAttributes,
    JobAttributesCustomAttributes,
    JobSpec,
    ResourceSpec,
)
from sfapi_client import AsyncClient
from sfapi_client.compute import Machine
from trame.widgets import vuetify3 as vuetify, html
from utils import timer, load_config_dict, create_date_filter
from calibration_manager import build_inferred_calibration
from execution_mode_manager import EXECUTION_MODE_ITEMS, remote_backend_unavailable_expr
from error_manager import add_error
from sfapi_manager import monitor_sfapi_job
from iriapi_manager import monitor_iriapi_job
from state_manager import state

LOGO_DIR = Path(__file__).parent / "logos"
AMSC_MLFLOW_URL = "https://mlflow.american-science-cloud.org"
MODEL_TYPE_GP = "Gaussian Process"
MODEL_TYPE_NN_SINGLE = "Neural Network (single)"
MODEL_TYPE_NN_ENSEMBLE = "Neural Network (ensemble)"
AMSC_LOGO_PATH = LOGO_DIR / "AmSC_300px.png"
AMSC_LOGO_URL = LocalFileManager(LOGO_DIR).url("amsc_logo", AMSC_LOGO_PATH)
MODEL_DOWNLOAD_ACTIVE_EXPR = "model_downloading"
AMSC_MLFLOW_LINK_ACTIVE_EXPR = (
    f"model_available && model_mlflow_tracking_uri === '{AMSC_MLFLOW_URL}'"
)
AMSC_MLFLOW_MODEL_URL_EXPR = (
    f"'{AMSC_MLFLOW_URL}/#/models/synapse-' + experiment + '_' + "
    f"(model_type_verbose === '{MODEL_TYPE_GP}' ? 'GP' : "
    f"model_type_verbose === '{MODEL_TYPE_NN_SINGLE}' ? 'NN' : "
    f"model_type_verbose === '{MODEL_TYPE_NN_ENSEMBLE}' ? 'ensemble_NN' : "
    "model_type_verbose)"
)
_NO_PRELOADED_MODEL = object()

model_type_dict = {
    MODEL_TYPE_GP: "GP",
    MODEL_TYPE_NN_SINGLE: "NN",
    MODEL_TYPE_NN_ENSEMBLE: "ensemble_NN",
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
IRI_TRAINING_LAUNCH_PREFIX = ("srun", "podman-hpc", "run")  # Container launch marker
TRAINING_REMOTE_DIR = "/global/cfs/cdirs/m558/superfacility/model_training"
TRAINING_CONFIG_REMOTE_PATH = f"{TRAINING_REMOTE_DIR}/config.yaml"
TRAINING_CONFIG_CONTAINER_PATH = "/app/ml/config.yaml"
TRAINING_CONFIG_MOUNT_RE = re.compile(
    rf"(?P<prefix>(?:^|\s)-v\s+)"
    rf"(?P<host_quote>[\"']?)"
    rf"(?P<host>[^\"'\s:]+)"
    rf"(?P=host_quote):"
    rf"(?P<container_quote>[\"']?)"
    rf"{re.escape(TRAINING_CONFIG_CONTAINER_PATH)}"
    rf"(?P=container_quote)"
    rf"(?=\s|\\|$)"
)
# Match model=$1 or model=${1}, with optional comment
SCRIPT_MODEL_ARGUMENT_RE = re.compile(r"^model=\$\{?1\}?\s*(#.*)?$")
# Capture shell assignments like REGISTRY_NAME=value, ignoring trailing comments
SHELL_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+?)(?:\s+#.*)?$")


def line_starts_with_tokens(line, expected_tokens):
    # Remove a trailing shell continuation so shlex can inspect the line prefix
    candidate = line.strip()
    if candidate.endswith("\\"):
        candidate = candidate[:-1].rstrip()

    try:
        tokens = shlex.split(candidate, comments=True)
    except ValueError:
        return False

    return tuple(tokens[: len(expected_tokens)]) == expected_tokens


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
    # Normalize Slurm duration variants to the seconds expected by AmSC IRI API
    days = 0
    time_part = duration.strip()
    has_days = "-" in time_part

    if has_days:
        day_part, time_part = time_part.split("-", 1)
        days = int(day_part)

    time_values = [int(v) for v in time_part.split(":")]
    n = len(time_values)

    if n > 3:
        raise ValueError(f"Unsupported Slurm duration format: {duration}")

    # Slurm treats ambiguous durations differently with a day prefix:
    # D-H[:M[:S]], H:M:S, M:S, or M
    if has_days:
        time_values += [0] * (3 - n)
    elif n == 1:
        time_values = [0, time_values[0], 0]
    else:
        time_values = [0] * (3 - n) + time_values

    hours, minutes, secs = time_values
    return days * 86400 + hours * 3600 + minutes * 60 + secs


def parse_sbatch_submit_options(script_path):
    # Extract only SBATCH fields that map cleanly onto AmSC IRI API submit options
    submit_options = {}
    with open(script_path) as script_file:
        for line in script_file:
            line = line.strip()
            if not line.startswith("#SBATCH"):
                continue

            directive = line.removeprefix("#SBATCH").strip()
            if not directive:
                continue

            # Use shell parsing so quoted SBATCH values are handled correctly
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

    # Convert values to the types expected by the AmSC IRI API submit endpoint
    submit_options["duration"] = parse_slurm_duration(submit_options["duration"])
    submit_options["nodes"] = int(submit_options["nodes"])
    return submit_options


def build_remote_training_config_path(experiment, model_type):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_experiment = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment).strip("-")
    safe_model_type = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_type).strip("-")
    config_name = f"config-{safe_experiment}-{safe_model_type}-{timestamp}.yaml"
    return f"{TRAINING_REMOTE_DIR}/{config_name}"


def replace_training_config_mount(command, remote_config_path):
    """Rewrite only the host path bound to the fixed container config path."""

    def replace_mount(match):
        return (
            f"{match.group('prefix')}"
            f"{shlex.quote(remote_config_path)}:{TRAINING_CONFIG_CONTAINER_PATH}"
        )

    rewritten_command, replacement_count = TRAINING_CONFIG_MOUNT_RE.subn(
        replace_mount,
        command,
        count=1,
    )
    if replacement_count != 1:
        raise ValueError(
            "Could not find the training config bind mount in the launch command"
        )
    return rewritten_command


def _custom_attributes_from_submit_options(submit_options):
    custom_attributes = JobAttributesCustomAttributes()
    for key in ("constraint",):
        if key in submit_options:
            custom_attributes.additional_properties[key] = submit_options[key]
    return custom_attributes


def _gpu_cores_per_process(submit_options):
    gpus_per_node = int(submit_options["gpus-per-node"])
    tasks_per_node = int(submit_options["ntasks-per-node"])
    if gpus_per_node % tasks_per_node != 0:
        raise ValueError(
            "AmSC IRI API GPU resources require --gpus-per-node to be evenly "
            "divisible by --ntasks-per-node"
        )
    return gpus_per_node // tasks_per_node


def build_iri_training_job_spec(submit_options, launch_spec, directory):
    """Build an IRI JobSpec with GPU resources in the standard resource field."""
    tasks_per_node = int(submit_options["ntasks-per-node"])
    node_count = submit_options["nodes"]

    resources = ResourceSpec(
        node_count=node_count,
        process_count=node_count * tasks_per_node,
        processes_per_node=tasks_per_node,
        gpu_cores_per_process=_gpu_cores_per_process(submit_options),
    )
    attributes = JobAttributes(
        duration=submit_options["duration"],
        queue_name=submit_options["queue"],
        account=submit_options["account"],
        custom_attributes=_custom_attributes_from_submit_options(submit_options),
    )

    return JobSpec(
        executable=launch_spec["executable"],
        arguments=launch_spec["arguments"],
        directory=directory,
        name=submit_options["name"],
        resources=resources,
        attributes=attributes,
        stdout_path=submit_options["stdout_path"],
        stderr_path=submit_options["stderr_path"],
        pre_launch=launch_spec["pre_launch"],
        launcher=launch_spec["launcher"],
    )


def collapse_shell_command(lines):
    """Return the one command from a launch block.

    Blank and comment-only lines are ignored before enforcing the command count.
    """
    command_parts = []
    current_command = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.endswith("\\"):
            current_command += f"{stripped[:-1]} "
        else:
            command = f"{current_command}{stripped}".strip()
            if command:
                command_parts.append(command)
            current_command = ""

    if current_command:
        command = current_command.strip()
        if command:
            command_parts.append(command)

    if len(command_parts) != 1:
        raise ValueError(
            "Expected exactly one AmSC IRI API training launch command, "
            f"but found {len(command_parts)}"
        )

    return command_parts[0]


def parse_shell_variable_assignments(lines):
    # Keep static shell assignments that can be expanded without execution
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


def expand_iri_shell_command(command, model_type, variables):
    # Expand values we can resolve locally while preserving runtime shell
    # variables like $HOME for the remote login shell on Perlmutter
    expanded = command
    replacements = {**variables, "model": model_type}
    # Repeat so static assignment chains like IMAGE=${REGISTRY}/${NAME} resolve
    for _ in replacements:
        previous = expanded
        for name, value in replacements.items():
            expanded = expanded.replace(f"${{{name}}}", value)
            expanded = expanded.replace(f"${name}", value)
        if expanded == previous:
            break
    return expanded


def parse_iri_training_launch_spec(script_path, model_type, remote_config_path=None):
    # Keep setup as raw pre_launch lines; only the podman-hpc launch command
    # is collapsed and rewritten locally.
    pre_launch_lines = []
    launch_lines = []
    found_launch = False

    with open(script_path) as script_file:
        for raw_line in script_file:
            line = raw_line.rstrip()
            stripped = line.strip()

            if found_launch:
                launch_lines.append(line)
            elif line_starts_with_tokens(stripped, IRI_TRAINING_LAUNCH_PREFIX):
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
        raise ValueError("Could not find the AmSC IRI API training launch command")

    launch_command = collapse_shell_command(launch_lines)
    variables = parse_shell_variable_assignments(pre_launch_lines)
    launcher = IRI_TRAINING_LAUNCH_PREFIX[0]
    executable = "/bin/bash"
    shell_command = re.sub(
        rf"^\s*{re.escape(launcher)}\s+", "", launch_command, count=1
    )
    shell_command = expand_iri_shell_command(shell_command, model_type, variables)
    if remote_config_path is not None:
        shell_command = replace_training_config_mount(shell_command, remote_config_path)
    arguments = ["-lc", shell_command]

    return {
        "arguments": arguments,
        "executable": executable,
        "launcher": launcher,
        "pre_launch": "\n".join(pre_launch_lines),
    }


def build_mlflow_model_name(config_dict, model_type):
    """Return the registered MLflow model name for an experiment and model type."""
    return f"synapse-{config_dict['experiment']}_{model_type}"


def configure_mlflow_tracking(config_dict):
    """Configure MLflow tracking for an experiment when MLflow is available."""
    mlflow_cfg = config_dict.get("mlflow") or {}
    tracking_uri = mlflow_cfg.get("tracking_uri")
    if not tracking_uri:
        msg = (
            "No mlflow.tracking_uri in configuration file for "
            f"{config_dict['experiment']}; cannot load model from MLflow."
        )
        print(msg)
        return False

    mlflow.set_tracking_uri(tracking_uri)
    # When using the AmSC MLflow, inject the X-Api-Key to authenticate.
    # (See https://gitlab.com/amsc2/ai-services/model-services/intro-to-mlflow-pytorch)
    if tracking_uri == AMSC_MLFLOW_URL:
        enable_amsc_x_api_key(config_dict)
    return True


def load_model_from_mlflow(config_dict, model_type):
    """Load the latest registered MLflow model for an experiment configuration."""
    if not configure_mlflow_tracking(config_dict):
        return None

    model_name = build_mlflow_model_name(config_dict, model_type)
    return (
        mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        .unwrap_python_model()
        .model
    )


def is_model_available_on_mlflow(config_dict, model_type):
    """Return whether MLflow has a registered model version to download."""
    if not configure_mlflow_tracking(config_dict):
        return False

    model_name = build_mlflow_model_name(config_dict, model_type)
    try:
        versions = mlflow.MlflowClient().search_model_versions(
            f"name='{model_name}'",
            max_results=1,
        )
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            return False
        print(f"Unable to check MLflow model availability for {model_name}: {e}")
        return False
    return bool(versions)


@contextmanager
def mlflow_artifact_progress_to_state(loop):
    """Expose MLflow artifact download progress through dashboard state."""
    progress_bar_modules = [
        mlflow_file_utils,
        mlflow_artifact_repo,
        mlflow_cloud_artifact_repo,
    ]
    original_progress_bars = {
        module: module.ArtifactProgressBar for module in progress_bar_modules
    }
    original_progress_bar = mlflow_file_utils.ArtifactProgressBar

    def set_download_progress(progress, total):
        """Publish the current download completion percentage to the GUI."""

        def update_progress_state():
            if total:
                state.model_download_progress = min(100, progress / total * 100)
            else:
                state.model_download_progress = None
            state.flush()

        loop.call_soon_threadsafe(update_progress_state)

    class TrameArtifactProgressBar(original_progress_bar):
        def __init__(self, desc, total, step, **kwargs):
            super().__init__(desc, total, step, **kwargs)
            self.trame_progress = 0
            if desc.startswith("Downloading"):
                set_download_progress(self.trame_progress, self.total)

        def update(self):
            super().update()
            self.trame_progress = min(
                self.total,
                self.trame_progress + self.step,
            )
            if self.desc.startswith("Downloading"):
                set_download_progress(self.trame_progress, self.total)

    for module in progress_bar_modules:
        module.ArtifactProgressBar = TrameArtifactProgressBar
    try:
        yield
    finally:
        for module, progress_bar in original_progress_bars.items():
            module.ArtifactProgressBar = progress_bar


def load_model_from_mlflow_with_progress(config_dict, model_type, loop):
    """Load an MLflow model while reporting artifact download progress."""
    with mlflow_artifact_progress_to_state(loop):
        return load_model_from_mlflow(config_dict, model_type)


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
    if getattr(rest_utils.http_request, "_synapse_amsc_api_key", None) == api_key:
        return

    _orig = getattr(rest_utils, "_synapse_http_request", rest_utils.http_request)

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

    patched._synapse_amsc_api_key = api_key
    rest_utils._synapse_http_request = _orig
    rest_utils.http_request = patched


class ModelManager:
    def __init__(self, config_dict, model_type, loaded_model=_NO_PRELOADED_MODEL):
        print("Initializing model manager...")
        self.__model = None
        self.__model_type = model_type

        try:
            self.__model = (
                load_model_from_mlflow(config_dict, model_type)
                if loaded_model is _NO_PRELOADED_MODEL
                else loaded_model
            )
            if self.__model is None:
                return
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
        build_inferred_calibration(input_variables, alpha_inferred, beta_inferred)

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
        build_inferred_calibration(output_variables, alpha_inferred, beta_inferred)
        # Notify Trame that the dict was modified in-place, so the UI updates
        state.dirty("simulation_calibration")

    def _prepare_training_config(
        self,
        temp_dir,
        experiment,
        experiment_date_range,
        simulation_calibration,
    ):
        """Prepare a training configuration file in the given temporary directory,
        updated with information from the dashboard.

        Returns the path to the written configuration file.
        """
        config_dict = load_config_dict(experiment)
        if config_dict.get("experiment") != experiment:
            raise ValueError(
                "Selected experiment does not match config file experiment: "
                f"{experiment} != {config_dict.get('experiment')}"
            )
        config_dict["simulation_calibration"] = simulation_calibration
        date_filter = create_date_filter(experiment_date_range)
        config_dict["date_filter"] = date_filter
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)
        return config_path

    async def _training_kernel_sfapi(
        self,
        experiment,
        model_type,
        experiment_date_range,
        simulation_calibration,
    ):
        try:
            # create an authenticated client
            async with AsyncClient(
                client_id=state.sfapi_client_id, secret=state.sfapi_key
            ) as client:
                perlmutter = await client.compute(Machine.perlmutter)
                # upload the configuration file to NERSC
                with tempfile.TemporaryDirectory() as temp_dir:
                    config_path = self._prepare_training_config(
                        temp_dir,
                        experiment,
                        experiment_date_range,
                        simulation_calibration,
                    )
                    remote_config_path = build_remote_training_config_path(
                        experiment,
                        model_type,
                    )
                    [target_path] = await perlmutter.ls(
                        TRAINING_REMOTE_DIR, directory=True
                    )
                    with open(config_path, "rb") as temp_file:
                        print("Uploading configuration file to NERSC...")
                        temp_file.filename = PurePosixPath(remote_config_path).name
                        await target_path.upload(temp_file)
                    print(f"Uploaded configuration file to {remote_config_path}")

                # set the path of the script used to submit the training job on NERSC
                training_script_path = find_training_script_path()
                with open(training_script_path) as file:
                    training_script = file.read()

                # replace the --model argument in the python command with the current model type from the state
                training_script = re.sub(
                    pattern=r"--model \$\{model\}",
                    repl=rf"--model {model_type}",
                    string=training_script,
                )
                training_script = replace_training_config_mount(
                    training_script, remote_config_path
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

    async def _training_kernel_iriapi(
        self,
        experiment,
        model_type,
        experiment_date_range,
        simulation_calibration,
    ):
        try:
            # Create an authenticated client
            client = Client(auth_method="globus")
            # Connect to NERSC
            nersc = await asyncio.to_thread(client.facility, "nersc")
            # Get the compute resource (Perlmutter)
            perlmutter = await asyncio.to_thread(nersc.resource, "compute")
            # Get the CFS resource for uploading files shared with compute jobs
            cfs = await asyncio.to_thread(nersc.resource, "cfs")
            training_script_path = find_training_script_path()
            remote_config_path = build_remote_training_config_path(
                experiment,
                model_type,
            )
            # Reuse SBATCH directives so AmSC IRI API submissions match the batch script
            submit_options = parse_sbatch_submit_options(training_script_path)
            launch_spec = parse_iri_training_launch_spec(
                training_script_path,
                model_type,
                remote_config_path,
            )
            training_job_spec = build_iri_training_job_spec(
                submit_options, launch_spec, TRAINING_REMOTE_DIR
            )
            print(
                "AmSC IRI API training submit options: "
                f"account={submit_options['account']}, "
                f"queue={submit_options['queue']}, "
                f"constraint={submit_options['constraint']}, "
                f"gpus-per-node={submit_options['gpus-per-node']}, "
                f"gpu-cores-per-process="
                f"{training_job_spec.resources.gpu_cores_per_process}, "
                f"nodes={submit_options['nodes']}, "
                f"duration={submit_options['duration']}"
            )

            # Training script is parsed locally; only config.yaml is uploaded
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = self._prepare_training_config(
                    temp_dir,
                    experiment,
                    experiment_date_range,
                    simulation_calibration,
                )
                print("Uploading configuration file to NERSC...")
                upload_task = await asyncio.to_thread(
                    cfs.fs.upload,
                    config_path,
                    remote_config_path,
                    file_name=PurePosixPath(remote_config_path).name,
                )
                upload_task = await asyncio.to_thread(upload_task.wait)
                if upload_task.state != "completed":
                    raise RuntimeError(
                        "Uploading configuration file to NERSC failed "
                        f"(task {upload_task.id} ended with state "
                        f"{upload_task.state})"
                    )
                print(f"Uploaded training configuration to {remote_config_path}")

            # Submit the training job through the AmSC IRI API
            iriapi_job = await asyncio.to_thread(
                perlmutter.submit,
                body=training_job_spec,
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

    async def _training_kernel_local(
        self,
        experiment,
        model_type,
        experiment_date_range,
        simulation_calibration,
    ):
        try:
            ml_dir = (Path(__file__).parent / "../ml").resolve()
            train_model_path = ml_dir / "train_model.py"

            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = self._prepare_training_config(
                    temp_dir,
                    experiment,
                    experiment_date_range,
                    simulation_calibration,
                )
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
            experiment = state.experiment
            model_type = model_type_dict[state.model_type_verbose]
            training_mode = state.model_training_mode
            experiment_date_range = list(state.experiment_date_range or [])
            simulation_calibration = deepcopy(state.simulation_calibration)
            state.model_training = True
            state.model_training_status = "Submitting"
            state.flush()
            if training_mode == "local":
                result = await self._training_kernel_local(
                    experiment,
                    model_type,
                    experiment_date_range,
                    simulation_calibration,
                )
            elif training_mode == "sfapi":
                result = await self._training_kernel_sfapi(
                    experiment,
                    model_type,
                    experiment_date_range,
                    simulation_calibration,
                )
            elif training_mode == "iriapi":
                result = await self._training_kernel_iriapi(
                    experiment,
                    model_type,
                    experiment_date_range,
                    simulation_calibration,
                )
            else:
                raise ValueError(f"Unsupported training mode: {training_mode}")
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
            MODEL_TYPE_GP,
            MODEL_TYPE_NN_SINGLE,
            MODEL_TYPE_NN_ENSEMBLE,
        ]
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_model", 0)):
            with vuetify.VExpansionPanel(
                title="Control: Models",
                style="font-size: 20px; font-weight: 500;",
            ):
                with vuetify.VExpansionPanelText():
                    with vuetify.VRow(align="center"):
                        with vuetify.VCol(cols=8, classes="d-flex align-center"):
                            vuetify.VSelect(
                                v_model=("model_type_verbose",),
                                label="Model type",
                                items=(model_type_list,),
                                dense=True,
                                hide_details=True,
                            )
                        with vuetify.VCol(
                            cols=4,
                            classes="d-flex align-center justify-end",
                        ):
                            with html.A(
                                href=(
                                    f"{AMSC_MLFLOW_LINK_ACTIVE_EXPR} ? "
                                    f"{AMSC_MLFLOW_MODEL_URL_EXPR} : null",
                                ),
                                target="_blank",
                                rel="noopener noreferrer",
                                title=(
                                    f"{AMSC_MLFLOW_LINK_ACTIVE_EXPR} ? "
                                    "'Open selected model in AmSC MLflow' : "
                                    "'Selected model is not available in AmSC "
                                    "MLflow'",
                                ),
                                style=(
                                    f"{AMSC_MLFLOW_LINK_ACTIVE_EXPR} ? "
                                    "'display: block; width: 100%; "
                                    "max-width: 300px; margin-left: auto; "
                                    "cursor: pointer;' : "
                                    "'display: block; width: 100%; "
                                    "max-width: 300px; margin-left: auto; "
                                    "cursor: default;'",
                                ),
                            ):
                                vuetify.VImg(
                                    src=AMSC_LOGO_URL,
                                    alt="AmSC",
                                    max_width=300,
                                    max_height=72,
                                    contain=True,
                                    style="width: 100%; cursor: inherit;",
                                )
                    with vuetify.VRow(
                        no_gutters=True,
                        align="center",
                        style=(
                            "margin-top: -8px; margin-bottom: 8px; min-height: 32px;"
                        ),
                    ):
                        with vuetify.VCol():
                            with html.Div(
                                style=(
                                    f"{MODEL_DOWNLOAD_ACTIVE_EXPR} ? "
                                    "'visibility: visible; opacity: 1;' : "
                                    "'visibility: hidden; opacity: 0;'",
                                )
                            ):
                                with html.Div(
                                    classes=(
                                        "d-flex align-center text-caption "
                                        "text-medium-emphasis mb-1"
                                    )
                                ):
                                    vuetify.VIcon(
                                        "mdi-cloud-download-outline",
                                        size=16,
                                        classes="mr-1",
                                    )
                                    html.Span(v_text=("model_download_status",))
                                    vuetify.VSpacer()
                                    html.Span(
                                        v_text=(
                                            "model_download_progress === null ? "
                                            "'' : "
                                            "`${Math.round(model_download_progress)}%`",
                                        ),
                                        style="min-width: 3em; text-align: right;",
                                    )
                                vuetify.VProgressLinear(
                                    indeterminate=("model_download_progress === null"),
                                    model_value=("model_download_progress",),
                                    color="primary",
                                    height=4,
                                    rounded=True,
                                )
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VSelect(
                                v_model=("model_training_mode",),
                                label="Training backend",
                                items=(EXECUTION_MODE_ITEMS,),
                                dense=True,
                                hide_details=True,
                            )
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VTextField(
                                v_model_number=("model_training_status",),
                                label="Training status",
                                readonly=True,
                                dense=True,
                                hide_details=True,
                            )
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VBtn(
                                "Train",
                                block=True,
                                click=self.training_trigger,
                                disabled=(
                                    "model_training || "
                                    f"{remote_backend_unavailable_expr('model_training_mode')}",
                                ),
                                style="text-transform: none",
                            )
