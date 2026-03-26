"""MLflow utility functions for AmSC authentication and configuration."""

import os


def enable_amsc_x_api_key(config_dict):
    """
    MLflow authentication helper for the AmSC MLflow server.

    Standard MLflow does not automatically inject custom headers like 'X-Api-Key'.
    This patches the http_request function to ensure every request to the server
    includes the AmSC API key.

    Args:
        config_dict: Configuration dictionary containing mlflow settings

    Raises:
        ValueError: If required mlflow configuration is missing or invalid

    See https://gitlab.com/amsc2/ai-services/model-services/intro-to-mlflow-pytorch for more details.
    """
    import mlflow.utils.rest_utils as rest_utils

    mlflow_cfg = config_dict.get("mlflow") if config_dict is not None else None
    if not isinstance(mlflow_cfg, dict):
        raise ValueError(
            "Missing 'mlflow' configuration section required for AmSC MLFlow authentication."
        )

    api_key_env = mlflow_cfg.get("api_key_env")
    if not api_key_env:
        raise ValueError(
            "Missing 'api_key_env' in 'mlflow' configuration. "
            "Please specify the name of the environment variable containing the AmSC API key."
        )

    api_key = os.environ.get(api_key_env)
    if api_key is None:
        raise ValueError(
            f"The environment variable '{api_key_env}' specified in 'mlflow.api_key_env' "
            "is not set. Please export it with the AmSC MLFlow API key."
        )

    _orig = rest_utils.http_request

    def patched(host_creds, endpoint, method, *args, **kwargs):
        h = dict(kwargs.get("headers") or kwargs.get("extra_headers") or {})
        h["X-Api-Key"] = api_key
        kwargs["headers" if "headers" in kwargs else "extra_headers"] = h
        return _orig(host_creds, endpoint, method, *args, **kwargs)

    rest_utils.http_request = patched
