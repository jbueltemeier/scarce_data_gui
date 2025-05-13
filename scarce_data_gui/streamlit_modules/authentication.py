import os
from typing import Any, Dict, Tuple

import streamlit_authenticator as st_auth
import yaml
from yaml.loader import SafeLoader

from scarce_data_gui.misc import get_default_pages_path


__all__ = [
    "load_user_auth_config_file",
    "get_authenticator",
    "update_user_auth_config_file",
]


def load_user_auth_config_file() -> str:
    return os.path.join(get_default_pages_path(), "auth_config.yaml")


def get_authenticator() -> Tuple[Dict[str, Any], st_auth.Authenticate]:
    config_file_path = load_user_auth_config_file()
    # Load configuration from the YAML file
    with open(config_file_path) as file:
        config = yaml.load(file, Loader=SafeLoader)

    return config, st_auth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )


def update_user_auth_config_file(config: Dict[str, Any]) -> None:
    config_file_path = load_user_auth_config_file()
    with open(config_file_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
