import configparser
import os
from ast import literal_eval
from typing import Any, Dict, Optional

__all__ = [
    "process_dir",
    "make_folder_path",
    "make_output_filename",
    "get_default_path",
    "get_default_dataset_path",
    "get_default_pages_path",
    "get_default_example_path",
    "get_default_pages_images_path",
    "load_config",
    "load_settings",
]


def process_dir(directory: str) -> str:
    directory = os.path.abspath(os.path.expanduser(directory))
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


def make_output_filename(
    *parts: str,
    extension: Optional[str] = None,
) -> str:
    return (
        "__".join([*parts]) + extension
        if extension is not None
        else "__".join([*parts])
    )


def make_folder_path(
    *folders: str,
    filename: Optional[str] = None,
) -> str:
    folder_path = process_dir(os.path.join(*folders))
    if filename is None:
        return folder_path

    return os.path.join(folder_path, filename)


def get_default_path() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def get_default_data_path() -> str:
    return make_folder_path(get_default_path(), "Data")


def get_default_dataset_path() -> str:
    return process_dir(make_folder_path(get_default_data_path(), "Dataset"))


def get_default_pages_path() -> str:
    return process_dir(make_folder_path(get_default_path(), "..", "streamlit_pages"))


def get_default_example_path() -> str:
    return process_dir(make_folder_path(get_default_path(), "..", "example"))


def get_default_pages_images_path() -> str:
    return process_dir(make_folder_path(get_default_pages_path(), "images"))


def parse_value(value: Any) -> Any:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            try:
                return literal_eval(value)
            except (ValueError, SyntaxError):
                return value


def load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(
        os.path.join(get_default_pages_path(), "settings.ini"), encoding="utf-8"
    )
    return config


def init_settings(
    config: configparser.ConfigParser, setting_str: str
) -> Dict[str, Any]:
    settings = {}
    for key, value in config[setting_str].items():
        settings[key] = parse_value(value)
    return settings


def load_settings() -> Dict[str, Any]:
    config = load_config()
    return init_settings(config, setting_str="User Settings")
