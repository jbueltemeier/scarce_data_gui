import configparser
import os
from ast import literal_eval
from typing import Any, Dict, Optional

__all__ = [
    "process_dir",
    "make_folder_path",
    "make_output_filename",
    "get_default_path",
    "get_default_result_path",
    "get_default_train_path",
    "get_default_model_path",
    "get_default_dataset_path",
    "load_config",
    "with_settings",
]


def process_dir(dir: str) -> str:
    dir = os.path.abspath(os.path.expanduser(dir))
    os.makedirs(dir, exist_ok=True)
    return dir


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


def get_default_result_path() -> str:
    return process_dir(make_folder_path(get_default_data_path(), "result"))


def get_default_train_path() -> str:
    return process_dir(make_folder_path(get_default_data_path(), "train"))


def get_default_model_path() -> str:
    return process_dir(make_folder_path(get_default_data_path(), "model"))


def get_default_dataset_path() -> str:
    return make_folder_path(get_default_data_path(), "dataset")


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
    here = os.path.abspath(os.path.dirname(__file__))
    config = configparser.ConfigParser()
    config.read(os.path.join(here, "..", "scripts", "settings.ini"))
    return config


def init_settings(
    config: configparser.ConfigParser, setting_str: str
) -> Dict[str, Any]:
    settings = {}
    for key, value in config[setting_str].items():
        settings[key] = parse_value(value)
    return settings


def with_settings(func: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        config = load_config()
        settings = init_settings(config, setting_str="Benchmark")
        kwargs["settings"] = settings
        return func(*args, **kwargs)

    return wrapper
