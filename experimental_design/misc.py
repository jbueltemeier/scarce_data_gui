import os
from typing import Optional

__all__ = [
    "process_dir",
    "make_folder_path",
    "make_output_filename",
    "get_default_path",
]


def process_dir(directory: str) -> str:
    directory = os.path.abspath(os.path.expanduser(directory))
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
