import pickle
from typing import Any, cast, Dict

__all__ = [
    "save_to_pickle",
    "open_pickle",
]


def save_to_pickle(save_path: str, data: Dict[int, Any]) -> None:
    with open(save_path, "wb") as file:
        pickle.dump(data, file)


def open_pickle(save_path: str) -> Dict[int, Any]:
    with open(save_path, "rb") as file:
        data = pickle.load(file)
    return cast(Dict[int, Any], data)
