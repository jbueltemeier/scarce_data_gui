import os
import pickle
from typing import cast, List, Optional

from experimental_design.dataset import PartDesignDataset

from sqlalchemy.orm import Session

from scarce_data_gui.db.queries import (
    query_current_data_filenames,
    query_load_dataset,
    query_save_dataset,
)
from scarce_data_gui.misc import get_default_dataset_path

__all__ = [
    "list_folders",
    "list_datasets",
    "save_dataset",
    "load_dataset",
    "get_all_current_dataset_filenames",
]


def list_folders(directory: str) -> Optional[List[str]]:
    folders = [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]
    return None if not folders else folders


def list_datasets(directory: str) -> Optional[List[str]]:
    folders = [name for name in os.listdir(directory)]
    return None if not folders else folders


def save_dataset(
    dataset: PartDesignDataset, filename: str, db: Optional[Session] = None
) -> None:
    if db is not None:
        query_save_dataset(db, dataset=dataset, filename=filename)
    else:
        folder = get_default_dataset_path()
        with open(os.path.join(folder, filename), "wb") as f:
            pickle.dump(dataset, f)


def load_dataset(
    filename: str, db: Optional[Session] = None
) -> Optional[PartDesignDataset]:
    if db is not None:
        return query_load_dataset(db, filename=filename)
    else:
        folder = get_default_dataset_path()
        with open(os.path.join(folder, filename), "rb") as f:
            dataset = pickle.load(f)
        return cast(Optional[PartDesignDataset], dataset)


def get_all_current_dataset_filenames(db: Optional[Session]) -> Optional[List[str]]:
    if db is not None:
        return query_current_data_filenames(db)
    else:
        return list_datasets(get_default_dataset_path())
