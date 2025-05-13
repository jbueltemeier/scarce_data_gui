import pickle
from typing import cast, List, Optional

from experimental_design.dataset import PartDesignDataset

from sqlalchemy.orm import Session

from scarce_data_gui.db import FileDataset

__all__ = [
    "query_save_dataset",
    "query_load_dataset",
    "query_current_data_filenames",
]


def query_save_dataset(
    db: Session, *, dataset: PartDesignDataset, filename: str
) -> None:
    dataset_pickle = pickle.dumps(dataset)
    db_dataset = (
        db.query(FileDataset)  # type: ignore[attr-defined]
        .filter(FileDataset.data_filename == filename)
        .first()
    )
    if db_dataset is not None:
        db_dataset.dataset = dataset_pickle
    else:
        db_dataset = FileDataset(data_filename=filename, dataset=dataset_pickle)
        db.add(db_dataset)

    db.commit()
    db.refresh(db_dataset)


def query_load_dataset(db: Session, filename: str) -> Optional[PartDesignDataset]:
    db_dataset = (
        db.query(FileDataset)  # type: ignore[attr-defined]
        .filter(FileDataset.data_filename == filename)
        .first()
    )
    dataset = pickle.loads(db_dataset.dataset)
    return cast(Optional[PartDesignDataset], dataset)


def query_current_data_filenames(db: Session) -> Optional[List[str]]:
    file_names = [filename[0] for filename in db.query(FileDataset.data_filename).all()]
    file_names = list(set(file_names))
    return None if not file_names else file_names
