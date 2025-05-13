import pandas as pd


__all__ = [
    "remove_slice_column",
]


def remove_slice_column(design: pd.DataFrame) -> pd.DataFrame:
    if isinstance(design, pd.DataFrame):
        design = design.iloc[:, 1:].to_numpy(dtype=float)
    return design
