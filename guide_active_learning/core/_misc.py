from typing import Tuple

import numpy as np
import pandas as pd

__all__ = [
    "filtering_array_loh",
    "create_class_scatter_matrix",
    "one_hot_enc",
]


def filtering_array_loh(
    filter_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets, counts = np.unique(filter_array[:, -1], return_counts=True)

    set_arrays = []
    for enu, j in enumerate(targets):
        tmp_j = filter_array[filter_array[:, -1] == j]
        for xi in range(filter_array.shape[-1] - 1):
            tmp_j = tmp_j[
                abs(tmp_j[:, xi] - np.mean(tmp_j[:, xi])) <= 2 * np.std(tmp_j[:, xi])
            ]
        set_arrays.append(tmp_j)

    set_array = np.vstack(set_arrays)
    targets, counts = np.unique(set_array[:, -1], return_counts=True)
    return set_array, targets, counts


def create_class_scatter_matrix(
    set_array: np.ndarray, targets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mean_arr = np.empty(len(targets), dtype=object)
    # matrix_in - within class scatter matrix
    matrix_in = np.empty(
        (set_array.shape[-1] - 1, set_array.shape[-1] - 1), dtype=float
    )
    # matrix_out - between class scatter matrix
    matrix_out = np.empty(
        (set_array.shape[-1] - 1, set_array.shape[-1] - 1), dtype=float
    )
    outer_mean = np.mean(set_array[:, :-1], axis=0)

    # iteration through different target classes
    for enu, target in enumerate(targets):
        # inner class scatter
        val_arr = set_array[set_array[:, -1] == target][:, :-1]
        mean_arr[enu] = np.mean(val_arr, axis=0)
        tmp1 = val_arr - mean_arr[enu]
        tmp1 = tmp1.astype(float)
        for i in range(len(val_arr)):
            matrix_in += np.outer(tmp1[i], tmp1[i].T)

        # between classes scatter
        tmp2 = (mean_arr[enu] - outer_mean).astype(float)
        matrix_out += np.outer(tmp2, tmp2.T)

    return matrix_in, matrix_out


def one_hot_enc(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "target" in df.columns:
        # Zielvariable extrahieren, wenn vorhanden
        y = df["target"]
    else:
        # Leere Zielvariable, wenn 'target' nicht vorhanden ist
        y = pd.Series(dtype=float)

    if "target" in df:
        df = df.drop("target", axis=1)
    X_num = df.select_dtypes(include=["number"])
    X_cat = df.select_dtypes(include=["object", "category"])
    # Kategorische Daten one-hot encodieren
    if not X_cat.empty:
        X_cat = pd.get_dummies(X_cat)
    # Numerische und one-hot encodierte Daten wieder zusammenführen
    x = pd.concat([X_num, X_cat], axis=1)

    return x, y
