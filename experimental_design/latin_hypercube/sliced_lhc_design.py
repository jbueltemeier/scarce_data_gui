from typing import List, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "create_factor_mapping",
    "create_level_mapping",
    "create_design",
    "sliced_lhc_design",
]


def create_level_mapping(
    *,
    num_datapoints_per_slice: int,
    num_numerical_features: int,
    num_slices: int,
) -> np.ndarray:
    return (
        np.array(
            [
                np.random.permutation(num_datapoints_per_slice)
                for _ in range(num_numerical_features * num_slices)
            ]
        )
        .transpose()
        .reshape((num_datapoints_per_slice, num_numerical_features, num_slices))
    )


def create_factor_mapping(
    *,
    num_datapoints_per_slice: int,
    num_numerical_features: int,
    num_slices: int,
) -> np.ndarray:
    factor_mapping = np.zeros(
        (num_datapoints_per_slice, num_slices, num_numerical_features), dtype=int
    )
    for k in range(num_numerical_features):
        for i in range(num_datapoints_per_slice):
            factor_mapping[i, :, k] = np.random.permutation(num_slices) + i * num_slices
    return factor_mapping


def create_design(
    *,
    factor_level: np.ndarray,
    factor_mapping: np.ndarray,
    level_mapping: np.ndarray,
    slices: List[str],
    numerical_factors: List[str],
    num_datapoints: int,
    num_slices: int,
    num_numerical_features: int,
    num_datapoints_per_slice: int,
) -> pd.DataFrame:
    design = pd.DataFrame()
    for i in range(num_slices):
        mask = factor_mapping[:, i, :].ravel(order="F")
        indices = (
            level_mapping[:, :, i]
            + np.arange(num_numerical_features) * num_datapoints_per_slice
        ).ravel(order="F")
        matrix = mask[indices].reshape(
            num_datapoints_per_slice, num_numerical_features, order="F"
        )

        slice_indices = (
            matrix + np.arange(num_numerical_features) * num_datapoints
        ).ravel(order="F")
        slice_data = np.hstack(
            [
                np.tile("/".join(slices[i]), (num_datapoints_per_slice, 1)),
                factor_level[slice_indices].reshape(
                    num_datapoints_per_slice, num_numerical_features, order="F"
                ),
            ]
        )
        slice_df = pd.DataFrame(slice_data, columns=["slice", *numerical_factors])
        design = pd.concat([design, slice_df], axis=0)
    return design


def sliced_lhc_design(
    *,
    factor_level: np.ndarray,
    slices: List[str],
    numerical_factors: List[str],
    num_datapoints: int,
    num_slices: int,
    num_numerical_features: int,
    num_datapoints_per_slice: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:

    level_mapping = create_level_mapping(
        num_datapoints_per_slice=num_datapoints_per_slice,
        num_numerical_features=num_numerical_features,
        num_slices=num_slices,
    )

    factor_mapping = create_factor_mapping(
        num_datapoints_per_slice=num_datapoints_per_slice,
        num_numerical_features=num_numerical_features,
        num_slices=num_slices,
    )

    design = create_design(
        factor_level=factor_level.ravel(order="F"),
        factor_mapping=factor_mapping,
        level_mapping=level_mapping,
        slices=slices,
        numerical_factors=numerical_factors,
        num_datapoints=num_datapoints,
        num_slices=num_slices,
        num_numerical_features=num_numerical_features,
        num_datapoints_per_slice=num_datapoints_per_slice,
    )

    return design, level_mapping, factor_mapping
