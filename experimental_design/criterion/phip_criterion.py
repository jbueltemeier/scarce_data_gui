from typing import cast, Optional

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform

from experimental_design.core import remove_slice_column

__all__ = [
    "calculate_phip_value",
    "average_inter_point_distance_phip",
    "scaled_nn_phip",
    "sliced_nn_phip_v3",
]


def calculate_phip_value(*, distance: np.ndarray, p_norm: int) -> float:
    if p_norm == 2:
        return cast(float, 1 / (np.sqrt(np.dot(distance, distance)) ** p_norm))
    elif p_norm == 1:
        return cast(float, 1 / (np.sum(distance) ** p_norm))
    else:
        raise NotImplementedError(
            f"Norm {p_norm} is not implemented for this PhiP criterion."
        )


def average_inter_point_distance_phip(
    *, design: pd.DataFrame, num_datapoints: int, p_norm: int
) -> float:
    design = remove_slice_column(design)

    phip: float = 0.0
    for i in range(num_datapoints):
        datapoint_1 = design[i, :]
        for k in range(i + 1, num_datapoints):
            distance = np.abs(datapoint_1 - design[k, :])
            phip += calculate_phip_value(distance=distance, p_norm=p_norm)

    factor = 2 / (num_datapoints * (num_datapoints - 1))
    phi = (factor * phip) ** (1 / p_norm)
    return cast(float, phi)


def scaled_nn_phip(
    *, design: pd.DataFrame, num_datapoints: int, p_norm: float, scale_factor: float
) -> float:
    design = remove_slice_column(design) * scale_factor
    pairwise_distances = pdist(design, "minkowski", p=2)
    distance_matrix = squareform(pairwise_distances)
    np.fill_diagonal(distance_matrix, np.inf)
    nn_distance = np.min(distance_matrix, axis=1)

    nn_phip: float = 0.0
    for i in range(num_datapoints):
        nn_phip += 1 / (nn_distance[i] ** p_norm)

    nn_phip = ((1 / num_datapoints) * nn_phip) ** (1 / p_norm)
    return cast(float, nn_phip)


def calculate_global_phip(
    design: pd.DataFrame, num_datapoints: int, p_norm: int, beta: float = 0.5
) -> float:
    return (1 - beta) * scaled_nn_phip(
        design=design, num_datapoints=num_datapoints, p_norm=p_norm, scale_factor=1
    )


def calculate_slice_phip(
    design: pd.DataFrame,
    num_datapoints: int,
    num_datapoints_per_slice: int,
    p_norm: int,
    beta: float = 0.5,
    comb_metric: str = "sum",
) -> float:
    categorical_vars = design.iloc[:, 0].to_numpy()
    slices = np.unique(categorical_vars)
    weights = beta / len(slices)
    scale_factor = (1 / len(slices)) ** (1 / num_datapoints_per_slice)

    phip: float = 0.0
    for i in slices:
        if comb_metric == "sum":
            phip += weights * scaled_nn_phip(
                design=design[categorical_vars == i],
                num_datapoints=num_datapoints_per_slice,
                p_norm=p_norm,
                scale_factor=scale_factor,
            )
        elif comb_metric == "mult":
            phip += (
                scaled_nn_phip(
                    design=design[categorical_vars == i],
                    num_datapoints=num_datapoints_per_slice,
                    p_norm=p_norm,
                    scale_factor=scale_factor,
                )
                ** p_norm
            ) * (num_datapoints / len(slices))

    if comb_metric == "mult":
        phip += beta * (((1 / num_datapoints) * phip) ** (1 / p_norm))
    return phip


def sliced_nn_phip_v3(
    *,
    design: pd.DataFrame,
    num_datapoints: int,
    num_datapoints_per_slice: int,
    p_norm: int,
    beta: Optional[float] = None,
    comb_metric: str = "sum",
) -> float:
    beta = 0.5 if beta is None else beta

    if not 0 <= beta <= 1:
        raise ValueError(f"Warning: Beta should be between 0 and 1, not {beta}.")

    phip: float = 0.0
    if beta < 1:
        phip += calculate_global_phip(
            design=design, num_datapoints=num_datapoints, p_norm=p_norm, beta=beta
        )

    if beta > 0:
        phip += calculate_slice_phip(
            design=design,
            num_datapoints=num_datapoints,
            num_datapoints_per_slice=num_datapoints_per_slice,
            p_norm=p_norm,
            beta=beta,
            comb_metric=comb_metric,
        )

    return phip
