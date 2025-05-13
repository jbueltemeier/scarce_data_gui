from typing import cast, Tuple

import numpy as np

from scipy.stats import qmc


__all__ = [
    "calculate_centered_l2_discrepancy",
    "calculate_discrepancy",
    "calculate_sliced_discrepancy",
]


def calculate_centered_l2_discrepancy(design: np.ndarray) -> float:
    num_datapoints, num_numerical_features = design.shape

    sum_1 = 0
    for i in range(num_datapoints):
        prod_i1 = 1
        for k in range(num_numerical_features):
            term1 = (
                1
                + (1 / 2) * abs(design[i, k] - 0.5)
                - (1 / 2) * (design[i, k] - 0.5) ** 2
            )
            prod_i1 *= term1
        sum_1 += prod_i1

    sum_2 = 0
    for i in range(num_datapoints):
        sum_i2 = 0
        for j in range(num_datapoints):
            prod_i2 = 1
            for k in range(num_numerical_features):
                term2 = (
                    1
                    + (1 / 2) * abs(design[i, k] - 0.5)
                    + (1 / 2) * abs(design[j, k] - 0.5)
                    - (1 / 2) * abs(design[i, k] - design[j, k])
                )
                prod_i2 *= term2
            sum_i2 += prod_i2
        sum_2 += sum_i2
    return (
        (13 / 12) ** num_numerical_features
        - (2 / num_datapoints) * sum_1
        + (1 / (num_datapoints**2)) * sum_2
    )


def calculate_discrepancy(*, design: np.ndarray, method: str = "L2") -> float:
    if method == "L2":
        return calculate_centered_l2_discrepancy(design)
    else:
        if method not in ["CD", "L2-star", "MD", "WD"]:
            raise ValueError(f"Unrecognized method {method}")
        return cast(float, qmc.discrepancy(design, method=method))


def calculate_sliced_discrepancy(
    *,
    design: np.ndarray,
    num_slices: int,
    num_datapoints_per_slice: int,
    method: str = "L2",
) -> Tuple[float, float]:
    max_unif_global = calculate_discrepancy(design=design, method=method)

    av_max_unif_slice = 0.0
    for i in range(num_slices):
        lower_bound = i * num_datapoints_per_slice
        upper_bound = (i + 1) * num_datapoints_per_slice
        av_max_unif_slice += (
            calculate_discrepancy(
                design=design[lower_bound:upper_bound, :], method=method
            )
            / num_slices
        )
    return max_unif_global, av_max_unif_slice
