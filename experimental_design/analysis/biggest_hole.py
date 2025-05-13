from typing import cast, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sobol_seq import i4_sobol_generate


__all__ = [
    "calculate_biggest_hole",
]


def calculate_biggest_hole(
    design: pd.DataFrame,
    num_slices: int,
    num_datapoints_per_slice: int,
    ref_data: Optional[np.ndarray] = None,
    metric: str = "euclidean",
) -> Tuple[float, float]:
    if ref_data is None:
        ref_data = i4_sobol_generate(design.shape[1], 2**18)

    distribution = np.min(cdist(design, ref_data, metric=metric), axis=1)
    max_dist_global = np.max(distribution)

    max_dist_slices_vec = np.zeros(num_slices)
    for i in range(num_slices):
        lower_bound = i * num_datapoints_per_slice
        upper_bound = (i + 1) * num_datapoints_per_slice
        distribution_slice = np.min(
            cdist(design[lower_bound:upper_bound, :], ref_data, metric="euclidean"),
            axis=1,
        )
        max_dist_slices_vec[i] = np.max(distribution_slice)

    max_dist_slice = np.max(max_dist_slices_vec)

    return cast(float, max_dist_global), cast(float, max_dist_slice)
