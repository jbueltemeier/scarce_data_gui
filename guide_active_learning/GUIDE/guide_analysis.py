from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from guide_active_learning.core import (
    calculate_contribution_table,
    compute_wilsonhilferty,
)

__all__ = [
    "sf_main_effect",
]


def sf_main_effect(df: pd.DataFrame, target: str) -> Dict[str, float]:
    chi2_dict = {}

    for column in list(df.drop(columns=target).columns):
        if df.dtypes[column] != object:
            cont_table = calculate_contribution_table(
                df, target, column, use_interval=True
            )
        else:
            cont_table = calculate_contribution_table(df, target, column)

        chi2_tmp = chi2_contingency(cont_table)
        wm = np.max([0, compute_wilsonhilferty(chi2_tmp)])

        chi2_dict.update({column: wm})

    return chi2_dict
