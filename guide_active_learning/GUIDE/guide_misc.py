from typing import cast, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from guide_active_learning.core import (
    compute_lda,
    compute_wilsonhilferty,
    create_class_scatter_matrix,
    create_intervals,
    filtering_array_loh,
    linear_lda_transform,
)


__all__ = [
    "sfi_createbins",
    "sfi_extractinter",
    "sf_interaction",
    "split_dataframe",
    "sf_linear_lda_transform",
]


def sfi_createbins(
    *,
    datapoint: pd.Series,
    df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    tmp = datapoint[:]
    if len(tmp) < 45 * len(df[target].value_counts()):
        intervals = create_intervals(mean=tmp.mean(), std=tmp.std(), num_intervals=2)
    else:
        intervals = create_intervals(mean=tmp.mean(), std=tmp.std(), num_intervals=3)

    tmp_df = pd.concat([tmp, pd.cut(tmp, bins=intervals, labels=False)], axis=1)
    tmp_df = tmp_df.set_axis([tmp_df.columns[0], "interval"], axis=1)
    return tmp_df


def sfi_extractinter(
    *, df: pd.DataFrame, relevant_feature: str, target: str
) -> List[str]:
    interacting_features = []
    feature_list = list(df.drop(columns=target).columns)
    dict1 = dict(zip(list(range(len(feature_list))), feature_list))
    interacting_features.append(dict1.get(int(relevant_feature[0])))
    feature_list.remove(interacting_features[0])
    dict2 = dict(zip(list(range(len(feature_list))), feature_list))
    interacting_features.append(dict2.get(int(relevant_feature[1])))
    return cast(List[str], interacting_features)


def sf_interaction(df: pd.DataFrame, target: str) -> Dict[str, float]:
    chi2_dict = {}
    for cnt1, col1 in enumerate(list(df.drop(columns=target).columns)):
        tmp1 = df.loc[:, col1]
        for cnt2, col2 in enumerate(list(df.drop(columns=[target, col1]).columns)):
            # nichtkategorisch
            tmp2 = df.loc[:, col2]

            if tmp1.dtypes == float:
                binning1 = sfi_createbins(datapoint=tmp1, df=df, target=target)
            else:
                binning1 = pd.concat([tmp1, tmp1], axis=1)
                binning1 = binning1.set_axis(["vals", "interval"], axis=1)

            if tmp2.dtypes == float:
                binning2 = sfi_createbins(datapoint=tmp2, df=df, target=target)
            else:
                binning2 = pd.concat([tmp2, tmp2], axis=1)
                binning2 = binning2.set_axis(["vals", "interval"], axis=1)

            cont_table = pd.crosstab(list(df[target]), list(binning1["interval"]))
            cont_table = pd.concat(
                [cont_table, pd.crosstab(list(df[target]), list(binning2["interval"]))],
                axis=1,
            )

            chi2_tmp = chi2_contingency(cont_table)
            wm = np.max([0, compute_wilsonhilferty(chi2_tmp)])

            chi2_dict.update({str(cnt1) + str(cnt2): wm})

    return chi2_dict


def split_dataframe(
    df: pd.DataFrame,
    split_point: Optional[Union[str, List[str], float]],
    split_feature: Optional[Union[str, List[str], np.ndarray]],
    target: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # numerical splitting
    if isinstance(split_point, float) and not isinstance(split_feature, list):
        df1 = df[df[split_feature] <= split_point]
        df2 = df[df[split_feature] > split_point]

    elif isinstance(split_feature, list):
        lda = compute_lda(df=df, feature=split_feature, target=target)
        lda.index = df.index
        df1 = df[lda["LDA"] <= split_point]
        df2 = df[lda["LDA"] > split_point]

    # categorical splitting
    else:
        if isinstance(split_point, str):
            df1 = df[df[split_feature].isin([split_point])]
            df2 = df[~df[split_feature].isin([split_point])]

        else:
            df1 = df[df[split_feature].isin(split_point)]
            df2 = df[~df[split_feature].isin(split_point)]

    return df1, df2


def sf_linear_lda_transform(
    df: pd.DataFrame,
) -> Union[str, Tuple[np.ndarray, pd.DataFrame]]:
    arr = df[:].to_numpy()
    set_array, targets, counts = filtering_array_loh(arr)
    matrix_in, matrix_out = create_class_scatter_matrix(set_array, targets)

    if (
        (np.linalg.det(matrix_in) != 0)
        and (np.linalg.det(matrix_out) != 0)
        and not (np.any(matrix_in.round(2) == 0) or np.any(matrix_out.round(2) == 0))
        and not (
            np.any(np.isnan(matrix_in))
            or np.any(np.isinf(matrix_in))
            or np.any(np.isnan(matrix_out))
            or np.any(np.isinf(matrix_out))
        )
    ):
        return linear_lda_transform(
            matrix_in=matrix_in, matrix_out=matrix_out, arr=arr, df=df
        )
    else:
        return "singularity"
