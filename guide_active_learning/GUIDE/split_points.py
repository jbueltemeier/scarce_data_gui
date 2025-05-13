import numbers
from itertools import combinations
from typing import cast, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from guide_active_learning.core import compute_gini_np, compute_gini_pd
from guide_active_learning.GUIDE.splitting import (
    spi_categorical_splitting,
    spi_mixed_splitting_01,
    spi_mixed_splitting_02,
)


__all__ = [
    "select_numerical_feature",
    "sp_main_feature",
    "spi_numerical",
    "spi_mixed",
    "spi_categorical",
    "sp_interaction_features",
]


def select_numerical_feature(
    df: pd.DataFrame,
    feature: Union[List[str], str],
    target_expression: pd.DataFrame,
    class_cost: np.ndarray,
) -> float:
    vals = np.array(df[feature].value_counts().index.sort_values())
    vals_enum = vals[:-1]
    gini = np.ones(len(vals_enum))
    for cnt, split in enumerate(vals_enum):
        df1 = df[df[feature] <= split]
        df2 = df[df[feature] > split]
        wG = (len(df1) / len(df)) * compute_gini_pd(df1, target_expression, class_cost) + (
            len(df2) / len(df)
        ) * compute_gini_pd(df2, target_expression, class_cost)

        gini[cnt] = wG

    if len(vals) > 1:
        return cast(float, (vals[np.argmin(gini)] + vals[np.argmin(gini) + 1]) / 2)
    else:
        return cast(float, vals[np.argmin(gini)])


def select_categorical_feature(
    df: pd.DataFrame,
    feature: Union[List[str], str],
    target_expression: pd.DataFrame,
    class_cost: np.ndarray,
) -> float:
    entries = list(df[feature].unique())
    if len(entries) == 1:
        all_combs = entries

    else:
        all_combs = []
        for i in range(1, len(entries)):
            all_combs.extend(combinations(entries, i))

    # select only relevant from all_combs (Katze,Hund = Pfirsich)
    gini = np.ones(len(all_combs))
    for cnt, A in enumerate(all_combs):
        df1 = df[df[feature].isin(list(A))]
        df2 = df[~df[feature].isin(list(A))]

        wG = (len(df1) / len(df)) * compute_gini_pd(df1, target_expression, class_cost) + (
            len(df2) / len(df)
        ) * compute_gini_pd(df2, target_expression, class_cost)

        gini[cnt] = wG

    min_gini = all_combs[np.argmin(gini)]
    if type(min_gini) == tuple and len(min_gini) == 1:
        min_gini = min_gini[0]

    return cast(float, min_gini)


def sp_main_feature(
    df: pd.DataFrame,
    feature: Union[List[str], str],
    target_expression: pd.DataFrame,
    class_cost: np.ndarray,
) -> Tuple[Union[str, List[str]], float]:
    """Functions for splitpoints - one selected feature"""
    if df.dtypes[feature] in [float, int, "int64"]:
        min_gini = select_numerical_feature(df, feature, target_expression, class_cost)
    elif df.dtypes[feature] == object:
        min_gini = select_categorical_feature(df, feature, target_expression, class_cost)
    else:
        raise ValueError("Feature nicht kategorisch oder numerisch!")
    return feature, min_gini


def spi_numerical(
    df: pd.DataFrame,
    feature: List[str],
    target: str,
    d: int,
    target_expression: pd.DataFrame,
    class_cost: np.ndarray
) -> Tuple[str, float]:
    """Functions for splitpoint - two Numerical selected features"""
    columns = feature[:]
    columns.append(target)
    df = df[columns]
    df = df.to_numpy()
    Sk1 = np.linspace(np.min(df[:, 0]), np.max(df[:, 0]), d)
    Sk2 = np.linspace(np.min(df[:, 1]), np.max(df[:, 1]), d)
    tuples_1 = np.empty(d, dtype=tuple)
    weighted_ginis_1 = np.empty(d)

    for cntc, c in enumerate(Sk1):
        wG1 = np.zeros(len(Sk2))
        wG2 = np.zeros(len(Sk2))
        df_l = df[df[:, 0] <= c]
        df_r = df[df[:, 0] > c]
        for cnt, sp in enumerate(Sk2):
            df_ll = df_l[df_l[:, 1] <= sp]
            df_lr = df_l[df_l[:, 1] > sp]

            wG1[cnt] = (len(df_ll) / len(df)) * compute_gini_np(df_ll, target_expression, class_cost) + (
                len(df_lr) / len(df)
            ) * compute_gini_np(df_lr, target_expression, class_cost)

        for cnt, e in enumerate(Sk2):
            df_rl = df_r[df_r[:, 1] <= e]
            df_rr = df_r[df_r[:, 1] < e]

            wG2[cnt] = (len(df_rl) / len(df)) * compute_gini_np(df_rl, target_expression, class_cost) + (
                len(df_rr) / len(df)
            ) * compute_gini_np(df_rr, target_expression, class_cost)

        tuples_1[cntc] = (c, Sk2[np.argmin(wG1)], Sk2[np.argmin(wG2)])
        weighted_ginis_1[cntc] = np.min(wG1) + np.min(wG2)

    # if minimum of gini = 0 reached, don't test other way around
    if np.min(weighted_ginis_1) == 0:
        mins = np.where(weighted_ginis_1 == np.min(weighted_ginis_1))[0]
        min_tuple = tuples_1[mins[-1]]

        sf = feature[0]

    # test it other way around (Feature 2/Feature 1)
    else:
        Sk1 = np.linspace(np.min(df[:, 1]), np.max(df[:, 1]), d)
        Sk2 = np.linspace(np.min(df[:, 0]), np.max(df[:, 0]), d)
        tuples_2 = np.empty(d, dtype=tuple)
        weighted_ginis_2 = np.empty(d)

        for cntc, c in enumerate(Sk1):
            wG1 = np.zeros(len(Sk2))
            wG2 = np.zeros(len(Sk2))
            df_l = df[df[:, 0] <= c]
            df_r = df[df[:, 0] > c]
            for cnt, d in enumerate(Sk2):
                df_ll = df_l[df_l[:, 1] <= d]
                df_lr = df_l[df_l[:, 1] > d]

                wG1[cnt] = (len(df_ll) / len(df)) * compute_gini_np(df_ll, target_expression, class_cost) + (
                    len(df_lr) / len(df)
                ) * compute_gini_np(df_lr, target_expression, class_cost)

            for cnt, e in enumerate(Sk2):
                df_rl = df_r[df_r[:, 1] <= e]
                df_rr = df_r[df_r[:, 1] < e]

                wG2[cnt] = (len(df_rl) / len(df)) * compute_gini_np(df_rl, target_expression, class_cost) + (
                    len(df_rr) / len(df)
                ) * compute_gini_np(df_rr, target_expression, class_cost)

            tuples_2[cntc] = (c, Sk2[np.argmin(wG1)], Sk2[np.argmin(wG2)])
            weighted_ginis_2[cntc] = np.min(wG1) + np.min(wG2)

        if np.min(weighted_ginis_1) <= np.min(weighted_ginis_2):
            weighted_ginis = weighted_ginis_1
            tuples = tuples_1
        else:
            weighted_ginis = weighted_ginis_2
            tuples = tuples_2

        sf = feature[1]

        mins = np.where(weighted_ginis == np.min(weighted_ginis))[0]
        min_tuple = tuples[mins[int(len(mins) / 2)]]

    return sf, min_tuple[0]


def spi_mixed(
    df: pd.DataFrame,
    feature: List[str],
    target: str,
    d: int,
    target_expression: pd.DataFrame,
    class_cost: np.ndarray
) -> Tuple[str, Union[str, float]]:
    """One categorical, one numerical feature selected"""
    numerical_feature = np.where(((df.dtypes[feature] == float).values))[0][0]
    cat_feature = 1 - numerical_feature
    columns = feature[:]
    columns.append(target)
    df = df[columns]
    df = df.to_numpy()

    # first stage
    Sk1 = np.linspace(
        np.min(df[:, numerical_feature]), np.max(df[:, numerical_feature]), d
    )

    feature_left = np.empty(len(Sk1), dtype=list)
    gini_left = np.ones(len(Sk1))
    feature_right = np.empty(len(Sk1), dtype=list)
    gini_right = np.ones(len(Sk1))

    for cnt, c in enumerate(Sk1):
        df_l = df[df[:, numerical_feature] <= c]
        df_r = df[df[:, numerical_feature] > c]

        if len(df_l) > 0 and len(df_r) > 0:
            feature_l, gini_l = spi_mixed_splitting_01(df_l, cat_feature, target_expression, class_cost)
            feature_r, gini_r = spi_mixed_splitting_01(df_r, cat_feature, target_expression, class_cost)

            feature_left[cnt] = feature_l
            gini_left[cnt] = gini_l
            feature_right[cnt] = feature_r
            gini_right[cnt] = gini_r

    cstar = Sk1[np.argmin(gini_left + gini_right)]
    delta1 = np.min(gini_left + gini_right)

    # second stage part 1
    df_l = df[df[:, numerical_feature] <= cstar]
    df_r = df[df[:, numerical_feature] > cstar]

    if (
        len(np.unique(df_l[:, cat_feature])) == 1
        or len(np.unique(df_r[:, cat_feature])) == 1
    ):
        return feature[numerical_feature], cstar

    else:
        U, delta2 = spi_mixed_splitting_02(
            df=df_l,
            cat_feature=cat_feature,
            numerical_feature=numerical_feature,
            csplits=Sk1,
            target_expression=target_expression,
            class_cost=class_cost,
        )

        V, delta3 = spi_mixed_splitting_02(
            df=df_r,
            cat_feature=cat_feature,
            numerical_feature=numerical_feature,
            csplits=Sk1,
            target_expression=target_expression,
            class_cost=class_cost,
        )

        if delta1 <= min(delta2, delta3):
            sf = feature[numerical_feature]
            sp = cstar

        else:
            sf = feature[cat_feature]
            if delta3 > delta2:
                sp = V
            else:
                sp = U

        try:
            return sf, sp[0]
        except IndexError:
            return sf, sp


def spi_categorical(
    df: pd.DataFrame,
    feature: List[str],
    target: str,
    target_expression: pd.DataFrame,
    class_cost: np.ndarray
) -> Tuple[str, np.ndarray]:
    """Two categorical features selected"""
    columns = feature[:]
    columns.append(target)

    df = df[columns]
    df = df.to_numpy()

    sp1, delta1 = spi_categorical_splitting(df=df, feature1=0, feature2=1, target_expression=target_expression,
                                            class_cost=class_cost)

    if delta1 != 0:
        # do splitting the other way around only if no optimal solution is reached
        sp2, delta2 = spi_categorical_splitting(df=df, feature1=1, feature2=0, target_expression=target_expression,
                                                class_cost=class_cost)

        if delta1 <= delta2:
            return feature[0], sp1

        else:
            return feature[1], sp2

    else:
        return feature[0], sp1


def sp_interaction_features(
    df: pd.DataFrame,
    feature: List[str],
    target: str,
    target_expression: pd.DataFrame,
    class_cost: np.ndarray
) -> Tuple[Optional[str], Optional[Union[str, float, np.ndarray]]]:
    """Search for split point of interaction search"""
    split_feature: Optional[str]
    split_point: Optional[Union[str, float, np.ndarray]]
    d = 100
    try:
        if issubclass(df.dtypes[feature[0]].type, numbers.Number) and issubclass(
            df.dtypes[feature[1]].type, numbers.Number
        ):
            split_feature, split_point = spi_numerical(df, feature, target, d, target_expression, class_cost)

        elif len(df.dtypes[feature].value_counts()) == 2:
            # one categorical and one numerical feature
            split_feature, split_point = spi_mixed(df, feature, target, d, target_expression, class_cost)

        elif df.dtypes[feature[0]] == object and df.dtypes[feature[1]] == object:
            # two categorical features
            split_feature, split_point = spi_categorical(df, feature, target, target_expression, class_cost)

            if len(split_point) == 1:
                split_point = split_point[0]
        else:
            split_feature = None
            split_point = None
    except ValueError:
        print("here")

    return split_feature, split_point
