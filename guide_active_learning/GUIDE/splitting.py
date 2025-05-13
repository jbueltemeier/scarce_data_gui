from itertools import combinations
from typing import Any, Callable, cast, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from guide_active_learning.core import (
    compute_gini_np,
    compute_wilsonhilferty,
    create_intervals,
)
from guide_active_learning.GUIDE.guide_misc import (
    sf_linear_lda_transform,
    sfi_extractinter,
)
from scipy.stats import chi2, chi2_contingency


__all__ = [
    "split_feature_linear",
    "linear_split",
    "spi_mixed_splitting_01",
    "spi_mixed_splitting_02",
    "spi_categorical_splitting",
    "univariate_split",
]


def split_feature_linear(df: pd.DataFrame, target: str) -> Dict[float, Any]:
    tmp_cols = list(df.select_dtypes(exclude=object).columns)
    if target not in tmp_cols:
        tmp_cols.append(target)
    df = df[:]
    df = df[tmp_cols]
    chi2_dict: Dict[float, Any] = {}

    for cnt1, col1 in enumerate(list(df.drop(columns=target).columns)):
        for cnt2, col2 in enumerate(list(df.drop(columns=[target, col1]).columns)):
            tmp_df = df.loc[:, [col1, col2, target]]
            lda_transformation = sf_linear_lda_transform(tmp_df)
            if lda_transformation == "singularity":
                chi2_dict.update({0: "singularity"})
            else:
                weights, df_lda = cast(
                    Tuple[np.ndarray, pd.DataFrame], lda_transformation
                )
                describing_array = [weights, np.array([col1, col2])]

                mean = df_lda.drop(columns=target).mean()
                std = df_lda.drop(columns=target).std()

                # TODO: check Interval calculation divider 2 and 3 given
                if len(df_lda) >= 20 * len(df[target].value_counts()):
                    intervals = create_intervals(
                        mean.values[0], std.values[0], num_intervals=4, divider=2
                    )

                else:
                    intervals = create_intervals(
                        mean.values[0], std.values[0], num_intervals=3
                    )

                df_lda["interval"] = pd.cut(
                    df_lda["LDA_feature"],
                    bins=intervals,
                    labels=False,
                    duplicates="drop",
                )
                cont_table = pd.crosstab(df_lda[target], df_lda["interval"])

                # delete empty rows and columns
                cont_table = cont_table.loc[(cont_table != 0).any(axis=1)]
                cont_table = cont_table.loc[:, (cont_table != 0).any(axis=0)]

                chi2_tmp = chi2_contingency(cont_table)
                wm = np.max([0, compute_wilsonhilferty(chi2_tmp)])

                chi2_dict.update({wm: describing_array})

    return chi2_dict


def linear_split(
    df: pd.DataFrame,
    target: str,
    interaction: Dict[str, float],
    main_effect: Dict[str, float],
    len_number_columns: int,
    beta: float,
    gamma: float,
) -> Union[str, List[str]]:
    if max(interaction.values()) > chi2.isf(beta, 1):
        relevant_feature = max(
            interaction, key=cast(Callable[[str], int], interaction.get)
        )
        return sfi_extractinter(df=df, relevant_feature=relevant_feature, target=target)
    elif len_number_columns == 1:
        return max(main_effect, key=cast(Callable[[str], int], main_effect.get))
    else:  # len_number_columns > 1:
        linear = split_feature_linear(df, target)
        if max(linear.keys()) > chi2.isf(gamma, 1):
            return cast(str, linear.get(max(list(linear.keys()))))
        else:
            return max(main_effect, key=cast(Callable[[str], int], main_effect.get))


def spi_mixed_splitting_01(
    df: pd.DataFrame,
    cat_feature: int,
    target_expression: pd.DataFrame,
    class_cost: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Supporting function for numerical and categorical features"""
    entries = np.unique(df[:, -1])
    values = np.unique(df[:, cat_feature])

    if len(entries) > 2:
        minimal_gini = 1.0
        minimal_class = ""
        for val in entries:
            gini = 1 - (len(df[df[:, -1] == val]) / len(df)) ** 2
            if gini < minimal_gini:
                minimal_gini = gini
                minimal_class = val

        df[:, -1][df[:, -1] == minimal_class] = "Superclass 2"

    if len(df) == 1:
        all_combs = [df[0, cat_feature]]
    else:
        all_combs = []
        if len(values) == 1:
            all_combs = [values[0]]
        else:
            for i in range(1, len(values)):
                all_combs.extend(combinations(list(values), i))

    combs = np.empty(len(all_combs), dtype=list)
    ginis = np.empty(len(all_combs))
    for cnt, A in enumerate(all_combs):
        df1 = df[np.isin(df[:, cat_feature], A)]
        df2 = df[~np.isin(df[:, cat_feature], A)]

        w_g = (len(df1) / len(df)) * compute_gini_np(df1, target_expression, class_cost) + (
            len(df2) / len(df)
        ) * compute_gini_np(df2, target_expression, class_cost)
        combs[cnt] = A
        ginis[cnt] = w_g

    return cast(np.ndarray, combs[np.argmin(ginis)]), cast(np.ndarray, np.min(ginis))


def spi_mixed_splitting_02(
    df: pd.DataFrame,
    cat_feature: int,
    numerical_feature: int,
    csplits: List[Any],
    target_expression: pd.DataFrame,
    class_cost: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Second supporting function for numerical and categorical features"""
    values = np.unique(df[:, cat_feature])

    all_combs: List[Any] = []
    for i in range(1, len(values)):
        all_combs.extend(combinations(list(values), i))

    combs1 = np.empty(len(all_combs), dtype=tuple)
    combs2 = np.ones((len(all_combs), 2))

    for enu, A in enumerate(all_combs):
        df1 = df[np.isin(df[:, cat_feature], A)]
        df2 = df[~np.isin(df[:, cat_feature], A)]

        w_gs = np.ones(len(csplits))
        for cnt, c in enumerate(csplits):
            df_1l = df1[df1[:, numerical_feature] <= c]
            df_1r = df1[df1[:, numerical_feature] > c]
            df_2l = df2[df2[:, numerical_feature] <= c]
            df_2r = df2[df2[:, numerical_feature] > c]

            w_g = (
                (len(df_1l) / len(df)) * compute_gini_np(df_1l, target_expression, class_cost)
                + (len(df_1r) / len(df)) * compute_gini_np(df_1r, target_expression, class_cost)
                + (len(df_2l) / len(df)) * compute_gini_np(df_2l, target_expression, class_cost)
                + (len(df_2r) / len(df)) * compute_gini_np(df_2r, target_expression, class_cost)
            )

            w_gs[cnt] = w_g

        combs1[enu] = A
        combs2[enu, 0] = csplits[np.argmin(w_gs)]
        combs2[enu, 1] = np.min(w_gs)

    return combs1[np.argmin(combs2[:, 1])], np.min(combs2[:, 1])


def spi_categorical_splitting(
    df: pd.DataFrame,
    feature1: int,
    feature2: int,
    target_expression: pd.DataFrame,
    class_cost: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Supporting function for two categorical features"""
    values = np.unique(df[:, feature1])

    all_combs: List[Any] = []
    for i in range(1, len(values)):
        all_combs.extend(combinations(list(values), i))

    outer_comb = np.empty(len(all_combs), dtype=tuple)
    outer_gini = np.empty(len(all_combs))
    for enum, comb in enumerate(all_combs):
        df1 = df[np.isin(df[:, feature1], comb)]
        df2 = df[~np.isin(df[:, feature1], comb)]

        combs_df1: List[Any] = []
        for i in range(1, len(np.unique(df1[:, feature2]))):
            combs_df1.extend(combinations(list(np.unique(df1[:, feature2])), i))

        combs1 = np.empty(len(combs_df1), dtype=tuple)
        if len(combs1) > 0:
            wgini1 = np.zeros(len(combs_df1))
        else:
            wgini1 = np.zeros(1)
        for cnt, A in enumerate(combs_df1):
            df_1l = df1[np.isin(df1[:, feature2], A)]
            df_1r = df1[~np.isin(df1[:, feature2], A)]

            w_g = (len(df_1l) / len(df)) * compute_gini_np(df_1l, target_expression, class_cost) + (
                len(df_1r) / len(df)
            ) * compute_gini_np(df_1r, target_expression, class_cost)

            combs1[cnt] = A
            wgini1[cnt] = w_g

        combs_df2: List[Any] = []
        for i in range(1, len(np.unique(df2[:, feature2]))):
            combs_df2.extend(combinations(list(np.unique(df2[:, feature2])), i))

        combs2 = np.empty(len(combs_df2), dtype=tuple)
        if len(combs2) > 0:
            wgini2 = np.zeros(len(combs_df2))
        else:
            wgini2 = np.zeros(1)
        for cnt, A in enumerate(combs_df2):
            df_2l = df2[np.isin(df2[:, feature2], A)]
            df_2r = df2[~np.isin(df2[:, feature2], A)]

            w_g = (len(df_2l) / len(df)) * compute_gini_np(df_2l, target_expression, class_cost) + (
                len(df_2r) / len(df)
            ) * compute_gini_np(df_2r, target_expression, class_cost)

            combs2[cnt] = A
            wgini2[cnt] = w_g

        outer_comb[enum] = comb
        outer_gini[enum] = np.min(wgini1) + np.min(wgini2)

    sp = outer_comb[np.argmin(outer_gini)]
    gini = np.min(outer_gini)

    return cast(np.ndarray, sp), cast(np.ndarray, gini)


def univariate_split(
    df: pd.DataFrame,
    target: str,
    interaction: Dict[str, float],
    main_effect: Dict[str, float],
    beta: float,
) -> Union[str, List[str]]:
    if max(interaction.values()) > chi2.isf(beta, 1):
        relevant_feature = max(
            interaction, key=cast(Callable[[str], int], interaction.get)
        )
        return sfi_extractinter(df=df, relevant_feature=relevant_feature, target=target)
    else:
        return max(main_effect, key=cast(Callable[[str], int], main_effect.get))
