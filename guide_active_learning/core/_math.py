import math
from typing import Any, cast, Dict, List, Tuple, Union

import numpy as np
import pandas as pd


__all__ = [
    "cm2inch",
    "calculate_column_beta",
    "calculate_column_gamma",
    "calculate_mean_std",
    "compute_wilsonhilferty",
    "compute_gini_np",
    "compute_gini_pd",
    "compute_information_gain",
    "compute_lda",
    "create_intervals",
    "calculate_contribution_table",
    "linear_lda_transform",
]


def cm2inch(value: Union[int, float]) -> float:
    return value * 0.3937


def calculate_column_beta(len_columns: int) -> float:
    return 0.05 / (len_columns * (len_columns - 1)) if len_columns > 1 else 0


def calculate_column_gamma(len_number_columns: int) -> float:
    return (
        0.05 / (len_number_columns * (len_number_columns - 1))
        if len_number_columns > 1
        else 0
    )


def calculate_mean_std(
    result_array: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # mean_scores = [np.mean(result_array[i], axis=0) for i in range(len(result_array))]
    mean_scores = np.mean(result_array, axis=0)
    # std_scores = [np.std(result_array[i], axis=0) for i in range(len(result_array))]
    std_scores = np.std(result_array, axis=0)
    return mean_scores, std_scores


def compute_wilsonhilferty(chi2_result: Dict[int, Any]) -> float:
    chi2 = chi2_result[0]
    deg = chi2_result[2]
    if deg != 0 and deg > 1:
        return cast(
            float,
            ((7 / 9) + math.sqrt(deg) * ((chi2 / deg) ** (1 / 3) - 1 + (2 / (9 * deg))))
            ** 3,
        )

    return cast(float, chi2) if deg != 0 and deg <= 1 else (7 / 9) ** 3


def compute_gini_np(df: np.ndarray, target_expression: pd.DataFrame, class_cost: np.ndarray) -> float:
    if df.size == 0:
        return 1.0  # Maximale Unsicherheit für leere Arrays

    # Wandelt target_expression (DataFrame) in eine flache Liste um
    target_classes = target_expression.iloc[:, 0].tolist()  # Falls es nur eine Spalte gibt

    # Zähle die Klassenhäufigkeiten, aber erzwinge die Reihenfolge von target_classes
    unique, counts = np.unique(df, return_counts=True)
    sum_classes = dict(zip(unique, counts))  # Erstellt ein Dictionary {Klasse: Häufigkeit}

    # Erstelle ein sortiertes Array mit den Häufigkeiten in target_classes-Reihenfolge
    sorted_counts = np.array([sum_classes.get(cls, 0) for cls in target_classes], dtype=float)

    probabilities = sorted_counts / df.size  # Wahrscheinlichkeiten berechnen

    # Erzeuge die Matrix mit P(i) * P(j) für alle Klassen
    prob_matrix = np.outer(probabilities, probabilities)

    # Berechnung der gewichteten Gini-Impurity
    gini_value = np.sum(class_cost * prob_matrix)

    return float(gini_value)

# def compute_gini_np(df: pd.DataFrame, target_expression: pd.DataFrame, class_cost: np.ndarray
# ) -> float:
#     if len(df) == 0:
#         return 1
#     else:
#         unique, counts = np.unique(df[:, -1], return_counts=True)
#         return cast(float, 1 - np.sum((counts / len(df)) ** 2))


def compute_gini_pd(
        df: pd.DataFrame, target_expression: pd.DataFrame, class_cost: np.ndarray
) -> float:
    target = "target"
    gini_value = 0.0

    # calculate numbered order of classes
    sum_classes = df[target].value_counts()
    sum_classes = sum_classes.reindex(target_expression, fill_value=0)

    for i in range(len(sum_classes)):
        for j in range(len(sum_classes)):
            sum_classes_i = sum_classes[i]
            sum_classes_j = sum_classes[j]

            gini_value = gini_value + class_cost[i, j] * (sum_classes_i/len(df)) * (sum_classes_j/len(df))

    return cast(float, gini_value)
    # return cast(float, 1 - sum((df[target].value_counts() / len(df)) ** 2))


def compute_information_gain(
    df_par: pd.DataFrame,
    df_c1: pd.DataFrame,
    df_c2: pd.DataFrame,
    target_expression: pd.DataFrame,
    class_cost: np.ndarray,
) -> float:
    gini_parent = compute_gini_pd(df_par, target_expression, class_cost)
    gini_c1 = compute_gini_pd(df_c1, target_expression, class_cost)
    gini_c2 = compute_gini_pd(df_c2, target_expression, class_cost)
    return (
        gini_parent
        - (len(df_c1) / len(df_par)) * gini_c1
        - (len(df_c2) / len(df_par)) * gini_c2
    )


def compute_lda(
    df: Union[pd.Series, pd.DataFrame],
    feature: Union[List[str], np.ndarray],
    target: str,
) -> pd.DataFrame:
    tmp = [str(i) for i in list(feature[1])]
    tmp.append(target)
    df = df[:]
    df = df[tmp]

    if type(df) == pd.DataFrame:  # multiple datapoints
        # check order of features
        if list(df.columns[:-1]) == list(feature[1]):
            lda_arr = df.drop(columns=target).to_numpy().dot(feature[0])

        else:
            lda_arr = df.drop(columns=target).to_numpy().dot(np.flip(feature[0]))

    elif type(df) == pd.Series:  # single datapoint
        # check order of features
        if np.all(df.index[:-1] == list(feature[1])):
            lda_arr = df.drop(index=target).to_numpy().dot(feature[0])

        else:
            lda_arr = df.drop(index=target).to_numpy().dot(np.flip(feature[0]))
        return lda_arr

    else:
        raise TypeError("compute_lda only with pandas series or dataframe")

    lda_df = pd.DataFrame(
        np.hstack((lda_arr.reshape(-1, 1), df[target].values.reshape(-1, 1))),
        columns=["LDA", target],
    )
    lda_df["LDA"] = pd.to_numeric(lda_df["LDA"])

    return lda_df


def create_intervals(
    mean: float, std: float, num_intervals: int = 3, divider: int = 3
) -> List[float]:
    intervals = [-math.inf, math.inf]

    if num_intervals % 2 == 0 or num_intervals == 2:
        intervals.append(mean)

    if num_intervals != 2:
        for i in range(1, math.ceil(num_intervals / 2)):
            intervals.append(mean - i * std * (math.sqrt(3) / divider))
            intervals.append(mean + i * std * (math.sqrt(3) / divider))

    return sorted(intervals)


def calculate_contribution_table(
    df: pd.DataFrame,
    target: str,
    column: str,
    use_interval: bool = False,
) -> pd.DataFrame:
    tmp = df.loc[:, [column, target]]

    if use_interval:
        mean = tmp.drop(columns=target).mean()
        std = tmp.drop(columns=target).std()
        if len(tmp) >= 20 * len(df[target].value_counts()):
            intervals = create_intervals(
                mean.values[0], std.values[0], num_intervals=4, divider=2
            )

        else:
            intervals = create_intervals(
                mean.values[0], std.values[0], num_intervals=3, divider=2
            )

        tmp["interval"] = pd.cut(
            tmp[column], bins=intervals, labels=False, duplicates="drop"
        )
        cont_table = pd.crosstab(tmp[target], tmp["interval"])
    else:
        cont_table = pd.crosstab(tmp[target], tmp[column])

    # delete empty rows and columns
    cont_table = cont_table.loc[(cont_table != 0).any(axis=1)]
    cont_table = cont_table.loc[:, (cont_table != 0).any(axis=0)]
    return cont_table


def linear_lda_transform(
    *, matrix_in: np.ndarray, matrix_out: np.ndarray, arr: np.ndarray, df: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    # starting with LDA from:
    # https://sebastianraschka.com/Articles/2014_python_lda.html#preparing-the-sample-data-set
    # other possibility sklearn: deviations between methods
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(matrix_in).dot(matrix_out))
    # project data on new axis
    max_eig_vec = eig_vecs[:, np.argmax(eig_vals)]  # changed
    if isinstance(max_eig_vec[0], complex) or isinstance(max_eig_vec[1], complex):
        max_eig_vec = np.array([i.real for i in max_eig_vec])

    arr_lda = arr[:, :-1].dot(max_eig_vec)
    tar = arr[:, -1].reshape(-1, 1)
    arr_lda = np.hstack((arr_lda.reshape(-1, 1), tar))

    df_lda = pd.DataFrame(data=arr_lda, columns=["LDA_feature", df.columns[-1]])
    return max_eig_vec, df_lda
