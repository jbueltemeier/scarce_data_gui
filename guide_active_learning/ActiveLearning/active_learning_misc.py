from typing import Any, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats.qmc as qmc
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from guide_active_learning.core import compute_lda, one_hot_enc

from guide_active_learning.GUIDE.guide_ensemble import GUIDEEnsemble
from guide_active_learning.GUIDE.guide_tree import DecisionTreeClassifierGUIDE, TreeNode

__all__ = [
    "compute_distance",
    "cartesian",
    "compute_threshold_distance",
    "create_sobol_sequence",
    "create_numerical_sobol_samples",
    "add_categorical_to_samples",
    "receive_tree_nodes",
    "qbc_exec_loop",
    "calc_uncertainty",
]


def compute_distance(
    df_train: pd.DataFrame,
    df_unlabeled: pd.DataFrame,
    max_distance: float,
    relevant_datapoints: pd.DataFrame = None,
    norm: int = 2,
) -> np.ndarray:
    # separate numeric and categorical features
    if "target" in df_train.columns:
        df_train = df_train.drop(columns="target")
    if "target" in df_unlabeled.columns:
        df_unlabeled = df_unlabeled.drop(columns="target")
    train_numerical = df_train.select_dtypes(include="number").to_numpy()
    train_categorical = df_train.select_dtypes(include="object").to_numpy()
    unlabeled_numerical = df_unlabeled.select_dtypes(include="number").to_numpy()
    unlabeled_categorical = df_unlabeled.select_dtypes(include="object").to_numpy()
    if relevant_datapoints is not None:
        if "target" in relevant_datapoints.columns:
            relevant_datapoints = relevant_datapoints.drop(columns="target")
        relevant_numerical = relevant_datapoints.select_dtypes(
            include="number"
        ).to_numpy()
        relevant_categorical = relevant_datapoints.select_dtypes(
            include="object"
        ).to_numpy()

    # scale numerical features to [0, 1]
    scaler = MinMaxScaler()
    scaler.fit(np.vstack((train_numerical, unlabeled_numerical)))
    train_numerical = scaler.transform(train_numerical)
    if relevant_datapoints is not None:
        unlabeled_numerical = scaler.transform(relevant_numerical)
        unlabeled_categorical = relevant_categorical
    else:
        unlabeled_numerical = scaler.transform(unlabeled_numerical)

    # compute numerical distance between unlabeled pool and training data
    # (normalized by max_distance)
    numerical_distances = (
        cdist(unlabeled_numerical, train_numerical, "minkowski", p=norm) / max_distance
    )

    # compute categorical distances
    if unlabeled_categorical.size > 0:
        categorical_distances = np.ones(
            (len(unlabeled_categorical), len(train_categorical))
        )
        for j in range(len(unlabeled_categorical)):
            intersection = np.sum(
                np.isin(train_categorical, unlabeled_categorical[j, :]), axis=1
            )
            union = np.array(
                [
                    len(
                        np.union1d(train_categorical[i, :], unlabeled_categorical[j, :])
                    )
                    for i in range(len(train_categorical))
                ]
            )

            categorical_distances[j, :] = 1 - intersection / union
        # combine distances
        combined_distances = numerical_distances + categorical_distances
    else:
        combined_distances = numerical_distances

    return cast(np.ndarray, combined_distances)


def cartesian(arrays: List[np.ndarray], out: Optional[np.ndarray] = None) -> np.ndarray:
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


def compute_threshold_distance(
    node: TreeNode,
    df_test: pd.DataFrame,
    df_train: pd.DataFrame,
    df_unlabeled: pd.DataFrame,
    max_distance: float,
    norm: int = 2,
) -> Union[str, np.ndarray]:
    feature = node.feature
    threshold = node.threshold
    distances = None

    if isinstance(threshold, float) and not isinstance(
        feature, list
    ):  # univariate split
        whole_samples = np.hstack((df_train[feature], df_unlabeled[feature]))
        min_val = np.min(whole_samples)
        max_val = np.max(whole_samples)
        scaled_queries = (df_test[feature] - min_val) / (max_val - min_val)
        scaled_threshold = (threshold - min_val) / (max_val - min_val)
        distances = (abs(np.array(scaled_queries) - scaled_threshold)) / max_distance

    if isinstance(node.feature, list):  # multivariate split
        whole_samples = compute_lda(
            df=pd.concat([df_train, df_unlabeled]),
            feature=cast(List[str], feature),
            target="target",
        )
        whole_samples = cast(pd.DataFrame, whole_samples).iloc[:, 0]
        min_val = np.min(whole_samples)
        max_val = np.max(whole_samples)
        lda_sample = compute_lda(
            df=df_test, feature=cast(List[str], feature), target="target"
        )
        scaled_queries = (lda_sample.iloc[:, 0] - min_val) / (max_val - min_val)
        scaled_threshold = (threshold - min_val) / (max_val - min_val)
        distances = (abs(np.array(scaled_queries) - scaled_threshold)) / max_distance

    if distances is None:
        return "categorical"

    return cast(np.ndarray, distances)


def create_sobol_sequence(
    *,
    num_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    random_state: int,
    n_samples: int,
    active_learning_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> pd.DataFrame:
    # compute bounds of dataset if no bounds are given
    bounds = (
        (
            np.array(np.min(num_df, axis=0)) - abs(np.array(np.min(num_df, axis=0))),
            np.array(np.max(num_df, axis=0)) + abs(np.array(np.max(num_df, axis=0))),
        )
        if active_learning_bounds is None
        else active_learning_bounds
    )

    cat_array = cat_df.to_numpy()
    if cat_array.size == 0:
        # no categorical features are present and only one Sobol Sequence is drawn
        sobol = qmc.Sobol(d=num_df.shape[-1], seed=random_state)
        sobol_samples = sobol.random_base2(int(np.ceil(np.log2(n_samples))))
        samples_ready = qmc.scale(sobol_samples, bounds[0], bounds[1])
    else:
        # compute unique categorical value-combinations
        uniques = cast(
            List[np.ndarray],
            tuple(
                [list(np.unique(cat_array[:, i])) for i in range(cat_array.shape[-1])]
            ),
        )
        unique_categories = create_categorical_uniques_array(uniques)
        num_unique_categories = len(unique_categories)
        # determine the number sobol samples that must be drawn per
        # unique feature combination
        samples_per_sobol = int(np.ceil(np.log2(n_samples / num_unique_categories)))
        for i, unique_category in enumerate(unique_categories):
            # sample and scale Sobol Sequence
            sobol = qmc.Sobol(d=num_df.shape[-1], seed=i * random_state)
            try:
                sobol_samples = sobol.random_base2(samples_per_sobol)
            except TypeError:
                print("error")
            sobol_samples = qmc.scale(sobol_samples, bounds[0], bounds[1])
            # create the combined samples-array
            if i == 0:
                samples_ready = np.hstack(
                    (
                        sobol_samples,
                        np.tile(unique_categories[i, :], (len(sobol_samples), 1)),
                    )
                )
            else:
                tmp_cat = np.hstack(
                    (
                        sobol_samples,
                        np.tile(unique_categories[i, :], (len(sobol_samples), 1)),
                    )
                )
                samples_ready = np.vstack((samples_ready, tmp_cat))

    # create a dataframe again
    columns_all = list(num_df.columns)
    columns_all.extend(list(cat_df.columns))
    samples_ready = pd.DataFrame(samples_ready, columns=columns_all)
    samples_ready["target"] = [None] * len(samples_ready)
    samples_ready = samples_ready.astype(
        dict(zip(num_df.columns, [float] * len(num_df.columns)))
    )
    return samples_ready


def create_numerical_sobol_samples(
    *,
    num_df: pd.DataFrame,
    random_state: int,
    num_snowballs: int,
    active_learning_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    extrapolation_factor: float = 0.0,
) -> np.ndarray:
    sobol = qmc.Sobol(d=num_df.shape[-1], seed=random_state)
    samples = sobol.random(num_snowballs)

    bounds = (
        (
            np.array(np.min(num_df, axis=0))
            - abs(np.array(np.min(num_df, axis=0))) * extrapolation_factor,
            np.array(np.max(num_df, axis=0))
            + abs(np.array(np.max(num_df, axis=0))) * extrapolation_factor,
        )
        if active_learning_bounds is None
        else active_learning_bounds
    )
    return cast(np.ndarray, qmc.scale(samples, bounds[0], bounds[1]))


def create_categorical_uniques_array(
    uniques: List[np.ndarray], uniques_array_creation: str = "meshgrid"
) -> np.ndarray:
    if uniques_array_creation == "meshgrid":
        return np.array(np.meshgrid(*list(uniques))).T.reshape(-1, len(uniques))
    else:  # tmp_array_str == "cartesian":
        return cartesian(uniques)


def add_categorical_to_samples(
    *,
    categories: pd.DataFrame,
    samples: np.ndarray,
    columns: List[str],
    uniques_array_creation: str = "meshgrid",
) -> pd.DataFrame:
    samples_ready = samples
    cat_np = np.array(categories)
    uniques = cast(
        List[np.ndarray],
        tuple([list(np.unique(cat_np[:, i])) for i in range(cat_np.shape[-1])]),
    )
    if len(uniques) > 0:
        tmp = create_categorical_uniques_array(
            uniques, uniques_array_creation=uniques_array_creation
        )
        for i in range(len(tmp)):
            if i == 0:
                samples_ready = np.hstack(
                    (samples, np.tile(tmp[i, :], (len(samples), 1)))
                )
            else:
                tmp_cat = np.hstack((samples, np.tile(tmp[i, :], (len(samples), 1))))
                samples_ready = np.vstack((samples_ready, tmp_cat))

    columns_all = list(columns)
    columns_all.extend(list(categories.columns))

    samples_ready = pd.DataFrame(samples_ready, columns=columns_all)
    samples_ready["target"] = [None] * len(samples_ready)
    samples_ready = samples_ready.astype(dict(zip(columns, [float] * len(columns))))
    return samples_ready


def receive_tree_nodes(
    df: pd.DataFrame,
    single_model: DecisionTreeClassifierGUIDE,
    ensemble: Union[GUIDEEnsemble, RandomForestClassifier, None],
    synth_pool: pd.DataFrame = None,
) -> Tuple[List[TreeNode], np.ndarray, np.ndarray]:
    if synth_pool.empty:
        single_model.run_through_tree(df=df)
    else:
        single_model.run_through_tree(df=synth_pool)

    nodes = single_model.return_all_nodes(only_split=True)

    nodes_new = []
    for node in nodes:
        if synth_pool.empty:
            if not cast(pd.DataFrame, node.pool).empty:
                nodes_new.append(node)
        else:
            path_check = [
                single_model.ison_prediction_path(x=df.iloc[i, :], check_node=node)
                for i in range(len(df))
            ]
            if any(path_check):
                nodes_new.append(node)

    nodes = nodes_new

    if isinstance(ensemble, GUIDEEnsemble):
        uncertainties = np.array(
            [ensemble.predict_uncertainties(node.pool, averaged=True) for node in nodes]
        )
    elif isinstance(ensemble, RandomForestClassifier):
        uncertainties = np.array(
            [np.mean(calc_uncertainty(ensemble, node.pool)) for node in nodes]
        )
    else:
        raise NotImplementedError()

    volumes = np.array(
        [single_model.compute_rel_volume_node(node, df.shape[0]) for node in nodes]
    )
    return nodes, uncertainties, volumes


def qbc_exec_loop(
    *,
    nodes: List[TreeNode],
    uncertainties: np.ndarray,
    volumes: np.ndarray,
    single_model: DecisionTreeClassifierGUIDE,
    unlabeled_pool: pd.DataFrame,
    alpha_weight: np.ndarray,
    max_distance: float,
    ensemble: Union[GUIDEEnsemble, RandomForestClassifier, None],
) -> Tuple[pd.Series, Dict[str, Any]]:
    initVal = 2
    # unique_classes = np.unique(single_model.train_df["target"])
    uncertainties_std = uncertainties / np.max(uncertainties)
    volumes_std = volumes / np.max(volumes)

    # Calculate uncertainty for each sample of unlabeled pool
    if isinstance(ensemble, GUIDEEnsemble):
        uncertainty_query = np.array(ensemble.predict_uncertainties(unlabeled_pool))

    elif isinstance(ensemble, RandomForestClassifier):
        uncertainty_query = calc_uncertainty(ensemble, unlabeled_pool)

    else:
        raise NotImplementedError()
    uncertainty_query_std = uncertainty_query

    # Calculate NN-distance between samples of unlabeled pool and already selected
    # training samples
    distances = compute_distance(
        df_train=single_model.train_df,
        df_unlabeled=unlabeled_pool,
        relevant_datapoints=unlabeled_pool,
        max_distance=max_distance,
    )
    min_dp_distances = np.min(distances, axis=1)
    min_dp_distances_std = min_dp_distances

    al_exex_res = {
        "nodes": nodes,
        "uncertainties_std": uncertainties_std,
        "volumes_std": volumes_std,
    }

    # Run a loop over all nodes to assign real pool indexes and the
    # corresponding uncertainty
    column_names = [
        "vol_per_sample",
        "av_uncertainty",
        "nn_dist",
        "sample_uncertainty",
        "threshold_dist",
    ]
    for k in range(len(nodes)):
        path_check = [
            single_model.ison_prediction_path(
                x=unlabeled_pool.iloc[i, :], check_node=nodes[k]
            )
            for i in range(len(unlabeled_pool))
        ]

        # select unlabeled pool with real index numbers
        nodes[k].unl_pool = unlabeled_pool[path_check]

        # Calculate metrics for each node:
        nodes[k].metrics = pd.DataFrame(
            initVal,
            index=range(len(cast(pd.DataFrame, nodes[k].unl_pool))),
            columns=column_names,
        )

        cast(pd.DataFrame, nodes[k].metrics)[column_names[0]] = volumes_std[k]
        cast(pd.DataFrame, nodes[k].metrics)[column_names[1]] = uncertainties_std[k]
        cast(pd.DataFrame, nodes[k].metrics)[column_names[2]] = min_dp_distances_std[
            path_check
        ]
        cast(pd.DataFrame, nodes[k].metrics)[column_names[3]] = uncertainty_query_std[
            path_check
        ]

        if type(nodes[k].threshold) != str:
            threshold_distances = compute_threshold_distance(
                node=nodes[k],
                df_test=nodes[k].unl_pool,
                df_train=single_model.train_df,
                df_unlabeled=unlabeled_pool,
                max_distance=1,
            )

            if type(threshold_distances) != str:
                # weird case of two categorical values instead of one and therefore the
                # first if fails to detect categorical splits
                scaler = MinMaxScaler()
                # TODO threshold distance can have the same values for all queries if
                #  partition is small and split feature
                threshold_distances_std = scaler.fit_transform(
                    cast(np.ndarray, threshold_distances).reshape(-1, 1)
                ).flatten()
                cast(pd.DataFrame, nodes[k].metrics)[
                    column_names[4]
                ] = threshold_distances_std

        if k == 0:
            data_pool = unlabeled_pool[path_check]
            metrics_pool = nodes[k].metrics
        else:
            data_pool = pd.concat([data_pool, unlabeled_pool[path_check]])
            metrics_pool = pd.concat([metrics_pool, nodes[k].metrics])

    scaler = MinMaxScaler()
    cast(pd.DataFrame, metrics_pool).iloc[:, 0:4] = scaler.fit_transform(
        cast(pd.DataFrame, metrics_pool).iloc[:, 0:4]
    )
    weighted_metrics = np.multiply(
        cast(pd.DataFrame, metrics_pool).to_numpy(), alpha_weight
    )  # weight the metric-matrix by specific factors to control exploration / exploitation

    if (
        weighted_metrics[
            np.argmax(np.sum(weighted_metrics[:, 0:4], axis=1), axis=0), -1
        ]
        == alpha_weight[-1] * initVal
    ):  # query in a node of a categorical split is best
        dp = data_pool.reset_index().iloc[
            np.argmax(np.sum(weighted_metrics[:, 0:4], axis=1), axis=0), :
        ]
    else:  # query in a node of a numerical or linear split is best
        dp = data_pool.reset_index().iloc[
            np.argmax(np.sum(weighted_metrics, axis=1), axis=0), :
        ]

    result_qbc_exex = {
        "al_exex_res": al_exex_res,
        # "datapoints": (relevant_datapoints, min_dp_distances),
    }

    return dp, result_qbc_exex

def calc_uncertainty(ensemble: RandomForestClassifier, df: pd.DataFrame) -> np.ndarray:
    # convert pool into input space on which ensemble was trained
    if df.empty:
        return cast(np.ndarray, np.nan)
    else:
        X_pool, y_pool = one_hot_enc(df)
        missing_features = set(ensemble.feature_names_in_) - set(X_pool.columns)
        for feature in missing_features:
            X_pool[feature] = False
        X_pool = X_pool[ensemble.feature_names_in_]
        # collect predictions from all trees
        all_predictions = np.zeros((X_pool.shape[0], ensemble.n_estimators))
        for i, tree in enumerate(ensemble.estimators_):
            all_predictions[:, i] = tree.predict(X_pool.to_numpy())
        # count the different predictions for each datapoint and compute
        # the relative occurrences of them
        probs = np.zeros((X_pool.shape[0], ensemble.n_classes_))
        for i, predictions in enumerate(all_predictions):
            counts = np.unique(predictions, return_counts=True)[1]
            if len(counts) < ensemble.n_classes_:
                probs[i] = (
                    np.append(counts, np.zeros(ensemble.n_classes_ - len(counts)))
                    / ensemble.n_estimators
                )
            else:
                probs[i] = counts / ensemble.n_estimators
        probs[probs == 0] = 1e-9  # logarithm 0 undefined

        return cast(np.ndarray, -np.sum(probs * np.log(probs), axis=1))
