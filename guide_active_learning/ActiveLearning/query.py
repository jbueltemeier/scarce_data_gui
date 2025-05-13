from typing import Any, cast, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from guide_active_learning.ActiveLearning.active_learning_misc import (
    add_categorical_to_samples,
    calc_uncertainty,
    compute_distance,
    create_numerical_sobol_samples,
    create_sobol_sequence,
    qbc_exec_loop,
    receive_tree_nodes,
)
from guide_active_learning.GUIDE.guide_ensemble import GUIDEEnsemble
from guide_active_learning.GUIDE.guide_tree import DecisionTreeClassifierGUIDE
from sklearn.ensemble import RandomForestClassifier

__all__ = [
    "qbc_uncertainties_query",
    "rf_uncertainty_query",
    "qbc_distance",
    "qbc_exex",
    "qbc_exex_2",
    "qbc_exex_2cat",
    "qbc_exex_rf",
    "qbc_exex_rf_reg",
]


def qbc_uncertainties_query(
    *,
    active_learning_scenario: str,
    guide_ensemble: GUIDEEnsemble,
    unlabeled_pool: pd.DataFrame,
) -> pd.Series:
    uncertainties = np.array(guide_ensemble.predict_uncertainties(unlabeled_pool))
    args = np.argwhere(uncertainties == np.max(uncertainties)).reshape(1, -1)[0]
    datapoint_arg = np.array([np.random.choice(args)])
    return unlabeled_pool.iloc[datapoint_arg, :]


def rf_uncertainty_query(
    *,
    active_learning_scenario: str,
    ensemble: RandomForestClassifier,
    unlabeled_pool: pd.DataFrame,
) -> pd.Series:
    uncertainties = calc_uncertainty(ensemble, unlabeled_pool)
    args = np.argwhere(uncertainties == np.max(uncertainties)).reshape(1, -1)[0]
    datapoint_arg = np.array([np.random.choice(args)])
    return unlabeled_pool.iloc[datapoint_arg, :]


def qbc_distance(
    *,
    single_model: DecisionTreeClassifierGUIDE,
    unlabeled_pool: pd.DataFrame,
    max_distance: float,
) -> pd.Series:
    distances = compute_distance(
        df_train=single_model.train_df,
        df_unlabeled=unlabeled_pool,
        max_distance=max_distance,
    )
    min_dp_distances = np.min(distances, axis=1)
    args = np.argwhere(min_dp_distances == np.max(min_dp_distances))[0]
    return unlabeled_pool.iloc[args, :]


def qbc_exex(
    *,
    single_model: DecisionTreeClassifierGUIDE,
    unlabeled_pool: pd.DataFrame,
    random_state: int,
    num_snowballs: int,
    guide_ensemble: GUIDEEnsemble,
    alpha_weight: np.ndarray,
    active_learning_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    extrapolation_factor: float = 0.1,
    max_distance: float,
) -> Tuple[pd.Series, Optional[Dict[str, Any]]]:
    # TODO: Original leicht anders, nochmal genau prüfen
    num_df = cast(pd.DataFrame, single_model.train_df).select_dtypes(include="number")

    samples = create_numerical_sobol_samples(
        num_df=num_df,
        random_state=random_state,
        num_snowballs=num_snowballs,
        active_learning_bounds=active_learning_bounds,
        extrapolation_factor=extrapolation_factor,
    )

    categories = (
        cast(pd.DataFrame, single_model.train_df)
        .drop(columns="target")
        .select_dtypes(include="object")
    )

    if not categories.empty:
        samples_ready = add_categorical_to_samples(
            categories=categories,
            samples=samples,
            columns=num_df.columns,
            uniques_array_creation="cartesian",
        )
    else:
        columns = list(num_df.columns)
        samples_ready = pd.DataFrame(samples, columns=columns)
        samples_ready["target"] = [None] * len(samples_ready)
        samples_ready = samples_ready.astype(dict(zip(columns, [float] * len(columns))))

    nodes, uncertainties, volumes = receive_tree_nodes(
        df=samples_ready,
        single_model=single_model,
        ensemble=guide_ensemble,
    )

    if type(unlabeled_pool) == (pd.DataFrame or pd.Series):
        dp, result_qbc_exex = qbc_exec_loop(
            nodes=nodes,
            uncertainties=uncertainties,
            volumes=volumes,
            single_model=single_model,
            unlabeled_pool=unlabeled_pool,
            alpha_weight=alpha_weight,
            max_distance=max_distance,
            ensemble=guide_ensemble,
        )
    else:
        print("unlabeled pool None")
        result_qbc_exex = None
        distances = compute_distance(
            df_train=single_model.train_df,
            df_unlabeled=unlabeled_pool,
            max_distance=max_distance,
        )
        min_dp_distances = np.min(distances, axis=1)
        dp = samples_ready.reset_index().iloc[np.argmax(min_dp_distances), :]

    return dp, result_qbc_exex


def qbc_exex_2(
    *,
    single_model: DecisionTreeClassifierGUIDE,
    unlabeled_pool: pd.DataFrame,
    random_state: int,
    num_snowballs: int,
    guide_ensemble: GUIDEEnsemble,
    alpha_weight: np.ndarray,
    active_learning_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_distance: float,
    pool_synth: bool = True,
) -> Tuple[pd.Series, Optional[Dict[str, Any]]]:
    if pool_synth:
        samples_ready = create_sobol_sequence(
            num_df=pd.concat([single_model.train_df, unlabeled_pool]).select_dtypes(
                include="number"
            ),
            cat_df=pd.concat([single_model.train_df, unlabeled_pool])
            .drop(columns="target")
            .select_dtypes(include="object"),
            random_state=random_state,
            n_samples=num_snowballs,
            active_learning_bounds=active_learning_bounds,
        )
    else:
        samples_ready = unlabeled_pool

    nodes, uncertainties, volumes = receive_tree_nodes(
        df=samples_ready,
        single_model=single_model,
        ensemble=guide_ensemble,
    )

    if type(unlabeled_pool) == (pd.DataFrame or pd.Series):
        dp, result_qbc_exex = qbc_exec_loop(
            nodes=nodes,
            uncertainties=uncertainties,
            volumes=volumes,
            single_model=single_model,
            unlabeled_pool=unlabeled_pool,
            alpha_weight=alpha_weight,
            max_distance=max_distance,
            ensemble=guide_ensemble,
        )
    else:
        result_qbc_exex = None
        distances = compute_distance(
            df_train=single_model.train_df,
            df_unlabeled=unlabeled_pool,
            max_distance=max_distance,
        )
        min_dp_distances = np.min(distances, axis=1)
        dp = samples_ready.reset_index().iloc[np.argmax(min_dp_distances), :]
    return dp, result_qbc_exex


def qbc_exex_2cat(
    *,
    single_model: DecisionTreeClassifierGUIDE,
    unlabeled_pool: pd.DataFrame,
    random_state: int,
    num_snowballs: int,
    guide_ensemble: GUIDEEnsemble,
    alpha_weight: np.ndarray,
    active_learning_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    pool_synth: bool = True,
    max_distance: float,
) -> Tuple[pd.Series, Optional[Dict[str, Any]]]:
    return qbc_exex_2(
        single_model=single_model,
        unlabeled_pool=unlabeled_pool,
        random_state=random_state,
        num_snowballs=num_snowballs,
        guide_ensemble=guide_ensemble,
        alpha_weight=alpha_weight,
        active_learning_bounds=active_learning_bounds,
        max_distance=max_distance,
        pool_synth=pool_synth,
    )


def qbc_exex_rf(
    *,
    single_model: DecisionTreeClassifierGUIDE,
    unlabeled_pool: pd.DataFrame,
    random_state: int,
    num_snowballs: int,
    ensemble: RandomForestClassifier,
    alpha_weight: float,
    max_distance: float,
    active_learning_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    pool_synth: bool = True,
) -> Tuple[pd.Series, Optional[Dict[str, Any]]]:

    if pool_synth:
        samples_ready = create_sobol_sequence(
            num_df=pd.concat([single_model.train_df, unlabeled_pool]).select_dtypes(
                include="number"
            ),
            cat_df=pd.concat([single_model.train_df, unlabeled_pool])
            .drop(columns="target")
            .select_dtypes(include="object"),
            random_state=random_state,
            n_samples=num_snowballs,
            active_learning_bounds=active_learning_bounds,
        )

    else:
        samples_ready = pd.DataFrame([])

    nodes, uncertainties, volumes = receive_tree_nodes(
        df=unlabeled_pool,
        single_model=single_model,
        ensemble=ensemble,
        synth_pool=samples_ready,
    )

    if (
        type(unlabeled_pool) == (pd.DataFrame or pd.Series)
        and len(nodes) > 0
        and volumes.size > 0
        and uncertainties.size > 0
    ):
        dp, result_qbc_exex = qbc_exec_loop(
            nodes=nodes,
            uncertainties=uncertainties,
            volumes=volumes,
            single_model=single_model,
            unlabeled_pool=unlabeled_pool,
            alpha_weight=np.array(
                [
                    alpha_weight,
                    1 - alpha_weight,
                    alpha_weight,
                    1 - alpha_weight,
                    -(1 - alpha_weight),
                ]
            ),
            max_distance=max_distance,
            ensemble=ensemble,
        )
    else:
        result_qbc_exex = None
        distances = compute_distance(
            df_train=single_model.train_df,
            df_unlabeled=unlabeled_pool,
            max_distance=max_distance,
        )
        min_dp_distances = np.min(distances, axis=1)
        dp = unlabeled_pool.reset_index().iloc[np.argmax(min_dp_distances), :]

    return dp, result_qbc_exex


def qbc_exex_rf_reg(
    *,
    unlabeled_pool: pd.DataFrame,
    ensemble: RandomForestClassifier,
) -> Tuple[pd.Series, Optional[Dict[str, Any]]]:

    uncertainty_query = calc_uncertainty(ensemble, unlabeled_pool)
    dp = unlabeled_pool.reset_index().iloc[np.argmax(uncertainty_query), :]
    result_qbc_exex = None

    return dp, result_qbc_exex
