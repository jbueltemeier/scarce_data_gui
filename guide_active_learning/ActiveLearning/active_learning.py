import warnings
from typing import Any, cast, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.spatial.distance
from guide_active_learning.core import one_hot_enc

from guide_active_learning.ActiveLearning.query import (
    qbc_distance,
    qbc_exex,
    qbc_exex_2,
    qbc_exex_2cat,
    qbc_exex_rf,
    qbc_exex_rf_reg,
    qbc_uncertainties_query,
    rf_uncertainty_query,
)
from guide_active_learning.core import save_active_learning_pickle
from guide_active_learning.GUIDE import DecisionTreeClassifierGUIDE, GUIDEEnsemble, M5PrimeTreeClassifier

from guide_active_learning.misc import make_output_filename
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    "ActiveLearning",
    "calculate_max_distance",
    "active_learning_step",
]


class ActiveLearning:
    """Class for implementation of different active learning scenarios"""

    def __init__(
        self,
        single_model: DecisionTreeClassifierGUIDE,
        guide_ensemble: Union[GUIDEEnsemble, RandomForestClassifier, None],
        unlabeled_pool: pd.DataFrame,
        max_distance: float,
        active_learning_scenario: str,
        alpha_weight: np.ndarray,
        random_state: Optional[int] = None,
        num_snowballs: int = 512,
        active_learning_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        extrapolation_factor: float = 0.1,
        pool_synth: bool = True,
    ):
        self.result_qbc_exex: Optional[Dict[str, Any]] = None
        self.single_model = single_model
        self.guide_ensemble = guide_ensemble
        self.unlabeled_pool = unlabeled_pool
        self.max_distance = max_distance
        self.active_learning_scenario = active_learning_scenario
        self.random_state = random_state
        self.alpha_weight = alpha_weight
        self.num_snowballs = num_snowballs
        self.active_learning_bounds = active_learning_bounds
        self.extrapolation_factor = extrapolation_factor
        self.pool_synth = pool_synth

        # active learning computations
        self.__request_label = self._select_query()

    def _select_query(self) -> Union[pd.Series, pd.DataFrame]:
        if self.random_state is None:
            self.random_state = 1

        if self.guide_ensemble is None:  # only for scenario "random"
            self.unlabeled_pool.sample(1, random_state=self.random_state)

        if "uncertainty_guide" in self.active_learning_scenario:
            return qbc_uncertainties_query(
                active_learning_scenario=self.active_learning_scenario,
                guide_ensemble=cast(GUIDEEnsemble, self.guide_ensemble),
                unlabeled_pool=self.unlabeled_pool,
            )
        elif "uncertainty_rf" in self.active_learning_scenario:
            return rf_uncertainty_query(
                active_learning_scenario=self.active_learning_scenario,
                ensemble=self.guide_ensemble,
                unlabeled_pool=self.unlabeled_pool,
            )

        elif self.active_learning_scenario == "distance":
            return qbc_distance(
                single_model=self.single_model,
                unlabeled_pool=self.unlabeled_pool,
                max_distance=self.max_distance,
            )

        elif self.active_learning_scenario == "qbc_exex":
            dp, self.result_qbc_exex = qbc_exex(
                single_model=self.single_model,
                unlabeled_pool=self.unlabeled_pool,
                random_state=self.random_state,
                num_snowballs=self.num_snowballs,
                guide_ensemble=cast(GUIDEEnsemble, self.guide_ensemble),
                alpha_weight=self.alpha_weight,
                active_learning_bounds=self.active_learning_bounds,
                extrapolation_factor=self.extrapolation_factor,
                max_distance=self.max_distance,
            )
            return dp

        elif self.active_learning_scenario == "qbc_exex_2":
            dp, self.result_qbc_exex = qbc_exex_2(
                single_model=self.single_model,
                unlabeled_pool=self.unlabeled_pool,
                random_state=self.random_state,
                num_snowballs=self.num_snowballs,
                guide_ensemble=cast(GUIDEEnsemble, self.guide_ensemble),
                alpha_weight=self.alpha_weight,
                active_learning_bounds=self.active_learning_bounds,
                pool_synth=self.pool_synth,
                max_distance=self.max_distance,
            )
            return dp

        elif self.active_learning_scenario == "qbc_exex_2cat":
            dp, self.result_qbc_exex = qbc_exex_2cat(
                single_model=self.single_model,
                unlabeled_pool=self.unlabeled_pool,
                random_state=self.random_state,
                num_snowballs=self.num_snowballs,
                guide_ensemble=cast(GUIDEEnsemble, self.guide_ensemble),
                alpha_weight=self.alpha_weight,
                active_learning_bounds=self.active_learning_bounds,
                pool_synth=self.pool_synth,
                max_distance=self.max_distance,
            )
            return dp

        elif self.active_learning_scenario == "qbc_exex_rf":
            dp, self.result_qbc_exex = qbc_exex_rf(
                single_model=self.single_model,
                unlabeled_pool=self.unlabeled_pool,
                random_state=self.random_state,
                num_snowballs=self.num_snowballs,
                ensemble=cast(RandomForestClassifier, self.guide_ensemble),
                alpha_weight=cast(float, self.alpha_weight),
                active_learning_bounds=self.active_learning_bounds,
                pool_synth=self.pool_synth,
                max_distance=self.max_distance,
            )
            return dp

        elif self.active_learning_scenario == "qbc_exex_rf_reg":
            dp, self.result_qbc_exex = qbc_exex_rf_reg(
                unlabeled_pool=self.unlabeled_pool,
                ensemble=cast(RandomForestClassifier, self.guide_ensemble),
            )
            return dp

        else:  # self.active_learning_scenario == "random":
            if self.active_learning_scenario != "random":
                warnings.warn(
                    f"Active Learning Scenario "
                    f"{self.active_learning_scenario} is not implemented. "
                    f"Scenario 'random' is used."
                )
            return self.unlabeled_pool.sample(1, random_state=self.random_state)

    def return_request_label(self) -> pd.Series:
        return self.__request_label


# TODO: Ab hier muss noch alles angepasst werden (Prüfen!!!!)


def calculate_max_distance(df: pd.DataFrame) -> float:
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    scaler = MinMaxScaler()
    normed_al_df = scaler.fit_transform(df.select_dtypes(include="number").to_numpy())
    return cast(
        float, np.max(scipy.spatial.distance.pdist(normed_al_df, metric="euclidean"))
    )


def active_learning_step(
    *,
    train_df: pd.DataFrame,
    df_unlabeled_pool: pd.DataFrame,
    foldername: str,
    max_distance: float,
    iteration: int = 0,
    alpha: float = 0.8,  # TODO: reduce alpha with every iteration
    ensemble_size: int = 100,
    max_depth: int = 10,
    min_info_gain: float = 0.075,
    use_linear_split: bool = True,
    split_type: str = "mixed",
    active_learning_method: str = "qbc_exex_rf",
    pool_synth: bool = False,
    calculate_ensemble: bool = True,
    save_active_learning: bool = False,
) -> pd.Series:
    if active_learning_method.endswith("_reg"):
        dtc = M5PrimeTreeClassifier()
        dtc.fit(train_df)
        use_regression = True
    else:
        dtc = DecisionTreeClassifierGUIDE(
            max_depth=max_depth,
            use_linear_split=use_linear_split,
            split_type=split_type,
            min_info_gain=min_info_gain,
        )
        dtc.fit(train_df, target="target")
        use_regression = False

    if calculate_ensemble and (active_learning_method.endswith("rf") or active_learning_method.endswith("rf_reg")):
        # print("RandomForest")
        X_train, y_train = one_hot_enc(train_df)
        ens = RandomForestClassifier(n_estimators=ensemble_size)
        ens.fit(X_train, y_train)
    elif calculate_ensemble:
        # print("Ensemble")
        ens = GUIDEEnsemble(
            max_depth=max_depth,
            use_linear_split=use_linear_split,
            split_type=split_type,
            min_info_gain=min_info_gain,
        )
        ens.fit(train_df, ensemble_size=ensemble_size, use_regression=use_regression)

    else:
        ens = None

    # print("Active Learning")
    al = ActiveLearning(
        single_model=dtc,
        guide_ensemble=ens,
        unlabeled_pool=df_unlabeled_pool,
        max_distance=max_distance,
        active_learning_scenario=active_learning_method,
        alpha_weight=np.array(alpha),
        pool_synth=pool_synth,
    )

    if save_active_learning:
        filename_ = make_output_filename(
            "gea", "exex_al", "result", f"step{str(iteration)}", extension=".pickle"
        )
        save_active_learning_pickle(foldername, filename=filename_, data=al)

    return al.return_request_label()
