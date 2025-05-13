from copy import deepcopy
from time import time
from typing import Any, cast, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.spatial.distance

from guide_active_learning.ActiveLearning import ActiveLearning
from guide_active_learning.core import one_hot_enc

from guide_active_learning.core import load_dataset, save_pickle
from guide_active_learning.GUIDE import DecisionTreeClassifierGUIDE, GUIDEEnsemble

from guide_active_learning.misc import make_output_filename
from joblib import delayed, Parallel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


__all__ = [
    "perform_active_learning",
    "investigation_loop",
]


def perform_active_learning(
    *,
    full_df_: pd.DataFrame,
    active_learning_method: str,
    initial_datapoints: int,
    alpha_weight: Optional[np.ndarray] = None,
    target: str = "target",
    split_type: str = "mixed",
    max_depth: int = 10,
    use_linear_split: bool = True,
    min_info_gain: float = 0.001,
    ensemble_size: int = 100,
    num_benchmark: int = 10,
    active_learning_steps: int = 20,
    parallel_computation: int = 1,
    calculate_ensemble: bool = True,
    pool_synth: bool = True,
) -> List[Dict[int, Any]]:
    def run_active_learning(
        full_df: pd.DataFrame,
        random_state: int,
    ) -> Dict[int, Any]:
        alpha = deepcopy(alpha_weight) if alpha_weight is not None else None
        inner_result_dict = {}

        al_df = full_df.sample(frac=0.7, random_state=random_state)
        test_df = full_df[~full_df.isin(al_df)].dropna()

        train_df = al_df.sample(initial_datapoints, random_state=random_state)
        unlabeled = al_df[~al_df.isin(train_df)].dropna()

        # compute max distance in the dataset for normalizing distances
        scaler = MinMaxScaler()
        normed_al_df = scaler.fit_transform(
            al_df.drop(columns=[target]).select_dtypes(include="number").to_numpy()
        )
        max_distance = np.max(
            scipy.spatial.distance.pdist(normed_al_df, metric="euclidean")
        )

        for i in range(1, active_learning_steps + 1):
            # alpha = alpha**i if alpha is not None else None
            print(i)
            dtc = DecisionTreeClassifierGUIDE(
                max_depth=max_depth,
                use_linear_split=use_linear_split,
                split_type=split_type,
                min_info_gain=min_info_gain,
            )
            dtc.fit(train_df, target=target)

            if calculate_ensemble and active_learning_method.endswith("rf"):
                X_train, y_train = one_hot_enc(train_df)
                ens = RandomForestClassifier(n_estimators=ensemble_size)
                ens.fit(X_train, y_train)
            elif calculate_ensemble:
                ens = GUIDEEnsemble(
                    max_depth=max_depth,
                    use_linear_split=use_linear_split,
                    split_type=split_type,
                    min_info_gain=min_info_gain,
                )
                ens.fit(train_df, ensemble_size=ensemble_size)
            else:
                ens = None

            al = ActiveLearning(
                single_model=dtc,
                guide_ensemble=ens,
                unlabeled_pool=unlabeled,
                max_distance=max_distance,
                active_learning_scenario=active_learning_method,
                alpha_weight=cast(np.ndarray, alpha),
                pool_synth=pool_synth,
            )
            dp = al.return_request_label()
            inner_result_dict.update(
                {
                    i: {
                        "single_model": dtc,
                        "ensemble": ens,
                        "active_learning": al,
                        "score": dtc.score(test_df),
                    }
                }
            )

            if active_learning_method in [
                "qbc_exex",
                "qbc_exex_2",
                "qbc_exex_2cat",
                "qbc_exex_rf",
                "qbc_exex_rf_reg",
            ]:
                train_df.loc[dp[train_df.index.name], :] = dp.drop(train_df.index.name)
                unlabeled.drop(index=dp[train_df.index.name], inplace=True)
            else:
                train_df = pd.concat([train_df, dp])
                unlabeled.drop(dp.index, inplace=True)

            # alpha = alpha**2 if alpha is not None else None

        return inner_result_dict

    results = Parallel(n_jobs=parallel_computation)(
        delayed(run_active_learning)(full_df_[:], random_state=random_state_)
        for random_state_ in range(num_benchmark)
    )

    return cast(List[Dict[int, Any]], results)


def investigation_loop(
    *,
    dataset_str: str,
    benchmark_str: str,
    active_learning_method: str,
    initial_datapoints: List[int],
    alpha_weights: Optional[List[np.ndarray]] = None,
    split_type: str = "mixed",
    max_depth: int = 10,
    use_linear_split: bool = True,
    min_info_gain: float = 0.001,
    ensemble_size: int = 100,
    num_benchmark: int = 10,
    active_learning_steps: int = 20,
    parallel_computation: bool = False,
    calculate_ensemble: bool = True,
    pool_synth: bool = True,
) -> None:
    print(f"Start: {active_learning_method}")
    dataset = load_dataset(dataset_str)

    for initial_datapoint in initial_datapoints:
        if alpha_weights is None:
            start_time = time()
            results_dict = perform_active_learning(
                full_df_=dataset,
                active_learning_method=active_learning_method,
                initial_datapoints=initial_datapoint,
                split_type=split_type,
                max_depth=max_depth,
                use_linear_split=use_linear_split,
                min_info_gain=min_info_gain,
                ensemble_size=ensemble_size,
                num_benchmark=num_benchmark,
                active_learning_steps=active_learning_steps,
                parallel_computation=parallel_computation,
                calculate_ensemble=calculate_ensemble,
                pool_synth=pool_synth,
            )
            print(time() - start_time)
            filename = make_output_filename(
                dataset_str,
                "bm_res",
                active_learning_method,
                split_type,
                str(initial_datapoint),
                f"ens{str(ensemble_size)}",
                f"{str(min_info_gain).replace('.', '')}",
            )

            save_pickle(
                f"{benchmark_str}/{dataset_str}_results",
                results=results_dict,
                filename=filename,
                foldername="result",
            )
        else:
            for alpha_weight in alpha_weights:
                start_time = time()
                results_dict = perform_active_learning(
                    full_df_=dataset,
                    active_learning_method=active_learning_method,
                    alpha_weight=alpha_weight,
                    initial_datapoints=initial_datapoint,
                    split_type=split_type,
                    max_depth=max_depth,
                    use_linear_split=use_linear_split,
                    min_info_gain=min_info_gain,
                    ensemble_size=ensemble_size,
                    num_benchmark=num_benchmark,
                    active_learning_steps=active_learning_steps,
                    parallel_computation=parallel_computation,
                    calculate_ensemble=calculate_ensemble,
                    pool_synth=pool_synth,
                )
                print(time() - start_time)
                filename = make_output_filename(
                    dataset_str,
                    "bm_res",
                    active_learning_method,
                    split_type,
                    str(initial_datapoint),
                    f"alpha{str(alpha_weight).replace('.', '')}",
                    f"ens{str(ensemble_size)}",
                    f"{str(min_info_gain).replace('.', '')}",
                )

                save_pickle(
                    f"{benchmark_str}/{dataset_str}_results",
                    results=results_dict,
                    filename=filename,
                    foldername="result",
                )
