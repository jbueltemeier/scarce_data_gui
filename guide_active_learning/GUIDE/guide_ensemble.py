from dataclasses import dataclass
from typing import cast, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import delayed, Parallel

from guide_active_learning.GUIDE.guide_tree import DecisionTreeClassifierGUIDE, M5PrimeTreeClassifier
__all__ = [
    "GUIDEEnsemble",
]


@dataclass
class GUIDEEnsemble:
    max_depth: int
    split_type: str
    use_linear_split: bool = True
    min_info_gain: float = 0.03
    guide_ensemble: Optional[List[DecisionTreeClassifierGUIDE]] = None
    train_df: Optional[pd.DataFrame] = None

    def load_trees(
        self, trees: List[DecisionTreeClassifierGUIDE], train_df: pd.DataFrame
    ) -> None:
        self.guide_ensemble = trees
        self.train_df = train_df

    def train_single_tree(
        self, train_df: pd.DataFrame, random_state: int, use_regression: bool = False
    ) -> Union[DecisionTreeClassifierGUIDE, M5PrimeTreeClassifier]:
        if use_regression:
            dtc = M5PrimeTreeClassifier()
            dtc.fit(train_df)
        else:
            dtc = DecisionTreeClassifierGUIDE(
                max_depth=self.max_depth,
                min_info_gain=self.min_info_gain,
                use_linear_split=self.use_linear_split,
                split_type=self.split_type,
            )
            dtc.fit(
                train_df.sample(len(train_df), replace=True, random_state=random_state),
                target="target",
            )
        return dtc

    def fit(self, train_df: pd.DataFrame, ensemble_size: int, use_regression: bool = False) -> None:
        self.train_df = train_df
        self.guide_ensemble = Parallel(n_jobs=1)(
            delayed(self.train_single_tree)(train_df, i, use_regression) for i in range(ensemble_size)
        )

    def score(self, df_test: pd.DataFrame) -> float:
        res = self.predict(df_test)
        df_test.reset_index(inplace=True, drop=True)
        rights = 0
        for r in range(len(res)):
            if res[r] == df_test.loc[r, "target"]:
                rights += 1
        return rights / len(df_test)

    def predict(
        self, df: pd.DataFrame, only_predictions: bool = True
    ) -> Union[List[str], np.ndarray]:
        if self.guide_ensemble is None:
            raise TypeError("Set Guide Ensemble before you can predict samples.")

        predictions = np.empty((len(df), len(self.guide_ensemble)), dtype=object)
        for enu, dt in enumerate(self.guide_ensemble):
            predictions[:, enu] = dt.predict(df)

        if only_predictions:
            preds = [
                np.unique(predictions[i, :])[
                    np.argmax(np.unique(predictions[i, :], return_counts=True)[1])
                ]
                for i in range(len(predictions))
            ]
            return preds
        else:  # use the calculations for probabilities and uncertainties
            return predictions

    def predict_probabilities(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        if self.guide_ensemble is None:
            raise TypeError("Set Guide Ensemble before you can predict samples.")

        predictions = self.predict(df, only_predictions=False)
        probas = []
        for i in range(len(predictions)):
            vals, counts = np.unique(
                cast(np.ndarray, predictions)[i, :], return_counts=True
            )
            counts = counts / len(self.guide_ensemble)
            probas.append(dict(zip(vals, counts)))

        return probas

    @staticmethod
    def _compute_vote_entropy(entry: Dict[str, float]) -> np.ndarray:
        probs = np.array(list(entry.values()))
        probs[probs == 0] = 1e-9  # logarithm 0 undefined
        return cast(np.ndarray, -np.sum(probs * np.log(probs)))

    def predict_uncertainties(
        self, df: pd.DataFrame, averaged: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        probas = self.predict_probabilities(df)
        uncertainties = []
        for prob in probas:
            uncertainties.append(self._compute_vote_entropy(prob))

        if averaged:
            return cast(np.ndarray, np.mean(uncertainties))

        return uncertainties
