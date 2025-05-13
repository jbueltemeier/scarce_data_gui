from typing import Any, cast, Generator, Optional

import numpy as np
from scipy.stats import special_ortho_group
from sklearn.decomposition import PCA
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state, gen_batches, resample


def random_feature_subsets(
    array: np.ndarray, batch_size: int, random_state: int = 1234
) -> Generator:
    """Generate K subsets of the features in X"""
    random_state = check_random_state(random_state)
    features = list(range(array.shape[1]))
    random_state.shuffle(features)
    for batch in gen_batches(len(features), batch_size):
        yield features[batch]


class RotationTreeClassifier(DecisionTreeClassifier):
    def __init__(
        self,
        n_features_per_subset: int = 3,
        rotation_algo: str = "pca",
        random_rotation: bool = True,
        criterion: str = "gini",
        splitter: str = "best",
        bootstrap: bool = False,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float = 1.0,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        class_weight: Optional[float] = None,
    ) -> None:

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo
        self.random_rotation = random_rotation
        self.bootstrap = bootstrap

        super(RotationTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
        )

    def rotate(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "rotation_matrix"):
            raise NotFittedError("The estimator has not been fitted")
        if X.shape[1] == 1:
            return X
        return cast(np.ndarray, np.dot(X, self.rotation_matrix))

    def pca_algorithm(self) -> PCA:
        """Determine PCA algorithm to use."""
        if self.rotation_algo == "randomized":
            return PCA(svd_solver="randomized", random_state=self.random_state)
        elif self.rotation_algo == "pca":
            return PCA()
        else:
            raise ValueError("`rotation_algo` must be either " "'pca' or 'randomized'.")

    def _fit_rotation_matrix(self, X: np.ndarray) -> None:
        if X.shape[1] == 1:
            self.rotation_matrix = np.array([])
        else:
            if not self.random_rotation:
                random_state = check_random_state(self.random_state)
                n_samples, n_features = X.shape
                self.rotation_matrix = np.zeros(
                    (n_features, n_features), dtype=np.float32
                )
                for i, subset in enumerate(
                    random_feature_subsets(
                        X, self.n_features_per_subset, random_state=random_state
                    )
                ):
                    # take a 75% bootstrap from the rows
                    x_sample = resample(
                        X,
                        n_samples=max(
                            int(n_samples * 0.75), self.n_features_per_subset
                        ),
                        random_state=None,
                    )
                    pca = self.pca_algorithm()
                    pca.fit(x_sample[:, subset])
                    self.rotation_matrix[np.ix_(subset, subset)] = pca.components_
            else:
                self.rotation_matrix = special_ortho_group.rvs(X.shape[1]).astype(
                    "float32"
                )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[float] = None,
        check_input: bool = True,
    ) -> None:
        if self.bootstrap:
            random_state = check_random_state(self.random_state)
            sample_indices = random_state.randint(
                0, X.shape[0], X.shape[0], dtype=np.int32
            )
            X = X[sample_indices]
            y = y[sample_indices]
        self._fit_rotation_matrix(X)
        super(RotationTreeClassifier, self).fit(
            self.rotate(X), y, sample_weight, check_input
        )

    def predict_proba(self, X: np.ndarray, check_input: bool = True) -> Any:
        return super(RotationTreeClassifier, self).predict_proba(
            self.rotate(X), check_input
        )

    def predict(self, X: np.ndarray, check_input: bool = True) -> Any:
        return super(RotationTreeClassifier, self).predict(self.rotate(X), check_input)

    def apply(self, X: np.ndarray, check_input: bool = True) -> Any:
        return super(RotationTreeClassifier, self).apply(self.rotate(X), check_input)

    def decision_path(self, X: np.ndarray, check_input: bool = True) -> Any:
        return super(RotationTreeClassifier, self).decision_path(
            self.rotate(X), check_input
        )


class RotationForestClassifier(ForestClassifier):
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        n_features_per_subset: int = 3,
        rotation_algo: str = "pca",
        random_rotation: bool = True,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float = 1.0,
        max_leaf_nodes: Optional[int] = None,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: Optional[float] = None,
    ):
        super(RotationForestClassifier, self).__init__(
            estimator=RotationTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "n_features_per_subset",
                "rotation_algo",
                "random_rotation",
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "random_state",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
        )

        self.n_features_per_subset = n_features_per_subset
        self.random_rotation = random_rotation
        self.rotation_algo = rotation_algo
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


class RotationTreeRegressor(DecisionTreeRegressor):
    def __init__(
        self,
        n_features_per_subset: int = 3,
        rotation_algo: str = "pca",
        random_rotation: bool = True,
        criterion: str = "squared_error",
        splitter: str = "best",
        bootstrap: bool = False,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float = 1.0,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
    ) -> None:

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo
        self.random_rotation = random_rotation
        self.bootstrap = bootstrap

        super(RotationTreeRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
        )

    def rotate(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "rotation_matrix"):
            raise NotFittedError("The estimator has not been fitted")
        if X.shape[1] == 1:
            return X
        return cast(np.ndarray, np.dot(X, self.rotation_matrix))

    def pca_algorithm(self) -> PCA:
        """Determine PCA algorithm to use."""
        if self.rotation_algo == "randomized":
            return PCA(svd_solver="randomized", random_state=self.random_state)
        elif self.rotation_algo == "pca":
            return PCA()
        else:
            raise ValueError("`rotation_algo` must be either " "'pca' or 'randomized'.")

    def _fit_rotation_matrix(self, X: np.ndarray) -> None:
        if X.shape[1] == 1:
            self.rotation_matrix = np.array([])
        else:
            if not self.random_rotation:
                random_state = check_random_state(self.random_state)
                n_samples, n_features = X.shape
                self.rotation_matrix = np.zeros(
                    (n_features, n_features), dtype=np.float32
                )
                for i, subset in enumerate(
                    random_feature_subsets(
                        X, self.n_features_per_subset, random_state=random_state
                    )
                ):
                    # take a 75% bootstrap from the rows
                    x_sample = resample(
                        X,
                        n_samples=max(
                            int(n_samples * 0.75), self.n_features_per_subset
                        ),
                        random_state=None,
                    )
                    pca = self.pca_algorithm()
                    pca.fit(x_sample[:, subset])
                    self.rotation_matrix[np.ix_(subset, subset)] = pca.components_

            else:
                self.rotation_matrix = special_ortho_group.rvs(X.shape[1]).astype(
                    "float32"
                )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[float] = None,
        check_input: bool = True,
    ) -> None:
        if self.bootstrap:
            random_state = check_random_state(self.random_state)
            sample_indices = random_state.randint(
                0, X.shape[0], X.shape[0], dtype=np.int32
            )
            X = X[sample_indices]
            y = y[sample_indices]
        self._fit_rotation_matrix(X)
        super(RotationTreeRegressor, self).fit(
            self.rotate(X), y, sample_weight, check_input
        )

    def predict(self, X: np.ndarray, check_input: bool = True) -> Any:
        return super(RotationTreeRegressor, self).predict(self.rotate(X), check_input)

    def apply(self, X: np.ndarray, check_input: bool = True) -> Any:
        return super(RotationTreeRegressor, self).apply(self.rotate(X), check_input)

    def decision_path(self, X: np.ndarray, check_input: bool = True) -> Any:
        return super(RotationTreeRegressor, self).decision_path(
            self.rotate(X), check_input
        )


class RotationForestRegressor(ForestRegressor):
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        n_features_per_subset: int = 3,
        random_rotation: bool = True,
        rotation_algo: str = "pca",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float = 1.0,
        max_leaf_nodes: Optional[int] = None,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
    ) -> None:
        super(RotationForestRegressor, self).__init__(
            estimator=RotationTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "n_features_per_subset",
                "rotation_algo",
                "random_rotation",
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "random_state",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )

        self.n_features_per_subset = n_features_per_subset
        self.random_rotation = random_rotation
        self.rotation_algo = rotation_algo
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
