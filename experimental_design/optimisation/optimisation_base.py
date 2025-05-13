from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from experimental_design.dataset import Dataset

__all__ = [
    "_Optimisationbase",
    "_NumericalOptimisationBase",
    "_SlicedOptimisationBase",
]


@dataclass  # type: ignore[misc]
class _Optimisationbase(ABC):
    t_start: float = 1
    max_operations: int = 200000
    conv_iterations: int = 800
    track_progress: bool = True
    illustrate: bool = False

    def __post_init__(self) -> None:
        if self.illustrate:
            self.maxOperations = 5000
            self.convIterations = 300

    @abstractmethod
    def optimisation_loop(
        self, dataset: Dataset, progress: Optional[List[pd.DataFrame]] = None
    ) -> Tuple[Dataset, List[float], Optional[List[pd.DataFrame]]]:
        pass

    @abstractmethod
    def plot_criterion(self, criterion_values: List[float]) -> None:
        pass

    def perform_optimisation(self, dataset: Dataset) -> Dataset:
        progress = [dataset.design] if self.track_progress else None
        dataset, criterion_values, progress = self.optimisation_loop(
            dataset=dataset, progress=progress
        )

        if self.illustrate:
            dataset.plot_input_space()
            self.plot_criterion(criterion_values)

        return dataset


class _NumericalOptimisationBase(_Optimisationbase):
    @staticmethod
    @abstractmethod
    def exchange(
        *,
        design: pd.DataFrame,
        selected_dim: int,
        exchange_partners: np.ndarray,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def optimisation_loop(
        self, dataset: Dataset, progress: Optional[List[pd.DataFrame]] = None
    ) -> Tuple[Dataset, List[float], Optional[List[pd.DataFrame]]]:
        pass

    @abstractmethod
    def plot_criterion(self, criterion_values: List[float]) -> None:
        pass


class _SlicedOptimisationBase(_Optimisationbase):
    @staticmethod
    @abstractmethod
    def within_slice_exchange(
        *,
        design: pd.DataFrame,
        selected_dim: int,
        selected_slice: int,
        exchange_partners: np.ndarray,
        num_datapoints_per_slice: int,
    ) -> pd.DataFrame:
        pass

    @staticmethod
    def between_slice_exchange(
        *,
        design: pd.DataFrame,
        level_mapping: np.ndarray,
        selected_dim: int,
        selected_factor: int,
        exchange_partners: np.ndarray,
        num_datapoints_per_slice: int,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def optimisation_loop(
        self, dataset: Dataset, progress: Optional[List[pd.DataFrame]] = None
    ) -> Tuple[Dataset, List[float], Optional[List[pd.DataFrame]]]:
        pass

    @abstractmethod
    def plot_criterion(self, criterion_values: List[float]) -> None:
        pass
