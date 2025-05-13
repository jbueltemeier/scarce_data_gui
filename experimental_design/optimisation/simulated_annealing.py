from dataclasses import dataclass
from typing import cast, List, Optional, Tuple

import numpy as np
import pandas as pd

from experimental_design.criterion import PhiPCriterion, SlicedPhiPCriterion

from experimental_design.dataset import Dataset
from experimental_design.optimisation.optimisation_base import (
    _NumericalOptimisationBase,
    _SlicedOptimisationBase,
)
from experimental_design.visualisation import plot_criterion_values

__all__ = [
    "PhiPSimulatedAnnealing",
    "SlicedPhiPSimulatedAnnealing",
]


@dataclass
class PhiPSimulatedAnnealing(_NumericalOptimisationBase, PhiPCriterion):
    alpha: float = 0.9
    threshold: float = 0.5

    @staticmethod
    def exchange(
        *, design: pd.DataFrame, selected_dim: int, exchange_partners: np.ndarray
    ) -> pd.DataFrame:
        (
            design.iloc[exchange_partners[0], selected_dim],
            design.iloc[exchange_partners[1], selected_dim],
        ) = (
            design.iloc[exchange_partners[1], selected_dim],
            design.iloc[exchange_partners[0], selected_dim],
        )
        return design

    def optimisation_loop(
        self, dataset: Dataset, progress: Optional[List[pd.DataFrame]] = None
    ) -> Tuple[Dataset, List[float], Optional[List[pd.DataFrame]]]:
        design_opt = dataset.design
        temp = self.t_start
        iterations = 0
        phi_exchanges: List[float] = []
        conv = 0

        phi_exchanges.append(
            self.calculate_criterion(
                design=dataset.design, num_datapoints=dataset.num_datapoints
            )
        )
        iterations += 1

        while iterations <= self.max_operations:
            print(f"Operation: {iterations}")
            selected_dim = np.random.randint(0, dataset.num_numerical_features) + 1
            exchange_partners = np.random.permutation(dataset.num_datapoints)
            design_candidate = self.exchange(
                design=dataset.design.copy(),
                selected_dim=selected_dim,
                exchange_partners=exchange_partners,
            )

            phi_p = self.calculate_criterion(
                design=design_candidate,
                num_datapoints=dataset.num_datapoints,
            )

            reject = phi_p < phi_exchanges[iterations - 1] or np.random.rand() < np.exp(
                -(phi_p - phi_exchanges[iterations - 1]) / temp
            )

            if reject:
                conv = 0
                dataset.design = design_candidate
                phi_exchanges.append(phi_p)
                if phi_p < phi_exchanges[iterations - 1]:
                    design_opt = dataset.design

                if self.track_progress:
                    cast(List[pd.DataFrame], progress).append(design_opt.copy())
            else:
                phi_exchanges.append(phi_exchanges[iterations - 1])
                conv += 1

            iterations += 1
            temp *= self.alpha

            if conv >= self.conv_iterations:
                break
        return dataset, phi_exchanges, progress

    def plot_criterion(self, criterion_values: List[float]) -> None:
        plot_criterion_values(
            criterion_values,
            title="Simulated Annealing with PhiP-Criterion",
            y_label="Phip Value",
        )


@dataclass
class SlicedPhiPSimulatedAnnealing(_SlicedOptimisationBase, SlicedPhiPCriterion):
    alpha: float = 0.9
    threshold: float = 0.5

    @staticmethod
    def within_slice_exchange(
        *,
        design: pd.DataFrame,
        selected_dim: int,
        selected_slice: int,
        exchange_partners: np.ndarray,
        num_datapoints_per_slice: int,
    ) -> pd.DataFrame:
        row_p1 = (selected_slice * num_datapoints_per_slice) + exchange_partners[0]
        row_p2 = (selected_slice * num_datapoints_per_slice) + exchange_partners[1]
        design.iloc[row_p1, selected_dim], design.iloc[row_p2, selected_dim] = (
            design.iloc[row_p2, selected_dim],
            design.iloc[row_p1, selected_dim],
        )
        return design

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
        dim = selected_dim - 1
        row_p1 = (exchange_partners[0] * num_datapoints_per_slice) + (
            level_mapping[:, dim, exchange_partners[0]] == selected_factor
        ).nonzero()[0][0]
        row_p2 = (exchange_partners[1] * num_datapoints_per_slice) + (
            level_mapping[:, dim, exchange_partners[1]] == selected_factor
        ).nonzero()[0][0]
        design.iloc[row_p1, selected_dim], design.iloc[row_p2, selected_dim] = (
            design.iloc[row_p2, selected_dim],
            design.iloc[row_p1, selected_dim],
        )
        return design

    def update_level_mapping(
        self,
        *,
        dataset: Dataset,
        exchange_partners: np.ndarray,
        selected_dim: int,
        selected_slice: int,
    ) -> Dataset:
        dim = selected_dim - 1
        (
            dataset.level_mapping[exchange_partners[0], dim, selected_slice],
            dataset.level_mapping[exchange_partners[1], dim, selected_slice],
        ) = (
            dataset.level_mapping[exchange_partners[1], dim, selected_slice],
            dataset.level_mapping[exchange_partners[0], dim, selected_slice],
        )
        return dataset

    def optimisation_loop(
        self, dataset: Dataset, progress: Optional[List[pd.DataFrame]] = None
    ) -> Tuple[Dataset, List[float], Optional[List[pd.DataFrame]]]:
        phi_exchanges: List[float] = []
        phi_exchanges.append(
            self.calculate_criterion(
                design=dataset.design,
                num_datapoints=dataset.num_datapoints,
                num_datapoints_per_slice=dataset.num_datapoints_per_slice,
            )
        )

        iterations = 1
        conv = 0
        temp = self.t_start
        design_opt = dataset.design

        while iterations <= self.max_operations:
            prob_operation = np.random.rand()
            selected_dim = np.random.randint(0, dataset.num_numerical_features) + 1

            if prob_operation <= self.threshold:
                selected_slice = np.random.randint(0, dataset.num_slices)
                exchange_partners = np.random.permutation(
                    dataset.num_datapoints_per_slice
                )
                design_candidate = self.within_slice_exchange(
                    design=dataset.design.copy(),
                    selected_dim=selected_dim,
                    selected_slice=selected_slice,
                    exchange_partners=exchange_partners,
                    num_datapoints_per_slice=dataset.num_datapoints_per_slice,
                )
            else:
                selected_factor = np.random.randint(0, dataset.num_datapoints_per_slice)
                exchange_partners = np.random.permutation(dataset.num_slices)
                design_candidate = self.between_slice_exchange(
                    design=dataset.design.copy(),
                    level_mapping=dataset.level_mapping,
                    selected_dim=selected_dim,
                    selected_factor=selected_factor,
                    exchange_partners=exchange_partners,
                    num_datapoints_per_slice=dataset.num_datapoints_per_slice,
                )

            phi_p = self.calculate_criterion(
                design=design_candidate,
                num_datapoints=dataset.num_datapoints,
                num_datapoints_per_slice=dataset.num_datapoints_per_slice,
            )

            accept_design = phi_p < phi_exchanges[
                iterations - 1
            ] or np.random.rand() < np.exp(
                -(phi_p - phi_exchanges[iterations - 1]) / temp
            )

            if accept_design:
                conv = 0
                if prob_operation <= self.threshold:
                    dataset = self.update_level_mapping(
                        dataset=dataset,
                        exchange_partners=exchange_partners,
                        selected_dim=selected_dim,
                        selected_slice=selected_slice,
                    )
                dataset.design = design_candidate
                phi_exchanges.append(phi_p)
                if phi_p < phi_exchanges[iterations - 1]:
                    design_opt = dataset.design

                if self.track_progress:
                    cast(List[pd.DataFrame], progress).append(design_opt.copy())
            else:
                phi_exchanges.append(phi_exchanges[iterations - 1])
                conv += 1

            iterations += 1
            temp *= self.alpha

            if conv >= self.conv_iterations:
                break
        return dataset, phi_exchanges, progress

    def plot_criterion(self, criterion_values: List[float]) -> None:
        plot_criterion_values(
            criterion_values,
            title="Sliced Simulated Annealing with PhiP-Criterion",
            y_label="Phip Value",
        )
