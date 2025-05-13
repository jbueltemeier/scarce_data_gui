from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from experimental_design.criterion.phip_criterion import (
    average_inter_point_distance_phip,
    sliced_nn_phip_v3,
)

__all__ = [
    "_Criterion",
    "PhiPCriterion",
    "SlicedPhiPCriterion",
]


class _Criterion(ABC):
    @abstractmethod
    def calculate_criterion(
        self, *, design: pd.DataFrame, num_datapoints: int
    ) -> float:
        pass


@dataclass
class PhiPCriterion(_Criterion):
    exp_phip: int = 2

    def calculate_criterion(
        self, *, design: pd.DataFrame, num_datapoints: int
    ) -> float:
        return average_inter_point_distance_phip(
            design=design,
            num_datapoints=num_datapoints,
            p_norm=self.exp_phip,
        )


@dataclass
class SlicedPhiPCriterion(_Criterion):
    exp_phip: int = 50
    p_norm: int = 2
    weight_phip: float = 0.5
    comb_metric: str = "sum"

    def calculate_criterion(  # type: ignore[override]
        self,
        *,
        design: pd.DataFrame,
        num_datapoints: int,
        num_datapoints_per_slice: int,
    ) -> float:
        return sliced_nn_phip_v3(
            design=design,
            num_datapoints=num_datapoints,
            num_datapoints_per_slice=num_datapoints_per_slice,
            p_norm=self.p_norm,
            beta=self.weight_phip,
            comb_metric=self.comb_metric,
        )
