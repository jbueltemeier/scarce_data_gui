from abc import ABC, abstractmethod
from typing import cast, Optional, Union

import lhsmdu

import numpy as np

__all__ = [
    "_GeneratorBase",
    "LHCGenerator",
]

import pandas as pd


class _GeneratorBase(ABC):
    def __init__(
        self,
        *,
        min_value: Union[float, int],
        max_value: Union[float, int],
        delay: Optional[float] = None,
    ) -> None:
        self.min_value = min_value
        self.max_value = max_value
        if delay is not None:
            assert (
                0 <= delay <= 1
            ), f"The number for the delay should be between 0 and 1, but the current value is {delay}."
        self.delay = delay

    def scale_number(self, *, samples: pd.DataFrame) -> pd.DataFrame:
        return self.min_value + samples * (self.max_value - self.min_value)

    @abstractmethod
    def generate_samples(self, num_samples: int) -> np.ndarray:
        pass

    def __repr__(self) -> str:
        return f"""
        <body>
            <p><strong>Minimal Wert:</strong> {self.min_value}</p>
            <p><strong>Maximal Wert:</strong> {self.max_value}</p>
            <p><strong>Delay:</strong> {self.delay}</p>
        </body>
        """


class LHCGenerator(_GeneratorBase):
    def generate_samples(self, num_samples: int) -> np.ndarray:
        if self.delay is None:
            samples = np.array(lhsmdu.sample(1, num_samples))
            return np.squeeze(samples, axis=0)
        else:
            delay_samples = int(num_samples * self.delay)
            zeros_array = np.zeros((1, delay_samples))
            sample_array = np.array(lhsmdu.sample(1, num_samples - delay_samples))
            result_array = np.concatenate((zeros_array, sample_array), axis=1)
            return cast(np.ndarray, np.squeeze(result_array, axis=0))
