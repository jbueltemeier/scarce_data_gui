import warnings
from typing import Generator, List

import numpy as np

__all__ = [
    "range_with_float",
    "is_divisible",
    "rescale_numerical",
    "is_int",
]


def range_with_float(*, start: float, stop: float, step: float) -> Generator:
    while stop > start:
        yield start
        start += step


def is_divisible(number: float, divisor: float) -> None:
    # TODO: check stack level
    if not number % divisor == 0:
        warnings.warn(
            f"{number} is not divisible by {divisor} without reminder.", stacklevel=3
        )


def rescale_numerical(*, bounds: List[int], factor_level: np.ndarray) -> np.ndarray:
    return bounds[0] + (bounds[1] - bounds[0]) * factor_level


def is_int(x):
    return isinstance(x, int) or (isinstance(x, float) and x.is_integer())
