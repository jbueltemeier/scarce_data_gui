import itertools
from typing import List, Tuple

__all__ = [
    "create_symbol_color_list",
    "remove_extension_df_columns",
]


def create_symbol_color_list(num_combinations: int = 5) -> List[Tuple[str, str]]:
    symbols = ["o", "s", "D", "*", "p", "h"]
    colors = ["b", "g", "r"]
    all_combinations = list(itertools.product(symbols, colors))  # TODO: reorder Tuples
    if num_combinations > len(all_combinations):
        raise ValueError(
            f"There are only {len(all_combinations)} possible combinations."
        )

    selected_combinations = all_combinations[:num_combinations]
    return selected_combinations


def remove_extension_df_columns(columns: List[str]) -> List[str]:
    return [
        col.replace("_static", "").replace("_numerical", "").replace("_categorical", "")
        for col in columns
    ]
