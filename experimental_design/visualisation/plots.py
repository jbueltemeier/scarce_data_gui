from typing import cast, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experimental_design.core import rescale_numerical

from experimental_design.visualisation import create_symbol_color_list

__all__ = [
    "plot_line_grid",
    "plot_slice_scatter",
    "plot_input_space_design",
    "plot_level_mapping",
    "plot_criterion_values",
    "plot_function_from_df",
]


def get_nearest_grid_center(
    val: float, bounds: List[int], line_positions: np.ndarray
) -> float:
    grid_lines = [bounds[0] + (bounds[1] - bounds[0]) * pos for pos in line_positions]
    grid_lines.insert(0, bounds[0])
    grid_lines.append(bounds[1])
    grid_centers = [
        (grid_lines[i] + grid_lines[i + 1]) / 2 for i in range(len(grid_lines) - 1)
    ]
    nearest_center = min(grid_centers, key=lambda center: abs(val - center))  # type: ignore[no-any-return]
    return cast(float, nearest_center)


def plot_line_grid(
    *,
    key1_bounds: List[int],
    key2_bounds: List[int],
    line_style: str = "--",
    line_width: Optional[float] = None,
    line_positions: np.ndarray,
) -> None:
    for pos in line_positions:
        plt.axvline(
            key1_bounds[0] + (key1_bounds[1] - key1_bounds[0]) * pos,
            linestyle=line_style,
            linewidth=line_width,
        )
        plt.axhline(
            key2_bounds[0] + (key2_bounds[1] - key2_bounds[0]) * pos,
            linestyle=line_style,
            linewidth=line_width,
        )


def plot_slice_scatter(
    *,
    design: pd.DataFrame,
    key1: str,
    key1_bounds: List[int],
    key2: str,
    key2_bounds: List[int],
    unique_slices: np.ndarray,
    line_positions: np.ndarray,
    center: bool = False,
) -> None:
    for group, marker_color in zip(
        unique_slices, create_symbol_color_list(len(unique_slices))
    ):
        subset = design[design["slice"] == group]
        x = rescale_numerical(
            bounds=key1_bounds, factor_level=subset[key1].to_numpy().astype(float)
        )
        y = rescale_numerical(
            bounds=key2_bounds, factor_level=subset[key2].to_numpy().astype(float)
        )
        if center:
            x = [  # type: ignore[assignment]
                get_nearest_grid_center(x_value, key1_bounds, line_positions)
                for x_value in x
            ]
            y = [  # type: ignore[assignment]
                get_nearest_grid_center(y_value, key2_bounds, line_positions)
                for y_value in y
            ]

        plt.scatter(
            x,
            y,
            label=group,
            marker=marker_color[0],
            color=marker_color[1],
        )


def plot_input_space_design(
    *,
    design: pd.DataFrame,
    key1: str,
    key1_bounds: List[int],
    key2: str,
    key2_bounds: List[int],
    line_positions_slice: np.ndarray,
    line_positions: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    center: bool = False,
) -> None:
    plot_slice_scatter(
        design=design,
        key1=key1,
        key1_bounds=key1_bounds,
        key2=key2,
        key2_bounds=key2_bounds,
        unique_slices=design["slice"].unique(),
        line_positions=line_positions,
        center=center,
    )

    plot_line_grid(
        key1_bounds=key1_bounds,
        key2_bounds=key2_bounds,
        line_style="--",
        line_positions=line_positions_slice,
    )

    plot_line_grid(
        key1_bounds=key1_bounds,
        key2_bounds=key2_bounds,
        line_style=":",
        line_width=0.25,
        line_positions=line_positions,
    )

    plt.title(title)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.xlim(key1_bounds)
    plt.ylim(key2_bounds)
    plt.legend()
    plt.show()


def plot_level_mapping(
    *,
    level_mapping: np.ndarray,
    slices: pd.Series,
    num_slices: int,
    num_datapoints_per_slice: int,
    key1_bounds: List[int],
    key2_bounds: List[int],
    line_positions_slice: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    dim1: List[float] = []
    dim2: List[float] = []
    for k in range(num_slices):
        temp_bounds = key1_bounds[1] - key1_bounds[0]
        time_bounds = key2_bounds[1] - key2_bounds[0]
        dim1.extend(
            key1_bounds[0]
            + temp_bounds * (level_mapping[:, 0, k]) * (1 / num_datapoints_per_slice)
            + temp_bounds * (0.5 * (1 / num_datapoints_per_slice))
        )
        dim2.extend(
            key2_bounds[0]
            + time_bounds * (level_mapping[:, 1, k]) * (1 / num_datapoints_per_slice)
            + time_bounds * (0.5 * (1 / num_datapoints_per_slice))
        )
    unique_slices = slices.unique()
    for group, marker_color in zip(
        unique_slices, create_symbol_color_list(len(unique_slices))
    ):
        mask = slices == group
        plt.scatter(
            np.array(dim1)[mask],
            np.array(dim2)[mask],
            label=group,
            color=marker_color[1],
            marker=marker_color[0],
            s=100,
            linewidths=1.5,
        )

    plot_line_grid(
        key1_bounds=key1_bounds,
        key2_bounds=key2_bounds,
        line_style="--",
        line_positions=line_positions_slice,
    )

    plt.title(title)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.xlim(key1_bounds)
    plt.ylim(key2_bounds)
    plt.legend()
    plt.show()


def plot_criterion_values(
    criterion_values: List[float], title: str, y_label: str
) -> None:
    plt.plot(criterion_values)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel(y_label)
    plt.show()


def plot_function_from_df(df: pd.DataFrame) -> plt.Figure:
    function_columns = [col for col in df.columns if col.endswith("_function")]
    times = [float(col.split("_")[0]) for col in function_columns]
    plt.figure(figsize=(10, 6))

    all_phases = set()
    for col in function_columns:
        for d in df[col]:
            all_phases.update(d.keys())
    all_phases = sorted(all_phases)

    for phase in all_phases:
        x_values = []
        y_values = []
        for time, col in zip(times, function_columns):
            column_data = df[col].apply(lambda x: x.get(phase, None))
            if not column_data.isnull().all():
                x_values.extend([time] * len(column_data))
                y_values.extend(column_data)
        plt.scatter(x_values, y_values, label=phase)

    plt.xlabel("Zeit")
    plt.ylabel("Funktionswert")
    plt.ylim(0, 100)
    plt.title("Funktionen über die Zeit")
    plt.legend()
    plt.grid(True)
    return plt
