from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

from guide_active_learning.core._math import cm2inch
from guide_active_learning.core.io import save_pdf, save_plot_figure
from guide_active_learning.misc import make_output_filename

__all__ = [
    "plot_df",
    "plot_benchmark_mean_std",
]


def get_color(number: int) -> str:
    tableau_colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "olive",
        "cyan",
        "gray",
        "brown",
        "pink",
    ]
    return tableau_colors[number % len(tableau_colors)]


def plot_df(data: pd.DataFrame, save_name: Optional[str]) -> None:
    plt.subplots(dpi=100, figsize=(cm2inch(13), cm2inch(9)))
    sns.heatmap(
        data.select_dtypes(float).corr(method="pearson").round(2),
        vmin=-1,
        vmax=1,
        annot=True,
    )

    if save_name is not None:
        save_pdf(filename=save_name)


def plot_benchmark_mean_std(
    data: List[Tuple[str, List[np.ndarray], List[np.ndarray]]],
    dataset_str: str,
    num_datapoints: str,
    save_figure: bool = False,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(dpi=dpi)
    ax2 = ax.twinx()

    for i, (label, mean_scores, std_scores) in enumerate(data):
        ax.plot(mean_scores, label=label, color=get_color(i))
        ax2.plot(
            std_scores,
            label=label,
            color=get_color(i),
            linestyle="dashed",
            alpha=0.2,
        )

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height * 0.5])
    ax.legend(loc="center", bbox_to_anchor=(0.5, 1.6))

    ax.set_title(f"{dataset_str}-Dataset Number Datapoints: {num_datapoints}")
    ax.set_xlabel("Iteration [-]")
    ax.set_ylabel("Classification Accuracy")
    ax2.set_ylabel("Std(Classification Accuracy)")
    fig.show()

    filename = make_output_filename(
        "benchmark_plot",
        f"initial_datapoints_{num_datapoints}",
        extension=".png",
    )
    if save_figure:
        save_plot_figure("benchmark", filename=filename, fig=fig)
