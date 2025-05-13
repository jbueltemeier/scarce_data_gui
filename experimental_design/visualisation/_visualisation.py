from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "function_visualisation",
    "visualize_centrifuge_test",
]


def function_visualisation(
    *,
    x: np.ndarray,
    y: np.ndarray,
) -> matplotlib.figure.Figure:

    plt.figure(figsize=(10, 6))
    for i in range(y.shape[1]):
        plt.plot(x, y[:, i], label=f"Param setting {i}")

    plt.title("Function curve for various parameter settings.")
    plt.xlabel("x-values")
    plt.ylabel("Function values")
    plt.grid(True)
    return plt


def visualize_centrifuge_test(
    phases: List[str], percentages: List[int]
) -> matplotlib.figure.Figure:
    colors = plt.cm.viridis(range(0, 256, int(256 / len(phases))))

    fig, ax = plt.subplots(figsize=(6, len(phases) * 1.5))
    bars = ax.barh(phases, percentages, color=colors, edgecolor="black")
    for bar, percentage in zip(bars, percentages):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{percentage}%",
            va="center",
            ha="left",
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Volumenprozent (%)")
    ax.set_title("Becherschleudertest - Phasenverteilung")

    plt.tight_layout()
    return plt
