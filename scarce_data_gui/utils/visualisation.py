import math
from typing import cast, List, Optional, Tuple

import altair as alt
import pandas as pd

__all__ = ["result_chart", "visualise_result_charts"]


def result_chart(
    source: pd.DataFrame,
    mark: str,
    mark_size: int,
    ylabel: str,
    color: str,
    axis_title: Optional[str] = None,
    filled: bool = True,
) -> alt.Chart:
    return cast(
        alt.Chart,
        alt.Chart(source)  # type: ignore[attr-defined]
        .mark_point(
            size=mark_size,
            shape=mark,
            filled=filled,
            stroke="black",
        )
        .encode(
            x=alt.X("Zeit (Sekunden):Q", scale=alt.Scale(domain=[0, 650])),
            y=alt.Y(
                ylabel,
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title=axis_title),
            ),
            color=alt.value(color),
        )
        .properties(
            width=800,
            height=400,
        )
        .transform_filter(alt.datum[ylabel] > 0),
    )


def filter_columns_by_keyword(*, df: pd.DataFrame, filter_keyword: str) -> List[str]:
    return [col for col in df.columns if filter_keyword in col]


def choose_altair_mark_and_color(
    index: int, init_mark_size: int = 100
) -> Tuple[str, int, str]:
    marks = [
        "circle",
        "square",
    ]

    colors = [
        "blue",
        "yellow",
        "red",
        "green",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
        "cyan",
    ]

    if 0 <= index < len(colors):
        mark = marks[index % len(marks)]
        mark_size = (math.ceil(index / len(marks)) + 1) * init_mark_size
        color = colors[index]
        return mark, mark_size, color
    else:
        raise ValueError("Index must be between 0 and 9")


def visualise_result_charts(
    df: pd.DataFrame, *, filter_keyword: str
) -> Tuple[Optional[alt.Chart], List[str]]:
    relevant_columns = filter_columns_by_keyword(df=df, filter_keyword=filter_keyword)
    charts = None
    legend_strings = []
    for i, column in enumerate(relevant_columns):
        mark, mark_size, color = choose_altair_mark_and_color(i)
        chart = result_chart(
            df,
            mark=mark,
            mark_size=mark_size,
            ylabel=column,
            axis_title="Merkmale",
            color=color,
        )
        charts = chart if charts is None else cast(alt.Chart, charts) + chart

        if color != "red":
            legend_strings.append(f"**{column}** :large_{color}_{mark}:")
        else:
            legend_strings.append(f"**{column}** :{color}_{mark}:")
    return charts, legend_strings
