import copy
import itertools
import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px

import plotly.graph_objects as go
from guide_active_learning.ActiveLearning import (
    active_learning_step,
    calculate_max_distance,
)
from scipy.stats import qmc

from experimental_design.core import is_int
from experimental_design.data_parts import _InstancePartBase, MetaPart, PartContainer
from experimental_design.latin_hypercube import sliced_lhc_design
from experimental_design.visualisation import (
    plot_input_space_design,
    plot_level_mapping,
    remove_extension_df_columns,
)

from .active_learning import ActiveLearning
from .part_design_handler import PartDesignHandler

__all__ = [
    "Dataset",
    "GeneratorDataset",
    "OvenToyDataset",
    "ArtificialDataset",
    "PartDesignDataset",
    "create_dataset",
]


@dataclass  # type: ignore[misc]
class Dataset(ABC):
    num_datapoints: int
    num_numerical_features: int
    numerical_columns: List[str]
    slices: List[str]
    num_slices: int
    num_datapoints_per_slice: int
    categorical_factors: Dict[str, List[str]]
    factor_level: np.ndarray

    def __post_init__(self) -> None:
        self.line_positions_slice = self.create_line_positions(
            num_lines=self.num_datapoints_per_slice
        )
        self.line_positions = self.create_line_positions(num_lines=self.num_datapoints)
        self.design, self.level_mapping, self.factor_mapping = self.create_sliced_lhd()

    @staticmethod
    def create_line_positions(*, num_lines: int) -> np.ndarray:
        return np.arange(1, num_lines) * (1 / num_lines)

    @staticmethod
    def form_cartesian_slices(values: List[List[str]]) -> List[str]:
        return cast(List[str], list(itertools.product(*values)))

    def create_sliced_lhd(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        return sliced_lhc_design(
            factor_level=self.factor_level,
            slices=self.slices,
            numerical_factors=self.numerical_columns,
            num_datapoints=self.num_datapoints,
            num_slices=self.num_slices,
            num_numerical_features=self.num_numerical_features,
            num_datapoints_per_slice=self.num_datapoints_per_slice,
        )

    @abstractmethod
    def plot_input_space(self) -> None:
        pass

    @abstractmethod
    def plot_level_mapping(self) -> None:
        pass


class GeneratorDataset(Dataset):
    def __init__(
        self,
        *,
        num_datapoints_per_slice: int,
        numerical_factors: Dict[str, List[int]],
        categorical_factors: Dict[str, List[str]],
    ) -> None:
        self.numerical_factors = numerical_factors
        num_numerical_features = len(numerical_factors)

        slices = self.form_cartesian_slices(list(categorical_factors.values()))
        num_slices = len(slices)

        num_datapoints = num_datapoints_per_slice * num_slices
        factor_level = self.generate_factor_levels(
            num_numerical_features=num_numerical_features,
            num_datapoints=num_datapoints,
        )
        super().__init__(
            num_datapoints=num_datapoints,
            num_numerical_features=num_numerical_features,
            numerical_columns=list(numerical_factors.keys()),
            slices=slices,
            num_slices=num_slices,
            num_datapoints_per_slice=num_datapoints_per_slice,
            categorical_factors=categorical_factors,
            factor_level=factor_level,
        )

    @staticmethod
    def generate_factor_levels(
        *, num_numerical_features: int, num_datapoints: int
    ) -> np.ndarray:
        lhs = qmc.LatinHypercube(d=num_numerical_features)
        return np.sort(lhs.random(n=num_datapoints), axis=0)

    @abstractmethod
    def plot_input_space(self) -> None:
        pass

    def _plot_input_space(
        self, key1: str, key2: str, title: str, xlabel: str, ylabel: str
    ) -> None:
        plot_input_space_design(
            design=self.design,
            key1=key1,
            key1_bounds=self.numerical_factors[key1],
            key2=key2,
            key2_bounds=self.numerical_factors[key2],
            line_positions_slice=self.line_positions_slice,
            line_positions=self.line_positions,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            center=True,
        )

    @abstractmethod
    def plot_level_mapping(self) -> None:
        pass

    def _plot_level_mapping(
        self, key1: str, key2: str, title: str, xlabel: str, ylabel: str
    ) -> None:
        plot_level_mapping(
            level_mapping=self.level_mapping,
            slices=self.design["slice"],
            num_slices=self.num_slices,
            num_datapoints_per_slice=self.num_datapoints_per_slice,
            key1_bounds=self.numerical_factors[key1],
            key2_bounds=self.numerical_factors[key2],
            line_positions_slice=self.line_positions_slice,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )


class OvenToyDataset(GeneratorDataset):
    def __init__(self, num_datapoints_per_slice: int) -> None:
        numerical_factors = {
            "temperature": [0, 1],
            "time": [0, 1],
        }
        categorical_factors = {
            "operation_mode": ["Slice 1", "Slice 2", "Slice 3"],
        }
        super().__init__(
            num_datapoints_per_slice=num_datapoints_per_slice,
            numerical_factors=numerical_factors,
            categorical_factors=categorical_factors,
        )

    def plot_input_space(self) -> None:
        title = ""
        xlabel = "Temperature in $[^\\circ C]$"
        ylabel = "Time in [min]"
        self._plot_input_space(
            key1="temperature", key2="time", title=title, xlabel=xlabel, ylabel=ylabel
        )

    def plot_level_mapping(self) -> None:
        title = ""
        xlabel = "Temperature in $[^\\circ C]$"
        ylabel = "Time in [min]"
        self._plot_level_mapping(
            key1="temperature", key2="time", title=title, xlabel=xlabel, ylabel=ylabel
        )


class ArtificialDataset(GeneratorDataset):
    def __init__(
        self,
        num_slices: int,
        num_datapoints_per_slice: int,
        num_numerical_features: int,
    ) -> None:
        numerical_factors = self.create_numerical_factors(num_numerical_features)
        categorical_factors = self.create_categorical_factors(num_slices)
        super().__init__(
            num_datapoints_per_slice=num_datapoints_per_slice,
            numerical_factors=numerical_factors,
            categorical_factors=categorical_factors,
        )

    @staticmethod
    def create_numerical_factors(num_numerical_features: int) -> Dict[str, List[int]]:
        alphabet = string.ascii_lowercase
        numerical_factors = {}
        for i in range(num_numerical_features):
            key = alphabet[i % len(alphabet)]
            value = [random.randint(0, 50), random.randint(51, 100)]
            numerical_factors[key] = value

        return numerical_factors

    @staticmethod
    def create_categorical_factors(num_slices: int) -> Dict[str, List[str]]:
        alphabet = string.ascii_lowercase
        categorical_factors: Dict[str, List[str]] = {"slice": []}
        for i in range(num_slices):
            value = alphabet[i % len(alphabet)]
            categorical_factors["slice"].append(value)

        return categorical_factors

    def plot_input_space(self) -> None:
        title = (
            "Input Space of Sliced LHC Design"
            if len(self.numerical_columns) != 0
            else "Input Space of LHC Design"
        )
        xlabel = "First Parameter"
        ylabel = "Second Parameter"
        self._plot_input_space(
            key1="a", key2="b", title=title, xlabel=xlabel, ylabel=ylabel
        )

    def plot_level_mapping(self) -> None:
        title = (
            "Input Space of Sliced LHC Design"
            if len(self.numerical_columns) != 0
            else "Input Space of LHC Design"
        )
        xlabel = "First Parameter"
        ylabel = "Second Parameter"
        self._plot_level_mapping(
            key1="a", key2="b", title=title, xlabel=xlabel, ylabel=ylabel
        )


class PartDesignDataset(Dataset, PartDesignHandler, ActiveLearning):
    def __init__(
        self,
        *,
        part_design_container: PartContainer,
        settings: PartContainer,
    ) -> None:
        PartDesignHandler.__init__(
            self, part_design_container=part_design_container, settings=settings
        )

        slices = self.form_cartesian_slices(list(self.categorical_factors.values()))
        num_slices = len(slices)

        num_datapoints = len(self.initial_df)
        num_datapoints_per_slice = int(num_datapoints / num_slices)
        self.is_multiple(num_datapoints=num_datapoints, num_slices=num_slices)
        num_numerical_features = len(self.numerical_columns)
        factor_level = self.get_factor_levels(self.initial_df)
        Dataset.__init__(
            self,
            num_datapoints=num_datapoints,
            num_numerical_features=num_numerical_features,
            numerical_columns=self.numerical_columns,
            slices=slices,
            num_slices=num_slices,
            num_datapoints_per_slice=num_datapoints_per_slice,
            categorical_factors=self.categorical_factors,
            factor_level=factor_level,
        )
        self.update_categorical_optimised_df()
        self.update_numerical_optimised_df()

        # Save initial df
        self.initial_df = self.optimised_df.copy()

        ActiveLearning.__init__(self)
        self.max_distance = calculate_max_distance(self.optimised_df)

        self.decision_trees = None
        self.custom_tree = None

    @staticmethod
    def is_multiple(*, num_datapoints: int, num_slices: int) -> None:
        if num_slices == 0:
            raise ValueError("Cannot have zero number of slices.")

        if num_datapoints % num_slices != 0:
            raise ValueError(
                f"{num_datapoints} datapoints should be a multiple of {num_slices} datapoints per slice."
            )

    def get_factor_levels(self, df: pd.DataFrame) -> np.ndarray:
        return np.sort(cast(np.ndarray, df[self.numerical_columns].to_numpy()), axis=0)

    def update_numerical_optimised_df(self) -> None:
        for key in self.numerical_columns:
            self.optimised_df[key] = self.design[key].values.astype(float)

    def update_categorical_optimised_df(self) -> None:
        slice_values = self.design["slice"].values
        flattened_list = [sublist.split("/") for sublist in slice_values]
        slice_df = pd.DataFrame(flattened_list)
        for i, column in enumerate(self.categorical_columns):
            self.optimised_df[column] = slice_df.iloc[:, i].values

    @staticmethod
    def remove_extension_df_columns(
        columns: Union[List[str], str]
    ) -> Union[List[str], str]:
        if isinstance(columns, str):
            return (
                columns.replace("_static", "")
                .replace("_numerical", "")
                .replace("_categorical", "")
            )
        return [
            col.replace("_static", "")
            .replace("_numerical", "")
            .replace("_categorical", "")
            for col in columns
        ]

    def get_dataframe(
        self,
        initial: bool = False,
        design_space: bool = False,
        remove_extension: bool = True,
        use_custom_tree: bool = False
    ) -> pd.DataFrame:

        if not hasattr(self, "custom_tree"):
            self.custom_tree = None

        if use_custom_tree and self.custom_tree is not None:
            return self.custom_tree.train_df

        if initial:
            self.part_design_container.assign_samples(self.initial_df)
        else:
            self.update_numerical_optimised_df()
            dtypes = self.optimised_df.dtypes
            df = pd.concat(
                [self.optimised_df, self.active_learning_datapoints], ignore_index=True
            )
            self.part_design_container.assign_samples(df.astype(dtypes))

        df = self.part_design_container.collect_samples(design_space=design_space)
        if remove_extension:
            df.columns = self.remove_extension_df_columns(df.columns)
        return df

    def get_labels(self) -> pd.DataFrame:
        labels = self.user_label_container.combine_labels()
        if self.active_learning_label_container is not None:
            active_learning_labels = (
                self.active_learning_label_container.combine_labels()
            )
            labels = pd.concat([labels, active_learning_labels], ignore_index=True)
        return labels

    def get_data(
        self, user_name: str, meta: bool = True, active_learning_mode: bool = False
    ) -> pd.DataFrame:
        design_columns = self.meta_columns
        columns = remove_extension_df_columns(design_columns)
        if active_learning_mode:
            position = (
                self.active_learning_label_container.num_samples  # type: ignore[union-attr]
                + self.user_label_container.num_samples
            )
        else:
            position = self.user_label_container.position(user_name)
        df = self.get_dataframe()

        return (
            df.loc[position, columns]
            if meta
            else df.loc[position, ~df.columns.isin(columns)]
        )

    def overwrite_df(self, df: pd.DataFrame) -> None:
        for key, value in df.items():
            self.initial_df[key] = value
            self.design[key] = value

        self.optimised_df = self.initial_df.copy()

    def init_active_learning(
        self,
        columns_to_keep: Optional[List[str]] = None,
        active_learning_label: Optional[str] = None,
    ) -> None:
        part_design_container = copy.deepcopy(self.part_design_container)
        self.init_active_learning_pool(
            part_design_container=part_design_container, settings=self.settings
        )
        self.columns_to_keep = columns_to_keep
        self.active_learning_label = active_learning_label
        self.perform_active_learning_step()

    def plot_input_space(
        self,
        initial: bool = False,
        user_name: Optional[str] = None,
        active_learning_mode: bool = False,
        x_axis_feature: Optional[str] = None,
        y_axis_feature: Optional[str] = None,
        labels: Optional[pd.DataFrame] = None,
    ) -> Any:
        df = self.get_dataframe(initial=initial)
        categorical_cols = self.remove_extension_df_columns(self.categorical_columns)
        df["slice"] = df[categorical_cols].agg("-".join, axis=1)

        # Numerische Spalten bestimmen
        x_axis_feature = x_axis_feature or self.numerical_columns[0]
        y_axis_feature = y_axis_feature or self.numerical_columns[1]
        x_axis_feature = self.remove_extension_df_columns(x_axis_feature)
        y_axis_feature = self.remove_extension_df_columns(y_axis_feature)

        # Daten für Active Learning und Augmentierung filtern
        df_active_learning = df[
            df["Berichtsnummer"].str.contains("active_learning", case=False, na=False)
        ]

        unique_slices = df["slice"].unique()
        symbols = [
            "circle",
            "square",
            "diamond",
            "cross",
            "x",
            "triangle-up",
            "triangle-down",
            "pentagon",
            "hexagon",
            "star",
        ]
        symbol_map = {
            slice_val: symbols[i % len(symbols)]
            for i, slice_val in enumerate(unique_slices)
        }

        unique_labels = labels.iloc[:, -1].unique() if labels is not None else None
        colors = (
            px.colors.qualitative.Safe + px.colors.qualitative.Bold
        )  # Farben für Farbenblinde geeignet
        color_map = (
            {
                label_val: colors[i % len(colors)]
                for i, label_val in enumerate(unique_labels)
            }
            if unique_labels is not None
            else colors
        )

        fig = go.Figure()

        for i, slice_val in enumerate(unique_slices):
            for j, label in (
                enumerate(unique_labels) if labels is not None else [(None, None)]
            ):
                if label is not None:
                    mask = labels.iloc[:, -1] == label  # type: ignore
                    label_color = color_map[label]
                    scatter_name = f"Slice: {slice_val}, Label: {label}"

                else:
                    mask = pd.Series(True, index=df.index)
                    label_color = list(color_map)[i]
                    scatter_name = f"Slice: {slice_val}"

                combined_mask = mask & (df["slice"] == slice_val)
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[combined_mask, x_axis_feature],
                        y=df.loc[combined_mask, y_axis_feature],
                        mode="markers",
                        marker=dict(
                            color=label_color,
                            symbol=symbol_map[slice_val],
                            size=12,
                            opacity=0.6,
                        ),
                        name=scatter_name,
                    )
                )

        # Active Learning Punkte hervorheben
        if not df_active_learning.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_active_learning[x_axis_feature],
                    y=df_active_learning[y_axis_feature],
                    mode="markers",
                    marker=dict(color="red", symbol="circle", size=30, opacity=0.5),
                    name="Active Learning",
                )
            )

        # Nutzer-Datenpunkt hervorheben
        highlight_index = (
            self.user_label_container.position(user_name=user_name)
            if user_name
            else None
        )
        if (
            highlight_index in df.index
            and not active_learning_mode
            and labels is None
            and not initial
        ):
            fig.add_trace(
                go.Scatter(
                    x=[df.loc[highlight_index, x_axis_feature]],
                    y=[df.loc[highlight_index, y_axis_feature]],
                    mode="markers",
                    marker=dict(color="orange", size=30, opacity=0.5),
                    name="Aktueller Datenpunkt",
                )
            )

        fig.update_layout(
            xaxis_title=x_axis_feature,
            yaxis_title=y_axis_feature,
            legend_title="Legende",
            template="plotly_white",
            paper_bgcolor="#2B4B60",
            font=dict(size=32),
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5
            ),
        )
        return fig

    def plot_level_mapping(self) -> None:
        pass

    def perform_active_learning_step(self) -> None:
        data_dtypes = self.optimised_df.dtypes
        train_df = pd.concat(
            [self.optimised_df, self.active_learning_datapoints], ignore_index=True
        )
        train_df = train_df.astype(data_dtypes)
        reduced_train_df = self.filter_columns(train_df)
        # TODO: make this for regression
        labels = cast(pd.DataFrame, self.user_label_container.combine_labels())
        active_learning_labels = cast(pd.DataFrame, self.active_learning_label_container.combine_labels())  # type: ignore[union-attr]
        all_labels = pd.concat([labels, active_learning_labels], ignore_index=True)
        if all_labels[self.active_learning_label].apply(is_int).all():  # regression
            self.active_learning_scenario = "qbc_exex_rf_reg"
        else:  # classification
            all_labels = labels.astype(str)
        if self.active_learning_label is not None:
            reduced_train_df["target"] = all_labels[self.active_learning_label]
        else:
            reduced_train_df["target"] = all_labels.iloc[:, 0]

        datapoint = active_learning_step(
            train_df=reduced_train_df.dropna(),  # entferne alle Zeilen mit Nan
            df_unlabeled_pool=self.filter_columns(self.active_learning_pool),
            foldername="Default",
            max_distance=self.max_distance,
            iteration=self.iteration,
            alpha=self.alpha_weight,
            ensemble_size=self.ensemble_size,
            max_depth=self.max_depth,
            min_info_gain=self.min_info_gain,
            split_type=self.guide_split,
            active_learning_method=self.active_learning_scenario,
        )
        self.update_active_learning_data(datapoint)
        self.iteration += 1


def create_dataset(
    meta_parts: Dict[str, _InstancePartBase],
    experimental_design_parts: Dict[str, _InstancePartBase],
    settings: Dict[str, _InstancePartBase],
) -> PartDesignDataset:
    meta = MetaPart(parts=meta_parts)  # type: ignore[arg-type]
    results = PartContainer(parts=experimental_design_parts)  # type: ignore[arg-type]
    dataset_parts = PartContainer(parts={"meta": meta, "experimental_design": results})
    label_parts = PartContainer(parts=settings)  # type: ignore[arg-type]
    label_parts.generate(num_samples=1)  # generate label settings
    return PartDesignDataset(part_design_container=dataset_parts, settings=label_parts)
