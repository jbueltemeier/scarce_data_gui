from __future__ import annotations

import copy


import os
import subprocess
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st
from graphviz import Digraph, Source
import graphviz
from sklearn.tree import export_graphviz, plot_tree

from guide_active_learning.core import (
    calculate_column_beta,
    calculate_column_gamma,
    compute_information_gain,
    compute_lda,
)
from guide_active_learning.core import one_hot_enc
from guide_active_learning.GUIDE.guide_analysis import sf_main_effect
from guide_active_learning.GUIDE.guide_misc import sf_interaction, split_dataframe
from guide_active_learning.GUIDE.split_points import (
    sp_interaction_features,
    sp_main_feature,
)
from guide_active_learning.GUIDE.splitting import linear_split, univariate_split
from scipy.stats import chi2
from sklearn.preprocessing import LabelEncoder
from m5py import M5Prime

pd.options.mode.chained_assignment = None
# Setzen des Pfades zu den Graphviz-Exe-Dateien
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"
# Suppress NumPy RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

COLOR_PALETTE = [
    "#117733",  # Dunkelgrün
    "#44AA99",  # Türkisgrün
    "#88CCEE",  # Hellblau
    "#DDCC77",  # Gelb
    "#CC6677",  # Rötlich
    "#AA4499",  # Lila
    "#882255",  # Dunkelrot
    "#332288",  # Dunkelblau
    "#999933",  # Oliv
    "#E17C05",  # Orange
]

__all__ = [
    "TreeNode",
    "DecisionTreeClassifierGUIDE",
    "train_decision_tree",
    "M5PrimeTreeClassifier",
    "train_regression_decision_tree",
]


@dataclass
class TreeNode:
    name: str
    dataset: field(default_factory=pd.DataFrame)
    depth: int
    feature: Optional[Union[List[str], str, np.ndarray]] = None
    threshold: Optional[Union[str, Tuple[str], float]] = None
    value: Optional[pd.Series] = None
    left_tree: Optional[TreeNode] = None
    right_tree: Optional[TreeNode] = None
    adapted: bool = False
    runs: int = 0
    pool: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)
    unl_pool: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)
    metrics: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.gini = self.compute_gini_node(self.dataset["target"])

    @staticmethod
    def compute_gini_node(target_feature: pd.Series) -> float:
        return 1 - sum((target_feature.value_counts() / len(target_feature)) ** 2)


@dataclass
class GUIDEDecisionTreeBuilder:
    root: Optional[TreeNode] = None
    train_df: Optional[pd.DataFrame] = None
    target: str = "target"
    max_depth: int = 5
    adapted: bool = False
    split_type: str = "mixed"
    use_linear_split: bool = True
    min_samples_split: int = 2
    min_info_gain: float = 0.05
    value: int = 0
    classCost: np.ndarray = 0
    balanceCosts: bool = False
    target_expression: Optional[pd.DataFrame] = None

    def _label_TreeNode(self, df: pd.DataFrame, name: str, depth: int) -> TreeNode:
        return TreeNode(
            name=name,
            value=df[self.target].value_counts(normalize=True),
            dataset=df,
            depth=depth,
            adapted=self.adapted,
        )

    def _build_tree(
        self, df: pd.DataFrame, depth: int = 0, name: str = "root"
    ) -> TreeNode:
        if (
            len(df[self.target].value_counts()) == 1
            or df.drop(columns=self.target).empty
            or depth >= self.max_depth
            or len(df) <= self.min_samples_split
        ):
            return self._label_TreeNode(df, name, depth)

        self.value += 1
        split_feature = self._select_split_feature(df, self.target)
        if split_feature is None:
            return self._label_TreeNode(df, name, depth)

        split_feature, split_point = self._select_split_point(
            df, split_feature, self.target
        )
        df1, df2 = split_dataframe(df, split_point[-1], split_feature, self.target)

        if compute_information_gain(df, df1, df2, self.target_expression, self.classCost) < self.min_info_gain:
            return self._label_TreeNode(df, name, depth)

        return TreeNode(
            name=name,
            feature=split_feature,
            threshold=split_point[-1],
            left_tree=self._build_tree(df1, depth + 1, name + "1"),
            right_tree=self._build_tree(df2, depth + 1, name + "2"),
            dataset=df,
            depth=depth,
            adapted=self.adapted,
        )

    def _select_split_feature(
        self, df: pd.DataFrame, target: str
    ) -> Optional[Union[str, List[str], np.ndarray]]:
        non_constant_cols = [col for col in df.columns if df[col].nunique() > 1]
        if target not in non_constant_cols:
            return None
        df = df[non_constant_cols]

        alpha = 0.05 / (len(df.columns) - 1)
        main_effect = sf_main_effect(df, target)

        if max(main_effect.values()) > chi2.isf(alpha, 1) or len(df.columns) == 2:
            return max(main_effect, key=cast(Callable[[str], int], main_effect.get))

        interaction = sf_interaction(df, target)
        len_number_columns = len(df.select_dtypes(include="number").columns)
        beta = calculate_column_beta(len(df.columns) - 1)
        gamma = calculate_column_gamma(len_number_columns)

        if self.use_linear_split:
            return linear_split(
                df, target, interaction, main_effect, len_number_columns, beta, gamma
            )

        return univariate_split(df, target, interaction, main_effect, beta)

    def _select_split_point(
        self, df: pd.DataFrame, feature: Union[str, List[str], np.ndarray], target: str
    ) -> Tuple:
        if isinstance(feature, str):
            return feature, sp_main_feature(df, feature, self.target_expression, self.classCost)

        if isinstance(feature[0], np.ndarray):
            lda_df = compute_lda(df, feature, target)
            return feature, sp_main_feature(lda_df, "LDA", self.target_expression, self.classCost)

        sf, _ = sp_interaction_features(df, cast(List[str], feature), target, self.target_expression, self.classCost)
        assert sf is not None, "sf should not be None here"
        return sf, sp_main_feature(df, sf, self.target_expression, self.classCost)

    def _collect_nodes(
        self, node: TreeNode, only_split: bool, nodes: List[TreeNode]
    ) -> List[TreeNode]:
        if not only_split or node.feature is not None:
            nodes.append(node)
        if node.left_tree:
            self._collect_nodes(node.left_tree, only_split, nodes)
        if node.right_tree:
            self._collect_nodes(node.right_tree, only_split, nodes)
        return nodes

    def return_all_nodes(self, only_split: bool = False) -> List[TreeNode]:
        return self._collect_nodes(self.root, only_split, []) if self.root else []

    def return_all_leaves(self) -> List[TreeNode]:
        return [node for node in self.return_all_nodes() if node.feature is None]

    def _mark_nodes(self, node: TreeNode, sample: pd.DataFrame) -> None:
        node.runs += 1
        node.pool = pd.concat([node.pool, sample], ignore_index=True)
        if node.feature is not None:
            if isinstance(node.threshold, float) and not isinstance(node.feature, list):
                next_node = (
                    node.left_tree
                    if sample[node.feature].iloc[0] <= node.threshold
                    else node.right_tree
                )
            elif isinstance(node.feature, list):
                lda_sample = compute_lda(sample, node.feature, self.target)
                next_node = (
                    node.left_tree
                    if lda_sample["LDA"].iloc[0] <= node.threshold
                    else node.right_tree
                )
            else:
                next_node = (
                    node.right_tree
                    if sample[node.feature].iloc[0] == node.threshold
                    else node.left_tree
                )
            self._mark_nodes(cast(TreeNode, next_node), sample)

    def run_through_tree(self, df: pd.DataFrame) -> None:
        for _, sample in df.iterrows():
            self._mark_nodes(cast(TreeNode, self.root), sample.to_frame().T)



@dataclass
class GUIDEDecisionTreeVisualiser:
    root: Optional[TreeNode] = None
    _tree_graph: Optional[Union[str, Digraph]] = None

    def _visualize_decision_tree(
        self, unique_labels: Optional[List[str]] = None
    ) -> Union[Digraph, plt.Figure]:
        try:
            subprocess.run(
                ["dot", "-V"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            dot = self._visualize_with_graphviz(unique_labels=unique_labels)
            return dot

        except FileNotFoundError:
            return self.plot_tree_with_matplotlib()

    def replace_umlauts(self, text: str) -> str:
        umlaut_map = {
            "ä": "ae",
            "ö": "oe",
            "ü": "ue",
            "ß": "ss",
            "Ä": "Ae",
            "Ö": "Oe",
            "Ü": "Ue",
            "°": "Grad ",
        }

        return "".join(umlaut_map.get(char, char) for char in text)

    def _visualize_with_graphviz(
        self, unique_labels: Optional[List[str]] = None
    ) -> Optional[Digraph]:
        def add_nodes_edges(
            dot: Digraph,
            node: TreeNode,
            parent: Optional[str] = None,
            edge_label: Optional[str] = None,
        ) -> None:
            node_id = str(id(node))
            if node.feature:
                if isinstance(node.feature, list) and isinstance(node.threshold, float):
                    label = (
                        f"Achsenschräger Split:\n"
                        f"{node.feature[0][0]:.3f} * {node.feature[1][0]} + "
                        f"{node.feature[0][1]:.3f} * {node.feature[1][1]}"
                    )
                else:
                    label = f"{node.feature}"
            else:
                max_target = cast(pd.Series, node.value).idxmax()
                label = f"Ausgangsgroesse:\n {max_target},\n Sicherheit:\n {cast(pd.Series, node.value).loc[max_target]:.2f}"
                fillcolor = label_colors.get(max_target, "#A0C4FF")

            color = "#D3D3D3" if node.feature else fillcolor
            shape = "box" if node.feature else "ellipse"

            dot.node(
                node_id,
                self.replace_umlauts(label),
                shape=shape,
                style="filled",
                fillcolor=color,
                fontname="Arial",
                fontsize="10",
            )

            if parent:
                dot.edge(
                    parent,
                    node_id,
                    label=self.replace_umlauts(cast(str, edge_label)),
                    color="black",
                    style="solid",
                    penwidth="2.0",
                    arrowhead="vee",
                    fontsize="10",
                    constraint="true",
                    splines="line",
                )

            if node.left_tree:
                if isinstance(node.threshold, float):
                    left_edge_label = f"<= {node.threshold:.2f}"
                else:
                    threshold = (
                        "\n".join(node.threshold)
                        if isinstance(node.threshold, tuple)
                        else node.threshold
                    )
                    left_edge_label = f"{threshold}"
                add_nodes_edges(
                    dot,
                    node.left_tree,
                    parent=node_id,
                    edge_label=left_edge_label,
                )
            if node.right_tree:
                right_edge_label = (
                    f"> {node.threshold:.2f}"
                    if isinstance(node.threshold, float)
                    else f"Andere {node.feature}"
                )

                add_nodes_edges(
                    dot,
                    node.right_tree,
                    parent=node_id,
                    edge_label=right_edge_label,
                )

        if "tree_filter_conditions" not in st.session_state:
            st.session_state["tree_filter_conditions"] = {}

        filtered_root = self.filter_tree(
            copy.deepcopy(self.root),
            filter_conditions=st.session_state["tree_filter_conditions"],
        )

        if unique_labels is not None:
            label_colors = {
                label: COLOR_PALETTE[i % len(COLOR_PALETTE)]
                for i, label in enumerate(list(unique_labels) or [])
            }
        else:
            label_colors = {}
        dot = Digraph(format="svg")
        dot.attr(rankdir="TB", nodesep="0.5", ranksep="0.5", forcelabels="true")
        add_nodes_edges(dot, cast(TreeNode, filtered_root))
        return dot

    def filter_tree(
        self,
        node: Optional[TreeNode],
        filter_conditions: Optional[Dict[str, float]] = None,
    ) -> Optional[TreeNode]:
        if filter_conditions is None:
            return node

        if not filter_conditions:
            return node

        if node is None:
            return None

        if node.feature is None:
            return node

        if isinstance(node.feature, list) and all(
            feature in filter_conditions for feature in node.feature[1]
        ):
            inputs = [
                cast(float, x) * filter_conditions[y]
                for x, y in zip(node.feature[0], node.feature[1])
            ]
            threshold = sum(inputs)
            comparison_value = threshold

        elif not isinstance(node.feature, list):
            if node.feature in filter_conditions:
                comparison_value = filter_conditions[cast(str, node.feature)]
            else:
                node.left_tree = self.filter_tree(node.left_tree, filter_conditions)
                node.right_tree = self.filter_tree(node.right_tree, filter_conditions)
                return node
        else:
            node.left_tree = self.filter_tree(node.left_tree, filter_conditions)
            node.right_tree = self.filter_tree(node.right_tree, filter_conditions)
            return node

        if isinstance(node.threshold, (str, tuple)):
            threshold = (
                " ".join(node.threshold)  # type: ignore
                if isinstance(node.threshold, tuple)
                else node.threshold
            )
            condition_met = cast(str, comparison_value) in cast(str, threshold)
        elif isinstance(node.threshold, float):
            condition_met = comparison_value <= node.threshold
        else:
            node.left_tree = self.filter_tree(node.left_tree, filter_conditions)
            node.right_tree = self.filter_tree(node.right_tree, filter_conditions)
            return node

        if condition_met:
            node.right_tree = None
            node.left_tree = self.filter_tree(node.left_tree, filter_conditions)
        else:
            node.left_tree = None
            node.right_tree = self.filter_tree(node.right_tree, filter_conditions)

        if node.left_tree is None and node.right_tree is None:
            return None

        return node

    def plot_tree_with_matplotlib(self, evaluation_mode: bool = False) -> plt.Figure:
        self.positions: Dict[str, tuple] = {}

        def add_nodes_edges(node: TreeNode, x: float, y: float, layer: int = 1) -> None:
            if node is not None:
                node_id = str(id(node))
                self.positions[node_id] = (x, y)

                if node.feature:
                    if isinstance(node.feature, list) and isinstance(
                        node.threshold, float
                    ):
                        label = (
                            f"Linearer Split:\n"
                            f"{node.feature[0][0]:.3f} * {node.feature[1][0]} + "
                            f"{node.feature[0][1]:.3f} * {node.feature[1][1]}:"
                        )
                    else:
                        label = f"{node.feature}:"
                else:
                    max_target = cast(pd.Series, node.value).idxmax()
                    label = f"Ausgangsgroesse: \n{max_target},\n Sicherheit: {cast(pd.Series, node.value).loc[max_target]:.2f}"

                if isinstance(node.threshold, float):
                    left_edge_label = f"\n<= {node.threshold:.2f}"
                else:
                    threshold = (
                        "\n".join(node.threshold)
                        if isinstance(node.threshold, tuple)
                        else node.threshold
                    )
                    left_edge_label = f"\n{threshold}"

                label += left_edge_label
                color = "lightblue" if node.feature else "lightblue"
                self.ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3", edgecolor="black", facecolor=color
                    ),
                )

                offset = 512 / (layer * 2)
                if node.left_tree is not None:
                    self.ax.plot([x, x - offset], [y - 0.1, y - 0.5], color="black")
                    self.ax.text(
                        (x + x - offset) / 2,
                        (y - 0.1 + y - 0.4) / 2,
                        "True",
                        ha="center",
                        va="bottom",
                    )
                    add_nodes_edges(node.left_tree, x - offset, y - 0.5, layer + 1)

                if node.right_tree is not None:
                    self.ax.plot([x, x + offset], [y - 0.1, y - 0.5], color="black")
                    self.ax.text(
                        (x + x + offset) / 2,
                        (y - 0.1 + y - 0.4) / 2,
                        "False",
                        ha="center",
                        va="bottom",
                    )
                    add_nodes_edges(node.right_tree, x + offset, y - 0.5, layer + 1)

        self.fig, self.ax = plt.subplots(figsize=(24, 8))
        self.ax.axis("off")
        add_nodes_edges(cast(TreeNode, self.root), 0.0, 0.0)
        return self.fig

    def build_tree_graph(
        self, unique_labels: Optional[List[str]] = None
    ) -> Union[Digraph, plt.Figure]:
        if self.root is None:
            raise ValueError("Kein Baum vorhanden. Bitte erst trainieren.")
        self._tree_graph = self._visualize_decision_tree(unique_labels=unique_labels)
        return self._tree_graph

    @property
    def tree_graph(self) -> Union[plt.Figure, Digraph]:
        if self._tree_graph is None:
            self._tree_graph = self.build_tree_graph()
        return self._tree_graph


@dataclass
class DecisionTreeClassifierGUIDE(
    GUIDEDecisionTreeBuilder, GUIDEDecisionTreeVisualiser
):
    # target_expression: Optional[pd.DataFrame] = None
    num_split_quantity: Optional[int] = None
    current_node: Optional[TreeNode] = None

    def __post_init__(self) -> None:
        if self.split_type == "univariat":
            self.use_linear_split = False

    def fit(self, df: pd.DataFrame, target: str) -> None:
        self.train_df = df
        self.target = target
        self.target_expression = df[self.target].unique()
        self._calculate_cost_matrix()
        self.root = self._build_tree(df, depth=0, name="0")

    def _predict_sample(
        self, node: TreeNode, sample: pd.Series, probabilities: bool = False
    ) -> Union[str, Dict[str, Any]]:
        if node.value is not None:
            if probabilities:
                if self.target_expression is None:
                    raise ValueError("Adjust trees before the prediction.")
                diff = np.setdiff1d(self.target_expression, np.array(node.value.index))
                return_dict = dict(zip(diff, [0] * len(diff)))
                return_dict.update(dict(node.value))
                return return_dict
            else:
                return cast(str, node.value.index[0])
        else:
            if isinstance(node.threshold, float) and not isinstance(node.feature, list):
                # univariate split
                if sample[node.feature] <= node.threshold:
                    return self._predict_sample(
                        cast(TreeNode, node.left_tree),
                        sample,
                        probabilities=probabilities,
                    )
                else:
                    return self._predict_sample(
                        cast(TreeNode, node.right_tree),
                        sample,
                        probabilities=probabilities,
                    )

            elif isinstance(node.feature, list):
                lda_sample = compute_lda(
                    df=sample, feature=node.feature, target=self.target
                )

                if lda_sample <= node.threshold:
                    return self._predict_sample(
                        cast(TreeNode, node.left_tree),
                        sample,
                        probabilities=probabilities,
                    )
                else:
                    return self._predict_sample(
                        cast(TreeNode, node.right_tree),
                        sample,
                        probabilities=probabilities,
                    )

            else:  # categorical splitting
                if node.threshold == sample[node.feature]:
                    return self._predict_sample(
                        cast(TreeNode, node.left_tree),
                        sample,
                        probabilities=probabilities,
                    )
                else:
                    return self._predict_sample(
                        cast(TreeNode, node.right_tree),
                        sample,
                        probabilities=probabilities,
                    )

    def predict(self, df: pd.DataFrame) -> List[str]:
        predictions = []
        for sample in range(len(df)):
            prediction = self._predict_sample(
                cast(TreeNode, self.root), df.iloc[sample, :]
            )
            predictions.append(cast(str, prediction))
        return predictions

    def predict_probabilities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        probabilities = []
        for sample in range(len(df)):
            prediction = self._predict_sample(
                cast(TreeNode, self.root), df.iloc[sample, :], probabilities=True
            )
            probabilities.append(cast(Dict[str, Any], prediction))
        return probabilities

    def score(self, df_test: pd.DataFrame) -> float:
        res = self.predict(df_test)
        rights = 0
        df_test.reset_index(inplace=True, drop=True)
        for r in range(len(res)):
            if res[r] == df_test.loc[r, "target"]:
                rights += 1
        return rights / len(df_test)

    def compute_rel_volume_node(self, node: TreeNode, num_snowballs: int) -> float:
        numerator = node.runs / num_snowballs
        denominator = len(node.dataset) / len(cast(pd.DataFrame, self.train_df))
        if denominator == 0:
            print("Warning: Denominator is zero. No training sample in tree node!")
            denominator = 1 / len(cast(pd.DataFrame, self.train_df))
        return numerator / denominator

    def ison_prediction_path(self, x: pd.Series, check_node: TreeNode) -> bool:
        result = self._run_prediction_path(cast(TreeNode, self.root), x, check_node)
        return result

    def _run_prediction_path(
        self, node: TreeNode, sample: pd.Series, check_node: TreeNode
    ) -> bool:
        if node == check_node:
            return True

        elif node.value is None:
            if isinstance(node.threshold, float) and not isinstance(node.feature, list):
                # univariate split
                if sample[node.feature] <= node.threshold:
                    return self._run_prediction_path(
                        cast(TreeNode, node.left_tree), sample, check_node
                    )
                else:
                    return self._run_prediction_path(
                        cast(TreeNode, node.right_tree), sample, check_node
                    )

            elif isinstance(node.feature, list):
                lda_sample = compute_lda(
                    df=sample, feature=node.feature, target=self.target
                )

                if lda_sample <= node.threshold:
                    return self._run_prediction_path(
                        cast(TreeNode, node.left_tree), sample, check_node
                    )
                else:
                    return self._run_prediction_path(
                        cast(TreeNode, node.right_tree), sample, check_node
                    )

            else:  # categorical splitting
                if node.threshold == sample[node.feature]:
                    return self._run_prediction_path(
                        cast(TreeNode, node.right_tree), sample, check_node
                    )
                else:
                    return self._run_prediction_path(
                        cast(TreeNode, node.left_tree), sample, check_node
                    )

        return False

    def evaluate(self) -> None:
        self._tree_graph = self.build_tree_graph()

    def _calculate_cost_matrix(self) -> None:
        # calculate number of different targets and their number of occurrence
        num_classes = len(self.target_expression)
        sum_classes = self.train_df[self.target].value_counts()
        sum_classes = sum_classes.reindex(self.target_expression, fill_value=0)

        self.classCost = np.ones((num_classes, num_classes), dtype=float)
        np.fill_diagonal(self.classCost, 0.0)

        if self.balanceCosts:
            for i in range(num_classes):
                for j in range(num_classes):
                    class_i = self.target_expression[i]
                    class_j = self.target_expression[j]

                    if class_i != class_j:
                        cost = max((sum_classes.get(class_i, 0) / sum_classes.get(class_j, 0)),1) # calculate missclassification costs
                    else:
                        cost = 0

                    self.classCost[i, j] = cost


def train_decision_tree(
    df: pd.DataFrame, min_info_gain: float = 0.075, use_linear_split: bool = True, balance_costs: bool = True
) -> DecisionTreeClassifierGUIDE:
    dtc = DecisionTreeClassifierGUIDE(
        min_info_gain=min_info_gain, use_linear_split=use_linear_split, balanceCosts=balance_costs, max_depth=3
    )
    dtc.fit(df, target="target")
    return dtc


class M5PrimeTreeClassifier(M5Prime):
    def preprocess_(self, df: pd.DataFrame) -> pd.DataFrame:
        df_train_encoded = df.copy()
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_train_encoded[col] = le.fit_transform(df[col].astype(str))
        return df_train_encoded

    def fit(self, df: pd.DataFrame) -> 'M5PrimeTreeClassifier':
        X_train, y_train = one_hot_enc(df)
        self.feature_names_ = X_train.columns.tolist()
        super().fit(X_train, y_train)
        return self

    def replace_special_characters(self, text: str) -> str:
        text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss').replace('°', 'Grad')
        return text

    def _visualize_decision_tree(self) -> Union[Digraph, Source, plt.Figure]:
        try:
            subprocess.run(
                ["dot", "-V"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            dot = self._visualize_with_graphviz()
            return dot

        except FileNotFoundError:
            return self.plot_tree_with_matplotlib()

    def _visualize_with_graphviz(self):
        feature_names_cleaned = [self.replace_special_characters(name) for name in self.feature_names_]
        dot_data = export_graphviz(
            self,
            out_file=None,
            feature_names=feature_names_cleaned,
            filled=True,
            rounded=True,
            special_characters=False,
        )
        return graphviz.Source(dot_data)

    def plot_tree_with_matplotlib(self):
        # Visualisierung des Entscheidungsbaums mit matplotlib
        plt.figure(figsize=(20, 10))  # Größe des Plots anpassen
        plot_tree(
            self,
            feature_names=self.feature_names_,
            filled=True,
            rounded=True,
            fontsize=12,
            precision=0,
        )
        return plt

    def build_tree_graph(
        self, unique_labels: Optional[List[str]] = None
    ) -> Union[Digraph, plt.Figure]:
        self._tree_graph = self._visualize_decision_tree()
        return self._tree_graph

    @property
    def tree_graph(self) -> Union[plt.Figure, Digraph]:
        if self._tree_graph is None:
            self._tree_graph = self.build_tree_graph()
        return self._tree_graph

    def evaluate(self) -> None:
        self._tree_graph = self.build_tree_graph()


def train_regression_decision_tree(df: pd.DataFrame, use_pruning: bool = True, use_smoothing: Optional[bool] = None) -> M5PrimeTreeClassifier:
    dtc = M5PrimeTreeClassifier(use_pruning=use_pruning, use_smoothing=use_smoothing)
    dtc.fit(df)
    return dtc