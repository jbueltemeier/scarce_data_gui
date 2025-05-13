import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st
from scarce_data_gui.page_parts import (
    add_funding_notice,
    add_redirection_buttons,
    display_evaluation_data,
    load_css,
    remove_sidebar,
)

if "tree_filter_conditions" not in st.session_state:
    st.session_state["tree_filter_conditions"] = None

if "use_custom_tree" not in st.session_state:
    st.session_state["use_custom_tree"] = False


st.set_page_config(layout="wide")
remove_sidebar()

load_css()
col1, col2 = st.columns([0.6, 0.4])
with col1:
    add_redirection_buttons("Evaluation")

with col2:
    add_funding_notice()

if st.session_state["dataset"].decision_trees is None:
    st.subheader("Es muss zunächst ein Model trainiert werden.")
else:
    st.title("Entscheidungsbaum-Evaluation")
    col1, _, col2 = st.columns([0.25, 0.1, 0.65])
    with col1:
        st.subheader("Auswahl Evaluation (Ausgangsgröße)")
        st.session_state["tree_evaluation_selection"] = st.selectbox(
            "Auswahl Ausgangsgröße:",
            list(st.session_state["dataset"].decision_trees.keys()),
        )

        st.subheader("Eingabe der Parameter")
        st.divider()
        if st.button("Restart", type="primary"):
            for column in st.session_state["tree_filter_conditions"].keys():
                column_name = f"{column}_evaluation"
                st.session_state[column_name] = None

            del st.session_state["tree_filter_conditions"]
            st.rerun()
        display_evaluation_data()
        st.divider()
        st.session_state["dataset"].decision_trees[
            st.session_state["tree_evaluation_selection"]
        ].evaluate()

    with col2:
        labels = st.session_state["dataset"].get_labels()
        label = st.session_state["tree_evaluation_selection"]
        decision_tree = st.session_state["dataset"].decision_trees[label]
        if label == "Maschinentyp" and st.session_state["use_custom_tree"]:
            tree = st.session_state["dataset"].custom_tree or decision_tree
        else:
            tree = decision_tree

        tree_graph = tree.build_tree_graph(
            unique_labels=pd.unique(labels[label])
        )
        if isinstance(tree_graph, plt.Figure):
            st.pyplot(tree_graph)
        else:
            st.image(tree_graph.render(format="svg"))
