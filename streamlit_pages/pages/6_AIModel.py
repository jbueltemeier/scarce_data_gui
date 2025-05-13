import os
import io
import pickle
import tempfile

import pandas as pd
import numpy as np
from m5py import M5Prime
from graphviz import Source
import streamlit as st
import matplotlib.pyplot as plt

from experimental_design.visualisation import remove_extension_df_columns
from guide_active_learning.GUIDE import DecisionTreeClassifierGUIDE
from scarce_data_gui.page_parts import (
    add_ai_model_train_button,
    add_funding_notice,
    add_redirection_buttons,
    load_css,
    remove_sidebar,
)

st.set_page_config(layout="wide")
remove_sidebar()


if "tree_filter_conditions" not in st.session_state:
    st.session_state["tree_filter_conditions"] = None
if st.session_state["tree_filter_conditions"] is not None:
    st.session_state["tree_filter_conditions"] = None

if "selected_columns" not in st.session_state:
    st.session_state["selected_columns"] = []

if "selected_df" not in st.session_state:
    st.session_state["selected_df"] = pd.DataFrame.empty

if "selected_labels" not in st.session_state:
    st.session_state["selected_labels"] = pd.DataFrame.empty

if "use_linear_split" not in st.session_state:
    st.session_state["use_linear_split"] = True

if "balance_costs" not in st.session_state:
    st.session_state["balance_costs"] = True


if "use_pruning" not in st.session_state:
    st.session_state["use_pruning"] = True

if "use_smoothing" not in st.session_state:
    st.session_state["use_smoothing"] = True

if "use_custom_tree" not in st.session_state:
    st.session_state["use_custom_tree"] = False

if "min_info_gain" not in st.session_state:
    st.session_state["min_info_gain"] = 0.05
    if "dataset_filename" in st.session_state:
        if "custom" in st.session_state["dataset_filename"]:
            st.session_state["min_info_gain"] = 0.01

load_css()
col1, col2 = st.columns([0.6, 0.4])
with col1:
    add_redirection_buttons("AIModel")

with col2:
    add_funding_notice()

col1, _, col2 = st.columns([0.36, 0.04, 0.6])

with col1:
    st.header("Einstellungen Guide Entscheidungsbaum:")

    df = st.session_state["dataset"].get_dataframe(
        design_space=False, remove_extension=False
    )
    filtered_columns = [
        col
        for col in df.columns
        if (col.endswith("_numerical") or col.endswith("_categorical"))
    ]

    filtered_columns = remove_extension_df_columns(filtered_columns)
    st.divider()
    st.header(
        "Wählen Sie die Parameter aus, die im Training berücksichtigt werden sollen:"
    )
    selected_columns = []
    for column in filtered_columns:
        if st.checkbox(column, value=True):
            selected_columns.append(column)
    df = st.session_state["dataset"].get_dataframe(design_space=False)
    st.session_state["selected_columns"] = selected_columns
    st.session_state["selected_df"] = df[st.session_state["selected_columns"]]

    labels = st.session_state["dataset"].get_labels()

    label_columns = labels.columns.tolist()
    st.divider()
    st.header("Wählen Sie die Label aus, für die ein Baum trainiert werden soll:")
    selected_label_columns = []
    for column in label_columns:
        if st.checkbox(column, value=True):
            selected_label_columns.append(column)
    st.session_state["selected_labels"] = labels[selected_label_columns]
    st.divider()
    st.header("Einstellungen Training (Klassifikation):")

    st.subheader("Minimaler Informationsgewinn für eine weitere Verzweigung")
    st.session_state["min_info_gain"] = st.number_input(
        label="Verringern Sie diesen Wert, wenn kein Baum gebildet werden kann oder nur wenige "
        "Verzweigungen vorhanden sind.",
        value=st.session_state["min_info_gain"],
    )

    st.subheader("Achsenschräger Split mehrerer Parameter aktivieren")
    st.session_state["use_linear_split"] = st.checkbox(
        label="Aktiviere Linear Split",
        value=st.session_state["use_linear_split"],
    )

    st.subheader("Unausgewogene Klassen ausgleichen")
    st.session_state["balance_costs"] = st.checkbox(
        label="Aktiviere ausgewogene Klassen",
        value=st.session_state["balance_costs"],
    )
    st.divider()
    st.header("Einstellungen Training (Regression):")

    st.subheader("Überanpassung vermeiden und die Vorhersagegenauigkeit auf neuen Daten verbessern")
    st.session_state["use_pruning"] = st.checkbox(
        label="Pruning der Bäume verwenden",
        value=st.session_state["use_pruning"],
    )

    st.subheader("Schätzung glätten und Ausreißer reduzieren")
    st.session_state["use_smoothing"] = st.checkbox(
        label="Smoothing des Baumes",
        value=st.session_state["use_smoothing"],
    )
    st.divider()


with col2:
    if (
        len(st.session_state["selected_columns"]) >= 2
        and not st.session_state["selected_labels"].empty
    ):
        col11, col12 = st.columns([0.5, 0.5])

        with col11:
            st.header("Entscheidungsbäume trainieren")
            add_ai_model_train_button()

            if "custom" in st.session_state["dataset_filename"]:
                st.session_state["use_custom_tree"] = st.checkbox(
                    label="Verwende den Custom Baum",
                    value=st.session_state["use_custom_tree"],
                )
                st.session_state["dataset"].use_custom_tree = st.session_state["use_custom_tree"]

            # integrate the trained custom tree
            uploaded_file = st.file_uploader(
                "Lade eine .pkl-Datei mit einem Entscheidungsbaum hoch. ",
                type="pkl",
            )

            if uploaded_file is not None:
                try:
                    file_buffer = io.BytesIO(uploaded_file.read())
                    tree = pickle.load(file_buffer)
                    if isinstance(tree, DecisionTreeClassifierGUIDE):
                        if "gea" in st.session_state["dataset_filename"]:
                            tree.train_df['target'] = tree.train_df['target'].replace({
                                'separator': 'Separator',
                                'decanter': 'Dekanter',
                                'both': 'Separator_Dekanter'
                            })
                            tree.train_df = tree.train_df.rename(
                                columns={
                                    'consSolids': 'Instanz_consSolids',
                                    'turbLightPhase': 'Instanz_turbLightPhase',
                                    'throughput': 'throughput (in m^3/h)',
                                    'tMax': 'Instanz_tMax (in Sekunden)',
                                    'abrasion': 'Instanz_abrasion',
                                    'targetPhase': 'Aufgabe',
                                }
                            )
                            # tree.train_df['Instanz_abrasion'] = tree.train_df['Instanz_abrasion'].astype(bool)
                            #
                            # consSolids_list = ['very soft', 'soft', 'slightly pasty', 'pasty', 'pasty to solid', 'solid',
                            #                    'hard']
                            # mapping_dict = {float(i + 1): wert for i, wert in enumerate(consSolids_list)}
                            # tree.train_df['Instanz_consSolids'] = tree.train_df['Instanz_consSolids'].map(mapping_dict)
                            #
                            # consSolids_list = ['clear', 'slightly turbid', 'turbid', 'heavily turbid']
                            # mapping_dict = {float(i + 1): wert for i, wert in enumerate(consSolids_list)}
                            # tree.train_df['Instanz_turbLightPhase'] = tree.train_df['Instanz_turbLightPhase'].map(
                            #     mapping_dict)

                            tree.target_expression = np.array(['Separator', 'Dekanter', 'Separator_Dekanter'])
                            tree.fit(tree.train_df, target="target")
                            tree.build_tree_graph(unique_labels=pd.unique(labels["Maschinentyp"]))
                        st.session_state["dataset"].custom_tree = tree
                    else:
                        st.warning("Die pickle Datei enthält keinen bekannten Entscheidungsbaum.")
                except Exception as e:
                    st.error(f"Fehler beim Laden der Datei: {e}")

        with col12:
            if st.session_state["dataset"].decision_trees is not None:
                st.header("Download Entscheidungsbäume")

                for name, tree in st.session_state["dataset"].decision_trees.items():
                    # Temporäre Datei speichern
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f"_{name}_tree.pkl"
                    ) as tmp_file:
                        pickle.dump(tree, tmp_file)
                        tmp_filename = tmp_file.name

                    # Download-Button
                    with open(tmp_filename, "rb") as file:
                        st.download_button(
                            label=f"Download {name}",
                            data=file,
                            file_name=os.path.basename(tmp_filename),
                            mime="application/octet-stream",
                            type="primary",
                        )

    else:
        st.warning(
            "Wählen Sie zunächst mehrere Parameter und mindestens ein Label aus!"
        )
    if (
        st.session_state["dataset"].decision_trees is not None
        and len(st.session_state["selected_columns"]) >= 2
        and not st.session_state["selected_labels"].empty
    ):
        for label, decision_tree in st.session_state["dataset"].decision_trees.items():
            try:
                if label == "Maschinentyp" and st.session_state["use_custom_tree"]:
                    tree = st.session_state["dataset"].custom_tree or decision_tree
                else:
                    tree = decision_tree
                st.divider()
                st.header(f"Entscheidungsbaum für Label: {label}")

                tree_graph = tree.build_tree_graph(
                    unique_labels=pd.unique(labels[label])
                )
                if isinstance(tree_graph, plt.Figure):
                    st.pyplot(tree_graph)
                else:
                    st.image(tree_graph.render(format="svg"))

            except ValueError:
                st.header("Das Model konnte nicht visualisiert werden.")
        st.divider()
    else:
        st.warning("Trainiere zunächst einen Baum!")
        st.divider()

    st.subheader("Ausgewählte Parameter:")
    if st.session_state["selected_columns"]:
        show_df = pd.concat(
            [st.session_state["selected_df"], st.session_state["selected_labels"]],
            axis=1,
        )
        st.dataframe(show_df.astype(str), width=2000, height=500)
    else:
        st.write("Keine Spalten ausgewählt.")
