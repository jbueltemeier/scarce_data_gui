import base64
import os
from typing import Union

import pandas as pd

import streamlit as st

from experimental_design.dataset import create_dataset
from experimental_design.optimisation import (
    PhiPSimulatedAnnealing,
    SlicedPhiPSimulatedAnnealing,
)
from guide_active_learning.GUIDE import train_decision_tree, train_regression_decision_tree
from st_bridge import bridge, html
from streamlit_extras.switch_page_button import switch_page

from scarce_data_gui.misc import get_default_pages_images_path, get_default_pages_path
from experimental_design.core import is_int
from scarce_data_gui.page_parts.page_utils import (
    check_design_condition,
    reset_data_instances,
    reset_design_data,
    reset_evaluation_data,
)
from scarce_data_gui.utils import (
    create_dataset_header,
    INSTANCE_META,
    INSTANCE_PARAMS,
    INSTANCE_SETTINGS,
    save_dataset,
)

__all__ = [
    "load_css",
    "add_funding_notice",
    "add_home_image_buttons",
    "add_redirection_buttons",
    "add_design_specific_button",
    "add_optimise_button",
    "add_dataset_save_button",
    "add_ai_model_train_button",
]


def get_base64_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def load_css() -> None:
    css_path = get_default_pages_path()
    filename = "gui_style.css"
    with open(os.path.join(css_path, filename)) as f:
        css_data = f.read()
    st.markdown(f"<style>{css_data}</style>", unsafe_allow_html=True)


def add_funding_notice() -> None:
    bild1 = get_base64_image(
        os.path.join(
            get_default_pages_images_path(), "Demonstratoren_Foerderhinweis.png"
        )
    )
    html_code = f"""
    <div style="background-color: white; padding: 20px; text-align: center; margin: 20px;">
        <img src="data:image/png;base64,{bild1}" class="image">
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)


def get_image_button_container() -> str:
    bild1 = get_base64_image(
        os.path.join(get_default_pages_images_path(), "workflow_ai4scada.png")
    )
    return f"""
    <div class="image-container-home">
        <img src="data:image/png;base64,{bild1}" class="image">
        <button class="button-home button1 its-owl-color" onClick="stBridges.send('my-bridge', 'page_design')">Versuchsplanung</button>
        <button class="button-home button2 its-owl-color"onClick="stBridges.send('my-bridge', 'page_labeling')">Labeling</button>
        <button class="button-home button3 its-owl-color"onClick="stBridges.send('my-bridge', 'page_active_learning')">Active Learning</button>
        <button class="button-home button4 its-owl-color"onClick="stBridges.send('my-bridge', 'page_ai_model')">GUIDE Baum</button>
        <button class="button-home button5 its-owl-color"onClick="stBridges.send('my-bridge', 'page_dataset')">tab. Datensatz</button>
        <button class="button-home button6 its-owl-color"onClick="stBridges.send('my-bridge', 'page_evaluation')">Evaluation</button>
    </div>
    """


def check_dataset_exists() -> bool:
    if st.session_state["dataset"] is not None:
        return True
    else:
        st.write("No dataset selected")
        return False


def add_home_image_buttons() -> None:
    data = bridge("my-bridge", default="no button is clicked")
    html(get_image_button_container())
    if data == "page_design":
        switch_page("Design")
    if data == "page_dataset":
        if check_dataset_exists():
            switch_page("Dataset")
    if data == "page_labeling":
        if check_dataset_exists():
            switch_page("Login")
    if data == "page_active_learning":
        if check_dataset_exists():
            switch_page("ActiveLearning")
    if data == "page_ai_model":
        if check_dataset_exists():
            switch_page("AIModel")
    if data == "page_evaluation":
        if check_dataset_exists():
            switch_page("Evaluation")


def get_redirection_buttons(exclude: str = "") -> str:
    buttons = {
        "Home": "Home",
        "Design": "Design",
        "Dataset": "Dataset",
        "Labeling": "Labeling",
        "AIModel": "Model",
        "Evaluation": "Evaluation",
        "ActiveLearning": "Active Learning",
    }

    button_html = '<div class="parent-container"><div class="button-container-red">\n'
    for key, label in buttons.items():
        if key != exclude:
            button_html += f"<button class=\"button-redirection its-owl-color\" onClick=\"stBridges.send('my-bridge-redirection', '{key}')\">{label}</button>\n"
        else:
            button_html += f'<button class="button-redirection-current its-owl-color">{label}</button>\n'
    button_html += "</div></div>"

    return button_html


def dataset_exists() -> bool:
    if st.session_state["dataset"] is not None:
        return True
    else:
        st.warning("Erstellen oder laden Sie zunächst einen Datensatz!")
        return False


def is_dataset_saved() -> bool:
    if "dataset_filename" not in st.session_state:
        st.warning("Speichern Sie zunächst den aktuellen Datensatz!")
        return False
    return True


def add_redirection_buttons(exclude: str = "") -> None:
    data = bridge("my-bridge-redirection", default="no button is clicked")
    html(get_redirection_buttons(exclude=exclude))
    if data == "Home":
        reset_data_instances(exclude)
        reset_evaluation_data(exclude)
        switch_page("Home")
    if data == "Design":
        switch_page("Design")
    if data == "Dataset":
        if dataset_exists():
            reset_data_instances(exclude)
            reset_evaluation_data(exclude)
            switch_page("Dataset")
    if data == "Labeling":
        if dataset_exists() and is_dataset_saved():
            reset_data_instances(exclude)
            reset_evaluation_data(exclude)
            switch_page("Login")
    if data == "AIModel":
        if dataset_exists() and is_dataset_saved():
            reset_data_instances(exclude)
            reset_evaluation_data(exclude)
            switch_page("AIModel")
    if data == "ActiveLearning":
        if dataset_exists() and is_dataset_saved():
            reset_data_instances(exclude)
            reset_evaluation_data(exclude)
            switch_page("ActiveLearning")
    if data == "Evaluation":
        if dataset_exists() and is_dataset_saved():
            reset_data_instances(exclude)
            reset_evaluation_data(exclude)
            switch_page("Evaluation")


def get_container_design_button() -> str:
    return """
    <div class="button-container vertical-center">
        <button class="button-redirection its-owl-color" onClick="stBridges.send('my-bridge-design', 'Neues Design')">Neues Design</button>
        <button class="button-redirection its-owl-color" onClick="stBridges.send('my-bridge-design', 'Design erstellen')">Design erstellen</button>
        <button class="button-redirection its-owl-color" onClick="stBridges.send('my-bridge-design', 'Design bearbeiten')">Design bearbeiten</button>
    </div>
    """


def add_design_specific_button() -> None:
    col_1, col_2, col_3 = st.columns([0.3, 0.3, 0.4])
    with col_1:
        st.subheader("Erstellen des Designs")
        if st.button("Design erstellen", type="primary"):
            try:
                if check_design_condition():
                    if (
                        st.session_state["dataset"] is None
                        or st.session_state["active_rework"]
                    ):

                        with st.spinner("Design wird initialisiert..."):
                            st.session_state["dataset"] = create_dataset(
                                meta_parts=st.session_state[
                                    create_dataset_header(header=INSTANCE_META)
                                ],
                                experimental_design_parts=st.session_state[
                                    create_dataset_header(header=INSTANCE_PARAMS)
                                ],
                                settings=st.session_state[
                                    create_dataset_header(header=INSTANCE_SETTINGS)
                                ],
                            )
                    switch_page("Dataset")
            except ValueError as e:
                st.error(e)
                st.warning(
                    "Bitte ändern Sie die Anzahl der Samples in ein Vielfaches der Slices."
                )
        if st.session_state["dataset"] is not None:
            st.write(
                "Das Design wird nur überschrieben, wenn die Bearbeitung ''Aktiv'' ist!"
            )
    with col_2:
        st.subheader("Design zurücksetzen")
        if st.button("Neues Design", type="primary"):
            reset_design_data()
    with col_3:
        st.subheader("Änderungen am Design vornehmen")
        if st.button("Design bearbeiten", type="primary"):
            st.session_state["active_rework"] = not st.session_state["active_rework"]

        if st.session_state["active_rework"]:
            st.warning(
                "Bearbeiten Aktiv: Erstelle zum Bearbeiten eine Instanz mit dem gleichen Namen.",
                icon="⚠️",
            )


def get_optimise_buttons() -> str:
    return """
    <div class="button-container vertical-center">
        <button class="button-redirection its-owl-color" onClick="stBridges.send('my-bridge-optimise', 'Optimise')">Datensatz Optimieren</button>
    </div>
    """


def add_optimise_button(confirmation: bool) -> None:
    optimizer: Union[PhiPSimulatedAnnealing, SlicedPhiPSimulatedAnnealing]
    if st.button("Optimieren", type="primary") and confirmation:
        if len(st.session_state["dataset"].categorical_columns) == 0:
            optimizer = PhiPSimulatedAnnealing()
        else:
            optimizer = SlicedPhiPSimulatedAnnealing()
        with st.spinner("Berechnung läuft..."):
            st.session_state["dataset"] = optimizer.perform_optimisation(
                st.session_state["dataset"]
            )


def get_dataset_save_button() -> str:
    return """
    <div class="button-container vertical-center">
        <button class="button-redirection its-owl-color" onClick="stBridges.send('my-bridge-dataset', 'Save')">Datensatz Speichern</button>
    </div>
    """


def add_dataset_save_button(filename: str) -> None:
    st.subheader("Generierten Datensatz abspeichern ")
    if st.button("Save", type="primary"):
        if filename:
            filename += ".pkl" if "pkl" not in filename else ""
            save_dataset(dataset=st.session_state["dataset"], filename=filename)
            st.write("Erfolgreich gespeichert.")
            st.session_state["dataset_filename"] = filename
        else:
            st.warning("Geben Sie einen Namen für den Datensatz an.")


def get_ai_model_train_button() -> str:
    return """
    <div class="button-container vertical-center">
        <button class="button-redirection its-owl-color" onClick="stBridges.send('my-bridge-train', 'Train')">Training</button>
    </div>
    """


def add_ai_model_train_button() -> None:
    if st.button("Training", type="primary"):
        labels = st.session_state["dataset"].user_label_container.combine_labels()
        if st.session_state["dataset"].active_learning_label_container is not None:
            active_learning_labels = st.session_state[
                "dataset"
            ].active_learning_label_container.combine_labels()
            labels = pd.concat([labels, active_learning_labels], ignore_index=True)
            labels.loc[len(labels)] = "unlabeled"
        reduced_labels = labels[
            ~labels.iloc[:, 0].astype(str).str.contains("unlabeled", na=False)
        ]
        if reduced_labels.empty:
            st.warning("Es sind noch keine Daten gelabelt!")
        else:
            num_labeled_datapoints = len(reduced_labels)
            num_datapoints = st.session_state["dataset"].num_datapoints
            st.write(
                f"{num_labeled_datapoints} Datenpunkte werden für das Training verwendet."
            )
            missing_labels = len(labels) - num_labeled_datapoints
            st.write(
                f"{missing_labels} Datenpunkte sind nicht gelabelt und werden daher nicht für das Training verwendet."
            )
            if num_labeled_datapoints < num_datapoints:
                st.write(
                    "Es sind noch nicht alle Datenpunkte gelabelt, eine Merkmalsraumabdeckung kann im AI "
                    "Model aber nur bei vollständiger Labelung gewährleistet werden!"
                )
            df_train = st.session_state["selected_df"].iloc[0:num_labeled_datapoints]

            with st.spinner("Berechnung läuft..."):
                trees = {}
                for label in st.session_state["selected_labels"]:
                    df_train = df_train.dropna()
                    target = reduced_labels[label]
                    if target.apply(is_int).all():  # regression
                        df_train["target"] = target
                        dtc = train_regression_decision_tree(
                            df_train,
                            use_pruning=st.session_state["use_pruning"],
                            use_smoothing=st.session_state["use_smoothing"],
                        )
                    else:  # classification
                        df_train["target"] = target.astype(str)
                        dtc = train_decision_tree(
                            df_train,
                            min_info_gain=st.session_state["min_info_gain"],
                            use_linear_split=st.session_state["use_linear_split"],
                            balance_costs=st.session_state["balance_costs"]
                        )
                    trees[label] = dtc
                st.session_state["dataset"].decision_trees = trees
                st.session_state["use_custom_tree"] = False
                save_dataset(
                    st.session_state["dataset"],
                    filename=st.session_state["dataset_filename"],
                )
