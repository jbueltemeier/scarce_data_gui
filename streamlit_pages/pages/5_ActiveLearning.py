import streamlit as st

from scarce_data_gui.page_parts import (
    add_funding_notice,
    add_redirection_buttons,
    display_active_learning_slider,
    display_results,
    load_css,
    remove_sidebar,
)
from scarce_data_gui.utils import save_dataset


def init_active_learning() -> None:
    try:
        with st.spinner("Initialisiere ungelabelten Datenpool für Active Learning..."):
            st.session_state["dataset"].init_active_learning(
                columns_to_keep=st.session_state["selected_active_learning_columns"],
                active_learning_label=st.session_state[
                    "selected_active_learning_label"
                ],
            )
            save_dataset(
                st.session_state["dataset"],
                st.session_state["dataset_filename"],
                db=st.session_state["db"],
            )
    except ValueError as e:
        print(f"Error: {e}")


def optional_reduce_al_params() -> None:
    col1, col2 = st.columns(2)
    with col1:
        filtered_columns = [
            col
            for col in st.session_state["dataset"].optimised_df.columns
            if (col.endswith("_numerical") or col.endswith("_categorical"))
        ]
        st.divider()
        st.header(
            "Wählen Sie die Parameter aus, die für das Active Learning berücksichtigt werden sollen:"
        )
        selected_columns = []
        for column in filtered_columns:
            if st.checkbox(column, value=True):
                selected_columns.append(column)
        st.session_state["selected_active_learning_columns"] = selected_columns

    with col2:
        labels = st.session_state["dataset"].get_labels()
        label_columns = labels.columns.tolist()
        st.divider()
        st.header(
            "Wählen Sie das Label, für das Active Learning durchgeführt werden soll:"
        )
        st.session_state["selected_active_learning_label"] = st.radio(
            "Label auswählen", label_columns
        )


def info_active_learning() -> None:
    st.divider()
    st.info(
        """
    ### ℹ️ Hinweis für die Initialisierung des Active Learning
    Bevor das Active Learning gestartet wird, müssen die Eingangsgrößen und eine Ausgangsgröße festgelegt werden, die für das Training verwendet werden sollen. Dadurch ist es möglich, bestimmte Eingangsgrößen, die sich als nicht relevant erwiesen haben, vom Training auszuschließen, um das Active Learning gezielt zu beeinflussen. Diese Anpassung sollte jedoch nur von erfahrenen Anwendern vorgenommen werden. Für alle anderen empfiehlt es sich, die Standardeinstellungen zu verwenden, bei denen automatisch alle Eingangsgrößen verwendet werden.

    Ist die Auswahl getroffen, kann mit dem Button *Initialisieren* die Initialisierung durchgeführt und das Active Learning initialisiert werden. Die Initialisierung kann einige Zeit in Anspruch nehmen, da zunächst ein größerer Pool an neuen Datenpunkten erzeugt werden muss.
    ### ℹ️ Für unerfahrene Benutzer
    Einfach die Standardeinstellungen verwenden und Active Learning durch *Initialisieren* initalisieren.
    """
    )
    st.divider()


# -------------------------Initalisation Page-------------------------------------------
st.set_page_config(layout="wide")
remove_sidebar()

load_css()
col1, col2 = st.columns([0.6, 0.4])
with col1:
    add_redirection_buttons(exclude="ActiveLearning")

with col2:
    add_funding_notice()

labels = st.session_state["dataset"].user_label_container.combine_labels()
if (labels == "unlabeled").any().any() and st.session_state[
    "dataset"
].active_learning_pool.empty:
    st.warning(
        "Der Datensatz enthält noch ungelabelte Daten. Active Learning trotzdem mit dem aktuellen Stand initialisieren?"
    )
    info_active_learning()
    optional_reduce_al_params()
    if st.button("Initialisieren"):
        init_active_learning()
        st.rerun()
else:
    if st.session_state["dataset"].active_learning_pool.empty:
        info_active_learning()
        optional_reduce_al_params()
        if st.button("Initialisieren"):
            init_active_learning()
            st.rerun()
    else:
        num_labeled_datapoints = len(labels)
        num_datapoints = st.session_state["dataset"].num_datapoints
        if num_labeled_datapoints < num_datapoints:
            st.warning("Es sind noch nicht alle Datenpunkte gelabelt!")
        else:
            col1, col2 = st.columns([3, 1], gap="large")
            display_results(col1, active_learning_mode=True)
            display_active_learning_slider(col2)
