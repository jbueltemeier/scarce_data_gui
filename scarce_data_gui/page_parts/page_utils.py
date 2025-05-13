from typing import Any, Optional

import streamlit as st

from scarce_data_gui.utils import (
    check_num_instances,
    create_dataset_header,
    INSTANCE_META,
    INSTANCE_PARAMS,
    INSTANCE_SETTINGS,
    load_dataset,
)

__all__ = [
    "check_design_condition",
    "load_dataset_initialisation",
    "reset_design_data",
    "reset_data_instances",
    "reset_evaluation_data",
]


def check_design_condition() -> bool:
    if len(st.session_state[create_dataset_header(header=INSTANCE_SETTINGS)]) == 0:
        st.warning(f"Erstelle zunächst die {INSTANCE_SETTINGS}.")
        return False
    elif not any(
        value.__class__.__name__ == "NumSamplesInstancePart"
        for value in st.session_state[
            create_dataset_header(header=INSTANCE_SETTINGS)
        ].values()
    ):
        st.warning(
            f"Erstelle zunächst eine *Anzahl der Samples* Instanz in {INSTANCE_SETTINGS}."
        )
        return False
    elif len(st.session_state[create_dataset_header(header=INSTANCE_SETTINGS)]) < 2:
        st.warning(f"Erstelle zunächst eine weitere {INSTANCE_SETTINGS} Instanz.")
        return False
    elif len(st.session_state[create_dataset_header(header=INSTANCE_PARAMS)]) == 0:
        st.warning(f"Zuerst Instanzen zu {INSTANCE_PARAMS} hinzufügen.")
        return False
    else:
        num_numerical_instances = check_num_instances(
            instance_type="numerical",
            current_instances=st.session_state[
                create_dataset_header(header=INSTANCE_PARAMS)
            ].values(),
        )

        if num_numerical_instances <= 1:
            st.warning("Erstelle zunächst eine weitere numerische Instanz.")
            return False

    return True


def load_dataset_initialisation(dataset: Optional[Any] = None) -> None:
    if dataset is None:
        st.session_state["dataset"] = load_dataset(
            st.session_state["dataset_filename"],
            db=st.session_state["db"],
        )
    else:
        st.session_state["dataset"] = dataset
    st.session_state[create_dataset_header(header=INSTANCE_META)] = (
        st.session_state["dataset"].part_design_container.sub_parts["meta"].sub_parts
    )
    st.session_state[create_dataset_header(header=INSTANCE_PARAMS)] = (
        st.session_state["dataset"]
        .part_design_container.sub_parts["experimental_design"]
        .sub_parts
    )
    st.session_state[
        create_dataset_header(header=INSTANCE_SETTINGS)
    ] = st.session_state["dataset"].settings.sub_parts

    st.session_state["trees"] = None


def reset_design_data() -> None:
    st.session_state["dataset"] = None
    for header in [INSTANCE_META, INSTANCE_PARAMS, INSTANCE_SETTINGS]:
        st.session_state[create_dataset_header(header=header)] = {}
    st.session_state["dataset_filename"] = ""
    st.session_state["trees"] = None


def reset_data_instances(current_page: str) -> None:
    if current_page == "Design":
        for header in [INSTANCE_META, INSTANCE_PARAMS, INSTANCE_SETTINGS]:
            data_instance = create_dataset_header(header=header)
            save_instance = f"{data_instance}_save"
            if save_instance in st.session_state:
                st.session_state[data_instance] = st.session_state[save_instance]
                st.session_state[save_instance] = None


def reset_evaluation_data(current_page: str) -> None:
    if current_page == "Evaluation":
        if st.session_state["tree_filter_conditions"] is not None:
            for column in st.session_state["tree_filter_conditions"].keys():
                column_name = f"{column}_evaluation"
                if column_name in st.session_state:
                    del st.session_state[column_name]
