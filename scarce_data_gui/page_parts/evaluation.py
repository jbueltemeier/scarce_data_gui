from typing import cast

import pandas as pd

import streamlit as st

from experimental_design.visualisation import remove_extension_df_columns

__all__ = [
    "display_evaluation_data",
]


def user_input_from_dataframe(df: pd.DataFrame) -> None:
    if st.session_state["tree_filter_conditions"] is None:
        st.session_state["tree_filter_conditions"] = {}
    filtered_columns = [
        col
        for col in st.session_state["dataset"].optimised_df.columns
        if (col.endswith("_numerical") or col.endswith("_categorical"))
    ]
    filtered_columns = remove_extension_df_columns(filtered_columns)
    if not st.session_state["use_custom_tree"]:
        df = df[[col for col in df.columns if col in filtered_columns]]
    for column in df.columns:
        column_name = f"{column}_evaluation"
        if pd.api.types.is_numeric_dtype(df[column]):
            st.number_input(
                f"{column} eingeben",
                format="%.2f",
                value=None,
                key=column_name,
            )
        else:
            unique_values = df[column].dropna().unique().tolist()

            st.selectbox(
                f"{column} auswählen",
                options=[None] + unique_values,
                index=0,
                key=column_name,
            )
        cast(dict, st.session_state["tree_filter_conditions"])[
            column
        ] = st.session_state[column_name]

    st.session_state["tree_filter_conditions"] = {
        k: v
        for k, v in cast(dict, st.session_state["tree_filter_conditions"]).items()
        if v is not None
    }


def display_evaluation_data() -> None:
    df = st.session_state["dataset"].get_dataframe(design_space=False, use_custom_tree=st.session_state["use_custom_tree"])
    user_input_from_dataframe(df)
