import streamlit as st

from experimental_design.data_parts import (
    CentrifugeVolumenPart,
    PartContainer,
    StringNumberInstancePart,
)
from experimental_design.visualisation import plot_function_from_df

st.title("Meine Streamlit App")

st.header("Willkommen zu meiner App")

fields = StringNumberInstancePart.render_fields(position="test")
instance = StringNumberInstancePart(fields=fields)
instance_html = f"<div>{instance}</div>"


fields_function = CentrifugeVolumenPart.render_fields(position="test_function")
instance_function = CentrifugeVolumenPart(fields=fields_function)
instance_html += f"<div>{instance_function}</div>"


st.markdown(
    f"<div> <h3> Design </h3><div class='flex-box-doe'>{instance_html} </div></div>",
    unsafe_allow_html=True,
)


container = PartContainer(
    parts={
        "stringnumber": instance,
        "function": instance_function,
    }
)
container.generate(num_samples=20)
df_design = container.collect_samples(design_space=True)
container.assign_samples(df_design)
df_design = container.collect_samples(design_space=False)
st.dataframe(df_design)

st.pyplot(plot_function_from_df(df_design.iloc[[0]]))
