from typing import cast, List, Tuple, Union

import streamlit as st

from experimental_design.data_parts import _InstancePartBase, get_part_list

from scarce_data_gui.utils import (
    check_duplicate_instances,
    create_dataset_header,
    INSTANCE_META,
    INSTANCE_PARAMS,
    INSTANCE_SETTINGS,
)


__all__ = [
    "add_data_part",
    "data_scrollable_container",
]


def check_instance_names(*, new_name: str, data_instance: str) -> Tuple[bool, str]:
    instance_dicts = {
        "INSTANCE_META": create_dataset_header(header=INSTANCE_META),
        "INSTANCE_PARAMS": create_dataset_header(header=INSTANCE_PARAMS),
        "INSTANCE_SETTINGS": create_dataset_header(header=INSTANCE_SETTINGS),
    }

    for _data_instance in instance_dicts.values():
        if any(
            new_name == part.name for part in st.session_state[_data_instance].values()
        ):
            return True, _data_instance

    return False, data_instance


def add_instance(
    new_instance: _InstancePartBase,
    instance_type: str,
    data_instance: str,
    data_check_box: str,
) -> None:
    name_exists, data_instance = check_instance_names(
        new_name=new_instance.name, data_instance=data_instance
    )
    if check_duplicate_instances(
        new_instance=new_instance,
        current_instances=st.session_state[
            create_dataset_header(header=INSTANCE_SETTINGS)
        ].values(),
    ):
        st.warning(
            "Es sind keine doppelten Instanzen von diesem Typ erlaubt.", icon="⚠️"
        )
    else:
        if name_exists and not st.session_state["active_rework"]:
            st.warning(
                "Doppelte Namen sind nicht erlaubt. Zum Ändern das Kontrollkästchen 'Design bearbeiten'.",
                icon="⚠️",
            )

        else:  # name_exists and st.session_state["active_rework"]:
            if name_exists:
                st.warning(
                    f"Die Instanz {new_instance.name} wurde überschrieben. "
                    f"Erstelle ein neues Design, um die Änderungen zu übernehmen.",
                    icon="⚠️",
                )
            st.session_state[data_instance][new_instance.name] = new_instance
            st.success(f"{instance_type} erstellt: {new_instance}")
            st.session_state[data_check_box] = False
            st.rerun()


def create_position_str(data_instance: str) -> str:
    position = len(st.session_state[data_instance].values())
    if data_instance == create_dataset_header(header=INSTANCE_META):
        return f"meta_{position}"
    elif data_instance == create_dataset_header(header=INSTANCE_PARAMS):
        return f"param_{position}"
    else:
        return f"label_{position}"


def add_data_part(header: str, data_instance: str) -> None:
    data_check_box = f"checkbox_{header.lower()}"
    with st.expander(f"Neue {header} Instanz erstellen", icon="➕"):
        instance_types: Union[str, List[str]] = "all"
        if header == INSTANCE_META:
            instance_types = ["static"]
        elif header == INSTANCE_SETTINGS:
            instance_types = ["label"]

        instances = get_part_list(instance_types)
        selected_class_name = st.selectbox(
            f"Wähle den Typ der {header} Instanz aus:",
            [instance["name"] for instance in instances],
        )
        selected_class = next(
            item["class"] for item in instances if item["name"] == selected_class_name
        )
        position = create_position_str(data_instance)
        fields = cast(_InstancePartBase, selected_class).render_fields(
            position=position
        )

        if st.button(f"Erstelle {header} {selected_class_name}"):
            new_instance = cast(_InstancePartBase, selected_class)(fields=fields)  # type: ignore[operator]

            add_instance(
                new_instance=new_instance,
                instance_type=cast(str, selected_class_name),
                data_instance=data_instance,
                data_check_box=data_check_box,
            )


def data_scrollable_container(header: str) -> None:
    data_instance = create_dataset_header(header=header)
    if data_instance not in st.session_state:
        st.session_state[data_instance] = {}
    else:
        save_instance = f"{data_instance}_save"
        if st.session_state.get(save_instance) is None:
            st.session_state[save_instance] = st.session_state[data_instance].copy()

    if len(st.session_state[data_instance]) > 0:
        instance_html = ""
        for idx, (key, instance) in enumerate(st.session_state[data_instance].items()):
            instance_html += f'<div class="instance-box-doe">{instance}</div>'
    else:
        instance_html = "<b>Noch keine Instanzen erstellt.</b>"

    st.markdown(
        f"<div class='scrollable-box-doe'> <h3> {header} </h3><div class='flex-box-doe'>{instance_html} </div></div>",
        unsafe_allow_html=True,
    )

    if st.session_state["active_rework"]:
        with st.expander(f"{header} Instanz entfernen", icon="🗑️"):
            for idx, (key, instance) in enumerate(
                st.session_state[data_instance].items()
            ):
                if st.button(f"🗑️ Löschen {key}", key=f"delete_{header}_{idx}"):
                    del st.session_state[data_instance][key]
                    st.rerun()

    add_data_part(header=header, data_instance=data_instance)
