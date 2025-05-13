from abc import ABC, abstractmethod
from typing import Any, cast, Dict, List, NewType

import streamlit as st

__all__ = [
    "info",
    "DataFieldBase",
]


info = NewType("info", str)


class DataFieldBase(ABC):
    required_fields = {
        "name": {"type": str, "default": "Instanz", "label": "Name der Instanz"}
    }
    _instance_suffix: str

    def __init__(self, name: str, fields: Dict[str, Any]) -> None:
        self._name = name
        self._instance_name = name.replace(self._instance_suffix, "")
        self.fields = fields

    @property
    def name(self) -> str:
        return self._name

    @property
    def instance_name(self) -> str:
        return self._instance_name

    @classmethod
    def render_fields(
        cls, position: str = "00", use_columns: bool = False
    ) -> Dict[str, Any]:
        from experimental_design.data_parts.container import _FunctionPartContainerBase

        # don't remove (circular import workaround)
        from experimental_design.data_parts.instance_part import _InstancePartBase

        fields = {}
        all_fields = {**cls.required_fields, **cls.get_own_fields()}

        if use_columns:
            columns = st.columns(len(all_fields))
        else:
            columns = [st.container()] * len(all_fields)

        for index, (field, properties) in enumerate(all_fields.items()):
            with columns[index]:
                field_type = properties.get("type", str)
                default_value = properties.get("default", None)
                label = properties.get("label", None)
                label = f"{label} ({field_type.__name__})"  # type:ignore[attr-defined]
                widget_key = f"{field}_{position}"

                if field_type == int:
                    fields[field] = st.number_input(  # type: ignore[call-overload]
                        label, value=default_value, format="%d", step=1, key=widget_key
                    )
                elif field_type == float:
                    fields[field] = st.number_input(
                        label, value=default_value, format="%.2f", key=widget_key
                    )
                elif field_type == info:
                    fields[field] = st.checkbox(
                        label, value=bool(default_value), key=widget_key
                    )
                    if fields[field]:
                        if issubclass(cls, _FunctionPartContainerBase):
                            st.write(cls.function_str())
                        st.pyplot(cls.info(fields=fields))  # type: ignore[attr-defined]
                elif field_type == list:
                    label += "(mit ',' separieren)"
                    value = ",".join(cast(List[str], default_value))
                    fields[field] = st.text_input(label, value=value, key=widget_key)
                    fields[field].replace(" ", "")
                    fields[field] = [x.strip() for x in fields[field].split(",")]
                elif issubclass(field_type, _InstancePartBase):  # type: ignore[arg-type]
                    fields[field] = cast(_InstancePartBase, field_type).render_fields(
                        position=field, use_columns=True
                    )
                elif issubclass(field_type, _FunctionPartContainerBase):  # type: ignore[arg-type]
                    fields[field] = cast(
                        _FunctionPartContainerBase, field_type
                    ).render_fields(position=field, use_columns=False)
                else:
                    if default_value is not None:
                        value = (
                            f"{cast(str, default_value)}_{position}"
                            if field == "name"
                            else cast(str, default_value)
                        )
                        fields[field] = st.text_input(
                            label, value=value, key=widget_key
                        )
                    else:
                        label = properties.get("label", None)  # type: ignore[assignment]
                        text = f"Parameter: **{position}**" if label is None else label
                        st.write(text)
                        fields[field] = position

        return fields

    @classmethod
    def get_own_fields(cls) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def __repr__(self) -> str:
        pass
