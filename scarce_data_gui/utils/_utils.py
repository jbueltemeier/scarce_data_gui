from typing import ValuesView

from experimental_design.data_parts import (
    _InstancePartBase,
    get_part_list,
    NumSamplesInstancePart,
)

__all__ = [
    "INSTANCE_META",
    "INSTANCE_PARAMS",
    "INSTANCE_SETTINGS",
    "create_dataset_header",
    "check_num_instances",
    "check_duplicate_instances",
]


INSTANCE_META = "Meta"
INSTANCE_PARAMS = "Versuchsparameter"
INSTANCE_SETTINGS = "Label"


def create_dataset_header(header: str) -> str:
    return f"instances_{header.lower()}"


def check_num_instances(instance_type: str, current_instances: ValuesView) -> int:
    instances = {
        instance["class"].__name__ for instance in get_part_list([instance_type])  # type: ignore[attr-defined]
    }
    return sum(
        1 for value in current_instances if value.__class__.__name__ in instances
    )


def check_duplicate_instances(
    new_instance: _InstancePartBase, current_instances: ValuesView
) -> bool:
    if isinstance(new_instance, NumSamplesInstancePart):
        for value in current_instances:
            if "NumSamplesInstancePart" in [value.__class__.__name__]:
                if new_instance.name == value.name:
                    return False
                return True
    return False
