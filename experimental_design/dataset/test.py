from experimental_design.data_parts import (
    LabelInstancePart,
    LabelListInstancePart,
    NumSamplesInstancePart,
    PartContainer,
)

from experimental_design.dataset import LabelContainer

container = PartContainer(
    parts={
        "num_samples": NumSamplesInstancePart(
            fields={
                "name": "num_samples",
                "num_samples": 10,
            }
        ),
        "instance": LabelInstancePart(
            fields={
                "name": "label_instance",
                "min_value": 20,
                "max_value": 80,
            }
        ),
        "instance2": LabelInstancePart(
            fields={
                "name": "label_instance2",
                "min_value": 20,
                "max_value": 50,
            }
        ),
        "percent_list": LabelListInstancePart(
            fields={
                "name": "percentage",
                "values": ["dog", "cat", "horse"],
            }
        ),
        "percent_list2": LabelListInstancePart(
            fields={
                "name": "percentage2",
                "values": ["rot", "grün", "gelb"],
            }
        ),
    }
)

container.generate(num_samples=1)
label_container = LabelContainer(label_settings_df=container.collect_samples())

user_name = "julian"
label_dict_parts1 = {
    "instance_label": {"label_instance_label": 40},
    "instance2_label": {"label_instance_label": 30},
    "percentage_label": {"dog": 80, "cat": 40, "horse": 10},
    "percentage2_label": {"rot": 100, "grün": 0, "gelb": 0},
}
label_dict_parts2 = {
    "instance_label": {"label_instance_label": 60},
    "instance2_label": {"label_instance_label": 20},
    "percentage_label": {"dog": 0, "cat": 40, "horse": 100},
    "percentage2_label": {"rot": 0, "grün": 100, "gelb": 0},
}
label_dict_parts3 = {
    "instance_label": {"label_instance_label": 50},
    "instance2_label": {"label_instance_label": 10},
    "percentage_label": {"dog": 0, "cat": 100, "horse": 30},
    "percentage2_label": {"rot": 100, "grün": 10, "gelb": 50},
}
user_name1 = "julian"
user_name2 = "name2"
user_name3 = "name3"

label_container.update_label(user_name=user_name, label_dict=label_dict_parts1)
label_container.change_position(user_name=user_name)
label_container.update_label(user_name=user_name, label_dict=label_dict_parts2)
label_container.change_position(user_name=user_name)
label_container.update_label(user_name=user_name, label_dict=label_dict_parts3)

label_container.update_label(user_name=user_name2, label_dict=label_dict_parts3)
label_container.change_position(user_name=user_name2)
label_container.update_label(user_name=user_name2, label_dict=label_dict_parts2)
label_container.change_position(user_name=user_name2)
label_container.update_label(user_name=user_name2, label_dict=label_dict_parts1)

label_container.update_label(user_name=user_name3, label_dict=label_dict_parts2)
label_container.change_position(user_name=user_name3)
label_container.update_label(user_name=user_name3, label_dict=label_dict_parts1)
label_container.change_position(user_name=user_name3)
label_container.update_label(user_name=user_name3, label_dict=label_dict_parts3)

label_dict = label_container.combine_labels()

print()
