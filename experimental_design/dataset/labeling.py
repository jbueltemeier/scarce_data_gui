from abc import ABC, abstractmethod
from typing import Any, cast, Dict, List, Optional

import streamlit as st


__all__ = [
    "UserLabel",
    "_UserLabelContainer",
    "CombinedPercentageLabelContainer",
    "UncombinedLabelContainer",
    "LabelContainer",
    "ActiveLearningLabelContainer",
]

import pandas as pd


class UserLabel:
    def __init__(self, *, user_name: str) -> None:
        self.user_name = user_name
        self.user_labels: Dict[int, Dict[str, Any]] = {}
        self.position = 0

    def get_user_labels(self) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(self.user_labels, orient="index")
        if "reason" in df.columns:
            df = df.drop(columns=["reason"])
        return df

    def __len__(self) -> int:
        return len(self.user_labels)

    def change_position(self, *, value: int = 1) -> None:
        if value > 0:
            self.position += value

        if self.position > 0 > value:
            self.position += value

    def update_label(self, label_dict: Dict[str, Any]) -> None:
        self.user_labels[self.position] = label_dict

    def load_label(self) -> Optional[Dict[str, Any]]:
        return (
            self.user_labels[self.position]
            if self.position in self.user_labels.keys()
            else None
        )

    def update_all_labels(self, complete_label_dict: Dict[int, Dict[str, Any]]) -> None:
        for pos, label_dict in complete_label_dict.items():
            self.user_labels[pos] = label_dict


class _UserLabelContainer(ABC):
    def __init__(
        self, name: str, labels: List[str], min_value: int, max_value: int
    ) -> None:
        self.name = name
        self.user_labels: Dict[str, UserLabel] = {}
        self.labels = labels
        self.min_value = min_value
        self.max_value = max_value

        self._settings: Dict[str, Any] = {
            "labels": self.labels,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    def _create_user(self, user_name: str) -> None:
        if user_name not in self.user_labels.keys():
            self.user_labels[user_name] = UserLabel(user_name=user_name)

    def update_label(self, user_name: str, label_dict: Dict[str, Any]) -> None:
        self._create_user(user_name)
        self.user_labels[user_name].update_label(label_dict)

    def update_all_labels(
        self, user_name: str, complete_label_dict: Dict[int, Dict[str, Any]]
    ) -> None:
        self._create_user(user_name)
        self.user_labels[user_name].update_all_labels(complete_label_dict)
        self.load_label(user_name=user_name)

    def load_label(self, user_name: str) -> None:
        slider_dict_name = f"{self.name}_slider_dict"
        st.session_state[slider_dict_name] = None

        if user_name in list(self.user_labels.keys()):
            st.session_state[slider_dict_name] = self.user_labels[
                user_name
            ].load_label()

        label_dict: Dict[str, int] = {}
        for label in self.labels:
            slider_name = f"{self.name}_slider_{label}"
            if st.session_state[slider_dict_name] is None:
                st.session_state[slider_name] = self.min_value
                label_dict[label] = self.min_value
            else:
                st.session_state[slider_name] = st.session_state[slider_dict_name][
                    label
                ]

        if st.session_state[slider_dict_name] is None:
            st.session_state[slider_dict_name] = label_dict

    def change_position(self, user_name: str, value: int = 1) -> None:
        self.user_labels[user_name].change_position(value=value)
        self.load_label(user_name=user_name)

    def position(self, user_name: str) -> int:
        self._create_user(user_name)
        return self.user_labels[user_name].position

    @abstractmethod
    def combine_labels(self) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def format_labels(self, label: str) -> str:
        pass

    def display_sliders(self, user_name: str) -> None:
        slider_dict_name = f"{self.name}_slider_dict"
        if slider_dict_name not in st.session_state:
            st.session_state[slider_dict_name] = None
        if st.session_state[slider_dict_name] is None:
            self.load_label(user_name=user_name)

        st.divider()
        st.subheader(f"{self.name}:")
        for label in self.labels:
            label_str = self.format_labels(label)
            slider_name = f"{self.name}_slider_{label}"
            st.slider(label_str, self.min_value, self.max_value, key=slider_name)
            st.session_state[slider_dict_name][label] = st.session_state[slider_name]

        self.update_label(
            user_name=user_name, label_dict=st.session_state[slider_dict_name]
        )


class CombinedPercentageLabelContainer(_UserLabelContainer):
    @staticmethod
    def generate_label(row: pd.DataFrame, columns: List[str]) -> str:
        labels = []
        max_value = row.max()
        if max_value < 5:
            return "unlabeled"
        for col in columns:
            if row[col] >= 0.8 * max_value:
                labels.append(col)
        return "_".join(labels)

    def combine_labels(self) -> Optional[pd.DataFrame]:
        dfs = []
        for user_name in self.user_labels.keys():
            dfs.append(self.user_labels[user_name].get_user_labels())

        if all(df.empty for df in dfs):
            return None
        else:
            combined_df = pd.concat(dfs, axis=0)
            mean_df = combined_df.groupby(combined_df.index).mean()
            return mean_df.apply(
                lambda row: self.generate_label(row, mean_df.columns), axis=1
            )

    def format_labels(self, label: str) -> str:
        return f"Label Zuordnung für **{label}** (in %)"


class UncombinedLabelContainer(_UserLabelContainer):
    def combine_labels(self) -> Optional[pd.DataFrame]:
        dfs = []
        for user_name in self.user_labels.keys():
            dfs.append(self.user_labels[user_name].get_user_labels())

        if all(df.empty for df in dfs):
            return None
        else:
            combined_df = pd.concat(dfs, axis=0)
            mean_df = combined_df.groupby(combined_df.index).mean()
            return mean_df

    def format_labels(self, label: str) -> str:
        return f"Label Zuordnung für **{label}**"


class _LabelContainer:
    def __init__(self, label_settings_df: pd.DataFrame) -> None:
        self._num_samples = 0
        self.sub_label_parts: Dict[str, _UserLabelContainer] = {}
        label_settings_df.apply(self.create_label_part, axis=0)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value: int) -> None:
        self._num_samples = value

    def get_all_users(self) -> List[str]:
        sub_part = next(iter(self.sub_label_parts.values()))
        return list(sub_part.user_labels.keys())

    def create_label_part(self, row: pd.DataFrame) -> None:
        label_settings = row.iloc[0]
        label_settings = label_settings.split("-")
        if "num_samples" in label_settings:
            self._num_samples = int(label_settings[1])
        elif "list" in label_settings:
            self.sub_label_parts[label_settings[1]] = CombinedPercentageLabelContainer(
                name=label_settings[1],
                labels=label_settings[7].split(","),
                min_value=int(label_settings[3]),
                max_value=int(label_settings[5]),
            )
        else:
            self.sub_label_parts[label_settings[1]] = UncombinedLabelContainer(
                name=label_settings[1],
                labels=[label_settings[1]],
                min_value=int(label_settings[3]),
                max_value=int(label_settings[5]),
            )

    def update_label(self, user_name: str, label_dict: Dict[str, Any]) -> None:
        for sub_part, label_dict_part in zip(
            self.sub_label_parts.values(), label_dict.values()
        ):
            sub_part.update_label(user_name=user_name, label_dict=label_dict_part)

    def update_all_labels(
        self, user_name: str, label_dict: Dict[int, Dict[str, Any]]
    ) -> None:
        for sub_part, subpart_label_dict in zip(
            self.sub_label_parts.values(), label_dict.values()
        ):
            if bool(subpart_label_dict):
                sub_part.update_all_labels(
                    user_name=user_name,
                    complete_label_dict=cast(
                        Dict[int, Dict[str, Any]], subpart_label_dict
                    ),
                )

    def load_label(self, user_name: str) -> None:
        for key, sub_part in self.sub_label_parts.items():
            sub_part.load_label(user_name=user_name)

    @abstractmethod
    def change_position(self, user_name: str, value: int = 1) -> None:
        pass

    def position(self, user_name: str) -> int:
        sub_part = next(iter(self.sub_label_parts.values()))
        return sub_part.position(user_name=user_name)

    def combine_labels(self) -> Optional[pd.DataFrame]:
        def fill_unlabeled(
            df: pd.DataFrame, fill_value: str = "unlabeled"
        ) -> pd.DataFrame:
            if df is None:
                return pd.DataFrame({key: [fill_value] * self.num_samples})
            num_missing = self.num_samples - len(df)
            if num_missing <= 0:
                return df
            filler_df = pd.DataFrame({df.columns[0]: [fill_value] * num_missing})
            return pd.concat([df, filler_df], ignore_index=True)

        dfs = []
        for key, sub_part in self.sub_label_parts.items():
            df = sub_part.combine_labels()
            if isinstance(df, pd.Series):
                df = pd.DataFrame(df, columns=[key])
            dfs.append(fill_unlabeled(df))

        return pd.concat(dfs, axis=1)

    def display_sliders(self, user_name: str) -> None:
        for sub_part in self.sub_label_parts.values():
            sub_part.display_sliders(user_name=user_name)
        st.divider()


class LabelContainer(_LabelContainer):
    def change_position(self, user_name: str, value: int = 1) -> None:
        for sub_part in self.sub_label_parts.values():
            position = sub_part.position(user_name=user_name)
            if 0 < position < (self.num_samples - 1) or (
                (value > 0 and position == 0)
                or (value < 0 and position == (self.num_samples - 1))
            ):
                sub_part.change_position(user_name=user_name, value=value)


class ActiveLearningLabelContainer(_LabelContainer):
    def __init__(self, label_settings_df: pd.DataFrame) -> None:
        super().__init__(label_settings_df)
        self.num_samples = 0

    def change_position(self, user_name: str, value: int = 1) -> None:
        for sub_part in self.sub_label_parts.values():
            for user_name in self.get_all_users():
                sub_part.change_position(user_name=user_name, value=value)
        self.num_samples = self.num_samples + 1
