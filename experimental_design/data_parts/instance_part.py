from abc import abstractmethod
from typing import Any, cast, Dict, List, Union

import numpy as np

import pandas as pd

from experimental_design.core import is_divisible
from experimental_design.data_parts.data_field import DataFieldBase
from experimental_design.data_parts.generator import LHCGenerator


__all__ = [
    "_InstancePartBase",
    "_StaticInstancePartBase",
    "StaticStringInstancePart",
    "StaticNumberInstancePart",
    "_GeneratorInstancePartBase",
    "NumberInstancePart",
    "FactorNumberInstancePart",
    "StringNumberInstancePart",
    "_ListInstancePartBase",
    "BooleanInstancePart",
    "StringInstancePart",
    "FunctionNumberInstancePart",
    "FactorFunctionNumberInstancePart",
    "FunctionTimeInstancePart",
    "_LabelInstancePartBase",
    "LabelInstancePart",
    "LabelListInstancePart",
    "NumSamplesInstancePart",
    "get_part_list",
]


class _InstancePartBase(DataFieldBase):
    _samples: pd.DataFrame

    @property
    def samples(self) -> pd.DataFrame:
        return self._samples

    @samples.setter
    def samples(self, df: pd.DataFrame) -> None:
        self._samples = df

    @abstractmethod
    def translate(self, design_space: bool = True) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate(self, num_samples: int) -> None:
        pass

    def __len__(self) -> int:
        return cast(int, self.samples.shape[1])


class _StaticInstancePartBase(_InstancePartBase):
    own_fields: Dict[str, Any] = {}

    def __init__(self, fields: Dict[str, Any]) -> None:
        self._instance_suffix = "_static"
        name = fields.get("name")
        super().__init__(f"{name}{self._instance_suffix}", fields)
        self.sample_value = fields.get("sample_value")

    @classmethod
    def get_own_fields(cls) -> Dict[str, Any]:
        return cls.own_fields

    def translate(self, design_space: bool = True) -> pd.DataFrame:
        return self.samples

    def generate(self, num_samples: int) -> None:
        self.samples = pd.DataFrame({self.name: [self.sample_value] * num_samples})

    def __repr__(self) -> str:
        return f"""
        <body>
            <h5>Statischer Wert</h5>
            <p><strong>Name:</strong> {self.instance_name}</p>
            <p><strong>Value:</strong> {self.sample_value}</p>
        </body>
        """


class StaticStringInstancePart(_StaticInstancePartBase):
    own_fields = {
        "sample_value": {"type": str, "default": "-", "label": "Wert der Instanz"},
    }


class StaticNumberInstancePart(_StaticInstancePartBase):
    own_fields = {
        "sample_value": {"type": int, "default": 0, "label": "Wert der Instanz"},
    }


class _GeneratorInstancePartBase(_InstancePartBase):
    own_fields: Dict[str, Any] = {}

    def __init__(self, fields: Dict[str, Any]) -> None:
        self._instance_suffix = "_numerical"
        name = fields.get("name")

        self.unit = fields.get("unit") if "unit" in fields.keys() else ""
        name_unit = "" if not self.unit else f"(in {fields.get('unit')})"

        super().__init__(f"{name} {name_unit}{self._instance_suffix}", fields)
        generator = LHCGenerator(
            min_value=cast(int, fields.get("min_value")),
            max_value=cast(int, fields.get("max_value")),
            delay=fields.get("delay"),
        )
        self.generator = generator

    @classmethod
    def get_own_fields(cls) -> Dict[str, Any]:
        return cls.own_fields

    def generate(self, num_samples: int) -> None:
        self.samples = pd.DataFrame(
            {self.name: self.generator.generate_samples(num_samples)}
        )

    @abstractmethod
    def translate(self, design_space: bool = True) -> List[Any]:
        pass

    def __repr__(self) -> str:
        return f"""
        <body>
            <h5>{self.__class__.__name__}</h5>
            <p><strong>Name:</strong> {self.instance_name}</p>
            <p><strong>Generator:</strong> {self.generator}</p>
        </body>
        """


class NumberInstancePart(_GeneratorInstancePartBase):
    own_fields = {
        "min_value": {"type": int, "default": 0, "label": "Minimale Wert der Instanz"},
        "max_value": {"type": int, "default": 1, "label": "Maximale Wert der Instanz"},
        "unit": {"type": str, "default": "", "label": "Einheit"},
        "delay": {"type": int, "default": 0, "label": "Delay der Instanz"},
        "rounding_points": {"type": int, "default": 2, "label": "Rundungsstellen"},
    }

    def __init__(self, fields: Dict[str, Any]) -> None:
        super().__init__(fields)
        self.rounding_points = fields.get("rounding_points")

    def round_numbers(self) -> pd.DataFrame:
        samples = self.generator.scale_number(samples=self.samples)
        return samples.round(decimals=self.rounding_points)

    def translate(self, design_space: bool = True) -> pd.DataFrame:
        if design_space:
            return self.samples
        return self.round_numbers()

    def __repr__(self) -> str:
        return f"""
        <body>
            <h5>Numerischer Versuchspunkt</h5>
            <p><strong>Name:</strong> {self.instance_name}</p>
            <p><strong>Generator:</strong> {self.generator}</p>
            <p><strong>Einheit:</strong> {self.unit}</p>
        </body>
        """


class FactorNumberInstancePart(NumberInstancePart):
    own_fields = {
        "factor": {"type": int, "default": 0, "label": "Factor der Instanz"},
        "min_value": {"type": int, "default": 0, "label": "Minimale Wert der Instanz"},
        "max_value": {"type": int, "default": 1, "label": "Maximale Wert der Instanz"},
        "unit": {"type": str, "default": "", "label": "Einheit"},
        "delay": {"type": int, "default": 0, "label": "Delay der Instanz"},
        "rounding_points": {"type": int, "default": 2, "label": "Rundungsstellen"},
    }

    def __init__(self, fields: Dict[str, Any]) -> None:
        super().__init__(fields=fields)
        self.factor = fields.get("factor")
        self.rounding_points = fields.get("rounding_points")

    def generate(self, num_samples: int) -> None:
        super().generate(num_samples=num_samples)
        self.samples = self.samples * self.factor

    def __repr__(self) -> str:
        return f"""
        <body>
            <h5>Numerischer Versuchspunkt mit Faktor</h5>
            <p><strong>Name:</strong> {self.instance_name}</p>
            <p><strong>Generator:</strong> {self.generator}</p>
            <p><strong>Faktor:</strong> {self.factor}</p>
            <p><strong>Einheit:</strong> {self.unit}</p>
        </body>
        """


class StringNumberInstancePart(_GeneratorInstancePartBase):
    own_fields = {
        "values": {
            "type": list,
            "default": [],
            "label": "Liste mit kategorischen Ausprägungen",
        },
    }

    def __init__(self, fields: Dict[str, Any]) -> None:
        fields["min_value"] = 1
        fields["max_value"] = len(cast(Dict[str, Any], fields.get("values"))) + 1
        super().__init__(fields)
        self._values = cast(List[str], fields.get("values"))

    @property
    def values(self) -> List[str]:
        return self._values

    @values.setter
    def values(self, value_list: List[str]) -> None:
        self._values = value_list

    def map(self) -> pd.DataFrame:
        samples = self.generator.scale_number(samples=self.samples)
        n_intervals = len(self.values)
        bins = np.linspace(0, 1, n_intervals + 1) * n_intervals + 1

        def map_value(value: float) -> str:
            index = int(np.digitize(value, bins, right=False) - 1)
            index = max(0, min(index, len(self.values) - 1))
            return self.values[index]

        return samples.applymap(map_value)

    def translate(self, design_space: bool = True) -> pd.DataFrame:
        if design_space:
            return self.samples
        return self.map()

    def __repr__(self) -> str:
        return f"""
        <body>
            <h5>Numerischer Versuchspunkt mit kategorischen Mapping</h5>
            <p><strong>Name:</strong> {self.instance_name}</p>
            <p><strong>Generator:</strong> {self.generator}</p>
            <p><strong>Mapping Kategorien:</strong> {self.values}</p>
        </body>
        """


class _ListInstancePartBase(_InstancePartBase):
    own_fields: Dict[str, Any] = {}

    def __init__(self, values: List[str], fields: Dict[str, Any]) -> None:
        self._instance_suffix = "_categorical"
        name = fields.get("name")
        super().__init__(f"{name}{self._instance_suffix}", fields)
        self._values = values

    @classmethod
    def get_own_fields(cls) -> Dict[str, Any]:
        return cls.own_fields

    def generate(self, num_samples: int) -> None:
        is_divisible(num_samples, len(self.values))
        factor = (num_samples + len(self.values) - 1) // len(self.values)
        self.samples = pd.DataFrame({self.name: (self.values * factor)[:num_samples]})

    @property
    def values(self) -> List[str]:
        return self._values

    @values.setter
    def values(self, value_list: List[str]) -> None:
        self._values = value_list

    def translate(self, design_space: bool = True) -> pd.DataFrame:
        return self.samples

    def __repr__(self) -> str:
        return f"""
        <body>
            <h5>Liste</h5>
            <p><strong>Name:</strong> {self.instance_name}</p>
            <p><strong>Liste:</strong> {self.values}</p>
        </body>
        """


class BooleanInstancePart(_ListInstancePartBase):
    def __init__(self, fields: Dict[str, Any]) -> None:
        bool_list = ["True", "False"]
        super().__init__(values=bool_list, fields=fields)


class StringInstancePart(_ListInstancePartBase):
    own_fields = {
        "values": {
            "type": list,
            "default": [],
            "label": "Liste mit kategorischen Werten",
        },
    }

    def __init__(self, fields: Dict[str, Any]) -> None:
        values = fields.get("values")
        super().__init__(values=cast(List[str], values), fields=fields)


class FunctionNumberInstancePart(_GeneratorInstancePartBase):
    own_fields = {
        "name": {"type": str, "default": None, "label": None},
        "min_value": {
            "type": float,
            "default": 0.0,
            "label": "Minimale Wert der Instanz",
        },
        "max_value": {
            "type": float,
            "default": 1.0,
            "label": "Maximale Wert der Instanz",
        },
    }

    def translate(self, design_space: bool = True) -> pd.DataFrame:
        if design_space:
            return self.samples
        return self.generator.scale_number(samples=self.samples)

    def __repr__(self) -> str:
        return f"""
        <body>
            <p><strong>Parameter:</strong> {self.instance_name}</p>
            <p><strong>Generator:</strong> {self.generator}</p>
        </body>
        """


class FactorFunctionNumberInstancePart(_GeneratorInstancePartBase):
    own_fields = {
        "name": {"type": str, "default": None, "label": None},
        "min_value": {
            "type": float,
            "default": 0.0,
            "label": "Minimale Wert der Instanz",
        },
        "max_value": {
            "type": float,
            "default": 1.0,
            "label": "Maximale Wert der Instanz",
        },
        "factor": {"type": float, "default": 1.0, "label": "Factor der Instanz"},
    }

    def __init__(self, fields: Dict[str, Any]) -> None:
        super().__init__(fields)
        self.factor = fields.get("factor")

    def generate(self, num_samples: int) -> None:
        super().generate(num_samples=num_samples)
        self.samples = self.samples * self.factor

    def translate(self, design_space: bool = True) -> pd.DataFrame:
        if design_space:
            return self.samples
        return self.generator.scale_number(samples=self.samples)

    def __repr__(self) -> str:
        return f"""
        <body>
            <p><strong>Parameter:</strong> {self.instance_name}</p>
            <p><strong>Generator:</strong> {self.generator}</p>
        </body>
        """


class FunctionTimeInstancePart(_InstancePartBase):
    own_fields = {
        "name": {"type": str, "default": None, "label": "Name der Instanz"},
        "timestamps": {
            "type": list,
            "default": ["10", "20", "30"],
            "label": "Liste mit Auswertezeitpunkten",
        },
    }

    def __init__(self, fields: Dict[str, Any]) -> None:
        self._instance_suffix = "_time"
        name = fields.get("name")
        super().__init__(f"{name}{self._instance_suffix}", fields)
        self._times = cast(List[str], fields.get("timestamps"))

    def __len__(self) -> int:
        return 0

    @classmethod
    def get_own_fields(cls) -> Dict[str, Any]:
        return cls.own_fields

    def translate(self, design_space: bool = True) -> pd.DataFrame:
        pass

    @staticmethod
    def filter_integer(time_list: List[str]) -> List[str]:
        valid_numbers = []
        for time in time_list:
            try:
                float(time)
                valid_numbers.append(time)
            except ValueError:
                continue
        return valid_numbers

    @property
    def times(self) -> List[str]:
        self._times = self.filter_integer(self._times)
        return self._times

    @times.setter
    def times(self, time_list: List[str]) -> None:
        self._times = self.filter_integer(time_list)

    def generate(self, num_samples: int) -> None:
        times = ",".join(self.times)
        self.samples = pd.DataFrame({self.name: [times] * num_samples})

    def __repr__(self) -> str:
        return f"""
        <body>
            <p><strong>Zeitpunkte:</strong> {self.times}</p>
        </body>
        """


class _LabelInstancePartBase(_InstancePartBase):
    own_fields: Dict[str, Any] = {}

    def __init__(self, fields: Dict[str, Any]) -> None:
        self._instance_suffix = "_label"
        name = fields.get("name")
        super().__init__(f"{name}{self._instance_suffix}", fields)

    @classmethod
    def get_own_fields(cls) -> Dict[str, Any]:
        return cls.own_fields

    def translate(self, design_space: bool = True) -> pd.DataFrame:
        return self.samples

    @abstractmethod
    def generate(self, num_samples: int) -> None:
        pass

    def __repr__(self) -> str:
        return f"""
        <body>
            <h5>Label</h5>
            <p><strong>Name:</strong> {self.instance_name}</p>
        </body>
        """


class LabelInstancePart(_LabelInstancePartBase):
    own_fields = {
        "min_value": {"type": int, "default": 0, "label": "Minimale Wert der Instanz"},
        "max_value": {
            "type": int,
            "default": 100,
            "label": "Maximale Wert der Instanz",
        },
    }

    def __init__(self, fields: Dict[str, Any]) -> None:
        name = fields.get("name")
        fields["name"] = f"{name}_reg"
        super().__init__(fields)
        self.min_value = fields.get("min_value")
        self.max_value = fields.get("max_value")

    def generate(self, num_samples: int) -> None:
        label_settings = f"name-{self.instance_name}-min_value-{self.min_value}-max_value-{self.max_value}"
        self.samples = pd.DataFrame({self.name: [label_settings] * num_samples})

    def __repr__(self) -> str:
        return f"""
        <body>
            <h5>Label</h5>
            <p><strong>Name:</strong> {self.instance_name}</p>
            <p><strong>Minimale Wert:</strong> {self.min_value}</p>
            <p><strong>Maximale Wert:</strong> {self.max_value}</p>
        </body>
        """


class LabelListInstancePart(_LabelInstancePartBase):
    own_fields = {
        "values": {
            "type": list,
            "default": [],
            "label": "Liste mit den Label",
        },
    }

    def __init__(self, fields: Dict[str, Any]) -> None:
        super().__init__(fields)
        self._values = cast(List[str], fields.get("values"))

    @property
    def values(self) -> List[str]:
        return self._values

    @values.setter
    def values(self, value_list: List[str]) -> None:
        self._values = value_list

    def generate(self, num_samples: int) -> None:
        labels = ",".join(self.values)
        label_settings = f"name-{self.instance_name}-min_value-{0}-max_value-{100}-labels-{labels}-list"
        self.samples = pd.DataFrame({self.name: [label_settings] * num_samples})

    def __repr__(self) -> str:
        return f"""
        <body>
            <h5>Label Liste</h5>
            <p><strong>Name:</strong> {self.instance_name}</p>
            <p><strong>Labels:</strong> {self.values}</p>
        </body>
        """


class NumSamplesInstancePart(_LabelInstancePartBase):
    own_fields = {
        "num_samples": {"type": int, "default": 20, "label": "Anzahl der Samples"},
    }

    def __init__(self, fields: Dict[str, Any]) -> None:
        super().__init__(fields)
        self.num_samples = fields.get("num_samples")

    def generate(self, num_samples: int) -> None:
        num_samples_str = f"num_samples-{self.num_samples}"
        self.samples = pd.DataFrame({"num_samples": [num_samples_str] * num_samples})

    def __repr__(self) -> str:
        return f"""
        <body>
            <h5>Anzahl der Versuchspunkte</h5>
            <p><strong>Name:</strong> {self.instance_name}</p>
            <p><strong>Anzahl der Samples:</strong> {self.num_samples}</p>
        </body>
        """


def get_part_list(instance_types: Union[str, List[str]]) -> List[Dict[str, object]]:
    from experimental_design.data_parts.additional_data_parts import (
        additional_data_parts,
    )

    instance_map = {
        "static": [
            {"class": StaticNumberInstancePart, "name": "Statische Nummer"},
            {"class": StaticStringInstancePart, "name": "Statischer String"},
        ],
        "numerical": [
            {"class": NumberInstancePart, "name": "Numerischer Versuchsparameter"},
            {
                "class": StringNumberInstancePart,
                "name": "Numerischer Versuchsparameter (wird gemappt auf kategorische Liste)",
            },
        ],
        "categorical": [
            {
                "class": BooleanInstancePart,
                "name": "Kategorischer boolischer Versuchsparameter (True/False)",
            },
            {"class": StringInstancePart, "name": "Kategorischer Versuchsparameter"},
        ],
        "label": [
            {
                "class": LabelInstancePart,
                "name": "Eigenständiges Label für Regression",
            },
            {
                "class": LabelListInstancePart,
                "name": "Liste mit Label (Auswertung Prozent)",
            },
            {"class": NumSamplesInstancePart, "name": "Anzahl der Samples"},
        ],
        "additional": additional_data_parts(),
    }
    if isinstance(instance_types, str):
        instance_types = list(instance_map.keys())
        instance_types.remove("label")

    return [
        part
        for type_ in instance_types
        if type_ in instance_map
        for part in instance_map[type_]
    ]
