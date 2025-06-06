# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

import json
import os
import pickle
from copy import deepcopy
from typing import Dict

from loguru import logger

from pyQCat.qubit import Qubit, Coupler, QubitPair
from pyQCat.qubit.qubit_pair import build_cz_gate_struct
from pyQCat.structures import QDict
import datetime


class VisageComponent:
    empty_bit = []
    comp_count = 0

    style_level = {
        "json": 0,
        "qubit": 1,
        "coupler": 2,
        "qubit_pair": 3,
        "dat": 4,
        "bin": 5,
    }

    def __init__(self, data: Dict, count: int = 0):
        self._id = None
        self._name = None
        self._username = None
        self._sample = None
        self._env_name = None
        self._point_label = None
        self._style = None
        self._update_time = None
        self._view_data = None
        self._data = None
        self._count = count
        self._format(data)
        self._edit_records = []

    @property
    def sort_level(self):
        return self.style_level.get(self.style, 1000)

    @property
    def edit_records(self):
        return self._edit_records

    def _format(self, data: Dict):
        VisageComponent.comp_count += 1
        self._id = data.get("id")
        self._name = data.get("name") or data.get("filename")
        self._username = data.get("username")
        self._sample = data.get("sample")
        self._env_name = data.get("env_name")
        self._point_label = data.get("point_label") or ""
        self._update_time = data.get("update_time") or data.get("create_time")
        if self._name.endswith("json"):
            self._style = "json"
            view_data = data.get("json")
            if self._name == "instrument.json":
                self._view_data = self._instrument_to_view(view_data)
            else:
                self._view_data = view_data
        elif self._name.endswith("dat"):
            self._style = "dat"
            self._view_data = {"dat": data.get("dat")}

        elif self._name.endswith("bin"):
            self._style = "bin"
            # bin_data = data.get('bin')
            bin_data = QDict(data.get("bin_abbr", {}))
            # try:
            #     iq_discriminator = pickle.loads(bin_data)
            #     dcm_data = iq_discriminator.to_dict()
            # except Exception as e:
            #     logger.warning(f'Import {self._name} error, Because {e}')
            #     VisageComponent.empty_bit.append(self.name.split(".")[0])
            #     dcm_data = {}
            self._view_data = bin_data

        elif self._name.startswith("q"):
            if self._name.find("q", 1) == -1:
                self._style = "qubit"
            else:
                self._style = "qubit_pair"
            self._view_data = data.get("parameters")
            self._view_data["qid"] = self.qid

        elif self._name.startswith("c"):
            self._style = "coupler"
            self._view_data = data.get("parameters")
            self._view_data["qid"] = self.qid

        else:
            pass

        self._data = data

        if VisageComponent.comp_count == self._count != 0:
            VisageComponent.bit_count = 0
            if VisageComponent.empty_bit:
                logger.warning(f"*.bin is empty in {VisageComponent.empty_bit}")
                VisageComponent.empty_bit.clear()

    @property
    def qid(self):
        return self._id

    @property
    def username(self):
        return self._username

    @property
    def sample(self):
        return self._sample

    @property
    def env_name(self):
        return self._env_name

    @property
    def point_label(self):
        return self._point_label

    @property
    def style(self):
        return self._style

    @property
    def name(self):
        return self._name

    @property
    def update_time(self):
        return self._update_time

    @property
    def data(self):
        return self._data

    @property
    def view_data(self):
        return self._view_data

    def to_file(self, dirname: str, use_time_flag=False):
        data = self.data

        if self.data.get("bin") is not None:
            data = deepcopy(self.data)
            bin_data = data["bin"]
            data["bin"] = str(bin_data)

        try:
            if use_time_flag:
                update_time = datetime.datetime.strptime(
                    self.update_time, "%Y-%m-%d %H:%M:%S"
                ).strftime("%Y-%m-%d_%H_%M_%S")
                filename = os.path.join(
                    dirname,
                    f"{self.name}-{update_time}.json",
                )
            else:
                filename = os.path.join(
                    dirname,
                    f"{self.name}.json",
                )
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                logger.info(f"{self.name} to file success, save in {filename}")
        except Exception as e:
            logger.error(f"{self.name} to file error, because\n{e}")

    @classmethod
    def from_base_qubit(cls, bit):
        data_dict = QDict(class_name=bit.__class__.__name__)
        data_dict.update(bit.to_dict())
        return cls(data_dict)

    @classmethod
    def from_json_file(cls, path):
        data_dict = QDict(class_name="Config")
        with open(path, "r") as f:
            data = json.load(f)
            data_dict.update(data)
        return cls(data_dict)

    @classmethod
    def from_dict(cls, data_dict: Dict, count: int = 0):
        return cls(data_dict, count)

    @staticmethod
    def _instrument_to_view(view_data: Dict):
        trans_view_data = deepcopy(view_data)

        def _update_module(name: str):
            trans_view_data[name] = dict()
            for module in view_data[name]:
                trans_view_data[name][f"channel{module['channel']}"] = module

        _update_module("Z_flux_control")
        _update_module("XY_control")
        _update_module("Read_out_control")

        return trans_view_data

    def to_data(self):
        """get object of Bit or config data"""
        if self.style == "qubit":
            qubit = Qubit.from_dict(self.view_data)
            return self.style, qubit
        elif self.style == "coupler":
            coupler = Coupler.from_dict(self.view_data)
            return self.style, coupler
        elif self.style == "json":
            return self.name, self.data.get("json")
        elif self.style == "dat":
            return self.name, self.view_data.get("dat")
        elif self.style == "bin":
            return self.name, self.data.get("bin")
        elif self.style == "qubit_pair":
            qubit_pair = build_cz_gate_struct(QubitPair.from_dict(self.view_data))
            return self.style, qubit_pair
        else:
            raise NameError(f"Can not tackle style {self.style}")

    def to_dict_conf(self):
        """To save config data in batches"""
        res = self.to_data()
        return {"filename": res[0], "file_data": res[1]}
