# -*- coding: utf-8 -*-

# This code is part of pyQCat Visage.
#
# Copyright (c) 2022-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/09/16
# __author:       SS Fang

"""
Define ExperimentMap dict.

Map experiment name to Monster experiment class.
Unified import experiment class.
"""

import re
from pathlib import Path
from typing import Union, Dict
from pyQCat.preliminary import *
from pyQCat.experiments.base_experiment import BaseExperiment
from pyQCat.experiments.batch_experiment import BatchExperiment
from pyQCat.experiments.single import *
from pyQCat.experiments.composite import *
from pyQCat.experiments.calibration import *
from pyQCat.experiments.batch import *
from pyQCat.pulse import CUSTOM_PULSE_MODELS, STANDARD_PULSE_MODELS
from pyQCat.tools.utilities import qarange

attr_list = dir()
attr_dict = locals()

ExperimentMap = {}
PulseMap = {value.__class__.__name__: value for value in STANDARD_PULSE_MODELS}
for pulse_module in CUSTOM_PULSE_MODELS:
    PulseMap.update({pulse_module.__class__.__name__: pulse_module})

for attr in attr_list:
    if not attr.startswith("__"):
        value = attr_dict.get(attr)
        if isinstance(value, type) and (issubclass(value, BaseExperiment) or issubclass(value, BatchExperiment)):
            ExperimentMap.update({attr: value})


def deal_exp_options(opt_dict: Dict):
    if opt_dict and isinstance(opt_dict, dict):
        new_dict = {}
        for key, value in opt_dict.items():
            if key == 'validator':
                continue
            elif key == 'readout_point_model':
                if value in PulseMap:
                    new_dict.update({key: PulseMap[key]})
                continue
            elif key in ["child_exp_options", "child_ana_options"]:
                value = deal_exp_options(value)
            elif isinstance(value, Dict):
                value = deal_exp_options(value)
            elif isinstance(value, str):
                if value.startswith("normal |"):
                    value = eval(value.split(" | ")[-1])
                elif value.startswith("qarange |"):
                    value = eval(value.split(" | ")[-1])
                    value = qarange(*value)
                elif value.startswith("Points"):
                    cv = value.split(" | ")
                    if len(cv) == 3:
                        if cv[1] == "normal":
                            value = eval(cv[2])
                        elif cv[1] == "qarange":
                            value = qarange(*eval(cv[2]))
            new_dict.update({key: value})
        return new_dict
    return opt_dict


def auto_parse_file(simulator_data_path):
    all_file = list(Path(simulator_data_path).glob('*.dat'))
    pattern = r'=(-?\d+(\.\d+)?)'
    x_list = []
    for file in all_file:
        match = re.search(pattern, file.name)
        if match:
            x = match.group(1)
            x = float(x)
            x_list.append(x)
    x_list = sorted(x_list)
    return x_list


def deal_exp_x_data(opt_dict: Dict, exp_opt: Dict):
    if opt_dict and isinstance(opt_dict, dict):
        for key, value in opt_dict.items():
            if key == 'validator':
                continue
            elif isinstance(value, str):
                if value.startswith("normal |"):
                    value = eval(value.split(" | ")[-1])
                    if isinstance(value, list) and not value:
                        x_data = auto_parse_file(exp_opt.get("simulator_data_path"))
                        exp_opt.update({key: x_data})
        return exp_opt
