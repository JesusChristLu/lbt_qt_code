# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/25
# __author:       Lang Zhu
"""
tool.
"""
import abc
import builtins
import configparser
import json
import os
import uuid
from ast import literal_eval
from enum import Enum
from pathlib import Path
from traceback import format_exc
from typing import Dict

from pyQCat.analysis import ParameterRepr
from pyQCat.config import PyqcatConfig
from pyQCat.experiments import batch
from pyQCat.experiments import calibration
from pyQCat.experiments import composite as composite_exp
from pyQCat.experiments import single as base_exp
from pyQCat.gate.notable_gate import SingleQgate
from pyQCat.log import pyqlog
from pyQCat.preliminary import library as preliminary_exp
from pyQCat.structures import QDict
from pyQCat.tools import DATA_PATH, parse_yaml_args
from pyqcat_visage.tool.utilies import transform_array_to_str, FATHER_OPTIONS

EXP_MAP = {
    "PreliminaryExperiment": preliminary_exp,
    "BaseExperiment": base_exp,
    "CompositeExperiment": composite_exp,
    "CalibrationExperiment": calibration,
    "CustomerExperiment": {},
    # 'ParallelExperiment': {}
}


class ExperimentType(str, Enum):

    preliminary = "PreliminaryExperiment"
    base = "BaseExperiment"
    composite = "CompositeExperiment"
    calibration = "CalibrationExperiment"
    batch = "BatchExperiment"
    custom = "CustomerExperiment"


def comparing_monster_version(local_version: str, courier_version: str) -> bool:
    """
    if local version <= courier version, return True
    else:
    return False
    """

    local_version = local_version.split(".")
    courier_version = courier_version.split(".")
    min_len = min([len(local_version), len(courier_version)])
    flag = True
    try:
        for x in range(min_len):
            if int(local_version[x]) > int(courier_version[x]):
                return False
        return True
    except Exception:
        pyqlog.error("comparing monster version error.")
        pyqlog.error(format_exc())
    return True


def check_options(exp_options, exp):
    """Check monster options to support visage

    Args:
        exp_options:
        exp:

    Returns:

    """
    validator = exp_options.get("validator", {})
    exp_options.pop("validator")

    new_options = QDict()

    for key, value in exp_options.items():
        if key in ["child_exp_options", "child_ana_options"]:
            new_options[key] = check_options(value, exp)
        elif key in validator:
            is_main = True
            if key in FATHER_OPTIONS:
                is_main = False

            vt = validator[key][0]

            if type(vt).__name__ == "type":
                vt = vt.__name__
            elif isinstance(vt, tuple):
                vt = f"spin{vt}"

            if vt == "dict":
                new_options[key] = value or {}
            elif vt == "list" and key not in ["quality_bounds"]:
                value = transform_array_to_str(value)
                new_options[key] = value
            else:
                new_options[key] = [value, vt, is_main]

    return new_options


def update_validator(validators):
    for key in validators.keys():
        v_types = validators.get(key)
        if v_types[0] == bool:
            new_v_types = ["False", "True"], v_types[1]
            validators[key] = new_v_types

    return validators


def format_dict_dumps(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if key not in ["result_parameters", "gate_map", "config"]:
                if key == "validator" and not value:
                    continue
                new_dict[key] = format_dict_dumps(value)
        return new_dict
    elif isinstance(data, tuple) or isinstance(data, list):
        list_iter = []
        for v in data:
            # print(type(v), v, type(v).__name__, v.__name__)
            list_iter.append(format_dict_dumps(v))
        if isinstance(data, tuple):
            return tuple(list_iter)
        return list_iter
    elif isinstance(data, abc.ABCMeta) or isinstance(data, PyqcatConfig):
        return str(data).split(".")[-1].strip(">").strip("'")
    elif isinstance(data, ParameterRepr):
        return str(data)
    elif isinstance(data, SingleQgate):
        return "Gate"
    elif data == "NoneType":
        return None
    elif type(data).__name__ == "type":
        return data.__name__
    else:
        return data


def format_dict_loads(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            new_dict[key] = format_dict_loads(value)
        return new_dict
    elif isinstance(data, list):
        list_iter = []
        for v in data:
            list_iter.append(format_dict_loads(v))
        return list_iter
    elif isinstance(data, tuple):
        list_iter = []
        for v in data:
            if isinstance(v, str) and hasattr(builtins, v):
                list_iter.append(getattr(builtins, v))
            else:
                list_iter.append(v)
        return tuple(list_iter)
    else:
        return data


def exp_class_to_model(cls):
    exp_name = cls.__name__
    exp_options = cls._default_experiment_options()
    ana_options = cls._default_analysis_options()
    exp_options = check_options(exp_options, exp_name)
    ana_options = check_options(ana_options, exp_name)
    # style validate remove to visage leaf node vali type
    # exp_options['validator']['style'] = (['qarange', 'normal'], False)
    # ana_options['validator']['style'] = (['qarange', 'normal'], False)
    return exp_name, exp_options, ana_options


def get_monster_exp_cls(exp_name: str):
    for exp_module_name, exp_module in EXP_MAP.items():
        if exp_name in [exp for exp in dir(exp_module) if exp[0].isupper()]:
            exp_class = getattr(exp_module, exp_name)
            return exp_class


def get_monster_exps():
    exp_dict = {}
    exp_names = []
    _temp_exp_map = {}

    # extract base experiment options
    for exp_module_name, exp_module in EXP_MAP.items():
        exp_dict[exp_module_name] = []
        for exp_name in sorted([exp for exp in dir(exp_module) if exp[0].isupper()]):
            exp_names.append(exp_name)
            exp_class = getattr(exp_module, exp_name)
            exp_name, exp_options, ana_options = exp_class_to_model(exp_class)
            exp_data = {
                "exp_name": exp_name,
                "exp_params": {
                    "experiment_options": exp_options.to_dict(),
                    "analysis_options": ana_options.to_dict(),
                },
            }
            exp_dict[exp_module_name].append(exp_data)
            _temp_exp_map[exp_name] = exp_data

    # extract batch experiment options
    batch_experiments = []
    for exp_name in sorted([exp for exp in dir(batch) if exp[0].isupper()]):
        exp_class = getattr(batch, exp_name)
        exp_name, exp_options, ana_options = exp_class_to_model(exp_class)
        dag_data = dag_from_batch_experiment(exp_class, _temp_exp_map)
        exp_data = {
            "exp_name": exp_name,
            "exp_params": {
                "experiment_options": exp_options.to_dict(),
                "analysis_options": ana_options.to_dict(),
                "dag": dag_data
            },
        }
        batch_experiments.append(exp_data)
    exp_dict.update({"BatchExperiment": batch_experiments})

    return exp_dict


def get_official_dags():
    abs_path = Path(__file__).absolute().parent.parent
    json_file = abs_path.joinpath("data").joinpath("QubitCalibrationDAG.json")
    with open(json_file, encoding="utf-8") as fp:
        data = json.load(fp)
    return data


def read_visage_config(file_name) -> QDict:
    config_parser = configparser.ConfigParser(inline_comment_prefixes="#")
    config_parser.read(file_name, encoding="utf-8")
    sec = config_parser.sections()
    q_setting = QDict.fromkeys(sec)
    for op in sec:
        ops = config_parser.options(op)
        temp_dict = QDict.fromkeys(ops)
        q_setting[op] = temp_dict
        for key in ops:
            value = literal_eval(config_parser.get(op, key))
            temp_dict[key] = value

    return q_setting


def save_config(filename: str, data: Dict):
    """
    save config to local.
    """
    os.chdir(os.path.dirname(filename))
    cf = configparser.ConfigParser()

    for module_name, module in data.items():
        cf.add_section(module_name)
        for key, value in module.items():
            if isinstance(value, str):
                a = value.replace("\\", "\\\\")
                cf.set(module_name, key, f'"{a}"')
            else:
                cf.set(module_name, key, str(value))
    try:
        with open(filename, "w+", encoding="utf-8") as f:
            cf.write(f)
    except ImportError as e:
        print(e)
        pass


def get_runtime_exp_id() -> bytes:
    """
    get runtime experiment id by user .pyqcat data.yaml
    """
    if os.path.exists(DATA_PATH):
        data = parse_yaml_args(DATA_PATH)
        if isinstance(data, dict):
            exp_id = data.get("id", "")
            return str(exp_id).encode(encoding="utf-8")
    return b""


def dag_from_batch_experiment(exp, exp_libs):
    dag_json = {
        "dag_name": exp.__name__,
        "official": True,
        "execute_params": {
            "is_traceback": False,
            "is_report": False,
            "start_node": None,
            "search_type": "weight_search",
        },
        "id": "".join(str(uuid.uuid4()).split("-")),
    }

    node_edges = {}
    node_params = {}
    exp_id_list = []
    length = len(exp._std_flow)

    for i, e in enumerate(exp._std_flow):
        e_name = e.__name__
        exp_id = f"{e_name}_{str(uuid.uuid4()).split('-')[-1]}"
        exp_id_list.append(exp_id)
        exp_options, ana_options = get_exp_data(e_name, exp_libs)
        node_params.update(
            {
                exp_id: {
                    "exp_name": e_name,
                    "exp_params": {
                        "experiment_options": exp_options,
                        "analysis_options": ana_options,
                        "adjust_params": {},
                        "location": [-500 + i * 200, 100],
                        "port_pos": ["left", "right"],
                    },
                }
            }
        )

    for i, e in enumerate(exp_id_list):
        if i == length - 1:
            node_edges[e] = {}
        else:
            node_edges[e] = {exp_id_list[i + 1]: {"weight": 1}}

    dag_json["node_edges"] = node_edges
    dag_json["node_params"] = node_params

    return dag_json


def get_exp_data(exp_name: str, exp_libs: Dict):

    if exp_name in exp_libs:
        try:
            exp_options = exp_libs.get(exp_name).get("exp_params").get("experiment_options")
            ana_options = exp_libs.get(exp_name).get("exp_params").get("analysis_options")
            return exp_options, ana_options
        except AttributeError:
            return {}, {}
    return {}, {}
