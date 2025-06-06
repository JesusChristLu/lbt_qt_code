# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/1/12
# __author:       Lang Zhu
"""
Load Experiment
"""

import pyqcat_developer
import os
import shutil
import traceback
import importlib
from pyQCat import experiments
from pyQCat.log import pyqlog
from typing import Union, List, AnyStr, Dict
from pyqcat_visage.exceptions import OperationError


def developer_base_local() -> str:
    developer_path = os.path.dirname(pyqcat_developer.__file__)
    return developer_path


def load_py_to_str(py_path: str):
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def write_cus_exp(params: Dict):
    exp_path = params.get('exp_params').get('exp_path')
    exp_describes = params.get('exp_params').get('exp_describes')
    developer_path = developer_base_local()

    filename = os.path.join(developer_path, f'{exp_path}.py')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(exp_describes)


def clear_developer(exp_path: str = None):
    developer_path = developer_base_local()

    if exp_path:
        if os.path.exists(f'{developer_path}\\{exp_path}.py'):
            os.remove(f'{developer_path}\\{exp_path}.py')
    else:
        names = os.listdir(developer_path)
        for name in names:
            if not name.startswith('__init__'):
                filename = f'{developer_path}\\{name}'
                if os.path.isdir(filename):
                    shutil.rmtree(filename)
                else:
                    os.remove(filename)


def _load_file(file_path: str) -> Union[int, List[AnyStr]]:
    if not os.path.exists(file_path):
        return -1
    file_name = os.path.basename(file_path)
    module_name = file_name.split(".", maxsplit=2)[0]
    if " " in module_name or "import" in module_name:
        return -4
    if not file_name.endswith(".py"):
        return -2
    base_path = developer_base_local()
    final_path = os.path.join(base_path, file_name)
    if os.path.exists(final_path):
        return -3
    shutil.copy(file_path, base_path)
    return module_name, final_path


def _load_module_experiment(file_name) -> list:
    base_path = developer_base_local()
    file_path = f"{file_name}.py"
    file_path = os.path.join(base_path, file_path)
    if os.path.exists(file_path):
        exp_module_name = f"pyqcat_developer.{file_name}"
        importlib.import_module(exp_module_name)
        importlib.reload(eval(exp_module_name))

        return _get_experiment_class(eval(exp_module_name))
    return []


def _get_experiment_class(exp_module):
    module_dir = [
        x for x in dir(exp_module)
        if not (x.startswith("__") and x.endswith("__"))
    ]

    experiment_class = []

    for tmp_name in module_dir:
        if tmp_name.endswith("Base"):
            continue
        if tmp_name in dir(experiments):
            continue
        tmp_obj = getattr(exp_module, tmp_name)
        if type(tmp_obj).__name__ == "type" and issubclass(
                tmp_obj, experiments.BaseExperiment):
            experiment_class.append(tmp_obj)

    return experiment_class


def load_experiment(local_path: str) -> dict:
    """load experiment from local file.

    Parameters
    ----------
    local_path : str
        the local file path.

    Returns
    -------
    dict
        return the load experiment info, if load failed, 
        will turn the failed code and failed reason in dict.
        such as:{
            "code":FF401,
            "msg":"file path or file type error, must python file",
            "exp_path": "rabi"
            "exp_file_path": "C:\\Users\\Desktop\\rabi.py",
            "exp_list": [],
        }
    """
    exp_info_dict = dict(code=400, msg="")
    res = _load_file(local_path)
    if res in [-1, -2, -3]:
        res_msg = {
            -1: "file path not exist",
            -2: "file type error, must python file",
            -3: "file has load.",
            -4: "Filenames are not compliant.",
        }
        exp_info_dict.update({
            "code":
                401,
            "msg":
                res_msg[res]
        })
        return exp_info_dict
    file_name, file_path = res
    exp_list = None
    try:
        exp_list = _load_module_experiment(file_name)
    except ImportError:
        exp_info_dict.update({
            "code":
                402,
            "msg":
                "parse file failed, find ImportError, please use absolute path"
        })
        pyqlog.debug("reload failed detail info:\n" + traceback.format_exc())
    except Exception:
        exp_info_dict.update({
            "code": 403,
            "msg": "parse file failed,Unexpected errors"
        })
        pyqlog.debug("reload failed detail info:\n" + traceback.format_exc())
    finally:
        if exp_list is None:
            os.remove(file_path)
            return exp_info_dict
        if not exp_list:
            os.remove(file_path)
            exp_info_dict.update({
                "code":
                    404,
                "msg":
                    "parse file can't find experiment class"
            })
            return exp_info_dict
    exp_info_dict.update({
        "exp_path": file_name,
        "exp_file_path": file_path,
        "exp_list": exp_list,
        "code": 200,
        "msg": "Success"
    })
    return exp_info_dict


__all__ = [
    "developer_base_local",
    "load_experiment",
    "load_py_to_str",
    "write_cus_exp",
    "clear_developer"
]
