# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

import numpy as np

from pyQCat.structures import Options
import subprocess
import os
from pyQCat.log import pyqlog
from pyQCat.structures import QDict
from pyqcat_visage.exceptions import DataServerError
from loguru import logger

PRE = "PreliminaryExperiment"
BASE = "BaseExperiment"
COPS = "CompositeExperiment"
CUSTOM = "CustomExperiment"


def kill_process_by_pid(pid):
    pid = str(pid)
    pyqlog.info(f"kill old process, it's id:{pid}")
    result = subprocess.run(["TASKKILL", "/F", "/PID", pid], stdout=subprocess.PIPE)
    if result.returncode == 0:
        pyqlog.info(f"kill process success, it's id:{pid!r}")
    else:
        pyqlog.warning(f"kill process fail, it's id:{pid!r}")


def kill_old_process(port: int):
    """
    kill old process which keep the tcp port.
    the function could get the common execute stderr info.
    """
    with subprocess.Popen(
            f"netstat -ano|findstr {port} | findstr LISTENING | findstr TCP",
            shell=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
    ) as cmd:
        cmd.wait()
        # err_msg = cmd.stderr.read()
        # if err_msg and err_msg != b"":
        #     pyqlog.debug(f"kill old process failed, detail:{err_msg}")
        #     return
        res = cmd.stdout.readline()
        if res:
            process_id = res.split("LISTENING")[-1].strip(" ").strip("\n")
            kill_process_by_pid(process_id)


def transform_str_to_array_options(array_str):
    options = Options(start="", end="", step="", method="normal", detail=[])

    if not array_str.startswith("qarange"):
        options.detail = array_str
    else:
        start, end, step = array_str[8:-1].split(",")
        options.start = start
        options.end = end
        options.step = step
        options.method = "qarange"

    return options


def transform_array_to_str_pre(array):
    list_infos = {
        "start": None,
        "end": None,
        "step": None,
        "details": array,
        "style": "normal",
        "describe": array,
    }

    if array and isinstance(array, list) or isinstance(array, np.ndarray):
        if isinstance(array[0], int) or isinstance(array[0], float):
            array = np.array(array)
            gaps = list(set(list(np.round(array[1:] - array[:-1], 3))))
            if len(gaps) == 1:
                list_infos["start"] = round(array[0], 3)
                list_infos["end"] = round(array[-1], 3)
                list_infos["step"] = round(gaps[0], 3)
                list_infos["style"] = "qarange"
                list_infos[
                    "describe"
                ] = f"qarange({round(array[0], 3)}, {round(array[-1], 3)}, {round(gaps[0], 3)})"

    return list_infos


def transform_array_to_str(array):
    new_data = QDict(
        start=["", "str", True],
        end=["", "str", True],
        step=["", "str", True],
        style=["normal", ["qarange", "normal"], True],
        details=[None, "list", True],
        describe=["Points(0) | normal | None", "str", True],
    )

    if array is not None:
        if isinstance(array, np.ndarray):
            array = list(array)

        if not isinstance(array, list):
            raise ValueError(f"Only support list, but your list is {array}")

        if len(array) == 0:
            return new_data

        if not isinstance(array[0], int) and not isinstance(array[0], float):
            new_data.details[0] = array
            new_data.describe[0] = f"Points({len(array)}) | normal | {array}"
            return new_data

        n_array = np.array(array)
        gaps = list(set(list(np.round(n_array[1:] - n_array[:-1], 3))))
        if len(gaps) == 1:
            new_data.start[0] = str(round(n_array[0], 3))
            new_data.end[0] = str(round(n_array[-1], 3))
            new_data.step[0] = str(round(gaps[0], 3))
            new_data.style[0] = "qarange"
            new_data.details[0] = array
            new_data.describe[
                0
            ] = f"Points({len(array)}) | qarange | ({round(n_array[0], 3)}, {round(n_array[-1], 3)}, {round(gaps[0], 3)})"
            return new_data

        new_data.details[0] = array
        new_data.describe[0] = f"Points({len(array)}) | normal | {array}"
        return new_data
    else:
        return new_data


SYS_OPT = [
    "config",
    "crosstalk_dict",
    "ac_prepare_time",
    "idle_qubits",
    "readout_point_model",
    "record_text",
    "fake_pulse",
    "bind_baseband_freq",
]

SCHEDULE_OPT = [
    "schedule_flag",
    "schedule_save",
    "schedule_measure",
    "schedule_type",
    "schedule_index",
    "schedule_show_measure",
    "schedule_show_real",
    "schedule_show_bits",
    "schedule_show_params",
    "schedule_hide_ac",
]

SIMULATOR_OPT = ["simulator_shape", "simulator_data_path", "simulator_remote_path"]

INST_OPT = ["bind_dc", "bind_drive", "bind_probe", "file_flag"]

FUN_OPT = [
    "show_result",
    "multi_readout_channels",
    "repeat",
    "use_dcm",
    "data_type",
    "enable_one_sweep",
    "register_pulse_save",
    "measure_bits",
    "save_label",
    "is_dynamic",
    "fidelity_matrix",
    "loop_num",
    "iq_flag",
    "parallel",
    "fidelity_correct_type",
    "post_select_type",
    "save_result",
    "is_amend",
    "add_f12",
    "plot_iq",
    "save_context"
]

ANA_OPT = [
    "figsize",
    "is_plot",
    "raw_data_format"
]

FATHER_OPTIONS = []
FATHER_OPTIONS.extend(SYS_OPT)
FATHER_OPTIONS.extend(SCHEDULE_OPT)
FATHER_OPTIONS.extend(SIMULATOR_OPT)
FATHER_OPTIONS.extend(INST_OPT)
FATHER_OPTIONS.extend(FUN_OPT)
FATHER_OPTIONS.extend(ANA_OPT)


def to_gui_name(name: str) -> str:
    """Convert python code name to gui name."""
    nice_name = name.replace("_", " ").title()
    return nice_name


def to_python_name(name: str) -> str:
    """Convert gui name to python code name."""
    py_name = name.replace(" ", "_").lower()
    return py_name


def convert_to_hex_color(color_tuple: tuple) -> str:
    """Convert color tuple to strings.

    Args:
        color_tuple (tuple):Color tuple object. like (0, 128, 255, 255).

    Returns:
        color strings.
    """
    hex_color_string = "#"
    for color in color_tuple:
        hex_color_string += hex(color).strip("0x")
    return hex_color_string


def courier_response(ret_data, is_raise: bool = True, describe: str = ""):
    e = DataServerError.from_ret_data(ret_data)

    if e:

        if describe:
            logger.error(f"Courier Response: {describe} filed!")

        if is_raise:
            raise e

    else:
        if describe:
            logger.info(f"Courier Response: {describe} success!")
