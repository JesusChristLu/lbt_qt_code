# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

"""Data structure for experiment which used in Visage."""

import json
import os
import time
from typing import Dict, List, Union

from loguru import logger

from pyQCat.structures import QDict
from pyQCat.tools import qarange
from pyQCat.types import StandardContext
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.structures import VisageMeta
from pyqcat_visage.tool.utilies import FATHER_OPTIONS

Dict = Union[Dict, QDict]


class VisageExperiment:
    """Visage experiment data structure."""

    def __init__(
        self,
        gid: str,
        name: str,
        tab: str,
        exp_options: Dict = None,
        ana_options: Dict = None,
        ctx_options: Dict = None,
        location: list = None,
        port_pos: list = None,
        adjust_params: Dict = None,
        role: str = None,
    ):
        self.gid = gid
        self.name = name
        self.tab = tab
        self.location = location
        self.port_pos = port_pos or ["left", "right"]
        self.role = role
        self.adjust_params = adjust_params
        self.exp_path = None
        self.exp_describes = None

        self._model_exp_options = exp_options or QDict()
        self._model_ana_options = ana_options or QDict()
        self._context_options = ctx_options or QDict()

        self.parallel_options = QDict(
            model_exp_options=QDict(),
            model_ana_options=QDict(),
            ctx_options=QDict(
                name="",
                physical_unit=[],
                readout_type="",
            ),
        )
        self.is_parallel = False
        self.dag = None

    def __repr__(self):
        id_str = self.gid if self.gid else ""
        return f"{self.name}{id_str}"

    @property
    def model_exp_options(self):
        return self._model_exp_options

    @property
    def model_ana_options(self):
        return self._model_ana_options

    @property
    def context_options(self):
        return self._context_options

    @property
    def is_parallel_same(self):
        return self.parallel_options.model_exp_options.get("same_options", [False])[0] is True and self.is_parallel

    @property
    def physical_unit(self):
        return self._context_options.get("physical_unit", [""])[0]

    def rebuild(self):
        pass

    @staticmethod
    def trans_options(options: Dict, is_full: bool = True):
        new_options = {}
        for k, v in options.items():
            if is_full or k not in FATHER_OPTIONS:
                if isinstance(v, list) and isinstance(v[-1], bool):
                    new_options[k] = v[0]
                elif isinstance(v, dict):
                    if "describe" in v:
                        new_options[k] = v["describe"][0]
                    else:
                        new_options[k] = VisageExperiment.trans_options(v, is_full)
                else:
                    new_options[k] = v

        return new_options

    def get_flash_options(self, parallel_mode: bool, is_full: bool = True):
        if parallel_mode:
            experiment_options = self.trans_options(self.parallel_options.model_exp_options, is_full)
            analysis_options = self.trans_options(self.parallel_options.model_ana_options, is_full)
            context_options = self.trans_options(self.context_options, is_full)
        else:
            experiment_options = self.trans_options(self.model_exp_options, is_full)
            analysis_options = self.trans_options(self.model_ana_options, is_full)
            context_options = self.trans_options(self.context_options, is_full)

        return experiment_options, analysis_options, context_options

    def to_flash_dag_dict(self, parallel_mode=False):
        experiment_options, analysis_options, context_options = self.get_flash_options(parallel_mode)
        return {
            "exp_name": self.name,
            "exp_params": {
                "experiment_options": experiment_options,
                "analysis_options": analysis_options,
                "context_options": context_options,
                "sub_analysis_options": {},
                "exp_path": self.exp_path,
                "parallel_mode": parallel_mode
            },
            "adjust_params": self.adjust_params,
            "location": self.location,
            "port_pos": self.port_pos,
            "role": self.role,
        }

    def to_run_exp_dict(self, parallel_mode=False):

        experiment_options, analysis_options, context_options = self.get_flash_options(parallel_mode)

        return {
            "exp_name": self.name,
            "exp_params": {
                "experiment_options": experiment_options,
                "analysis_options": analysis_options,
                "context_options": context_options,
                "exp_path": self.exp_path,
                "parallel_mode": parallel_mode,
                "dag": self.dag.to_run_exp_dict(parallel_mode) if self.dag else None
            },
        }

    def to_file(self, dirname: str, meta: VisageMeta, is_full: bool = True, describe: str = None, is_save: bool = True):
        """Save options parameters to local file.

        Args:
            dirname: Local directory name.
            meta: Visage Metadata.
            is_full: Save all options to file.
            describe: Save all options to file.
            is_save: Save all options to file.
        """
        def _generate_file_data(_exp):
            # single
            eop, aop, ctx = _exp.get_flash_options(parallel_mode=False, is_full=is_full)

            # parallel
            p_eop, p_aop, _ = _exp.get_flash_options(parallel_mode=True, is_full=is_full)
            p_eop.pop("same_options", None)

            data = {
                "context_options": ctx,
                "options_for_regular_exec": {
                    "experiment_options": eop,
                    "analysis_options": aop,
                },
                "options_for_parallel_exec": {
                    "experiment_options": p_eop,
                    "analysis_options": p_aop,
                },
            }

            return data

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        exp_name = f"{self.name}_{describe}" if describe else self.name
        filename = os.path.join(dirname, f"{exp_name}.json")

        exp_data = {}
        visage_meta_data = meta.to_dict()
        visage_meta_data.update({
            "exp_class_name": self.name,
            "export_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        })
        exp_data.update({"meta": visage_meta_data})
        exp_data.update(_generate_file_data(self))

        if self.dag:
            exp_data["dag"] = {}

            for exp in self.dag.node_params.values():

                exp_name = exp.name
                index = 1
                while exp_name in exp_data["dag"]:
                    exp_name = f"{exp_name}_{index}"

                exp_data["dag"].update(
                    {
                        exp_name: _generate_file_data(exp)
                    }
                )

        if is_save:
            key = f"{self.name}_{describe}" if describe else self.name
            save_data = {key: exp_data}
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

            logger.log("UPDATE", f"Save {self.name} to {filename}!")

        return exp_data

    @classmethod
    def from_dict(cls, params: QDict, tab: str = "experiments"):
        exp = cls(
            gid="0x00",
            name=params.exp_name,
            tab=tab,
            exp_options=params.exp_params.experiment_options,
            ana_options=params.exp_params.analysis_options,
            ctx_options=params.exp_params.context_options,
            location=params.exp_params.get("location") or params.get("location"),
            port_pos=params.exp_params.get("port_pos") or params.get("port_pos"),
            adjust_params=params.exp_params.get("adjust_params") or params.get("adjust_params"),
            role=params.exp_params.get("role") or params.get("role"),
        )
        exp.exp_path = params.exp_params.get("exp_path")
        exp.exp_describes = params.exp_params.get("exp_describes")
        if params.exp_params.get("parallel_options"):
            exp.parallel_options = params.exp_params.parallel_options
        if params.exp_params.get("is_parallel"):
            exp.is_parallel = params.exp_params.is_parallel

        exp._context_validator()

        return exp

    def format_this(self, params: QDict):

        def _format_options(outer_data: Dict, local_data: Dict):
            if not isinstance(outer_data, dict) or not isinstance(local_data, dict):
                return

            for k, v in outer_data.items():
                if k not in local_data:
                    logger.warning(f"{self.name} Key({k}) not in std context options!")
                    continue

                std_v = local_data.get(k)
                if isinstance(std_v, List) and len(std_v) == 3:
                    if isinstance(std_v[1], List) and v not in std_v[1]:
                        logger.warning(f"{self.name} Key({k}) | Value({v}) not in std context options!")
                    else:
                        std_v[0] = v
                elif isinstance(std_v, dict) and isinstance(v, dict):
                    _format_options(v, std_v)
                elif isinstance(std_v, QDict) and v.startswith("Points("):
                    point, mode, data = v.split(" | ")
                    std_v.style[0] = mode
                    std_v.describe[0] = v
                    if mode == "normal":
                        std_v.details[0] = eval(data)
                    else:
                        sr, en, se = eval(data)
                        std_v.start[0] = float(sr)
                        std_v.end[0] = float(sr)
                        std_v.step[0] = float(sr)
                        std_v.details[0] = qarange(std_v.start[0], std_v.end[0], std_v.step[0])
                else:
                    logger.warning(f"Key({k}) | Value({v}) import failed!")

        # update base experiment
        _format_options(params.context_options, self.context_options)
        _format_options(params.options_for_regular_exec.experiment_options, self.model_exp_options)
        _format_options(params.options_for_regular_exec.analysis_options, self.model_ana_options)
        _format_options(params.options_for_parallel_exec.experiment_options, self.parallel_options.model_exp_options)
        _format_options(params.options_for_parallel_exec.analysis_options, self.parallel_options.model_ana_options)

    def to_save_dict(self, save_all: bool = False):
        name = "name" if save_all else "exp_name"
        params = "params" if save_all else "exp_params"
        return {
            name: self.name,
            params: {
                # use for experiment
                "experiment_options": self.model_exp_options.to_dict(),
                "analysis_options": self.model_ana_options.to_dict(),
                "context_options": self.context_options.to_dict(),
                "exp_path": self.exp_path,
                "exp_describes": self.exp_describes,
                "dag": self.dag.to_save_dag() if self.dag else None,
                "parallel_options": self.parallel_options.to_dict(),
                "is_parallel": self.is_parallel,

                # use for dag
                "adjust_params": self.adjust_params,
                "location": self.location,
                "port_pos": self.port_pos,
                "role": self.role,
            },
        }

    def _context_validator(self):
        if len(self.context_options) == 0:
            context_names = GUI_CONFIG.context_map.get(self.name)

            if not context_names:
                logger.warning(f'{self.name} no have default context name, support all!')
                context_names = [e.value for e in list(StandardContext)]

            cur_name = context_names[0]
            read_type = GUI_CONFIG.std_context.get(cur_name)
            if read_type:
                read_type_list = list(read_type.values())
                cur_read_type = [read_type_list[0], read_type_list, True]
            else:
                cur_read_type = ["", str.__name__, True]

            context_options = QDict(
                name=[cur_name, context_names, True],
                readout_type=cur_read_type,
                physical_unit=["", str.__name__, True]
            )

            self._context_options = context_options
