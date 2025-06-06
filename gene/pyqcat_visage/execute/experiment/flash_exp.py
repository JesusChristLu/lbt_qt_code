# -*- coding: utf-8 -*-

# This code is part of pyQCat Visage.
#
# Copyright (c) 2022-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/09/16
# __author:       SS Fang

import importlib
import os
import traceback
from copy import deepcopy
from typing import Dict, Any, Union

from prettytable import PrettyTable

import pyqcat_developer
from pyQCat.errors import ExperimentParallelError, PyQCatError, UnCollectionError
from pyQCat.experiments.proc_parallel_experiment import ParallelExperiment
from pyQCat.tools.utilities import (
    generate_exp_list,
    judge_exp_failed,
    transform_child_exp_options,
)
from pyqcat_visage.exceptions import LogicError
from pyqcat_visage.options import Options, RunOptions
from .exp_map import BaseExperiment, ExperimentMap, deal_exp_options, deal_exp_x_data
from ..log import logger

IGNORE_FILE = ["__init__.py", "__pycache__"]

f"""
Main Run Experiment class.

Visage backend used to process run experiment logic.
"""


class ExpExecutor(RunOptions):
    """Experiment Executor class."""

    def __init__(
        self, name: str = None, params: Dict = None, exp_path: Union[str, Dict] = None
    ):
        """Initial `ExpExecutor` object."""

        self._id = None
        self._name = name
        self._path = exp_path
        self._params = params if isinstance(params, dict) else {}

        self._results = {}  # monster experiment analysis.results
        self._quality = None  # monster experiment analysis.quality

        # Run experiment middle operate options
        self._run_options = self._default_run_options()
        self.update_params = {}
        self.status = 0
        self.monster_obj = None
        self.error = None

    @classmethod
    def from_dict(cls, data: Dict) -> "ExpExecutor":
        """From dict data create an Exp object.

        Args:
            data (dict): Create exp dict data.

        Returns:
            BaseDag: The dag object.
        """
        kwargs = {
            "name": data.get("exp_name"),
            "params": data.get("exp_params"),
            "exp_path": data.get("exp_params", {}).get("exp_path", None),
        }
        exp_obj = cls(**kwargs)
        return exp_obj

    def to_dict(self) -> Dict:
        """To dict data.

        Returns:
            dict: The exp object some main information to dict.
        """
        data = {
            "exp_name": self._name,
            "exp_params": self._params,
        }
        return data

    @classmethod
    def _default_run_options(cls) -> Options:
        """Set run experiment operate options.

        Options:
            context: An object of context...
            belong (str): Mark the experiment belong to dag or not.
            quality: An object of analysis quality...

        """
        options = super()._default_run_options()
        options.context = None
        options.belong = "normal"  # `normal` or `dag`
        options.simulator = False
        options.simulator_base_path = None
        options.update_context = True

        return options

    @property
    def id(self) -> str:
        """Return experiment register id."""
        return self._id

    @property
    def name(self) -> str:
        """Return experiment name."""
        return self._name

    @property
    def results(self) -> Dict:
        """Return results, which monster experiment analysis result."""
        return self._results

    @property
    def quality(self) -> Any:
        """Return results, which monster experiment analysis quality."""
        return self._quality

    def get_exp_cls(self) -> Any:

        exp_cls = None

        if self._name in ExperimentMap:
            exp_cls = ExperimentMap.get(self._name)
        elif self._path:
            try:
                developer_path = os.path.dirname(pyqcat_developer.__file__)
                exp_file_list = os.listdir(developer_path)
                exp_file_list = [
                    x.split(".", maxsplit=2)[0]
                    for x in (set(exp_file_list) - set(IGNORE_FILE))
                ]
                if self._path in exp_file_list:
                    exp_module_name = f"pyqcat_developer.{self._path}"
                    importlib.import_module(exp_module_name)
                    importlib.reload(eval(exp_module_name))
                    if self._name in dir(eval(exp_module_name)):
                        exp_cls = getattr(eval(exp_module_name), self._name)
            except Exception as e:
                logger.debug(f"Load custom exp ({self._name}) error, detail:\n{e}")

        if exp_cls is None:
            raise LogicError(f"Unable to load experiment ({self._name})")

        # if not issubclass(exp_cls, BaseExperiment):
        #     raise LogicError(f"Exp({self._name}) no inherit from BaseExperiment")

        return exp_cls

    def run(self):
        """Run Experiment."""
        try:

            exp_cls = self.get_exp_cls()

            # define parameters
            experiment_options = deal_exp_options(self._params.get("experiment_options", {}))
            analysis_options = deal_exp_options(self._params.get("analysis_options", {}))
            update_context = self.run_options.update_context
            is_parallel = self._params.get("parallel_mode", False)
            context_options = self._params.get("context_options", {})
            if not context_options:
                raise LogicError("No find any context options!")

            # check options
            if not self._params.get("dag"):

                # for BaseExperiment

                if self.run_options.simulator:
                    if not self.run_options.simulator_base_path or not experiment_options.get("simulator_data_path", None):
                        logger.warning(
                            f"use simulator, but path is None, so will be use simulator from courier!"
                        )
                        experiment_options.update({"simulator_data_path": None})
                    else:
                        experiment_options["simulator_data_path"] = os.path.join(
                            self.run_options.simulator_base_path,
                            experiment_options["simulator_data_path"],
                        )
                    experiment_options = deal_exp_x_data(self._params.get("experiment_options", {}), experiment_options)
                else:
                    experiment_options.update({"simulator_data_path": None})
                experiment_options["use_simulator"] = self.run_options.simulator

                # creat experiment context
                context_obj = self.run_options.context
                context_ = deepcopy(context_obj.generate_context(**context_options, use_parallel=is_parallel))

                # create monster experiment object
                if not is_parallel:
                    # for base experiment
                    monster_exp_obj = exp_cls.from_experiment_context(context_)
                    monster_exp_obj.set_experiment_options(**experiment_options)
                    monster_exp_obj.set_analysis_options(**analysis_options)
                else:
                    # for parallel experiment
                    experiment_options.pop("same_options", None)
                    experiment_options.pop("exp_name", None)

                    options = {
                        "experiment_options": experiment_options,
                        "analysis_options": analysis_options,
                    }

                    exps = generate_exp_list(exp_cls, context_, options)

                    monster_exp_obj = ParallelExperiment(exps)
                    self._name = exp_cls.__class__.__name__
                self.monster_obj = monster_exp_obj

                self._run_monster_exp()
                self._update_result_to_ctx(context_obj, update_context)
            else:
                # for Batch Experiment
                if not is_parallel:
                    batch_exp = exp_cls(self.run_options.context)
                    batch_exp.set_experiment_options(**experiment_options)
                    batch_exp.set_analysis_options(**analysis_options)
                    batch_exp.set_standard_dag(
                        self._params.get("dag"), self.run_options.simulator
                    )
                    batch_exp.run()
                    self.status = 1
                else:
                    if exp_cls.__name__ == "RBSpectrum":
                        exp_data = {}
                        qubit_names = list(
                            experiment_options.get("physical_unit").keys()
                        )
                        sweep_dict = experiment_options.get("freq_list")
                        dag_data = self._params.get("dag")
                        for key, data in dag_data.items():
                            exp_data[key] = {}
                            for qubit in qubit_names:
                                exp_options = transform_child_exp_options(
                                    qubit, data.get("experiment_options")
                                )
                                ana_options = transform_child_exp_options(
                                    qubit, data.get("analysis_options")
                                )
                                exp_options.update(
                                    {"use_simulator": self.run_options.simulator}
                                )
                                exp_options.pop("same_options", None)
                                exp_options.pop("exp_name", None)
                                exp_data[key].update(
                                    {
                                        qubit: {
                                            "experiment_options": deal_exp_options(exp_options),
                                            "analysis_options": deal_exp_options(ana_options),
                                        }
                                    }
                                )

                        # run parallel rb spectrum
                        from pyQCat.experiments.batch.rb_spectrum import (
                            parallel_rb_spectrum,
                        )

                        parallel_rb_spectrum(
                            self.run_options.context, qubit_names, sweep_dict, exp_data
                        )
                    else:
                        logger.warning(
                            f"Parallel Batch Experiment only support rb spectrum"
                        )

        except Exception as err:
            logger.error(f"{self.name} run error: {err}")
            logger.debug(f"detail:\n{traceback.format_exc()}")

            if not isinstance(err, PyQCatError):
                err = UnCollectionError("flash exp run")

            if isinstance(self.monster_obj, BaseExperiment):
                self.monster_obj.update_execute_exp(2)

            self.status = -1
            self.error = err
        finally:
            if self.monster_obj:
                self._id = str(self.monster_obj.id)
                self.monster_obj = None

    def _run_monster_exp(self):
        """ Run monster experiment and update execute status."""
        self.monster_obj.run()

        # wait all file thread pool finish
        if isinstance(self.monster_obj, ParallelExperiment):
            for exp in self.monster_obj.experiments:
                if exp.file:
                    exp.file.wait_thread_clear()
                    break
        else:
            self.monster_obj.file.wait_thread_clear()

        # collect exp run status
        if not isinstance(self.monster_obj, ParallelExperiment):
            if hasattr(self.monster_obj.analysis, "results"):
                self._results = self.monster_obj.analysis.results
            if hasattr(self.monster_obj.analysis, "quality"):
                self._quality = self.monster_obj.analysis.quality

            failed_flag = judge_exp_failed(self._quality)

            if failed_flag:
                self.status = 2
            else:
                self.status = 1
        else:
            # todo optimize error handler.

            # default 2 fail status
            self.status = 2

            # collect error
            parallel_error = None
            for exp in self.monster_obj.experiments:
                *_, err = exp.parallel_state
                if isinstance(err, Exception):
                    if parallel_error is None:
                        parallel_error = ExperimentParallelError()
                    self.status = -1
                    parallel_error.add_unit_state(exp.parallel_state)
            self.error = parallel_error

            # collect result
            for index, anas in enumerate(self.monster_obj.analyses):
                if anas:
                    _results = anas.results
                    _quality = anas.quality
                    failed_flag = judge_exp_failed(_quality)
                    # if not failed_flag or _quality is None:
                    if not failed_flag:
                        self.status = 1
                        for k, v in _results.items():
                            self._results[f"{k}-{index}"] = v

    def _update_result_to_ctx(self, context_obj, update_context: bool = True):

        # update experiment result to context.
        if self.status == 1 and update_context:

            table = PrettyTable()
            table.field_names = ["Physical unit", "Field", "Before update", "After update"]

            pre_key = None
            for result in self.results:

                update_info = context_obj.update_single_result_to_context(
                    self.results[result]
                )

                if update_info:
                    for key, uv_map in update_info.items():
                        if key == "Hardware":
                            continue
                        for pn, v in uv_map.items():
                            is_divide = True if pre_key and pre_key == str(key) else False
                            table.add_row(
                                [str(key), str(pn), str(v[0]), str(v[1])],
                                divider=is_divide
                            )
                            pre_key = str(key)
                        if key in self.update_params:
                            self.update_params[key].update(uv_map)
                        else:
                            self.update_params.update({key: uv_map})

                        # if key in context_obj.records:
                        #     context_obj.records[key].update(uv_map)
                        # else:
                        #     context_obj.records.update({key: uv_map})

            logger(f"The following experiment ({self.name}) results are used to update context:\n{table}", name="UPDATE")
            if self.update_params:
                update_names = list(self.update_params.keys())
                logger(f"Will Update Num: {len(update_names)}, Names: {update_names}")

    def update_dcm(self, ctx, name):
        new_dcm = self.results[name].value
        dcm_name = f"{new_dcm.name}_{new_dcm.level_str}.bin"
        old_value = ctx.chip_data.cache_dcm.get(dcm_name)
        ctx.chip_data.cache_dcm[dcm_name] = new_dcm
        
        if "discriminators" not in self.update_params:
            self.update_params.update({"discriminators": {}})

        self.update_params["discriminators"].update(
            {name: [old_value, self.results[name].value]}
        )

        if "discriminators" not in ctx.records:
            ctx.records.update({"discriminators": {}})

        ctx.records["discriminators"].update(
            {name: [old_value, self.results[name].value]}
        )
