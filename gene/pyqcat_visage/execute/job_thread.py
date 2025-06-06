# -*- coding: utf-8 -*-
# This code is part of pyQCat Visage.
#
# Copyright (c) 2022-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/18
# __author:       Lang Zhu
"""
executor run quantum experiment thread.
"""
import ctypes
import inspect
import pickle
import threading
from traceback import format_exc
from typing import Dict, Union
from pyQCat.executor import MonsterContextManager
from pyQCat.structures import QDict
from pyQCat.invoker import DataCenter
from pyqcat_visage import md
from pyqcat_visage.structures import Job
from pyqcat_visage.execute.tools import ExecuteContext, get_run_data
from pyqcat_visage.execute.network_client import InterComClient
from pyqcat_visage.execute.dag.flash_dag import Dag
from pyqcat_visage.execute.experiment.flash_exp import ExpExecutor
from pyqcat_visage.execute.log import logger
from pyqcat_visage.protocol import ExecuteInterOp
from pyQCat.tools.utilities import qarange
from pyqcat_visage.backend.backend import get_fake_task_context
from pyQCat.analysis.fit import amp2freq_formula, freq2amp_formula

Dict = Union[Dict, QDict]

SUPPORT_EXECUTE_TYPE = ["experiment", "dag", "sweep_dag"]


class JobThread(threading.Thread):
    def __init__(self, job: Job) -> None:
        threading.Thread.__init__(self)
        self.job = job
        self.records = None
        self.run_type = ""
        self.run_obj = None

    def close(self, msg: str = "press"):
        self.stop_thread()

    def stop_thread(self, exctype=SystemExit):

        def _clear_exp_thread(exp):
            if hasattr(exp, "td_list"):
                for td in exp.td_list:
                    logger.info(f"Close {exp} thread - {td}")
                    td.close()
            if hasattr(exp, "rc_sock"):
                if exp.rc_sock:
                    exp.rc_sock.send_multipart(["<Stop Parallel>, parallel".encode()])
                    exp.rc_sock.close()

        tid = ctypes.c_long(self.ident)
        if not inspect.isclass(exctype):
            exctype = type(exctype)

        # bugfix: dag no monster obj cause thread stop error
        if self.run_type == "experiment":
            _clear_exp_thread(self.run_obj.monster_obj)
        elif self.run_type == "dag":
            _clear_exp_thread(self.run_obj.current_node.exp_obj.monster_obj)

        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if self.run_type == "experiment":
            if self.run_obj and self.run_obj.monster_obj:
                self.run_obj.monster_obj.update_execute_exp(3)
        self.run_obj = None
        if res == 0:
            msg = f"stop thread error: invalid thread id, res: {res}"
            logger.error(msg)
            raise EnvironmentError(msg)
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            msg = f"PyThreadState_SetAsyncExc failed, res:{res}"
            logger.error(msg)
            raise EnvironmentError(msg)

    def run(self):
        if not self.job.run_config:
            self.job.run_config = {}
        self.records = self.execute(
            context=self.job.task_context_manage,
            run_type=self.job.task_type.value,
            run_data=self.job.run_data,
            **self.job.run_config,
        )
        self.job.records = self.records

    def execute(
        self,
        context: Union[MonsterContextManager, Dict],
        run_type: str,
        run_data: Dict = None,
        run_id: str = None,
        report: Dict = None,
        run: Dict = None,
    ) -> Union[dict, None]:
        update_context: bool = True if run.get("exp_save_mode", 0) == 1 else False
        simulator: bool = run.get("use_simulator", False)
        simulator_base_path: str = run.get("simulator_data_path", None)
        register: bool = run.get("register_dag", True)
        intercom = InterComClient()
        self.run_type = run_type
        if run_type not in SUPPORT_EXECUTE_TYPE:
            logger.error("execute got unsupported execute type")
            return
        try:
            with ExecuteContext(context) as ctx:
                if run_data is None:
                    if run_id:
                        run_data = get_run_data(
                            run_type=run_type, run_id=run_id, register=register
                        )

                    if not run_data:
                        logger.error("execute can't get execute run data.")
                        return

                run_id = None
                if run_type == "experiment":
                    run_obj = self.run_experiment(
                        ctx,
                        run_data,
                        simulator=simulator,
                        simulator_base_path=simulator_base_path,
                        update_context=update_context
                    )
                    run_id, run_error = run_obj.id, run_obj.error
                    if run_error is not None:
                        intercom.push_msg(
                            ExecuteInterOp.error.value,
                            ExecuteInterOp.error_exp.value,
                            pickle.dumps(run_error),
                        )
                    self.after_run(update_context, ctx, intercom)
                elif run_type == "dag":
                    run_data, update_context = self.pre_run_dag(
                        run_data, run_type, run, update_context
                    )
                    run_id = self.run_dag(
                        ctx,
                        run_data,
                        register,
                        simulator=simulator,
                        simulator_base_path=simulator_base_path,
                    )
                    self.after_run(update_context, ctx, intercom)
                elif run_type == "sweep_dag":
                    run_data, update_context = self.pre_run_dag(run_data, run_type, run, update_context)
                    sweep_list = run_data.get("sweep_list")
                    sweep_name = run_data.get("param")
                    qubit_name = run_data.get("qubit_name")
                    qubit = context.chip_data.cache_qubit.get(qubit_name)
                    fq_max, fc, M, _, d = qubit.ac_spectrum.standard
                    for var in sweep_list:
                        # set idle point and drive freq
                        if sweep_name == "freq":
                            qubit.drive_freq = var
                            left_result = freq2amp_formula(var, fq_max, fc, M, 0, d, "left")
                            right_result = freq2amp_formula(var, fq_max, fc, M, 0, d, "right")
                            if qubit.dc_max >= 0:
                                z_amp = left_result
                            else:
                                z_amp = right_result
                            qubit.idle_point = z_amp
                            qubit.readout_point.amp = -z_amp
                        else:
                            freq = round(amp2freq_formula(var, fq_max, fc, M, 0, d), 3)
                            qubit.drive_freq = freq
                            qubit.idle_point = var
                            qubit.readout_point.amp = -var
                        run_id = self.run_dag(
                            ctx,
                            run_data.get("dag"),
                            register,
                            simulator=simulator,
                            simulator_base_path=simulator_base_path,
                        )
                        self.after_run(update_context, ctx, intercom)

                if report is not None and report.get("is_report", False):
                    if run_id and run_type == "dag":
                        if isinstance(report, (dict, QDict)):
                            res = md.execute(id=run_id, **report)
                            report.update(file_name=f"dag_{run_id}")
                            file_path = md.save_report(report_doc=res, **report)
                            logger(f"report save path:{file_path}")
                logger.debug(f"Experiment is finished. (Run type: {run_type} | Run id: {run_id})")
        except Exception as err:
            logger(f"Execution of the experiment is crashed. (Exception in thread {self.__repr__()}: {err.__repr__()})")
            logger.error(format_exc())

    def run_experiment(
        self,
        context,
        exp_data: Dict,
        simulator: bool = False,
        simulator_base_path: str = None,
        update_context: bool = True
    ):
        """Run single experiment."""
        exp_obj = ExpExecutor.from_dict(exp_data)
        self.run_obj = exp_obj
        exp_obj.set_run_options(
            context=context,
            belong="normal",
            simulator=simulator,
            simulator_base_path=simulator_base_path,
            update_context=update_context
        )
        exp_obj.run()
        return exp_obj

    def run_dag(
        self,
        context,
        dag_data: Dict,
        register: bool = True,
        simulator: bool = False,
        simulator_base_path: str = None,
    ):
        """Run dag."""
        dag_obj = Dag.from_dict(dag_data)
        self.run_obj = dag_obj
        dag_obj.set_run_options(
            context=context,
            register=register,
            simulator=simulator,
            simulator_base_path=simulator_base_path,
        )
        dag_obj.run()
        return dag_obj.id

    def run_sweep_dag(
        self,
        context,
        dag_data: Dict,
        register: bool = True,
        simulator: bool = False,
        simulator_base_path: str = None,
    ):
        """Run dag."""
        dag_obj = Dag.from_dict(dag_data)
        self.run_obj = dag_obj
        dag_obj.set_run_options(
            context=context,
            register=register,
            simulator=simulator,
            simulator_base_path=simulator_base_path,
        )
        dag_obj.run()
        return dag_obj.id

    @staticmethod
    def pre_run_dag(run_data, run_type, run, update_context):
        if run_type == "sweep_dag":
            policy_opt = run_data["policy"].get("options")
            param = policy_opt.get("param")[0]
            scan_list = policy_opt.get("scan_list")
            style = scan_list.get("style")[0]
            if style == "qarange":
                start = scan_list.get("start")[0]
                end = scan_list.get("end")[0]
                step = scan_list.get("step")[0]
                sweep_list = qarange(start, end, step)
            else:
                sweep_list = scan_list.get("details")[0]
            res = get_fake_task_context(run_data.global_options, "dag", run_data.dag)
            for k, v in res.items():
                if v:
                    qubit_name = v[0]
            run_data.update({"sweep_list": sweep_list,
                             "param": param,
                             "qubit_name": qubit_name})
        else:
            execute_params = run_data.get("execute_params", {})
            if not execute_params.get("is_traceback", None):
                execute_params.update({"is_traceback": run.get("use_backtrace", False)})
            if not execute_params.get("node_update", None):
                dag_save_mode = run.get("dag_save_mode", 0)
                if dag_save_mode == 0:
                    update_context = False
                    execute_params.update({"node_update": False})
                else:
                    update_context = True
                    if dag_save_mode == 1:
                        execute_params.update({"node_update": True})

            run_data.update({"execute_params": execute_params})
        return run_data, update_context

    @staticmethod
    def after_run(update_context, ctx, intercom):
        if update_context:
            if ctx.update_records:
                res = ctx.extract_hot_data()
                if res:
                    db = DataCenter()
                    for k, v in res.items():
                        ret = db.update_single_config(k, file_data=v)
                        if ret.get("code") == 200:
                            logger.log("UPDATE", f"Update chip data {k}")

                if ctx.update_records:
                    record = pickle.dumps(ctx.update_records)
                    intercom.push_msg(
                        ExecuteInterOp.record.value,
                        ExecuteInterOp.record_update.value,
                        record,
                    )
                return ctx.update_records
