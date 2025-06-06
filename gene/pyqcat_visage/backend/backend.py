# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/24
# __author:       Lang Zhu
"""
Visage run backend.
"""
import json
import os
import re
import time
from collections import defaultdict
from functools import cmp_to_key
from multiprocessing import Process
from typing import Dict, List, Union
from datetime import datetime, timedelta

from prettytable import PrettyTable

from pyQCat.config import PyqcatConfig
from pyQCat.executor import ChipLineConnect
from pyQCat.executor.structures import ChipConfigField, generate_default_context_data
from pyQCat.hardware_manager import HardwareOffsetManager
from pyQCat.init_instrument_json import InitInstrument
from pyQCat.invoker import DataCenter, Invoker
from pyQCat.log import pyqlog
from pyQCat.parallel.parallel_utils import PARALLEL_PORT, run_parallel_server
from pyQCat.qubit import Coupler, Qubit, QubitPair, NAME_PATTERN
from pyQCat.structures import QDict
from pyQCat.tools import sort_bit
from pyQCat.tools.utilities import ac_spectrum_transform
from pyqcat_visage.backend.base_backend import BaseBackend, authentication
from pyqcat_visage.backend.component import VisageComponent
from pyqcat_visage.backend.experiment import VisageExperiment
from pyqcat_visage.backend.model_dag import ModelDag
from pyqcat_visage.backend.reload_exp import write_cus_exp, clear_developer
from pyqcat_visage.backend.utilities import ExperimentType
from pyqcat_visage.backend.utilities import (
    get_monster_exps,
    get_official_dags,
    read_visage_config,
)
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.exceptions import LogicError, OperationError
from pyqcat_visage.gui.types import DagRunMode, ExpRunMode
from pyqcat_visage.protocol import ExecuteOp, ExecutorJobType, CronType, ExecuteCronOp
from pyqcat_visage.structures import NormalJob, Cron, TemplateJob
from pyqcat_visage.tool.utilies import kill_old_process, courier_response


class Backend(BaseBackend):
    """
    Backend.
    """

    # ----------------------------- for login widget  -----------------------------------

    def init_config(self):
        cache_user_config_path = os.path.join(
            self.cache_user_path, GUI_CONFIG.cache_file_name.config
        )
        if os.path.exists(cache_user_config_path):
            # cache config
            self.context_builder.config = read_visage_config(cache_user_config_path)
            self.config.system.invoker_addr = self.invoker_addr
        else:
            # monster default config
            self.context_builder.config = PyqcatConfig(init_log=False)
            # report config
            self.config.report = QDict(
                is_report=False,
                id_type="dag",
                theme="light",
                system_theme=None,
                save_type="pdf",
                language="cn",
                report_detail="detail",
                file_path=self.config.system.local_root,
            )
            # run config
            self.config.run = QDict(
                exp_save_mode=ExpRunMode.un_save.value,
                simulator_data_path="",
                use_simulator=False,
                dag_save_mode=DagRunMode.save_process.value,
                use_backtrace=False,
                register_dag=False,
            )
            self.config.system.invoker_addr = self.invoker_addr

    def _init_exp(self):
        """
        get experiment msg from invoker, then translate exp_dict to visage experiment.
        """
        exp_res = self.db.query_exp_list()

        if exp_res and exp_res.get("code") != 200:
            pyqlog.warning(f"load experiment failed, detail:{exp_res.get('msg', None)}")

        exp_map = exp_res["data"] or {}
        for exp_module_name, exp_module in exp_map.items():
            child_exp_dict = QDict()
            for exp_class in exp_module:
                params = QDict(**exp_class)
                if params.exp_type == ExperimentType.custom.value:
                    write_cus_exp(params)
                exp_struct = VisageExperiment.from_dict(params)
                if params.exp_params.dag:
                    exp_struct.dag = ModelDag.from_dict(params.exp_params.dag)
                child_exp_dict[exp_struct.name] = exp_struct
            self._experiments[exp_module_name] = child_exp_dict

    def _init_dag(self):
        dag_res = self.db.query_dag_list()

        if dag_res and dag_res.get("code") != 200:
            pyqlog.warning(f"load dags failed, detail:{dag_res.get('msg', None)}")

        dag_list = dag_res["data"]

        for dag in dag_list:
            dag_obj = ModelDag.from_dict(dag)
            self._dag[dag_obj.name] = dag_obj

    def refresh_customer_exp(self, cus_exp_list):
        ret_data = self.db.init_customer_exp(cus_exp_list)
        courier_response(ret_data, "Save customer exp")
        self._init_exp()

    @staticmethod
    def _sync_courier():
        """
        if local version >= courier version, sync monster info to courier.
        """
        if Invoker.check_version() > 0:
            exp_data = get_monster_exps()
            dag_data = get_official_dags()
            db = DataCenter()
            res = db.init_experiment_and_dag(exp_data=exp_data, dag_data=dag_data)
            if res.get("code") == 405:
                pyqlog.error(res.get("msg"))
            return res

    def set_invoker_env(self):
        """Set invoker"""
        Invoker.set_env(
            self.config.system.invoker_addr,
            self.config.system.point_label,
            self.config.system.sample,
            self.config.system.env_name,
            enforcement_flag = True,
        )

    def init_backend(self):
        clear_developer()
        self._sync_courier()
        self._init_exp()
        self._init_dag()

    def _init_user_info(self):
        self.is_super = self.login_user.get("is_super")
        self.is_admin = self.login_user.get("is_admin")
        self.group_name = self.login_user.get("groups")
        self.username = self.login_user.get("username")

    @authentication
    def login(self, username, password):
        Invoker.set_env(invoker_addr=self.invoker_addr, enforcement_flag=True)
        if username and password:
            ret_data = Invoker.verify_account(username, password, enforcement_flag=True)
        else:
            ret_data = Invoker.load_account()
        # print(Invoker.invoker.job_client.token)
        if ret_data.get("code") == 200:
            self.login_user = ret_data.get("data")
            self._init_user_info()
        return ret_data

    def login_out(self, courier_exit: bool = False):
        # self._init_config()
        self._heatmap_components = QDict(
            Qubit=QDict(), Coupler=QDict(), QubitPair=QDict()
        )
        self._experiments = QDict()
        self._dag = QDict()
        self._components = []
        self.model_channels = None
        self.view_channels = QDict()
        # self.context_builder.clear_context()
        if not courier_exit:
            Invoker.logout_account()
            self.is_super = False
            self.is_admin = False

    # ----------------------------- for system widget  -----------------------------------

    @authentication
    def query_chip_line(self):
        chip_line_data = None

        if self.context_builder.chip_data:
            chip_line_data = self.context_builder.chip_data.cache_config.get(
                "chip_line_connect.json"
            )

        if chip_line_data is None:
            ret_data = self.db.query_chip_line()
            if ret_data.get("code") == 200:
                chip_line_data = ret_data.get("data")

        if chip_line_data:
            self.model_channels = ChipLineConnect(**chip_line_data)
            self.view_channels.update(chip_line_data.get("QubitParams"))
            self.view_channels.update(chip_line_data.get("CouplerParams"))

    # ----------------------------- for main widget -----------------------------------

    def _run_kwargs(self, run_id=None) -> Dict:
        run_dict = dict(
            run_id=run_id,
            report=self.config.report,
            run=self.config.run,
        )
        return run_dict

    def run_experiment(self, exp_name: str):
        visage_exp = None

        for key, module_exps in self._experiments.items():
            if exp_name in module_exps:
                visage_exp = module_exps.get(exp_name)
                break

        if visage_exp:
            if self.parallel_mode and "," not in visage_exp.physical_unit:
                raise OperationError(
                    f"Parallel mode no support current context physical unit!"
                )

            run_data = visage_exp.to_run_exp_dict(self.parallel_mode)
            normal_job = NormalJob(
                task_type=ExecutorJobType.EXPERIMENT,
                run_data=run_data,
                run_config=self._run_kwargs(),
                task_context_manage=self.context_builder,
            )
            self.execute_send(ExecuteOp.normal_job, job=normal_job)
            self.run_state = True
        else:
            raise LogicError(f"No find {exp_name} in experiment library!")

    def run_dag(self, dag: ModelDag):
        normal_job = NormalJob(
            task_type=ExecutorJobType.DAG,
            run_data=dag.to_dict(self.parallel_mode),
            run_config=self._run_kwargs(),
            task_context_manage=self.context_builder,
        )

        self.execute_send(ExecuteOp.normal_job, job=normal_job)
        self.run_state = True

    def generate_cron(self, tasks):
        cron_jobs = []
        if isinstance(tasks, dict):
            tasks = [tasks]
        for task_info in tasks:
            task = QDict(**task_info)
            policy = task.policy
            policy_type = policy.type
            enable = task.enable
            is_calibration = policy.options.is_calibration[0]
            repeat = 0
            unit = policy.options.unit[0]
            interval = policy.options.interval[0]
            if policy_type == "schedule":
                timing = None
                cron_type = CronType.INTERVAL
                repeat = policy.options.repeat[0]
                if unit == "min":
                    interval = interval * 60
                elif unit == "h":
                    interval = interval * 60 * 60
            elif policy_type == "timing":
                unit_map = {"s": "seconds", "min": "minutes", "h": "hours"}
                cron_type = CronType.TIMING
                current_date = datetime.now().date()
                total_nodes = policy.options.time_nodes[0]
                current_time = datetime(
                    current_date.year,
                    current_date.month,
                    current_date.day,
                    policy.options.hour[0],
                    policy.options.minute[0],
                    policy.options.second[0],
                )
                timing_nodes = [
                    current_time + timedelta(**{unit_map.get(unit): i * interval})
                    for i in range(total_nodes)
                ]
                timing = [
                    [timing_node.hour, timing_node.minute, timing_node.second]
                    for timing_node in timing_nodes
                ]
            else:
                return
            level = policy.options.priority[0]
            global_options = task.global_options
            sub_type = task.sub_type
            cron_id = task.id
            if sub_type == "exp":
                run_data = task.exp
                task_type = ExecutorJobType.EXPERIMENT
                fake_context = get_fake_task_context(global_options, sub_type, run_data)
            else:
                run_data = task.dag
                task_type = ExecutorJobType.DAG
                fake_context = get_fake_task_context(global_options, sub_type, run_data)

            normal_job = TemplateJob(
                task_type=task_type,
                task_name=task.task_name,
                task_desc=task.task_desc,
                policy=policy,
                sub_name=task.sub_name,
                run_data=run_data,
                fake_task_context=fake_context,
                global_context_params=global_options,
                run_config=self._run_kwargs(),
                level=level,
            )

            cron_job = Cron(
                cron_type=cron_type,
                cron_id=cron_id,
                interval=interval,
                timing=timing,
                repeat=repeat,
                level=level,
                calibration=is_calibration,
                job_template=normal_job,
                enable=enable,
            )
            cron_jobs.append(cron_job)
        return cron_jobs

    def run_cron(self, task_info):
        cron_jobs = self.generate_cron(task_info)
        self.execute_send(
            ExecuteOp.cron,
            secend_op=ExecuteCronOp.register_cron.value,
            cron_job=cron_jobs[0],
        )

    def remove_one_cron(self, cron_id):
        self.execute_send(
            ExecuteOp.cron, secend_op=ExecuteCronOp.remove_cron.value, cron_id=cron_id
        )

    def init_cron(self, task_list: List):
        cron_list = self.generate_cron(task_list)
        self.execute_send(
            ExecuteOp.cron, secend_op=ExecuteCronOp.init_cron.value, cron_list=cron_list
        )

    def run_sweep_dag(self, task_info):
        normal_job = NormalJob(
            task_type=ExecutorJobType.SWEEP_DAG,
            run_data=task_info,
            run_config=self._run_kwargs(),
            task_context_manage=self.context_builder,
        )

        self.execute_send(ExecuteOp.normal_job, job=normal_job)

    def init_execute_context(self):
        self.execute_send(op=ExecuteOp.init_config, config=self.context_builder.config)

    @authentication
    def save_one_exp_to_db(self, exp: "VisageExperiment"):
        return self.db.save_exp_options(**exp.to_save_dict())

    def save_all_exp(self, save_type: str, describe: str, items: list):
        if items:
            if save_type == "Database":
                exp_list = []
                for module in self._experiments.values():
                    for exp in module.values():
                        if exp.name in items:
                            exp_list.append(exp.to_save_dict(save_all=True))
                ret_data = self.db.save_exp_list(exp_list)
                courier_response(ret_data, describe="Save all exp")
            else:
                config_path = self.config.system.config_path
                dirname = os.path.join(config_path, "EXP", self.system_anno)

                exp_data = {}
                for module in self._experiments.values():
                    for exp in module.values():
                        if exp.name in items:
                            exp_data.update(
                                {
                                    exp.name: exp.to_file(
                                        dirname, self.meta, is_full=False, is_save=False
                                    )
                                }
                            )

                describe = describe or "options"
                save_file = os.path.join(dirname, f"{describe}.json")

                with open(save_file, "w", encoding="utf-8") as f:
                    json.dump(exp_data, f, indent=4, ensure_ascii=False)

                pyqlog.log("UPDATE", f"Save exp data to {save_file}!")
        else:
            pyqlog.warning(f"Please select the saved experiment!")

    def save_dag(self, dag: "ModelDag", mode: str = "DB"):
        dag.validate()
        dag_dict = dag.to_save_dag()

        if mode == "DB":
            ret_data = self.db.create_dag(
                name=dag.name,
                node_edges=dag_dict.get("node_edges"),
                execute_params=dag_dict.get("execute_params"),
                node_params=dag_dict.get("node_params"),
            )
            courier_response(ret_data, describe=f"Save dag ({dag.name})")
        else:
            config_path = self.config.system.config_path
            dirname = os.path.join(config_path, "DAG", self.system_anno)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            filename = os.path.join(dirname, f"{dag.name}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(dag_dict, f, indent=4, ensure_ascii=False)
            pyqlog.log("UPDATE", f"Save DAG {dag.name} in {filename}!")

    def delete_custom_exp(self, exp: "VisageExperiment"):
        ret_data = self.db.delete_customer_exp(exp.name)
        courier_response(ret_data, f"Delete custom exp {exp.name}")
        clear_developer(exp.exp_path)
        self.experiments.CustomerExperiment.pop(exp.name)

    # ----------------------------- for context widget -----------------------------------
    @authentication
    def initial_base_qubit_data(self, bit_names: List = None, delete: bool = False):
        """Initial base qubit data from chip line parameters"""
        # todo
        xy_baseband_freq = self.config.system.baseband_freq
        m_baseband_freq = self.config.system.m_baseband_freq
        params = QDict(
            m_baseband_freq=m_baseband_freq, xy_baseband_freq=xy_baseband_freq
        )
        fmt = QDict(qubit_fmt=Qubit(0).to_dict(), coupler_fmt=Coupler(0).to_dict())
        ret_data = self.db.init_base_qubit_data(params, fmt, bit_names, delete)
        return ret_data

    @authentication
    def initial_config_data(self, bit_names: List = None, delete: bool = False):
        """Initial config data from chip line parameters"""

        params = {
            "xy_baseband_freq": self.config.system.baseband_freq,
            "m_baseband_freq": self.config.system.m_baseband_freq,
        }

        qaio_type = self.config.system.qaio_type
        instrument_data = InitInstrument(unit_type=qaio_type).instrument
        ret_data = self.db.init_config_data(params, instrument_data, bit_names, delete)
        # todo
        return ret_data

    @authentication
    def set_env_bit(self, env_bit_list: List[str] = None, set_all: bool = True):
        if env_bit_list:
            bits = sorted(env_bit_list, key=cmp_to_key(sort_bit))
            self.context_builder.global_options.env_bits = bits
        else:
            if set_all:
                self.context_builder.global_options.env_bits = list(
                    self.view_channels.keys()
                )
            else:
                self.context_builder.global_options.env_bits = []

    @authentication
    def create_default_context(self):
        if self.experiment_context is None:
            return QDict(code=200, msg="Create a new context success")
        else:
            return QDict(code=600, msg="context is exited!")

    @authentication
    def context_add_bit(self, bit_name: str):
        if self.experiment_context:
            if bit_name.startswith("q"):
                bit_names = [qubit.name for qubit in self.experiment_context.qubits]
                bit_names.append(bit_name)
                self.experiment_context.configure_qubits(list(set(bit_names)))
            else:
                bit_names = [
                    coupler.name for coupler in self.experiment_context.couplers
                ]
                bit_names.append(bit_name)
                self.experiment_context.configure_couplers(list(set(bit_names)))
        else:
            return QDict(code=600, msg="The context is empty!")

    @authentication
    def context_add_dcm(self, dcm_name: str):
        if self.experiment_context:
            if not self.experiment_context.configure_dcm(dcm_name):
                return QDict(code=600, msg=f"{dcm_name} is empty!")
        else:
            return QDict(code=600, msg="The context is empty!")

    @authentication
    def context_add_crosstalk(self):
        if self.experiment_context:
            self.experiment_context.configure_crosstalk_dict()
        else:
            return QDict(code=600, msg="The context is empty!")

    @authentication
    def context_add_compensates(self, is_min: bool):
        if self.experiment_context:
            if is_min:
                self.experiment_context.minimize_compensate()
            else:
                self.experiment_context.maximize_compensate()
            self.experiment_context.configure_environment(
                self.context_builder.global_options.env_bits
            )
        else:
            return QDict(code=600, msg="The context is empty!")

    @authentication
    def save_chip_line(self):
        if self.model_channels:
            ret_data = self.db.create_chip_line(self.model_channels.to_dict())
            return ret_data
        else:
            return QDict(code=600, msg="channels is empty")

    @authentication
    def add_inst(self):
        if self.experiment_context:
            self.experiment_context.configure_inst()
            return QDict(code=200, msg="ok!")
        else:
            return QDict(code=600, msg="experiment context is none!")

    @authentication
    def build_sqc_context(self, pd):
        return QDict(
            code=600,
            msg="This interface has been deprecated!",
        )

    @authentication
    def build_cpc_context(self, pd):
        return QDict(
            code=600,
            msg="This interface has been deprecated!",
        )

    @authentication
    def build_nt_context(self, pd):
        return QDict(
            code=600,
            msg="This interface has been deprecated!",
        )

    @authentication
    def build_crosstalk_context(self, pd):
        return QDict(
            code=600,
            msg="This interface has been deprecated!",
        )

    @authentication
    def build_union_read_context(self, pd):
        return QDict(
            code=600,
            msg="This interface has been deprecated!",
        )

    @authentication
    def build_cz_calibration_context(self, pd):
        return QDict(
            code=600,
            msg="This interface has been deprecated!",
        )

    @authentication
    def build_coupler_cali_context(self, pd):
        return QDict(
            code=600,
            msg="This interface has been deprecated!",
        )

    @authentication
    def build_coupler_zz_shift_cali_context(self, pd):
        return QDict(
            code=600,
            msg="This interface has been deprecated!",
        )

    # ----------------------------- for component widget -----------------------------------

    def query_all_component(self):
        ret_data = self._query_components()
        if ret_data:
            self.context_builder.refresh_chip_data(ret_data.get("data"))

    def query_component(
        self, name, user=None, point_label=None, sample=None, env_name=None
    ):
        if name is None or name == "":
            raise OperationError(f"Please input query name!")

        ret_data = self._query_components(
            component_names=name,
            user=user,
            point_label=point_label,
            sample=sample,
            env_name=env_name,
        )
        if ret_data:
            self.context_builder.refresh_chip_data(ret_data.get("data"))
            return ret_data

    def query_component_by_id(self, qid):
        self._query_components(qid=qid)

    @authentication
    def query_history_component(
        self,
        name,
        user=None,
        point_label=None,
        sample=None,
        env_name=None,
        page: int = 1,
        volume: int = 10,
    ):
        user = user or self.login_user.get("username")
        sample = sample or self.config.system.sample
        point_label = point_label or self.config.system.point_label
        env_name = env_name or self.config.system.env_name

        ret_data = self.db.query_qcomponent_history(
            name=name,
            username=user,
            sample=sample,
            env_name=env_name,
            point_label=point_label,
            page_num=page,
            page_size=volume,
        )

        self._components.clear()

        if ret_data.get("code") == 200:
            cl = []
            for i, component_data in enumerate(ret_data.get("data")):
                cl.append(
                    VisageComponent.from_dict(component_data, len(ret_data.get("data")))
                )
            self._components = cl
        else:
            pyqlog.warning(f"query empty data, because {ret_data.get('msg')}")

    @authentication
    def save_component(self, name: Union[str, VisageComponent] = None):
        def save_note_log(ret_data: Dict, com_name):
            if ret_data.get("code") == 200:
                pyqlog.log("UPDATE", f"Save {com_name} success")
            else:
                pyqlog.warning(f'Save {com_name} failed, because {ret_data.get("msg")}')

        def save_data_single(com_obj: VisageComponent):
            if com_obj.style in ["qubit", "coupler", "qubit_pair"]:
                save_flag = True
                # if not self.login_user.get("is_super"):
                #     for x in com_obj.edit_records:
                #         if x.split(".")[-1] in UserForbidden.change_chip_params:
                #             ret_data = QDict(
                #                 code=800,
                #                 msg=f"You are not super user, can not save some chip param",
                #             )
                #             save_flag = False
                #             break
                if save_flag:
                    bit_name, bit = com_obj.to_data()
                    ret_data = bit.save_data(
                        close_log=True, update_list=com_obj.edit_records
                    )
                    com_obj.data["parameters"] = bit.to_dict()
                    com_obj.edit_records.clear()
            elif com_obj.name == "instrument.json":
                com_obj.data["json"]["pulse_period"] = com_obj.view_data["pulse_period"]
                com_obj.data["json"]["trig_way"] = com_obj.view_data["trig_way"]
                ret_data = self.db.update_single_config(*com_obj.to_data())
            elif com_obj.name == "chip_line_connect.json":
                if self.login_user.get("is_super"):
                    ret_data = self.db.update_single_config(*com_obj.to_data())
                else:
                    ret_data = QDict(
                        code=800,
                        msg=f"You are not super user, can not edit chip line connect json!",
                    )
            else:
                ret_data = self.db.update_single_config(*com_obj.to_data())
            save_note_log(ret_data, com_obj.name)
            return ret_data

        def save_data_many(components: List):
            bit_list = []
            conf_list = []
            for com_obj in components:
                if com_obj.style in ["qubit", "coupler"]:
                    bit_name, bit = com_obj.to_data()
                    bit_list.append(bit.to_dict_sock())
                else:
                    conf_list.append(com_obj.to_dict_conf())
            if bit_list:
                ret_data = self.db.update_qcomponent_list(bit_list)
                save_note_log(ret_data, [x["name"] for x in bit_list])
            if conf_list:
                ret_data = self.db.update_many_config(conf_list)
                save_note_log(ret_data, [x["filename"] for x in conf_list])

        if name:
            save_ok = False
            if isinstance(name, str):
                for component in self._components:
                    if name == component.name:
                        save_data_single(component)
                        save_ok = True
                        break
            else:
                res = save_data_single(name)
                if res["code"] == 200:
                    save_ok = True
                else:
                    return res
            if not save_ok:
                return QDict(
                    code=600, msg=f"{name} doesn't exist, please query it first!"
                )
        else:
            if len(self._components) == 0:
                return QDict(code=600, msg="component is empty!")
            else:
                save_data_many(self._components)

        return QDict(code=200, msg="save success!")

    @authentication
    def save_component_as(self, point_label: str, sample: str, env_name: str):
        Invoker.set_env(
            self.config.system.invoker_addr,
            point_label or self.config.system.point_label,
            sample or self.config.system.sample,
            env_name or self.config.system.env_name,
            enforcement_flag = True,
        )

        self.save_component()

        self.set_invoker_env()

        return QDict(
            code=200,
            msg=f"Save As | Sample({sample}) | "
            f"Point({point_label}) | ENV({env_name}) | Success!",
        )

    @authentication
    def refresh_component(self):
        if len(self._components) == 0:
            return QDict(code=600, msg="No find any component, please query first!")

        component_names = [c.name for c in self._components]
        return self._query_components(component_names=component_names)

    @authentication
    def save_to_file(self, dirname: str, save_list=None):
        if len(self._components) == 0:
            return QDict(code=600, msg="No find any component")

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        save_list = save_list or self._components
        cache_name = []
        for component in save_list:
            component.to_file(
                dirname, use_time_flag=True if component.name in cache_name else False
            )
            cache_name.append(component.name)

        return QDict(code=200, msg="To file success!")

    @authentication
    def import_components(self, filenames: List[str]):
        new_components = []
        if filenames:
            for filename in filenames:
                try:
                    with open(filename, encoding="utf-8") as a:
                        data = json.load(a)
                        cname = data.get("name")
                        new_components.append(VisageComponent.from_dict(data))
                        pyqlog.info(f"Import {cname} success!")
                except Exception as e:
                    pyqlog.error(f"{filename} import failed!")
                    pyqlog.debug(f"{filename} import failed! Because\n{e}")

            self._components = new_components
            return QDict(code=200, msg="Import success!")
        else:
            return QDict(code=600, msg="No find any file")

    def _query_components(
        self,
        qid: str = None,
        component_names: Union[str, List] = None,
        user: str = None,
        point_label: str = None,
        sample: str = None,
        env_name: str = None,
    ) -> Dict:
        def sort_component(c1, c2):
            # c1: VisageComponent = c1[1]
            # c2: VisageComponent = c2[1]

            if c1.style == c2.style:
                if c1.style == "qubit":
                    return -1 if int(c1.name[1:]) < int(c2.name[1:]) else 1
                elif c1.style == "coupler":
                    return (
                        -1
                        if int(c1.name.split("-")[0][1:])
                        < int(c2.name.split("-")[0][1:])
                        else 1
                    )
                return 0
            else:
                return -1 if c1.sort_level < c2.sort_level else 1

        user = user or self.login_user.get("username")
        sample = sample or self.config.system.sample
        point_label = point_label or self.config.system.point_label
        env_name = env_name or self.config.system.env_name

        ret_data = self.db.query_chip_all(
            qid=qid,
            name=component_names,
            username=user,
            sample=sample,
            point_label=point_label,
            env_name=env_name,
        )

        self._components.clear()

        if ret_data.get("code") == 200:
            for key, component_data in ret_data.get("data").items():
                name = component_data.get("name")

                if (
                    "." not in name
                    and name not in self.view_channels.keys()
                    and component_data.get("bit_type") != "QubitPair"
                ):
                    continue

                if (
                    name.endswith("dat")
                    and name.split(".")[0].split("_")[-1]
                    not in self.view_channels.keys()
                ):
                    continue

                self._components.append(
                    VisageComponent.from_dict(
                        component_data, len(ret_data.get("data").items())
                    )
                )

            self._components = sorted(self._components, key=cmp_to_key(sort_component))
        else:
            pyqlog.warning(f"query empty data, because {ret_data.get('msg')}")

        return ret_data

    # ----------------------------- for heatmap widget -----------------------------------

    @authentication
    def save_heatmap_to_local(self, dirname: str):
        def _to_dict(single_class_dict):
            r_dict = {}
            for key_, component in single_class_dict.items():
                r_dict[str(key_)] = component.to_dict()
            return r_dict

        local_heatmap = dict()
        for key, value in self._heatmap_components.items():
            local_heatmap[key] = _to_dict(value)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        filename = f"{self.system_anno}-heatmap.json"
        save_path = os.path.join(dirname, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(local_heatmap, f, indent=4, ensure_ascii=False)

        pyqlog.info(f"heatmap records save in {save_path}")
        return QDict(code=200, msg="Save success!")

    @authentication
    def import_heatmap_from_file(self, filename: str):
        with open(filename, encoding="utf-8") as a:
            data = json.load(a)

        if data:
            qubit_dict = {}
            for key, value in data.get("Qubit").items():
                qubit_dict[key] = Qubit.from_dict(value)

            coupler_dict = {}
            for key, value in data.get("Coupler").items():
                coupler_dict[key] = Coupler.from_dict(value)

            pair_dict = {}
            for key, value in data.get("QubitPair").items():
                pair_dict[key] = QubitPair.from_dict(value)

            new_heatmap = QDict(
                Qubit=QDict(**qubit_dict),
                Coupler=QDict(**coupler_dict),
                QubitPair=QDict(),
            )

            self._heatmap_components = new_heatmap
            return QDict(code=200, msg="Import heatmap success!")
        else:
            return QDict(code=600, msg="data is empty!")

    @authentication
    def save_heatmap_as(self, point_label: str, sample: str, env_name: str):
        point_label = point_label or self.config.system.point_label
        sample = sample or self.config.system.sample
        env_name = env_name or self.config.system.env_name

        Invoker.set_env(self.config.system.invoker_addr, point_label, sample, env_name, enforcement_flag=True)
        bits_data = []
        for component in self._heatmap_components.Qubit.values():
            bits_data.append(component.to_dict_sock())

        for component in self._heatmap_components.Coupler.values():
            bits_data.append(component.to_dict_sock())
        ret_dict = self.db.update_qcomponent_list(bits_data)
        msg = f"Save As | Sample({sample}) | Point({point_label}) | ENV({env_name}) | "
        if ret_dict.get("code") != 200:
            msg_ = msg + "Fail !"
            pyqlog.error(msg_)
            return QDict(code=600, msg=msg_)
        else:
            msg_ = msg + "Success !"
            pyqlog.log("UPDATE", msg_)
        self.set_invoker_env()

        return QDict(code=200, msg=msg_)

    # ----------------------------- for user manager -----------------------------------

    @authentication
    def query_all_groups(self):
        return self.db.query_all_groups()

    @authentication
    def query_group_info(self, target_group: str = None):
        return self.db.query_group_info(target_group)

    @authentication
    def create_group(self, group_name: str, description: str):
        return self.db.create_group(group_name, description)

    @authentication
    def change_user_group(self, target_user: str, target_group: str):
        return self.db.change_user_group(target_user, target_group)

    @authentication
    def change_group_leader(self, target_user: str, target_group: str, is_admin: bool):
        return self.db.change_group_leader(target_user, target_group, is_admin)

    @authentication
    def change_password(self, new_password: str):
        return self.db.change_passwd(new_password)

    # ----------------------------- for other -----------------------------------

    def import_experiments(self, filenames):
        def _update_exp(name: str, data: QDict):
            for modules in self.experiments.values():
                for exp in modules.values():
                    if exp.name == name:
                        # if _validate(exp, data):
                        exp.format_this(data)
                        pyqlog.info(f"Import {exp_name} success!")

        for filename in filenames:
            with open(filename, encoding="utf-8") as a:
                exp_name = filename.split("/")[-1].split(".")[0]
                exp_data = json.load(a)
                _update_exp(exp_name, QDict(**exp_data))

    def import_dags(self, filenames: str):
        def _update_dag(name: str, data: Dict):
            if name in self.dags.keys():
                anno = "UPDATE"
            else:
                anno = "ADD"
            self.dags[name] = ModelDag.from_dict(data, self.experiments)
            pyqlog.info(f"Import DAG {name} ({anno}) success!")

        for filename in filenames:
            dag_name = filename.split("/")[-1].split(".")[0]
            with open(filename, encoding="utf-8") as a:
                dag_data = json.load(a)
                _update_dag(dag_name, dag_data)

    def close_save(self):
        # save config information
        # Fix: save config move to system config widget save config action
        # self.save_config()

        # save all exp
        self.save_all_exp(save_type="DB", describe="", items=self.experiment_names)

        # save all no official dag
        for dag in self._dag.values():
            if not dag.official:
                self.save_dag(dag)

    @authentication
    def query_other_user_data(
        self,
        username: str,
        type_name: str = None,
        sample: str = None,
        env_name: str = None,
        point_label: str = None,
    ):
        return self.db.query_other_user_data(
            username, type_name, sample, env_name, point_label
        )

    @authentication
    def copy_other_user_data(
        self,
        from_user: str,
        from_sample: str,
        from_env_name: str,
        from_point_label: str,
        local: bool = True,
        to_user: str = None,
        to_sample: str = None,
        to_env_name: str = None,
        to_point_label: str = None,
        element_names: List[str] = None,
        element_configs: List[str] = None,
        copy_qubit: bool = True,
    ):
        return self.db.copy_other_user_data(
            from_user,
            from_sample,
            from_env_name,
            from_point_label,
            local,
            to_user,
            to_sample,
            to_env_name,
            to_point_label,
            element_names,
            element_configs,
            copy_qubit,
        )

    @authentication
    def query_username_list(self):
        return self.db.query_usernames()

    @authentication
    def query_sample_list(self, username: str):
        return self.db.query_sample_list(username)

    @authentication
    def query_env_name_list(self, username: str, sample: str):
        return self.db.query_env_name_list(username, sample)

    @authentication
    def query_point_label_list(self, username: str, sample: str, env_name: str):
        return self.db.query_point_label_list(username, sample, env_name)

    @authentication
    def query_conf_type_list(self):
        return self.db.query_conf_type_list()

    def refresh_parallel_server(self):
        ip = self.config.mongo.inst_host
        port = self.config.mongo.inst_port
        qaio_type = self.config.system.qaio_type
        if self.parallel_proc is None:
            kill_old_process(PARALLEL_PORT)
            self.parallel_proc = Process(
                target=run_parallel_server, args=(self.config.to_dict(),)
            )
            self.parallel_proc.daemon = True
            self.parallel_proc_records = QDict(ip=ip, port=port, qaio_type=qaio_type)
        elif self.parallel_proc.is_alive():
            if (
                ip == self.parallel_proc_records.ip
                and port == self.parallel_proc_records.port
                and qaio_type == self.parallel_proc_records.qaio_type
            ):
                return
            else:
                self.parallel_proc.kill()
                self.parallel_proc = Process(
                    target=run_parallel_server, args=(self.config.to_dict(),)
                )
                self.parallel_proc.daemon = True
                self.parallel_proc_records = QDict(
                    ip=ip, port=port, qaio_type=qaio_type
                )
        else:
            pyqlog.error(f"Parallel server is killed！")
            return

        self.parallel_proc.start()

    def band_lo_num(self, units):
        module, lo, bits = units
        for bit in bits:
            qubit = self.context_builder.chip_data.cache_qubit.get(bit)
            pre_lo = qubit.inst.get(f"{module}_lo")
            qubit.inst.update({f"{module}_lo": lo})
            pyqlog.log("UPDATE", f"{bit} {module} lo from {pre_lo} to {lo}!")
            qubit.save_data(close_log=True)

    def band_bus_num(self, units):
        bus, bits = units
        for bit in bits:
            qubit = self.context_builder.chip_data.cache_qubit.get(bit)
            pre_bus = qubit.inst.bus
            qubit.inst.update({"bus": int(bus)})
            pyqlog.log("UPDATE", f"{bit} bus from {pre_bus} to {bus}!")
            qubit.save_data(close_log=True)

    def refresh_lo_info(self, log: bool = False):
        self.lo_map = defaultdict(list)
        self.bus_map = defaultdict(list)

        for key, bit in self.context_builder.chip_data.cache_qubit.items():
            if isinstance(bit.inst.xy_lo, int):
                self.lo_map[f"xy-lo-{bit.inst.xy_lo}"].append(bit.name)

            if isinstance(bit.inst.m_lo, int):
                self.lo_map[f"m-lo-{bit.inst.m_lo}"].append(bit.name)

            if isinstance(bit.inst.bus, int):
                self.bus_map[f"Bus-{bit.inst.bus}"].append(bit.name)

        if log:
            table = PrettyTable()
            table.field_names = ["Label", "Qubits"]

            for k, v in self.lo_map.items():
                v.sort(key=cmp_to_key(sort_bit))
                table.add_row([k, str(v)])
            for k, v in self.bus_map.items():
                v.sort(key=cmp_to_key(sort_bit))
                table.add_row([k, str(v)])

            pyqlog.info(f"XY/M LO Layout as follow:\n{table}")

    def export_inst_information(self, dirname: str):
        data = {
            "bus": defaultdict(list),
            "xy-lo": defaultdict(list),
            "m-lo": defaultdict(list),
        }
        for name, bit in self.context_builder.cache_bits._cache.items():
            if isinstance(bit, Qubit):
                bus = bit.inst.bus
                xy_lo = bit.inst.xy_lo
                m_lo = bit.inst.m_lo

                if bus:
                    data["bus"][bus].append(bit.name)

                if xy_lo:
                    data["xy-lo"][xy_lo].append(bit.name)

                if m_lo:
                    data["m-lo"][m_lo].append(bit.name)
        if data:
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            filename = os.path.join(dirname, f"{self.system_anno}-inst.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                pyqlog.info(f"inst to file success, save in {filename}")

    def import_inst_information(self, filename: str):
        if os.path.isfile(filename):
            with open(filename, encoding="utf-8") as a:
                data = json.load(a)
                for num, bit_names in data.get("bus").items():
                    self.band_bus_num([int(num), bit_names])
                for num, bit_names in data.get("xy-lo").items():
                    self.band_lo_num(["xy", int(num), bit_names])
                for num, bit_names in data.get("m-lo").items():
                    self.band_lo_num(["m", int(num), bit_names])

    def view_hardware_offset(self, save_records: bool = False, view_unlink: bool = False):
        def relative_delay(*delays):
            min_v = min(delays)
            anno = ""
            for delay in delays:
                anno += f" {round(delay - min_v, 4)} |"
            return anno[:-1]

        chip_data = self.context_builder.chip_data.cache_config.get(
            ChipConfigField.chip_line
        )
        hardware_data = self.context_builder.chip_data.cache_config.get(
            ChipConfigField.hardware_offset
        )
        records_data = {}

        if chip_data and hardware_data:
            qubits = chip_data.get("QubitParams")
            couplers = chip_data.get("CouplerParams")
            manager = HardwareOffsetManager.from_data(hardware_data)

            table1 = PrettyTable()
            table1.field_names = ["unit", "xy", "z", "relative"]
            for qubit, params in qubits.items():
                xy_channel = params.get("xy_channel")
                z_channel = params.get("z_flux_channel")
                x_delay = manager.xy_delay[xy_channel]
                z_delay = manager.z_delay[z_channel]
                if manager.link.is_xz_link(xy_channel, z_channel):
                    table1.add_row(
                        [
                            qubit,
                            f"{xy_channel} | {x_delay}",
                            f"{z_channel} | {z_delay}",
                            relative_delay(x_delay, z_delay),
                        ]
                    )
                    records_data[qubit] = {
                        "xy": {"channel": xy_channel, "delay": x_delay},
                        "z": {"channel": z_channel, "delay": z_delay},
                    }
                elif view_unlink:
                    table1.add_row(
                        [
                            f"{qubit}-unlink",
                            f"{xy_channel} | {x_delay}",
                            f"{z_channel} | {z_delay}",
                            relative_delay(x_delay, z_delay),
                        ]
                    )

            table2 = PrettyTable()
            table2.field_names = ["unit", "zc", "zp", "zd", "relative"]
            for coupler, params in couplers.items():
                z_channel = params.get("z_flux_channel")
                probe_qubit = f"q{params.get('probe_bit')}"
                drive_qubit = f"q{params.get('drive_bit')}"
                probe_qubit_z = qubits.get(probe_qubit).get("z_flux_channel")
                drive_qubit_z = qubits.get(drive_qubit).get("z_flux_channel")
                _, link_infos = manager.link.mul_z_link_check(
                    z_channel, probe_qubit_z, drive_qubit_z
                )
                d_c = manager.z_delay[z_channel]
                d_p = manager.z_delay[probe_qubit_z]
                d_d = manager.z_delay[drive_qubit_z]
                if "unlink" not in link_infos:
                    table2.add_row(
                        [
                            coupler,
                            f"{z_channel} | {d_c}",
                            f"{probe_qubit_z} | {d_p}",
                            f"{drive_qubit_z} | {d_d}",
                            relative_delay(d_c, d_p, d_d),
                        ]
                    )
                    records_data[coupler] = {
                        "zc": {"channel": z_channel, "delay": d_c},
                        "zp": {"channel": probe_qubit_z, "delay": d_p},
                        "zd": {"channel": drive_qubit_z, "delay": d_d},
                    }
                elif view_unlink:
                    table2.add_row(
                        [
                            f"{coupler}-unlink",
                            f"{z_channel} | {d_c}",
                            f"{probe_qubit_z} | {d_p}",
                            f"{drive_qubit_z} | {d_d}",
                            relative_delay(d_c, d_p, d_d),
                        ]
                    )

            pyqlog.info(f"XYZTiming as follows: \n{table1}")
            pyqlog.info(f"ZZTiming as follows: \n{table2}")

            if save_records:
                config_path = self.config.get("system").get("config_path")
                timestamp_str = datetime.fromtimestamp(time.time()).strftime(
                    "%Y-%m-%d %H-%M-%S"
                )
                filename = os.path.join(
                    config_path, f"{self.system_anno}-line-delay-{timestamp_str}.json"
                )
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(records_data, f, indent=4, ensure_ascii=False)
                    pyqlog.info(f"hardware records to file success, save in {filename}")

    def ac_spectrum_transform(self, unit: str, value: float, mode: int = 0):
        physical_unit = self.context_builder.chip_data.get_physical_unit(unit)
        ac_spectrum_transform(physical_unit, value, mode)

    def refresh_chip_context(self):
        ret_data = self.db.query_chip_all(
            username=self.login_user.get("username"),
            sample=self.config.system.sample,
            point_label=self.config.system.point_label,
            env_name=self.config.system.env_name,
        )
        if ret_data.get("code") == 200:
            chip_data = ret_data.get("data")
            self.context_builder.refresh_chip_data(chip_data)
            self.context_builder.chip_data.sorted_qubits()

    def load_user_cache_context(self):
        cache_data = self.db.query_cache_data()
        if cache_data.get("code") == 200:
            cache_data = cache_data.get("data")
            context_default_data = cache_data.get("context_data")
            if not context_default_data:
                cache_data["context_data"] = generate_default_context_data()
            self.context_builder.set_global_options(**cache_data)

    def cache_context_manager(self):
        self.db.update_cache_data(self.context_builder.global_options.to_dict())


def generate_exp(exp_class: QDict) -> VisageExperiment:
    """
    generate VisageExperiment from exp
    """
    return VisageExperiment.from_dict(exp_class)


def get_fake_task_context(global_options, sub_type, data_dict):
    ctx_names = []
    qubits = []
    couplers = []
    pairs = []
    physical_dict = {}
    fake_context = {}
    context_data = global_options.context_data
    if not data_dict:
        return
    if sub_type == "exp":
        exp_params = data_dict.exp_params
        context_options = exp_params.context_options
        physical_dict[context_options.name] = context_options.physical_unit
    else:
        node_params = data_dict.node_params
        for node, params in node_params.items():
            ctx_name = params.exp_params.context_options.name
            ctx_names.append(ctx_name)
            physical_dict[ctx_name] = params.exp_params.context_options.physical_unit

    for ctx_name, ctx_conf in context_data.items():
        if ctx_name in ctx_names:
            if ctx_conf.default:
                physical_unit = ctx_conf.physical_unit
            else:
                physical_unit = physical_dict.get(ctx_name)
            if re.match(NAME_PATTERN.qubit, physical_unit):
                qubits.append(physical_unit)
            elif re.match(NAME_PATTERN.coupler, physical_unit):
                couplers.append(physical_unit)
            elif re.match(NAME_PATTERN.qubit_pair, physical_unit):
                pairs.append(physical_unit)
    fake_context.update({"qubits": qubits, "couplers": couplers, "pairs": pairs})
    return fake_context
