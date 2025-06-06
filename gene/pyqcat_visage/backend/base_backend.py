# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/21
# __author:       Lang Zhu
"""
Base backend.
"""
import os
import pickle
import time
from abc import abstractmethod
from copy import deepcopy
from functools import wraps
from traceback import format_exc
from typing import List, Union
from uuid import uuid1

import zmq

from pyQCat import __version__ as monster_version
from pyQCat.executor import MonsterContextManager, ChipLineConnect
from pyQCat.invoker import DataCenter, Invoker
from pyQCat.log import pyqlog
from pyQCat.qubit import Coupler, Qubit, QubitPair
from pyQCat.structures import QDict
from pyqcat_visage import __version__ as visage_version
from pyqcat_visage.backend.utilities import save_config
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.protocol import TIMEOUT_WRONG, EXECUTE_ADDR, TASK_ADDR
from pyqcat_visage.structures import VisageMeta


def authentication(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = None
        try:
            res = func(*args, **kwargs)
        except Exception:
            detail_msg = format_exc()
            pyqlog.error(
                f"function {getattr(func, '__name__')} running crash, details:\n{detail_msg}"
            )
            res = {"code": 600, "data": None, "msg": detail_msg}

        if not isinstance(res, dict) or "code" not in res:
            res = QDict(code=200, data=res, msg="default success")
        if isinstance(res, dict):
            res = QDict(**res)
        return res

    return wrapper


class BaseBackend:

    def __init__(self):
        self._experiments = QDict()
        self._dag = QDict()
        self._components = []

        self.model_channels: ChipLineConnect = None
        self.view_channels = QDict()
        self.db = DataCenter()

        # record login user
        self.login_user = None
        self.invoker_addr = None

        # record current user permission
        self.is_admin = False
        self.is_super = False
        self.group_name = ""
        self.username = ""

        # experiment context
        # self.context_builder = ContextBuilder(init_link=False)
        self.context_builder = MonsterContextManager()

        # remove init config, Initialize the configure
        # file after the user login successful
        # self._init_config()

        # run process
        self.run_exp_process = None
        self.run_dag_process = None

        # head map struct
        self._heatmap_struct = None
        self._init_heatmap_struct()

        # heat map component
        self._heatmap_components = QDict(Qubit=QDict(),
                                         Coupler=QDict(),
                                         QubitPair=QDict())

        # filer str
        self._filter_str = ''

        # execute sock
        self._sock_execute = None
        self._sock_execute_cache = {}
        self.execute_heart = time.time()
        self.execute_is_alive = False

        # async task.
        self._async_task_sock = None

        self.sub_proc = None
        self.parallel_proc = None
        self.parallel_proc_records = None
        self._init_sock()

        # run state
        self.run_state = False

        # user state
        self.user_state = False

        # current experiment save path
        self.current_dirs = None

        # cache user data
        self.cache_user_path = None
        self._retry_times = 0
        self._retry_timestamp = time.time_ns()

        # parallel mode
        self._parallel_mode = False

        # lo_map
        self.lo_map = None
        self.bus_map = None

        # courier ip port
        self.courier_ip = ""
        self.courier_port = 8088

        # visage meta data
        self.meta = None

    @property
    def parallel_mode(self):
        return self._parallel_mode

    @property
    def experiment_names(self):
        exp_list = []
        for team in self.experiments.values():
            exp_list.extend(list(team.keys()))
        return exp_list

    def trans_parallel_mode(self):
        self._parallel_mode = not self._parallel_mode

    def _init_sock(self):
        _ctx = zmq.Context.instance()
        self._sock_execute = _ctx.socket(zmq.DEALER)
        self._sock_execute.connect(EXECUTE_ADDR)

        self._async_task_sock = _ctx.socket(zmq.PUSH)
        self._async_task_sock.connect(TASK_ADDR)

    def async_task_send(self, async_work: str, recall: Union[List[str], str] = None, *args, **kwargs):
        if recall and isinstance(recall, str):
            recall = [recall]
        task_json = {
            "op_code": "register",
            "async_work": async_work,
            "recall": recall,
            "args": args,
            "kwargs": kwargs
        }
        self._async_task_sock.send_json(task_json)

    def execute_send(self, op: bytes, secend_op: bytes = b"", need_recv: bool = False, time_out: float = 3, **kwargs):
        """
        execute send msg to execute process.
        """
        if self.sub_proc is None or not self.sub_proc.is_alive():
            print("execute process is not exist.")
            return None
        data = pickle.dumps(kwargs)
        request_id = str(uuid1())[:-13].encode()
        self._sock_execute.send_multipart([op, secend_op, request_id, data])
        if need_recv:
            self._sock_execute_cache.update({request_id: None})
            start_time = time.perf_counter()
            while True:
                if self._sock_execute_cache[request_id] is None:
                    if time.perf_counter() - start_time > time_out:
                        self._sock_execute_cache.pop(request_id)
                        time.sleep(0.05)
                        if request_id in self._sock_execute_cache:
                            self._sock_execute_cache.pop(request_id)
                        return TIMEOUT_WRONG
                    else:
                        time.sleep(0.05)
                else:
                    data = self._sock_execute_cache.pop(request_id)
                    return data
        else:
            return None

    def _init_heatmap_struct(self):
        qubit_struct_map = deepcopy(Qubit.qubit_unit_map)
        coupler_struct_map = deepcopy(Coupler.coupler_unit_map)
        pair_struct_map = deepcopy(QubitPair.unit_map)

        qubit_struct_map["dcm"] = {
            "K": "",
            "FM": "",
            "outlier": "",
        }
        qubit_struct_map["freq_max"] = "MHz"
        qubit_struct_map["freq_min"] = "MHz"

        self._heatmap_struct = {
            'Qubit': qubit_struct_map,
            'Coupler': coupler_struct_map,
            'QubitPair': pair_struct_map
        }

    # ----------------------------- property -----------------------------------

    @property
    def system_anno(self):
        sample = self.config.system.sample or '#'
        point_label = self.config.system.point_label or '#'
        env_name = self.config.system.env_name or '#'
        username = self.username

        return "-".join((username, env_name, sample, point_label))

    @property
    def config(self):
        return self.context_builder.config

    @property
    def experiment_context(self):
        return self.context_builder.context

    @property
    def filter_str(self):
        return self._filter_str

    @filter_str.setter
    def filter_str(self, name: str):
        self._filter_str = name

    @property
    def components(self):
        return self._components

    @property
    def experiments(self):
        """Get the experiment dict."""
        return self._experiments

    @property
    def dags(self):
        """Get the dags."""
        return self._dag

    @property
    def view_context(self):
        view_context = QDict()
        return view_context

    # ----------------------------- for login widget  -----------------------------------

    @abstractmethod
    def init_config(self):
        """
        init config
        """

    @abstractmethod
    def init_backend(self):
        """

        sync_courier
        init_exp
        init_dag
        init_component
        init_config
        """

    def login(self, username, password) -> None:
        """
        user login
        """

    @staticmethod
    def register(username, password, rep_pwd, email, group):
        ret_data = Invoker.register_account(username, password, rep_pwd, email, group)
        pyqlog.info(f"Register normal user account result \n{ret_data}")
        return ret_data

    @staticmethod
    def find_account(username, pre_pwd, new_pwd, email) -> bool:
        db = DataCenter()
        return db.reset_passwd(username, pre_pwd, new_pwd, email)

    @staticmethod
    def test_connect(ip, port) -> bool:
        conn_test = Invoker.test_connect(ip, int(port))
        if conn_test.get("code") == 200:
            return True
        else:
            return False

    @staticmethod
    def get_his_env():
        return Invoker.get_env()

    def set_env(self, ip, port):
        Invoker.set_env(invoker_addr=f"tcp://{ip}:{port}", enforcement_flag=True)
        self.courier_ip = ip
        self.courier_port = int(port)

    # ----------------------------- for system widget  -----------------------------------
    @abstractmethod
    def create_default_context(self) -> None:
        """
        init context.
        """

    @abstractmethod
    def query_chip_line(self) -> None:
        """
        query chip line, update in channels.
        """

    # ----------------------------- for main widget -----------------------------------

    def get_library(self, style: str):

        def _filter_exp(name):
            name = name.lower()
            new_exp = QDict(
                PreliminaryExperiment=QDict(),
                BaseExperiment=QDict(),
                CompositeExperiment=QDict(),
                CalibrationExperiment=QDict(),
                CustomerExperiment=QDict(),
                BatchExperiment=QDict()
            )
            for key, module in self._experiments.items():
                if key not in new_exp:
                    continue
                for exp_name, exp in module.items():
                    if exp_name.lower().startswith(name):
                        new_exp[key][exp_name] = exp
            return new_exp

        def _filter_dag(name):
            name = name.lower()
            new_dag = QDict()
            for key, dag in self._dag.items():
                if key.lower().startswith(name):
                    new_dag[key] = dag
            return new_dag

        if style.lower() == 'experiments':
            if self._filter_str:
                return _filter_exp(self._filter_str)
            else:
                new_dict = self._experiments.copy()
                if "ParallelExperiment" in new_dict:
                    new_dict.pop("ParallelExperiment")
                return new_dict
        elif style.lower() == 'dags':
            if self._filter_str:
                return _filter_dag(self._filter_str)
            else:
                return self._dag
        else:
            raise NameError(f'style is {style}')

    @abstractmethod
    def run_experiment(self, exp_name: str) -> bool:
        """
        execute experiment.
        if execute return True else False.
        """

    @abstractmethod
    def run_dag(self, dag):
        """
        execute dag.
        """

    # ----------------------------- for context widget -----------------------------------
    @abstractmethod
    def create_chip_line(self, row: str, col: str, layout_style: str):
        """
        create new chip line, check user right before.
        """

    # experiment and dag

    @abstractmethod
    def initial_config_data(self):
        """Initial config data from chip line parameters"""

    @abstractmethod
    def initial_base_qubit_data(self):
        """Initial base qubit data from chip line parameters"""

    @abstractmethod
    def set_env_bit(self, env_bit_list: List[str], set_all: bool = True):
        """
        set env bit.
        """

    def get_env_bit(self) -> List[str]:
        """
        get env bit.
        """
        return self.context_builder.env_bits

    @abstractmethod
    def add_inst(self):
        """
        add inst.
        """

    # ----------------------------- for component widget -----------------------------------

    def query_all_component(self):
        """
        query all component.
        """

    def query_component(self, name, user, point_label, sample):
        """
        query component.
        """

    def query_component_by_id(self, qid):
        """
        query component by id.
        """

    def save_component(self, name):
        """

        save component.
        """

    # ----------------------------- for other -----------------------------------
    def save_config(self, filename: str = None):
        """
        save config to local.
        """
        if not filename:

            if self.cache_user_path is None:
                return

            filename = os.path.join(self.cache_user_path, GUI_CONFIG.cache_file_name.config)

        save_config(filename, self.config.to_dict())

    def refresh_parallel_server(self):
        """

        refresh parallel serve.
        """

    def refresh_lo_info(self):
        """
        Refresh lo divide map information.
        """

    def refresh_chip_context(self):
        """
        Refresh chip data to context.
        """

    def load_user_cache_context(self):
        """
        Load cache context information.
        """

    def cache_context_manager(self):
        """
        Cache context manager global options
        """

    def refresh_meta_data(self):
        self.meta = VisageMeta(
            username=self.username,
            visage_version=visage_version,
            monster_version=monster_version,
            chip={
                "sample": self.config.system.sample,
                "env_name": self.config.system.env_name,
                "point_label": self.config.system.point_label,
            }
        )
