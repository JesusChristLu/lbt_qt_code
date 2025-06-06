# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/02
# __author:       HanQing Shi
"""visage heart qthread."""
import asyncio
import json
import pickle
import time

import zmq
from PySide6.QtCore import QThread, Signal
from .backend import Backend
from enum import Enum
from pyqcat_visage.protocol import EXECUTE_HEART, EXECUTE_ADDR, ExecuteOp, EXECUTE_HEART_TIME, EXECUTE_START_TIMEOUT, \
    ExecuteMonsterOp, EXECUTE_CODE
from pyqcat_visage.structures import ProcInfo
from pyqcat_visage.tool.utilies import kill_process_by_pid
from zmq.asyncio import Context, Poller
from loguru import logger


class CommStatus(bytes, Enum):
    qubit = b"qubit"
    config = b"config"
    having_data = b"having_data"
    qs_state = b"qs_state"
    error_msg = b"error_msg"
    qs_probe = b"probe"
    logout = b"logout"


class HeartThead(QThread):
    qs_probe = Signal(str)
    qs_state_signal = Signal(bytes)
    cache = Signal(bytes, bool)
    logout = Signal()
    pull_remind = Signal()
    err_signal = Signal(bytes, list)
    execute_timeout = Signal(str)
    update_context = Signal(dict)
    execute_stats = Signal(str)
    report_error = Signal(bytes)
    update_dag_status = Signal(bytes)
    update_task_status = Signal(bytes)

    dynamic_graph = Signal(bytes, bytes)
    process_start = Signal()

    def __init__(self, backend: Backend) -> None:
        super().__init__()
        self._ctx = None
        self.sub_courier: zmq.Socket = None
        self.execute_sock: zmq.Socket = None
        self.poller: zmq.Poller = None
        self.backend: Backend = backend
        self.close_flag = False
        self.flag_close = False
        self.execute_restart_flag = False
        self.execute_need_force_close_flag: int = 0
        self.proc_info_dict = {}

    @property
    def user_name(self):
        return self.backend.login_user.get("username", "")

    def _create_sock(self):
        """
        create sock.
        """
        self._ctx = Context.instance()

        chip_info = f"{self.backend.config.system.sample}_|_{self.backend.config.system.env_name}".encode()
        courier_addr = f"tcp://{self.backend.courier_ip}:{self.backend.courier_port + 1}"
        self.sub_courier = self._ctx.socket(zmq.SUB)
        self.sub_courier.setsockopt(zmq.SUBSCRIBE, chip_info)
        if self.backend.login_user.get("token"):
            self.sub_courier.setsockopt(zmq.SUBSCRIBE, self.backend.login_user["token"].encode())
            self.sub_courier.setsockopt(zmq.SUBSCRIBE, self.backend.login_user["username"].encode())
        self.sub_courier.connect(courier_addr)

        self.execute_sock = self._ctx.socket(zmq.DEALER)
        self.execute_sock.setsockopt(zmq.IDENTITY, EXECUTE_HEART)
        self.execute_sock.connect(EXECUTE_ADDR)
        self.poller = Poller()
        self.poller.register(self.sub_courier, zmq.POLLIN)
        self.poller.register(self.execute_sock, zmq.POLLIN)

    def close(self):
        self.close_flag = True

    def deal_sub_sock(self, op, data, args):

        if op == CommStatus.qs_probe:
            self.qs_probe.emit(data.decode())
        elif op == CommStatus.qs_state:
            self.qs_state_signal.emit(data)
        elif op == CommStatus.qubit:
            self.cache.emit(data, True)
        elif op == CommStatus.config:
            self.cache.emit(data, False)
        elif op == CommStatus.logout:
            self.logout.emit()
        elif op == CommStatus.having_data:
            self.pull_remind.emit()
        elif op == CommStatus.error_msg:
            self.err_signal.emit(data, args)

    def deal_heart_sock(self, op, data):
        if op == ExecuteOp.heart:
            if not self.backend.execute_is_alive:
                self.backend.execute_is_alive = True
                self.backend.init_execute_context()

            self.backend.execute_heart = float(data[0].decode())
            execute_stats = data[1]
            if execute_stats == ExecuteOp.state_free:
                if self.backend.run_state:
                    self.execute_stats.emit("free")
                self.backend.run_state = False
            if execute_stats == ExecuteOp.state_running and not self.backend.run_state:
                self.execute_stats.emit("run")
        elif op in [ExecuteMonsterOp.dy_start.value, ExecuteMonsterOp.dy_loop.value, ExecuteMonsterOp.dy_end.value]:
            self.dynamic_graph.emit(op, data[0])
        elif op == ExecuteOp.update_context:
            context_info = pickle.loads(data[0])
            self.backend.context_builder.update_chip_records(context_info, -1)
        elif op == ExecuteOp.dag_context_rollback:
            context_info = pickle.loads(data[0])
            self.backend.context_builder.update_chip_records(context_info, 0)
        elif op == ExecuteOp.state_free:
            if self.backend.run_state:
                self.execute_stats.emit("free")
        elif op == ExecuteOp.update_dag_status:
            self.update_dag_status.emit(data[0])
        elif op == ExecuteOp.report_error:
            self.report_error.emit(data[0])
        elif op == ExecuteOp.stop_force:
            self.backend.execute_is_alive = False
            self.backend.run_state = False
            self.execute_need_force_close_flag = 1
            # self.execute_restart_flag = True
        elif op in ExecuteOp.need_recv_op:
            request, *rel_data = data
            if request in self.backend._sock_execute_cache:
                self.backend._sock_execute_cache.update({request: rel_data})
        elif op == ExecuteOp.init_config_ack:
            self.execute_stats.emit("free")
            self.process_start.emit()
        elif op == ExecuteOp.sync_proc_info:
            process_name, process_info, *_ = data
            if process_name in [EXECUTE_CODE]:
                process_info = ProcInfo(**json.loads(process_info))
                self.proc_info_dict.update({process_name: process_info})
        elif op == ExecuteOp.cron_info:
            self.update_task_status.emit(data[1])
        else:
            pass

    async def check_signal(self):
        while not self.close_flag:
            if self.backend.execute_is_alive:
                time_out_time = EXECUTE_HEART_TIME
            else:
                time_out_time = EXECUTE_START_TIMEOUT
            if self.execute_restart_flag:
                time_out_time = 1
            if time.time() - self.backend.execute_heart > time_out_time:
                # self.backend.execute_heart = time.time()
                if self.execute_restart_flag:
                    if self.execute_need_force_close_flag == 0:
                        self.execute_need_force_close_flag = 2
                    else:
                        if self.execute_need_force_close_flag == 2:
                            if EXECUTE_CODE in self.proc_info_dict:
                                kill_process_by_pid(self.proc_info_dict[EXECUTE_CODE].process_id)
                        logger.info(f"ready to start executor process")
                        self.execute_restart_flag = False
                        if EXECUTE_CODE in self.proc_info_dict:
                            self.proc_info_dict.pop(EXECUTE_CODE)
                        self.execute_timeout.emit("restart")
                else:
                    logger.warning(f"execute process running timeout: {time.time() - self.backend.execute_heart}")
            await asyncio.sleep(1)

    async def run_poller(self):
        while not self.close_flag:
            sock_signal = dict(await self.poller.poll(1000))
            if self.sub_courier in sock_signal:
                _, op, data, *args = await self.sub_courier.recv_multipart()
                self.deal_sub_sock(op, data, args)
            if self.execute_sock in sock_signal:
                op, *data = await self.execute_sock.recv_multipart()
                self.deal_heart_sock(op, data)

    def release_resource(self):
        if self.poller:
            self.poller.unregister(self.sub_courier)
            self.poller.unregister(self.execute_sock)
        if self.sub_courier:
            self.sub_courier.close()
        if self.execute_sock:
            self.execute_sock.close()

    async def async_run(self):

        poller_future = asyncio.create_task(self.run_poller())
        heart_future = asyncio.create_task(self.check_signal())

        await asyncio.gather(poller_future, heart_future)

    def run(self) -> None:
        self._create_sock()
        asyncio.run(self.async_run())
        self.release_resource()
