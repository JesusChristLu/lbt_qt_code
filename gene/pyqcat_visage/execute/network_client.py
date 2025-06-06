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
dynamic plot socket.
"""

import json
import os
import time

import zmq
from loguru import logger

from pyQCat.structures import Singleton
from pyqcat_visage.protocol import EXECUTE_DYNAMIC, EXECUTE_ADDR, ExecuteOp, ExecuteInterOp, ExecuteMonsterOp


class VisageDynamic(metaclass=Singleton):
    remove_parameters_list = ["exp"]
    monster_key = ExecuteOp.monster

    def __init__(self):
        self.sock = None
        self.is_connect = False

    def _init_sock(self):
        try:
            ctx = zmq.Context.instance()
            self.sock = ctx.socket(zmq.DEALER)
            self.sock.setsockopt(zmq.IDENTITY, EXECUTE_DYNAMIC)
            self.sock.setsockopt(zmq.CONNECT_TIMEOUT, 200)
            self.sock.setsockopt(zmq.RCVTIMEO, 1000)
            self.sock.connect(EXECUTE_ADDR)
            time.sleep(0.01)
            self.sock.send_multipart([ExecuteOp.monster, ExecuteMonsterOp.dy_init.value, b""])
            self.sock.recv_multipart()
            self.is_connect = True
        except zmq.error.Again:
            logger.error(f"Visage Dynamic sock test connect error!")
            self.is_connect = False

    def _pass_func(self, *args, **kwargs):
        if not self.is_connect:
            self._init_sock()

    def start_dynamic_plot(self, experiment_id, loop_counter: int, dirs=None, title=None):
        self._pass_func()
        if self.is_connect:
            msg = {
                "experiment_id": experiment_id,
                "title": title,
                "loop_counter": loop_counter,
                "dirs": dirs
            }
            msg = json.dumps(msg).encode()
            self.sock.send_multipart([ExecuteOp.monster, ExecuteMonsterOp.dy_start.value, msg])

    def end_dynamic_plot(self, experiment_id, status):
        if self.is_connect:
            msg = {
                "experiment_id": experiment_id,
                "status": status
            }
            msg = json.dumps(msg).encode()
            self.sock.send_multipart([ExecuteOp.monster, ExecuteMonsterOp.dy_end.value, msg])

    def dynamic_loop(self, data):
        if self.is_connect:
            if not isinstance(data, dict):
                print("dynamic loop data type error")
                return
            for param in self.remove_parameters_list:
                if param in data:
                    data.pop(param)
            msg = json.dumps(data).encode()
            self.sock.send_multipart([ExecuteOp.monster, ExecuteMonsterOp.dy_loop.value, msg])


class InterComClient:

    def __init__(self):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.DEALER)
        self.sock.connect(EXECUTE_ADDR)

    def push_msg(self, op: ExecuteInterOp, *msg):
        data = [ExecuteOp.intercom, op]
        data.extend(msg)
        self.sock.send_multipart(data)
