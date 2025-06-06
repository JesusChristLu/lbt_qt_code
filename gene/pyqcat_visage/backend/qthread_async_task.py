# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/02
# __author:       HanQing Shi
"""visage async task qthread."""
import asyncio

from PySide6.QtCore import QThread, Signal
from zmq.asyncio import Context, Socket, Poller
from pyqcat_visage.protocol import TASK_ADDR
from pyqcat_visage.backend.async_task import ASYNC_TASK_MAP
from loguru import logger

import zmq
from uuid import uuid4


class TaskQthread(QThread):
    component_query_all = Signal(dict)

    def __init__(self, backend) -> None:
        super().__init__()
        self.backend = backend
        self._ctx = None
        self._sock = None
        self._poll = None
        self._close_flag = True
        self.register_task_map = {}

    def _init_sock(self):
        self._ctx = Context.instance()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.bind(TASK_ADDR)
        self._poll = Poller()
        self._poll.register(self._sock, zmq.POLLIN)

    def release_resource(self):
        if self._poll:
            self._poll.unregister(self._sock)
            del self._poll
            self._poll = None
        if self._sock:
            self._sock.close()
            self._sock = None

        self.register_task_map.clear()

    async def async_job(self, task_info: dict, job_id: str):
        if task_info["async_work"] in ASYNC_TASK_MAP:
            try:
                run_data = await ASYNC_TASK_MAP[task_info["async_work"]](self.backend, *task_info["args"],
                                                                         **task_info["kwargs"])
            except Exception as err:
                import traceback
                msg = traceback.format_exc()
                logger.error(f"async job {task_info['async_work']} error:{str(err)} detail:\n {msg}")
                run_data = {}
        else:
            await asyncio.sleep(0.01)
        if task_info["recall"]:
            for recall in task_info["recall"]:
                if not hasattr(self, recall):
                    break
                if run_data is not None:
                    getattr(self, recall).emit(run_data)
                else:
                    getattr(self, recall).emit()
        self.register_task_map.pop(job_id)

    def do_async_job(self, task_info):
        job_id = str(uuid4())
        job_future = asyncio.create_task(self.async_job(task_info, job_id))
        self.register_task_map.update({job_id: job_future})

    async def async_run(self):

        while self._close_flag:
            res = dict(await self._poll.poll(10))
            if self._sock in res:
                task_info = await self._sock.recv_json()
                if task_info.get("op_code", None) == "register":
                    self.do_async_job(task_info)
        self.release_resource()

    def run(self) -> None:
        self._init_sock()
        asyncio.run(self.async_run())

    def close(self):
        self._close_flag = False
