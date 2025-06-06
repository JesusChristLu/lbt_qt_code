# -*- coding: utf-8 -*-
import time
# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/8/28
# __author:       Lang Zhu
from typing import Dict
import re


class ShowThreadType:
    ALL = "thread"
    HIGHER = "color"
    THREAD0 = "t0"
    THREAD1 = "t1"
    THREAD2 = "t2"
    THREAD3 = "t3"
    THREAD4 = "t4"


class ProbeStruct:
    """
    probe struct.
    """

    def __init__(
            self, timestamp, scheduler, core_status, core_thread, task_msg=None, **kwargs
    ):
        self.scheduler = scheduler
        self.core_status = core_status
        self.core_thread = self.deal_core_thread(probe_core_thread=core_thread)
        self.timestamp = timestamp
        self._task_msg = task_msg or {}
        self.kwargs = kwargs

    @property
    def task_list(self):
        """
        schduler task list.
        """
        if not self.scheduler.get("task_list", []):
            return []
        return  self.trans_task_list(self.scheduler["task_list"])

    @property
    def color(self):
        """
        chip map color.
        """
        return self.scheduler.get("color", {})

    @property
    def normal_task_list(self):
        """
        normal_task_list_msg
        """
        if not self._task_msg or not self._task_msg.get("normal_task"):
            return []
        return  self.trans_task_list(self._task_msg.get("normal_task"))

    @property
    def normal_task_len(self):
        """
        normal task len, if -1 is not support.
        """
        if not self._task_msg:
            return -1

        return self._task_msg.get("normal_task_len", 0)

    @property
    def vip_task_list(self):
        """
        vip_task list msg
        """
        if not self._task_msg or not self._task_msg.get("vip_task"):
            return []
        return self.trans_task_list(self._task_msg.get("vip_task"))

    @property
    def vip_task_len(self):
        """
        vip task len, if -1 is not support.
        """
        if not self._task_msg:
            return -1

        return self._task_msg.get("vip_task_len", 0)

    @property
    def low_task_list(self):
        """
        low_task_list_msg
        """
        if not self._task_msg or not self._task_msg.get("low_task"):
            return []
        return  self.trans_task_list(self._task_msg.get("low_task"))

    @property
    def low_task_len(self):
        """
        low task len, if -1 is not support.
        """
        if not self._task_msg:
            return -1

        return self._task_msg.get("low_task_len", 0)

    @property
    def wait_task_len(self):
        """
        waiting task len by out of schduler.
        """
        if not self._task_msg:
            return -1
        return self._task_msg.get("wait_task_len", 0)

    @property
    def max_wait_limit(self):
        """
        max wait limit by out of scheduler.
        """
        if not self._task_msg:
            return -1
        return self._task_msg.get("max_task_len", 0)

    def deal_core_thread(self, probe_core_thread: Dict) -> dict:
        core_msg = {}
        for tr_name, tr in probe_core_thread.items():
            if tr.get("status", "ready") != "ready":
                tr["env_bits"] = [self.trans_env_bit(x) for x in tr["env_bits"]]
                tr["use_bits"] = [self.trans_env_bit(x) for x in tr["use_bits"]]
                tr["run_time"]= 0
                tr["expected"] = None

                core_msg[tr_name] = tr
        remove_tasks = []
        for task in self.scheduler["task_list"]:
            for run_task in core_msg.values():
                if task["doc_id"] == run_task["task_id"]:
                    run_task['expected'] = task['expected']
                    remove_tasks.append(task)
        for task in remove_tasks:
            self.scheduler['task_list'].remove(task)
        return core_msg

    @staticmethod
    def trans_env_bit(bit: str):
        if bit.startswith("<"):
            patten = re.compile("<(.+)>")
            bit_str, *_ = patten.findall(bit)[0].split("||")
        else:
            bit_str = bit
        return bit_str

    @staticmethod
    def trans_task_list(task_list):
        if not task_list:
            return []
        task_list = sorted(task_list, key=lambda t: t.get("priority", 0), reverse=True)
        return task_list
        # return [
        #     f"{x.get('exp'):^20} | {x.get('user'):^10} | {round(x.get('priority', 0), 2):^8} | {round(x.get('expected', 0), 2):^8} | {x.get('doc_id'):^20}"
        #     for x in task_list
        # ]
