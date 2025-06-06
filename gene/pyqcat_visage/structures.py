# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/10/23
# __author:       Lang Zhu
"""File contains some structures by executor and process.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional
from uuid import uuid4

from pyQCat.executor.context_manager import MonsterContextManager, BaseContextManager
from pyQCat.executor.structures import FakeTaskContext
from pyqcat_visage.protocol import CronType, ExecutorJobType, ExecutorJobStatus


@dataclass
class BaseJob:
    task_type: ExecutorJobType
    run_data: dict
    global_context_params: dict = field(default_factory=dict)
    run_config: dict = field(default_factory=dict)
    level: int = 0
    task_id: str = str(uuid4())

    __annotations__ = {
        "task_type": ExecutorJobType,
        "run_data": dict,
        "global_context_params": dict,
        "run_config": dict,
        "level": int,
        "task_id": str
    }

    def to_dict(self):
        return {
            "task_type": self.task_type,
            "run_data": self.run_data,
            "run_config": self.run_config,
            "level": self.level,
            "task_id": self.task_id,

        }

    def to_all_dict(self):
        return asdict(self)

    @property
    def is_parallel(self):
        if self.task_type == ExecutorJobType.EXPERIMENT:
            return self.run_data.get("exp_params", {}).get("parallel_mode", False)
        elif self.task_type == ExecutorJobType.DAG:
            return self.run_data.get("parallel_mode", False)
        elif self.task_type == ExecutorJobType.SWEEP_DAG:
            if self.run_data.get("sub_type") == ExecutorJobType.DAG.value:
                return self.run_data.get("dag", {}).get("parallel_mode", False)
            elif self.run_data.get("sub_type") == ExecutorJobType.EXPERIMENT.value:
                return self.run_data.get("exp", {}).get("exp_params", {}).get("parallel_mode", False)


@dataclass
class NormalJob(BaseJob):
    task_context_manage: Union[MonsterContextManager, BaseContextManager] = None

    __annotations__ = {
        "task_context_manage": Union[MonsterContextManager, BaseContextManager],
    }


@dataclass
class TemplateJob(BaseJob):
    fake_task_context: dict = field(default_factory=dict)
    global_context_params: dict = field(default_factory=dict)
    task_name: str = None
    task_desc: str = ""
    policy: dict = field(default_factory=dict)
    sub_name: str = ""

    __annotations__ = {
        "fake_task_context": dict,
        "global_context_params": dict,
        "task_name": str,
        "task_desc": str,
        "policy": dict,
        "sub_name": str,
    }


@dataclass
class Job(BaseJob):
    task_context_manage: Union[MonsterContextManager, BaseContextManager] = None
    status: ExecutorJobStatus = ExecutorJobStatus.INIT
    records: dict = field(default_factory=dict)
    start_time: str = None
    end_time: str = None

    __annotations__ = {
        "task_context_manage": Union[MonsterContextManager, BaseContextManager],
        "status": ExecutorJobStatus,
        "records": dict,
        "start_time": str,
        "end_time": str,
    }

    def __hash__(self):
        return hash(self.task_id)


@dataclass
class CronJob(Job, TemplateJob):
    """
    Cron Job
    """
    cron_id: str = None
    end_flag: bool = False
    is_calibration: bool = False
    doc_id: str = None

    __annotations__ = {
        "cron_id": str,
        "end_flag": bool,
        "is_calibration": bool,
        "doc_id": str,
    }

    def get_task_data(self, task_status: ExecutorJobStatus = ExecutorJobStatus.INIT) -> dict:
        task_data = {
            "task_name": self.task_name,
            "task_type": int(self.is_calibration),
            "task_desc": self.task_desc,
            "global_options": self.global_context_params,
            "policy": self.policy,
            "status": task_status.value,
            "sub_type": self.task_type.value,
            "sub_name": self.sub_name,
        }
        return task_data

    def get_bit_data(self, update_records=None) -> dict:
        if self.fake_task_context:
            bit_data = FakeTaskContext(**self.fake_task_context)
        else:
            bit_data = FakeTaskContext(**self.fake_task_context)
        if update_records:
            bit_data.update_records(update_records)
        return bit_data.to_dict()


@dataclass
class Cron:
    """
    Cron
    """
    cron_type: CronType
    cron_id: str = str(uuid4())
    task_descript: str = None
    repeat: int = 0
    interval: int = 0
    timing: list = None
    enable: bool = True
    level: int = 0
    calibration: bool = False
    calibration_msg: dict = None
    last_register_time: Optional[datetime] = None
    last_cron_job: str = None
    executor_count: int = 0
    job_template: TemplateJob = None

    __annotations__ = {
        "cron_type": CronType,
        "cron_id": str,
        "task_descript": str,
        "repeat": int,
        "interval": int,
        "timing": list,
        "enable": bool,
        "level": int,
        "calibration": bool,
        "calibration_msg": dict,
        "last_register_time": Optional[datetime],
        "last_cron_job": str,
        "executor_count": int,
        "job_template": TemplateJob,
    }


@dataclass
class ProcInfo:
    """
    proc info.
    """
    process_id: int
    memery_use: int = 0
    cpu_use: float = 0

    __annotations__ = {
        "process_id": int,
        "memery_use": int,
        "cpu_use": float,
    }

    def to_dict(self):
        return asdict(self)


@dataclass
class VisageMeta:

    username: str
    visage_version: str
    monster_version: str
    chip: dict
    exp_class_name: str = ""
    export_datetime: str = None
    description: str = None

    __annotations__ = {
        "username": str,
        "visage_version": str,
        "monster_version": str,
        "chip": dict,
        "exp_class_name": str,
        "export_datetime": str,
        "description": str,
    }

    def to_dict(self):
        return asdict(self)
