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
async task scheduler.
"""
import asyncio
import enum
from datetime import datetime
from typing import Dict
from uuid import uuid4

from pyQCat.executor.context_manager import BaseContextManager
from pyQCat.invoker import Invoker, AsyncDataCenter
from pyQCat.parallel.parallel_utils import run_parallel_backend, PARALLEL_CACHE
from pyQCat.structures import QDict
from pyqcat_visage.execute.job_thread import JobThread
from pyqcat_visage.execute.log import logger
from pyqcat_visage.execute.tools import set_experiment_status
from pyqcat_visage.protocol import ExecutorJobStatus, ExecutorJobType, ExecuteCronOp
from pyqcat_visage.structures import Job, BaseJob, CronJob, Cron, NormalJob, CronType
from pyqcat_visage.tool.utilies import kill_process_by_pid


class JobPool:
    """
    execute running job pool.
    """

    def __init__(self):
        self.parallel_thread = None
        self._job_dict: Dict[str, Job] = {}

        self.config = None

    def __bool__(self):
        if self._job_dict:
            return True
        else:
            return False

    def __contains__(self, item):
        if isinstance(item, BaseJob):
            if item.task_id in self._job_dict:
                return True
        return False

    @property
    def job_list(self):
        return list(self._job_dict.values())

    def _sort(self, sort_type="level", order=-1):
        if not self._job_dict:
            return []
        job_list = list(self._job_dict.values())
        if not hasattr(job_list[0], sort_type):
            return job_list
        sorted(job_list, key=lambda x: getattr(x, sort_type))
        return job_list

    def next_job(self):
        job_list = self._sort()
        if job_list:
            job = job_list[0]
            self._job_dict.pop(job.task_id)
            return job
        else:
            return None

    def register_job(self, register_job: NormalJob):
        """
        register job to pool
        """
        if not isinstance(register_job, (NormalJob, CronJob)) or not isinstance(
            register_job.task_type, ExecutorJobType
        ):
            return False

        if isinstance(register_job, CronJob):
            for job in self._job_dict.values():
                if isinstance(job, CronJob) and job.cron_id == job.cron_id:
                    return False

        if not register_job.task_id:
            register_job.task_id = str(uuid4())
        register_job.status = ExecutorJobStatus.PENDING
        if not isinstance(register_job.level, int):
            register_job.level = 0
        self._job_dict.update({register_job.task_id: register_job})

        if (
            (self.parallel_thread is None or not self.parallel_thread.is_alive()) and
            register_job.is_parallel
        ):
            self.parallel_thread = run_parallel_backend(conf_file=self.config)
            logger.log("FLOW", "start parallel manager thread ...")

        return True

    def remove_job(
        self,
        job=None,
        job_id=None,
        cron_id=None,
        job_status: ExecutorJobStatus = ExecutorJobStatus.REMOVE,
    ):
        if job is None:
            if job_id and job_id in self._job_dict:
                job = self._job_dict.pop(job_id)
            if cron_id:
                for temp_cron_job in self._job_dict.values():
                    if (
                        isinstance(temp_cron_job, CronJob)
                        and temp_cron_job.cron_id == cron_id
                    ):
                        self._job_dict.popitem(temp_cron_job)
                        break
        if job:
            job.status = job_status
            # todo

    def clear_pool(self):
        if self:
            job_list = list(self._job_dict.values())
            for job in job_list:
                self.remove_job(job=job)


class CronManager:
    class CronManageType(enum.Enum):
        NEED_REMOVE = 0
        TRIGGER = 1
        PASS = 2

    def __init__(self):
        self._cron_dict: Dict[str, Cron] = {}
        self.db = AsyncDataCenter()

    def __bool__(self):
        return bool(self._cron_dict)

    def register_cron(self, cron_job, *args, **kwargs):
        if isinstance(cron_job, Cron) and cron_job.cron_id not in self._cron_dict:
            self._cron_dict.update({cron_job.cron_id: cron_job})
            logger.info(
                f"cron manager add cron job:{cron_job.cron_id} | {cron_job.cron_type.value} | {cron_job.enable}"
            )
            return True
        return False

    def remove_cron(
        self, cron_id, reason: str = "get remove operation", *args, **kwargs
    ):
        if cron_id in self._cron_dict:
            cron_job = self._cron_dict.pop(cron_id)
            logger.info(
                f"cron manager remove cron job< {cron_job.cron_id} >, reason:{reason}"
            )
        else:
            logger.info(f"cron manager can't find cron {cron_id} in cache")

    def init_cron(self, cron_list, *args, **kwargs):
        for cron in cron_list:
            self.register_cron(cron)

    def cron_trigger(self):
        if self:
            need_run_cron_job = []
            remove_cron = []
            for cron in self._cron_dict.values():
                check_cron_flag = self._cron_is_need_trigger(cron)
                if check_cron_flag == self.CronManageType.NEED_REMOVE:
                    remove_cron.append(cron)
                elif check_cron_flag == self.CronManageType.TRIGGER:
                    need_run_cron_job.append(self.create_cron_job(cron))
                elif check_cron_flag == self.CronManageType.PASS:
                    pass
                else:
                    pass
            for cron in remove_cron:
                self.remove_cron(
                    cron.cron_id, reason="check cron is over, need remove."
                )
        else:
            return []

        return need_run_cron_job

    def _cron_is_need_trigger(self, cron: Cron) -> CronManageType:
        def interval_cron_check_trigger(cron_job: Cron) -> CronManager.CronManageType:
            if cron_job.last_register_time is None:
                cron_job.last_register_time = datetime.now()
                cron_job.executor_count += 1
                return CronManager.CronManageType.TRIGGER
            elif (
                datetime.now() - cron_job.last_register_time
            ).total_seconds() > cron_job.interval:
                cron_job.executor_count += 1
                cron_job.last_register_time = datetime.now()
                return CronManager.CronManageType.TRIGGER
            return CronManager.CronManageType.PASS

        def timing_cron_check_trigger(cron_job) -> CronManager.CronManageType:
            current_time = [datetime.now().hour, datetime.now().minute, datetime.now().second]
            if cron_job.last_register_time:
                corn_job_time = [
                    cron_job.last_register_time.hour,
                    cron_job.last_register_time.minute,
                    cron_job.last_register_time.second,
                ]
                if current_time == corn_job_time:
                    cron_job.last_register_time = datetime.now()
                    return CronManager.CronManageType.PASS
            if current_time in cron_job.timing:
                cron_job.last_register_time = datetime.now()
                return CronManager.CronManageType.TRIGGER
            return CronManager.CronManageType.PASS

        def default_cron_check_trigger(cron_job) -> CronManager.CronManageType:
            return CronManager.CronManageType.PASS

        tigger_check_dict = {
            CronType.TIMING: timing_cron_check_trigger,
            CronType.INTERVAL: interval_cron_check_trigger,
        }
        if not cron.enable:
            return self.CronManageType.PASS
        if cron.repeat != 0 and cron.executor_count >= cron.repeat:
            return self.CronManageType.NEED_REMOVE

        return tigger_check_dict.get(cron.cron_type, default_cron_check_trigger)(cron)

    def create_cron_job(self, cron: Cron) -> Job:
        cron_job = CronJob(
            cron_id=cron.cron_id,
            is_calibration=cron.calibration,
            **cron.job_template.to_all_dict(),
        )
        return cron_job

    async def cron_job_pre(self, job: CronJob):
        # 1. cron job start deal and register to courier
        # 2. check is calibration job, if true, will send qubit lock to courier.
        logger.info(f"corn job:{job.cron_id} start")
        res = await self.db.add_custom_task_his(
            task_data=job.get_task_data(task_status=ExecutorJobStatus.RUNNING),
            bit_data=job.get_bit_data(),
        )
        if res["code"] == 200:
            job.doc_id = res["data"].get("doc_id")
        await asyncio.sleep(0.01)

    async def cron_job_end(self, job: CronJob):
        # 1. cron job end push to courier.
        # 2. if calibration, push to courier unlock resource and update context.
        if job.end_flag:
            logger.warn(f"cron job has running end!")
            return
        job.end_flag = True
        logger.info(f"cron job:{job.cron_id} end")

        await self.db.update_custom_task_his(
            doc_id=job.doc_id,
            task_data=job.get_task_data(task_status=ExecutorJobStatus.SUCCESS),
            bit_data=job.get_bit_data(job.records),
        )

        await asyncio.sleep(0.01)


class JobScheduler:
    def __init__(self):
        self.job_pool = JobPool()
        self.cron_manager = CronManager()
        self.job_execute_thread: JobThread = None
        self._running_status = False
        self._execute_status = False
        self._stop_experiment = None

        self._experiment_manager: BaseContextManager = None
        self.router_map = {
            ExecuteCronOp.init_cron.value: self.cron_manager.init_cron,
            ExecuteCronOp.register_cron.value: self.cron_manager.register_cron,
            ExecuteCronOp.remove_cron.value: self.cron_manager.remove_cron,
            # ExecuteCronOp.update_cron_job_status.value: None,
        }

    @property
    def is_running(self):
        return self._running_status

    @property
    def parallel_thread(self):
        return self.job_pool.parallel_thread

    @property
    def _thread_status(self):
        if not self._execute_status:
            return self._execute_status

        if self.job_execute_thread:
            if self.job_execute_thread.is_alive():
                return self._execute_status
            else:
                if isinstance(self.job_execute_thread.job, CronJob):
                    # await self.cron_manager.cron_job_end(self.job_execute_thread.job)
                    asyncio.create_task(
                        self.cron_manager.cron_job_end(self.job_execute_thread.job)
                    )
                    self._execute_status = False
                self._execute_status = False
        return self._execute_status

    async def init_experiment_manager(self, config):
        res = Invoker.load_account()
        if res["code"] != 200:
            logger.error("execute process init load account failed.")
            return False
        username = res.get("data").get("username", "")
        self._experiment_manager = BaseContextManager(config_data=config)
        self._experiment_manager.username = username
        self.job_pool.config = config
        res = await self.refresh_context()
        return res

    async def refresh_context(self):
        db = AsyncDataCenter()
        chip_data = await db.query_chip_all()
        if chip_data["code"] != 200:
            logger.warning("execute refresh context failed.")
            return False
        else:
            chip_data = chip_data.get("data")
            if chip_data:
                self._experiment_manager.refresh_chip_data(chip_data)
                return True

        return False

    async def _execute_thread(self, job: BaseJob):
        run_job = None
        if isinstance(job, CronJob):
            await self.cron_manager.cron_job_pre(job)
            self._experiment_manager.update_records.clear()
            self._experiment_manager.global_options = QDict(**job.global_context_params)
            if await self.refresh_context():
                job.task_context_manage = self._experiment_manager
                run_job = job
            else:
                logger.warning(
                    f"Cronjob {job.cron_id} | {job.task_id} executor run failed, can't refresh cache."
                )
        elif isinstance(job, NormalJob):
            run_job = Job(**job.to_dict(), task_context_manage=job.task_context_manage)
        if not run_job:
            logger.warning(f"Job:{job.task_id} can't build.")
            return
        self.job_execute_thread = JobThread(run_job)
        self.job_execute_thread.start()
        await asyncio.sleep(0.01)
        if self.job_execute_thread.is_alive():
            self._execute_status = True
            self._running_status = True
        else:
            print("job execute warning")

    async def scheduler_executing(self):
        send_list = self.cron_manager.cron_trigger()
        if send_list:
            for job in send_list:
                if not self.job_pool.register_job(job):
                    self.cron_manager.db.add_custom_task_his(
                        task_data=job.get_task_data(
                            task_status=ExecutorJobStatus.REMOVE
                        ),
                        bit_data=job.get_bit_data(),
                    )
        if self._stop_experiment is not None:
            await self._stop_execute_job()
        if not self._thread_status:
            next_job = self.job_pool.next_job()
            if next_job:
                await self._execute_thread(next_job)
            else:
                self.job_execute_thread = None
                self._running_status = False
                await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(0.5)

    async def _stop_execute_job(self):
        async def check_thread_status(job_thread):
            while True:
                if job_thread.is_alive():
                    await asyncio.sleep(0.05)
                else:
                    return

        gather_list = []
        experiment_id, use_simulator = self._stop_experiment
        if self._thread_status:
            self.job_execute_thread.close(msg="stop job")
            gather_list.append(
                asyncio.create_task(check_thread_status(self.job_execute_thread))
            )
            if len(self._stop_experiment[0]) <= 24:
                gather_list.append(
                    asyncio.create_task(
                        set_experiment_status(experiment_id, use_simulator)
                    )
                )

        self.job_pool.clear_pool()
        await asyncio.gather(*gather_list)
        self._stop_experiment = None

    def stop_execute_job(self, job_exp_id, use_simulator):

        print(job_exp_id, use_simulator)
        if self.job_execute_thread.job and self.job_execute_thread.job.is_parallel:
            self.close_parallel_thread()

        if not self._stop_experiment and self._execute_status:
            self._stop_experiment = [job_exp_id, use_simulator]

    def register_normal_job(self, job):
        return self.job_pool.register_job(job)

    def close_parallel_thread(self):
        if self.parallel_thread and self.parallel_thread.is_alive():
            # check parallel process pool pid list and close
            self.parallel_thread.close()

            for pid in PARALLEL_CACHE.get("pid_list"):
                kill_process_by_pid(pid)

            logger.warning("Close parallel manager thread...")
