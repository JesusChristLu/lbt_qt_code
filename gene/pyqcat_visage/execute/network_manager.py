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
visage execute process network manager.
"""
import json
import pickle
import time
import os
import zmq
import zmq.asyncio as async_zmq
from mongoengine import disconnect_all
from pyqcat_visage.execute.log import logger
from pyqcat_visage.protocol import (
    EXECUTE_ADDR,
    EXECUTE_HEART,
    EXECUTE_CODE,
    ExecuteOp,
    ExecuteInterOp,
    ExecuteMonsterOp,
)
from pyqcat_visage.structures import ProcInfo
from pyqcat_visage.tool.utilies import kill_old_process
from pyqcat_visage.execute.job_scheduler import JobScheduler
import asyncio
import traceback


class ExecuteScheduler:

    """
    Q1: execute thread no use

    """
    def __init__(self):
        self.execute_thread = None
        self.ctx = async_zmq.Context.instance()
        self.execute_sock = None
        self.job_scheduler = JobScheduler()
        self.execute_flag = False
        self.router_dict = {
            ExecuteOp.monster: self.monster_msg,
            ExecuteOp.intercom: self.check_intercom,
            ExecuteOp.cron: self.cron_msg,
        }

        self.process_pid = os.getpid()

    def _init_sock(self):
        self.execute_sock = self.ctx.socket(zmq.ROUTER)
        self.execute_sock.setsockopt(zmq.IDENTITY, EXECUTE_CODE)
        self.execute_sock.bind(EXECUTE_ADDR)

    def _pre_start(self):
        logger.info("start init execute process")
        port = int(EXECUTE_ADDR.split(":")[-1])
        kill_old_process(port)
        self._init_sock()
        self.execute_flag = True

    def stop(self):
        self.execute_flag = False
        disconnect_all()
        if self.execute_sock:
            self.execute_sock.close()
            self.execute_sock = None
        logger.info("execute process is stop")

    def stop_execute_thread(self, data):
        if data and isinstance(data, dict):
            exp_id = data.get("exp_id")
            use_simulator = data.get("use_simulator", False)
        else:
            exp_id = None
            use_simulator = True
        self.job_scheduler.stop_execute_job(
            job_exp_id=exp_id, use_simulator=use_simulator
        )

    async def check_intercom(self, client, op, msg):
        inner_op, *msg = msg
        if op == ExecuteInterOp.dag.value:
            if inner_op == ExecuteInterOp.dag_map.value:
                self.execute_sock.send_multipart(
                    [EXECUTE_HEART, ExecuteOp.update_dag_status, msg[0]]
                )
            elif inner_op == ExecuteInterOp.record_update.value:
                self.execute_sock.send_multipart(
                    [EXECUTE_HEART, ExecuteOp.update_context, msg[0]]
                )
            elif inner_op == ExecuteInterOp.record_rollback.value:
                self.execute_sock.send_multipart(
                    [EXECUTE_HEART, ExecuteOp.dag_context_rollback, msg[0]]
                )
        elif op == ExecuteInterOp.error.value:
            if inner_op == ExecuteInterOp.error_exp.value:
                self.execute_sock.send_multipart(
                    [EXECUTE_HEART, ExecuteOp.report_error, msg[0]]
                )
        elif op == ExecuteInterOp.record.value:
            if inner_op == ExecuteInterOp.record_update.value:
                self.execute_sock.send_multipart(
                    [EXECUTE_HEART, ExecuteOp.update_context, msg[0]]
                )

    async def diplomat(self, client, op, require_id, data):
        if op == ExecuteOp.stop_force:
            self.execute_sock.send_multipart([EXECUTE_HEART, ExecuteOp.stop_force])
            self.execute_flag = False
            return
        elif op in [ExecuteOp.normal_job, ExecuteOp.stop_experiment, ExecuteOp.exit]:
            if data and data != b"":
                data = data[0]
                data = pickle.loads(data)
            if op == ExecuteOp.normal_job:
                self.job_scheduler.register_normal_job(data.get("job", None))

            elif op == ExecuteOp.exit:
                if self.execute_thread:
                    self.execute_thread.join()
                self.job_scheduler.close_parallel_thread()
                self.execute_flag = False
            elif op == ExecuteOp.stop_experiment and self.job_scheduler.is_running:
                self.stop_execute_thread(data)
        elif op in [ExecuteOp.need_recv_op]:
            pass
        elif op == ExecuteOp.init_config:
            data = data[0]
            try:
                data = pickle.loads(data)
                config = data.get("config", None)
                if config:
                    res = await self.job_scheduler.init_experiment_manager(
                        config=config
                    )
                    if res:
                        self.execute_sock.send_multipart(
                            [
                                EXECUTE_HEART,
                                ExecuteOp.init_config_ack,
                                ExecuteOp.success,
                            ]
                        )
                        asyncio.create_task(self.execute_scheduler())
                        return
                self.execute_sock.send_multipart(
                    [EXECUTE_HEART, ExecuteOp.init_config_ack, ExecuteOp.failed]
                )
            except Exception:
                logger.warning(
                    f"executor init conext failed,details:\n{traceback.format_exc()}"
                )
                self.execute_sock.send_multipart(
                    [EXECUTE_HEART, ExecuteOp.init_config_ack, ExecuteOp.failed]
                )

    async def monster_msg(self, client, op, msg):
        if op == ExecuteMonsterOp.dy_init.value:
            self.execute_sock.send_multipart([client, ExecuteMonsterOp.dy_init.value])
            return
        send_msg = [EXECUTE_HEART, op]
        if isinstance(msg, list):
            send_msg.extend(msg)
        else:
            send_msg.append(msg)
        await self.execute_sock.send_multipart(send_msg)

    async def cron_msg(self, client, op, msg):
        if op in self.job_scheduler.router_map:
            data = msg[1]
            data = pickle.loads(data)
            self.job_scheduler.router_map[op](**data)
        else:
            logger.warning(f"cron scheduler get undefined operation{op}")

        await asyncio.sleep(0.001)

    async def schedule_execute_msg(self):
        while self.execute_flag:
            client, op, secend_op, *msg = await self.execute_sock.recv_multipart()
            if not msg:
                continue
            if op in self.router_dict:
                await self.router_dict[op](client, secend_op, msg)
            else:
                request_id, *data = msg
                await self.diplomat(client, op, request_id, data)

    async def heartbeat(self):
        while self.execute_flag:
            if self.job_scheduler.is_running:
                self.execute_sock.send_multipart(
                    [
                        EXECUTE_HEART,
                        ExecuteOp.heart,
                        str(time.time()).encode(),
                        ExecuteOp.state_running,
                    ]
                )
            else:
                self.execute_sock.send_multipart(
                    [
                        EXECUTE_HEART,
                        ExecuteOp.heart,
                        str(time.time()).encode(),
                        ExecuteOp.state_free,
                    ]
                )

            await asyncio.sleep(1)

    async def execute_scheduler(self):
        while self.execute_flag:
            await self.job_scheduler.scheduler_executing()

    async def sync_proc_info(self):
        while True:
            proc_info = ProcInfo(
                process_id=self.process_pid
            )

            proc_info = json.dumps(proc_info.to_dict()).encode()
            self.execute_sock.send_multipart([EXECUTE_HEART, ExecuteOp.sync_proc_info, EXECUTE_CODE, proc_info])
            await asyncio.sleep(10)

    async def sync_cron_info(self):
        while True:
            cron_map = {}
            if self.job_scheduler.job_execute_thread:
                job = self.job_scheduler.job_execute_thread.job
                if hasattr(job, "cron_id"):
                    cron_map.update({job.cron_id: "running"})
            cron_info = json.dumps(cron_map).encode()
            self.execute_sock.send_multipart([EXECUTE_HEART, ExecuteOp.cron_info, EXECUTE_CODE, cron_info])
            await asyncio.sleep(1)

    async def execute(self):
        logger.info(f"execute process pid:{self.process_pid}")
        try:
            self._pre_start()
            gather_list = [
                asyncio.create_task(self.schedule_execute_msg()),
                asyncio.create_task(self.heartbeat()),
                asyncio.create_task(self.sync_proc_info()),
                asyncio.create_task(self.sync_cron_info())
                # asyncio.create_task(self.execute_scheduler()),
            ]
            logger.info("execute start success")
            await asyncio.gather(*gather_list)
        except:
            logger.error(f"execute error:\n{traceback.format_exc()}")

        finally:
            self.stop()
