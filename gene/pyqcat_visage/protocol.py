# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/01
# __author:       HanQing Shi
"""File contains some internal config definitions.
"""
from pyQCat.invoker import DEFAULT_PORT
from enum import Enum

EXECUTE_CODE = b"visage_execute"
EXECUTE_HEART = b"execute_heart"
EXECUTE_DYNAMIC = b"dynamic"
TIMEOUT_WRONG = b"timeout"
EXECUTE_ADDR = f"tcp://127.0.0.1:{DEFAULT_PORT + 2}"
EXECUTE_HEART_TIME = 3
EXECUTE_START_TIMEOUT = 30
DYNAMIC_MIN_INTERVAL = 0.1
TASK_ADDR = f"tcp://127.0.0.1:{DEFAULT_PORT + 5}"


class ExecuteOp:
    """
    execute link op.

    need_recv_op: todo
    normal_job: 基本任务的协议，如 Exp，Dag，SweepDag
    stop_experiment: 停止任务的协议
    init_config: 为 visage execute 进程初始化 config 协议
    init_config_ack: 初始化 config 的回复协议
    success: 进程拉起后会将配置 config 发过去，并 refresh 新的实验管理器，如果创建成功会返回此标记
    failed: 进程拉起后会将配置 config 发过去，并 refresh 新的实验管理器，如果创建失败会返回此标记
    state_free: 执行进程会每秒发出一条执行任务的线程状态给到 Visage 的心跳管理器，如果执行线程中没有任务，则返回此标识，它是管理页面中
        Run、Stop 按钮状态的信号
    state_running: 执行进程会每秒发出一条执行任务的线程状态给到 Visage 的心跳管理器，如果执行线程中有正在执行的任务，则返回此标识，它
        是管理页面中 Run、Stop 按钮状态的信号
    heart: 心跳协议，和 state_freq 和 state_running 同时出现
    sync_proc_info: visage execute 进程每隔 10s 会发送一条进程 ID 出去，可用于 Restart 操作 Kill 掉原来的进程
    update_context: 用来跟心 backend 中的 context manager
    update_dag_status: 用来更新 Dag 执行的状态
    dag_context_rollback: 用来进行 Dag 回溯操作
    report_error: 用于传递执行线程中的异常信息
    exit: 页面退出协议，需要等待执行线程结束
    stop_force: 强制重新启动协议
    monster: 与 Monster 通信协议，主要应用在动态绘图的功能上
    intercom: 与Job线程之间的通信协议，主要用于返回异常信息和update record
    cron: Cron 定时任务通信码
    cron_info: Cron 定时任务 Info
    """
    need_recv_op = []
    normal_job = b"execute experiment"
    stop_experiment = b"stop experiment"
    init_config = b"init_config"
    init_config_ack = b"init_config_ack"

    success = b"ok"
    failed = b"fail"

    state_free = b"free"
    state_running = b"running"
    heart = b"heart"
    sync_proc_info = b"proc_info"
    update_context = b"update"
    update_dag_status = b"dag_run"
    dag_context_rollback = b"dag_back"
    report_error = b"error"
    exit = b"exit"
    stop_force = b"stop_force"

    monster = b"monster"
    intercom = b"intercom"
    cron = b"cron"
    cron_info = b"cron_info"


class ExecuteMonsterOp(Enum):
    """
    ExecuteMonsterOp.dy_init: 动态绘图 client 初始化，用户检测是否建立连接，没建立连接会捕获超时报错
    ExecuteMonsterOp.dy_start: 动态绘图开始，发送实验 ID， DIR 等信息
    ExecuteMonsterOp.dy_loop: 动态绘图过程，发送实验回传数据
    ExecuteMonsterOp.dy_end: 动态绘图结束
    """
    dy_start = b"dynamic_start"
    dy_end = b"dynamic_end"
    dy_loop = b"dynamic_loop"
    dy_init = b"dynamic_init"


class ExecuteCronOp(Enum):
    """
    ExecuteCronOp.register_cron: 注册定时任务协议
    ExecuteCronOp.remove_cron: 移除定时任务协议
    ExecuteCronOp.update_cron_job_status: 更新定时任务状态协议
    ExecuteCronOp.init_cron: 初始化定时任务
    """
    register_cron = b"register_cron_job"
    remove_cron = b"remove_cron_job"
    update_cron_job_status = b"update_cron_job_status"
    init_cron = b"init_cron"


class ExecuteInterOp(Enum):
    """
    Execute process inner link op.

    JobThread 与 Visage Execute 之间的通信协议

    ExecuteInterOp.error: JobThread 出现了执行异常
    ExecuteInterOp.error_exp: JobThread 出现了实验执行异常
    ExecuteInterOp.dag: Dag 执行流中的通信协议
    ExecuteInterOp.dag_end: Dag 执行结束协议
    ExecuteInterOp.dag_map: Dag 节点执行状态协议
    ExecuteInterOp.record: Job 更新记录协议
    ExecuteInterOp.record_update: Job 更新记录协议
    ExecuteInterOp.record_rollback: Job 更新回溯协议
    """

    error = b"error"
    error_exp = b"error_exp"

    dag = b"dag"
    dag_end = b"dag_end"
    dag_map = b"dag_map"
    record = b"record"
    record_update = b"record_update"
    record_rollback = b"record_back"


class ExecutorJobType(Enum):
    """
    任务类型

    ExecutorJobType.EXPERIMENT: 纯粹的实验任务
    ExecutorJobType.DAG: Dag 任务
    ExecutorJobType.SWEEP_DAG: 循环 Dag 任务
    """
    EXPERIMENT = "experiment"
    DAG = "dag"
    SWEEP_DAG = "sweep_dag"


class ExecutorJobStatus(Enum):
    """
    任务状态

    ExecutorJobStatus.PENDING: 等待
    ExecutorJobStatus.RUNNING: 正在执行
    ExecutorJobStatus.SUCCESS: 执行成功
    ExecutorJobStatus.FAILED: 执行失败，未启用
    ExecutorJobStatus.ERROR: 执行失败，未启动
    ExecutorJobStatus.REMOVE: 被移除的任务
    ExecutorJobStatus.CLEAR: 未启用
    ExecutorJobStatus.INIT: 初始化任务，未注册
    ExecutorJobStatus.CRON_JOB_EXIST: 定时任务已结束
    """
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"
    STOP = "stop"
    REMOVE = "remove"
    CLEAR = "clear"
    INIT = "init"
    CRON_JOB_EXIST = "cron_job_exist"


class CronType(Enum):
    """
    定时任务类型

    CronType.TIMING: 定时任务
    CronType.INTERVAL: 定时重复任务
    """

    TIMING = "timing"
    INTERVAL = "interval"


class CronStatus(Enum):
    """
    定时任务状态

    CronStatus.STOP: 停止状态
    CronStatus.PROCESS_CLOSE: 进程结束异常终止
    CronStatus.RUN_END: 执行结束
    CronStatus.RUNNING: 正在执行
    """
    STOP = "stop"
    PROCESS_CLOSE = "process_close"
    RUN_END = "end"
    RUNNING = "running"
