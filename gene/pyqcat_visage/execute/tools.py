# -*- coding: utf-8 -*-

# This code is part of pyQCat Visage.
#
# Copyright (c) 2022-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/09/16
# __author:       SS Fang
"""
execute tools.
"""
import asyncio
import sys
from typing import Dict, Union

from mongoengine import disconnect

from pyQCat.database.ODM import ExperimentDoc
from pyQCat.invoker import DataCenter, Invoker, AsyncDataCenter
from pyQCat.qaio_property import QAIO
from pyQCat.tools import connect_server
from pyQCat.types import RespCode
from pyqcat_visage.execute.log import logger


async def set_experiment_status(
        experiment_id: Union[str, bytes] = None, use_simulator=False
):
    if experiment_id is None or experiment_id in [b"", ""]:
        logger(f"change experiment id not exist {experiment_id}!!!")
        return
    else:
        if isinstance(experiment_id, bytes):
            experiment_id = experiment_id.decode()
    try:
        if not use_simulator:
            await AsyncDataCenter().delete_experiment(task=experiment_id)
        # todo wait all user use multi qstream, cloud remove change mongo status.
        # ExperimentDoc.objects(id=experiment_id).update(status=0)
        logger(f"change experiment {experiment_id} status to 0")
    except Exception as e:
        logger.error(f"Close exp {experiment_id} error, bcause {e}!")


def win_adapter():
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class ExecuteContext:
    def __init__(self, context):
        self.context = context
        self._init()

    def _init(self):
        logger("\n" + str(Invoker.get_env()))
        res = Invoker.load_account()
        if res["code"] != 200:
            logger(f"load account failed! \ncode: {res['code']}\nmsg:{res['msg']}")
        if self.context.config:
            ip = self.context.config.mongo.inst_host
            port = self.context.config.mongo.inst_port
            qaio_type = self.context.config.system.qaio_type
            connect_server(ip, port)
            QAIO.type = qaio_type

    def __enter__(self):
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        disconnect()


def get_run_data(run_type: str, run_id: str, register: bool = True) -> Dict:
    """get job run data by no run data but have run_id or experiment name.

    Parameters
    ----------
    run_type : str
        job type or
    run_id : str
        dag run_id or experiment name.
    register : bool, optional
        the dag job is register, by default True

    Returns
    -------
    Dict
        _description_
    """
    db = DataCenter()
    res = None
    if run_type == "experiment":
        res = db.query_exp_options(run_id)
    elif run_type == "dag":
        if register:
            res = db.query_dag_record(dag_id=run_id)
        else:
            res = db.query_dag_details(run_id)
    if not res or res.get("code") != 200:
        logger.warning(f"get run data failed, {res.get('msg', None)}")
        return None

    res_data = res["data"]
    res_data.update({"_id": run_id})
    return res_data


def put_dag_conf(
        dag_id: str, conf_pre: Dict = None, conf_sub: Dict = None, conf_work: Dict = None
):
    """put dag conf msg to courier before run dag.

    Parameters
    ----------
    dag_id : str
    conf_pre : Dict, optional
    conf_sub : Dict, optional
    conf_work : Dict, optional
    """
    db = DataCenter()
    res = db.set_dag_history_conf(
        dag_id=dag_id, conf_pre=conf_pre, conf_suf=conf_sub, conf_work=conf_work
    )
    if res.get("code") != 200:
        logger(f"dag{dag_id} put config info failed, detail:{res['msg']}")

# def save_dag(data: Dict):
#     """Save a dag data to Data Center DagStore,
#     if exist update, else create.
#
#     Args:
#         data (dict): The dag information data, normal like:
#             {
#                 "dag_name": "xxx",
#                 "official": False,
#                 "node_edges": {},
#                 "node_params": {},
#                 "execute_params": {}
#             }
#
#     Raises:
#         logger.error: If not login, or other.
#
#     """
#     name = data.get("dag_name")
#     node_edges = data.get("node_edges")
#     execute_params = data.get("execute_params")
#     node_params = data.get("node_params")
#
#     db = DataCenter()
#
#     all_dag_names = query_all_dag_names()
#     if name not in all_dag_names:
#         data = db.create_dag(name, node_edges, execute_params, node_params)
#     else:
#         data = db.update_dag(name, node_edges, execute_params, node_params)
#
#     if data and isinstance(data, dict):
#         code_num = data.get("code")
#         if code_num != RespCode.resp_success.value:
#             logger.error(f"Save {name} dag data error: {data.get('msg')}")
#     else:
#         logger.error("Please Login!")


def register_dag(data: Dict) -> str:
    """Register dag to Data Center, when run the dag.

    Args:
        data (dict): The dag information data, normal like:
            {
                "dag_name": "xxx",
                "official": False,
                "node_edges": {},
                "node_params": {},
                "execute_params": {}
            }

    Raises:
        logger.error: If not login, or other.

    """
    name = data.get("dag_name")
    node_edges = data.get("node_edges")
    execute_params = data.get("execute_params")
    node_params = data.get("node_params")
    is_save = data.get("is_save", False)

    db = DataCenter()
    data = db.execute_dag(name, node_edges, execute_params, node_params, is_save)

    register_dag_id = None
    if data and isinstance(data, dict):
        code_num = data.get("code")
        if code_num == RespCode.resp_success.value:
            target_data = data.get("data")
            register_dag_id = target_data.get("dag_id")
        else:
            logger.error(f"Register {name} dag data error: {data.get('msg')}")
    else:
        logger.error("Please Login!")

    return register_dag_id


def put_node_result(data: Dict):
    """Put node run result to Data Center DagHistory.

    Args:
        data (dict): The data dict of node run result. normal like:
            {
                "_id": "xxx",
                "node_id": "xxx",
                "result": {},
                "loop_flag": False or True
            }

    Raises:
        logger.error: If not login, or other.

    """
    _id = data.get("_id")
    node_id = data.get("node_id")
    result = data.get("result")
    loop_flag = data.get("loop_flag")
    # todo, invoker api add loop_flag parameter

    db = DataCenter()
    data = db.put_node_result(_id, node_id, result, loop_flag)

    if data and isinstance(data, dict):
        code_num = data.get("code")
        if code_num != RespCode.resp_success.value:
            logger.error(
                f"Put {_id} dag {node_id} node result " f"data error: {data.get('msg')}"
            )
    else:
        logger.error("Please Login!")
