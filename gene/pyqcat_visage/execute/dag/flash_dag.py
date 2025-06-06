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
Main Run Dag class.

Visage backend used to process run dag logic.
"""
import pickle
import time
from typing import Dict

import zmq
from pyQCat.context import ExperimentContext
from pyQCat.invoker import DataCenter
from pyqcat_visage.execute.tools import (put_dag_conf, put_node_result,
                                         register_dag)
from pyqcat_visage.execute.network_client import InterComClient
from pyqcat_visage.options import Options
from pyqcat_visage.protocol import ExecuteInterOp

from ..log import logger
from ..node.flash_node import Node
from .base_dag import BaseDag

# import random

NodeStatus = ["static", "success", "failed", "running", "failed"]


class Dag(BaseDag):
    """Visage backend main Dag class."""
    inter_type: bytes

    # default run dag parameters
    # `search_type` support `bfs`, `dfs`, `weight_search`
    default_execute_params = {
        "is_traceback": False,
        "trackback_self": False,
        "back_depth": 5,
        "node_update": False,
        "start_node": None
    }

    def __init__(self,
                 _id: str = None,
                 name: str = None,
                 official: bool = None,
                 adj: Dict = None,
                 node_params: Dict = None,
                 execute_params: Dict = None):
        super().__init__(_id, name, official, adj, node_params, execute_params)
        self.execute_path: list[str] = []
        self.backtrack_dict: Dict = {}
        self.backtrack_stack: list = []
        self.start_node = None
        self.wait_check_child_stack: list = []
        self.pre_deal_node_stack: list = []
        self.node_map: Dict = {}
        self.intercom: InterComClient = InterComClient()
        self.inter_type: bytes = ExecuteInterOp.dag.value
        self.node_map_msg: dict = {}
        self.current_node = None

    @classmethod
    def _default_run_options(cls) -> Options:
        """Set run dag operate options.

        Options:
            context: An object of context...
            node_map (dict): Map node and `Node` or `Context`.
            back_times (int): Note traceback times.
            execute_nodes (list): Note execute node list.

        """
        options = super()._default_run_options()

        options.context = None
        options.node_map = {}
        options.node_result = {}
        options.execute_nodes = []
        options.traceback_note = []
        options.register = True
        options.simulator = False
        options.simulator_base_path = None

        return options

    def register(self):
        """Register dag to data center DagHistory."""
        data = self.to_dict()
        dag_history_id = register_dag(data)
        self._id = dag_history_id

    def put_node_result(self, node_obj: Node, loop_flag: bool):
        """When run node end, put node result to data center."""
        data = {
            "_id": self._id,
            "node_id": node_obj.id,
            "result": node_obj.result,
            "loop_flag": loop_flag
        }
        put_node_result(data)

    def _update_conf(self, context: ExperimentContext, is_pre=False):
        if not self.id:
            return
        conf_dict = {}
        working_dc = None
        if context.qubit:
            conf_dict.update({context.qubit.name: context.qubit.to_dict()})
        if context.coupler:
            conf_dict.update({context.coupler.name: context.coupler.to_dict()})
        if context.working_dc:
            working_dc = context.working_dc

        if is_pre:
            put_dag_conf(self.id, conf_pre=conf_dict, conf_work=working_dc)
        else:
            put_dag_conf(self.id, conf_sub=conf_dict, conf_work=working_dc)

    def add_wight_to_pred(self):

        def sort_by_maopao(tmp_node_location):
            for node_ in self._pred:
                if self._pred[node_] and len(self._pred[node_]) > 1:
                    for i in range(1, len(self._pred[node_])):
                        for j in range(0, len(self._pred[node_]) - i):
                            if self._pred[node_][j] in tmp_node_location and self._pred[node_][
                                j + 1] in tmp_node_location:
                                if tmp_node_location[self._pred[node_][j]] > tmp_node_location[
                                    self._pred[node_][j + 1]]:
                                    self._pred[node_][j], self._pred[node_][j + 1] = self._pred[node_][j + 1], \
                                                                                     self._pred[node_][j]

        def sort_by_sorted(tmp_node_location):
            for node_ in self._pred:
                if self._pred[node_] and len(self._pred[node_]) > 1:
                    self._pred[node_] = [(x, tmp_node_location.get(x, -1)) for x in self._pred[node_]]
                    self._pred[node_] = sorted(self._pred[node_], key=lambda x: x[1], reverse=False)
                    self._pred[node_] = [x[0] for x in self._pred[node_]]

        node = self.start_node
        temp_dict = {}
        count = 1
        while True:
            temp_dict.update({node: count})
            count += 1
            # get next node
            self.node_map.update({node: {"status": 1}})
            node = next(
                self.recursion_d(node,
                                 status=1,
                                 is_backtrack=False,
                                 backtrack_self=False,
                                 max_backtrack_deep=1))
            if node is None:
                break

        sort_by_sorted(temp_dict)

        self.wait_check_child_stack: list = []
        self.pre_deal_node_stack: list = []
        self.node_map: Dict = {}
        self.execute_path: list[str] = []
        self.backtrack_dict: Dict = {}
        self.backtrack_stack: list = []

    def run(self):
        """
        new context update run.
        """
        # validate dag struct.
        if not self.run_options.register:
            self.validate()
            self.register()

        # update dag context before run.
        # self._update_conf(self.run_options.context, True)

        # prepare dag run.
        self.start_node = self._execute_params.get("start_node") or [
            k for k, v in self._in_degree.items() if v == 0
        ][0]
        # execute options
        max_trackback_deep = self._execute_params.get(
            "back_depth", self.default_execute_params.get("back_depth"))
        is_backtrack = self._execute_params.get(
            "is_traceback",
            self.default_execute_params.get("is_traceback", False))
        trackback_self = self._execute_params.get(
            "trackback_self",
            self.default_execute_params.get("trackback_self"))
        node_upadte = self._execute_params.get(
            "node_update", self.default_execute_params.get("node_update"))

        logger("dag execute options:\n{}: {}\n{}: {}\n{}: {}\n{}: {}\n".format(
            "max_trackback_deep", max_trackback_deep, "is_backtrack",
            is_backtrack, "node_upadte", node_upadte, "register",
            self.run_options.register))
        # while temp params
        node = self.start_node
        self.add_wight_to_pred()
        status = 0
        node_obj = None
        self.update_node_map("start")

        # run
        while True:
            # run node.
            node_obj = self.run_node(node, node_upadte=node_upadte)
            status = node_obj.status
            # get next node
            node = next(
                self.recursion_d(node,
                                 status=status,
                                 is_backtrack=is_backtrack,
                                 backtrack_self=trackback_self,
                                 max_backtrack_deep=max_trackback_deep))

            # update node and exit.
            if node is None or node in [-1, -2]:
                if node is None:
                    logger(f"DAG:{self.name} succ!")
                else:
                    logger(f"DAG:{self.name} failed! flag:{node}")

                logger.debug(f"DAG execute path:\n{self.execute_path}")
                self.put_node_result(node_obj, False)
                # self._update_conf(self.run_options.context)
                self.update_node_map("end")
                return
            else:
                self.put_node_result(node_obj, True)

    def update_node_map(self, status: str = "update"):

        def update_node_map():
            msg = pickle.dumps(self.node_map_msg)
            return [ExecuteInterOp.dag_map.value, msg]

        def start_dag():
            self.node_map_msg = {node: NodeStatus[0] for node in self.adj}
            msg = pickle.dumps(self.node_map_msg)
            return [ExecuteInterOp.dag_map.value, msg]

        def end_dag():
            time.sleep(3)
            return [ExecuteInterOp.dag_end.value, b""]

        update_status = {
            "update": update_node_map,
            "start": start_dag,
            "end": end_dag,
        }
        msg = [self.inter_type] + update_status[status]()
        self.intercom.push_msg(*msg)
        return

    def update_node_records(self, records, status: str = "update"):
        if status == "update":
            op = ExecuteInterOp.record_update.value
        elif status == "rollback":
            op = ExecuteInterOp.record_rollback.value
        self.intercom.push_msg(self.inter_type, op, pickle.dumps(records))

    def get_node_context(self, node_id) -> ExperimentContext:
        return self.run_options.context

    def run_node(self, node_id: str, node_upadte: bool = False):
        # prepare node params
        logger(f"\n{node_id:-^100}\n")
        node_params = self._node_params[node_id]
        node_params.update({"_id": node_id})
        node_execute = Node.from_dict(node_params)
        self.current_node = node_execute

        node_execute.set_run_options(
            context=self.get_node_context(node_id),
            simulator=self.run_options.simulator,
            simulator_base_path=self.run_options.simulator_base_path)
        #  run node
        for key, value in self.node_map.items():
            self.node_map_msg.update({key: NodeStatus[value["status"]]})
        self.node_map_msg.update({node_id: NodeStatus[3]})
        self.update_node_map()
        node_execute.run()
        #  pyqcat_developer random status.
        # if random.randint(1, 10) >= 7:
        #     node_execute._status = 2
        # else:
        #     node_execute._status = 1
        self.execute_path.append(node_id)
        if node_id in self.node_map and self.node_map[node_id]["status"] != 4:
            self.node_map[node_id]["status"] = node_execute.status
            self.node_map[node_id]["record"] = node_execute.ctx_update_record
        else:
            self.node_map.update({
                node_id:
                    dict(status=node_execute.status,
                         deep=0,
                         record=node_execute.ctx_update_record)
            })

        # save  ctx to courier.
        if node_upadte:
            if node_execute.ctx_update_record:
                res = self.run_options.context.extract_hot_data()
                if res:
                    db = DataCenter()
                    for k, v in res.items():
                        ret = db.update_single_config(k, file_data=v)
                        if ret.get("code") == 200:
                            logger.log("UPDATE", f"Update chip data {k}")
                self.run_options.context.update_records.clear()
            else:
                logger(f"node:{node_id} no params update.")

        logger(
            f"\n{node_id:=^100}\n" + \
            f"status:{node_execute.status_dict[node_execute.status]}\nrecord:{node_execute.ctx_update_record}\n" + \
            f"{node_id:=^100}\n")

        self.node_map_msg.update({node_id: NodeStatus[node_execute.status]})
        self.update_node_map()
        self.update_node_records(node_execute.ctx_update_record)

        return node_execute

    def recursion_d(self,
                    node: str,
                    status: int,
                    is_backtrack: bool = False,
                    backtrack_self: bool = False,
                    max_backtrack_deep: int = 3):
        """ recursion
        dfs + weight
        Parameters
        ----------
        node : str
            _description_
        status : int
            _description_
        is_backtrack : bool, optional
            _description_, by default False
        backtrack_self : bool, optional
            _description_, by default False
        max_backtrack_deep: int default 3
        node_edges_map = self._adj
        node_pred = self._pred


        Returns
        -------
        _type_
            _description_
        """

        def clear_stack(top_node: str, stack: list):
            for node in stack[:]:
                if self.has_the_child_node(top_node, node):
                    stack.remove(node)

        def get_node_childs(node_) -> list:
            if node_ in self._adj:
                child_dict = self._adj[node_]
                child_list = []
                for node, line_weight in child_dict.items():
                    child_list.append((node, line_weight["weight"]))
                sorted(child_list, key=lambda x: x[1], reverse=True)
                return [x[0] for x in child_list]
            else:
                return []

        def check_node_depends(node_: str) -> bool:
            depends_nodes = self._pred.get(node_, [])
            for de_node in depends_nodes:
                if not (de_node in self.node_map
                        and self.node_map[de_node]["status"] == 1):
                    return False
            return True

        def normal_next_by_pre_stack():
            for pre_node in reversed(self.pre_deal_node_stack[:]):
                if check_node_depends(pre_node):
                    self.pre_deal_node_stack.remove(pre_node)
                    return pre_node
            return None

        def normal_next(node_):
            if self.backtrack_stack:
                next_node = self.backtrack_stack.pop(0)
                adjust_stack(next_node)
                refrash_node_map(next_node)
                return next_node

            for child_node in get_node_childs(node_):
                if check_node_depends(child_node) and (
                        child_node not in self.node_map
                        or self.node_map[child_node]["status"] != 1
                ) and child_node not in self.pre_deal_node_stack:
                    self.pre_deal_node_stack.append(child_node)

            return normal_next_by_pre_stack()

        def push_trackback_node_stack(fail_node_):
            """depends failed node, push trackback nodes in backtrack stack.
            """
            if backtrack_self:
                if fail_node_ not in self.node_map:
                    self.node_map.update(
                        {fail_node_: dict(status=2, deep=1, record=None)})
                    return fail_node_
                else:
                    if self.node_map[fail_node_]["deep"] >= max_backtrack_deep:
                        return -1
                    elif self.node_map[fail_node_]["status"] == 4:
                        self.node_map[fail_node_]["status"] == 2
                        pass
                    else:
                        self.node_map[fail_node_]["deep"] += 1
                        self.node_map[fail_node_]["status"] = 4
                        return fail_node_
            depends_nodes = self._pred.get(fail_node_, [])
            if not depends_nodes:
                return -1

            # check trackback limit.
            for pre_back_node in depends_nodes:
                if pre_back_node not in self.node_map:
                    return None
                else:
                    if self.node_map[pre_back_node][
                        "deep"] >= max_backtrack_deep:
                        return -1

            self.backtrack_stack += depends_nodes
            for node_key in self.backtrack_stack:
                self.node_map[node_key]["status"] = 0
            if self.backtrack_stack:
                return self.backtrack_stack.pop(0)
            else:
                return None

        def adjust_stack(pre_execute_node: str):
            """
            refrash stack
            """
            # refrash backtrack stack

            clear_stack(pre_execute_node, self.backtrack_stack)
            clear_stack(pre_execute_node, self.pre_deal_node_stack)
            clear_stack(pre_execute_node, self.wait_check_child_stack)

        def refrash_node_map(next_node):
            # refrash_node_map_status
            ctx_rollback_stack = []
            ctx_rollback_stack.append(next_node)
            temp_dfs_stack = []
            temp_node = next_node
            while True:
                if not temp_node and not temp_dfs_stack:
                    break
                if not temp_node:
                    temp_node = temp_dfs_stack.pop(-1)
                temp_dfs_stack += get_node_childs(temp_node)
                if temp_node in self.node_map:
                    if self.node_map[temp_node]["status"] != 2:
                        self.node_map[temp_node]["status"] = -1
                    ctx_rollback_stack.append(temp_node)

                temp_node = None

            #  change deep.
            if next_node in self.node_map:
                self.node_map[next_node]["deep"] += 1
            else:
                self.node_map.update(
                    {next_node: dict(status=status, deep=1, record=None)})

            # refrash node context.
            for _node in reversed(ctx_rollback_stack):
                if self.node_map[_node]["record"] is not None:
                    self.rollback_ctx(self.node_map[_node]["record"])
                self.node_map[_node]["record"] = None
            if ctx_rollback_stack:
                self.put_node_result(ctx_rollback_stack)

        if self.start_node is None:
            return None

        while True:
            if status == 1:
                yield normal_next(node)
            elif status == 2:
                if not is_backtrack:
                    yield -2
                else:
                    next_node = push_trackback_node_stack(node)
                    if not next_node or next_node in [-1]:
                        yield next_node
                    else:
                        adjust_stack(next_node)
                        refrash_node_map(next_node)
                        yield next_node

            elif status == 0:
                if self.execute_path and node != self.execute_path:
                    yield node
                else:
                    yield -1
            elif status == -1:
                yield -1
            else:
                yield -1

    def has_the_child_node(self, node: str, child_node: str) -> bool:
        """the node has the child node? if has return True.

        this contain is not just node child node, need Recursive search the childnode is after the node.

        Parameters
        ----------
        node : str
            _description_
        child_node : str
            _description_

        Returns
        -------
        bool
            _description_
        """
        if node in self._adj:
            if child_node in self._adj[node]:
                return True
            for node_child in self._adj[node]:
                if self.has_the_child_node(node_child, child_node):
                    return True
        else:
            pass
        return False

    def rollback_ctx(self, record: dict):
        """
        rollbck ctx by record, get old value -> new value.
        """
        for record_type, record_details in record.items():
            if record_type == "discriminators":
                for bit_name, record_list in record_details:
                    if isinstance(self.run_options.context.discriminators, list):
                        for dsm in self.run_options.context.discriminators:
                            if dsm.name == bit_name:
                                self.run_options.context.discriminators.remove(dsm)
                                if record_list[0] is not None:
                                    self.run_options.context.discriminators.append(record_list[0])
                                break
                    else:
                        self.run_options.context.discriminators = record_list[0]
            else:
                logger(f"||--rollback ctx--|| {record_type}: {record}")
                self.run_options.context.rollback_ctx(record_type=record_type, record=record_details)

    # def recursion_b(self,
    #                 node: str,
    #                 status: int,
    #                 is_backtrack: bool = False,
    #                 trackback_self: bool = False,
    #                 max_trackback_deep: int = 3):
    #     """ recursion
    #     bfs + weight
    #     Parameters
    #     ----------
    #     node : str
    #         _description_
    #     status : int
    #         _description_
    #     is_backtrack : bool, optional
    #         _description_, by default False
    #     trackback_self : bool, optional
    #         _description_, by default False
    #
    #     node_edges_map = self._adj
    #     node_pred = self._pred
    #
    #     Returns
    #     -------
    #     _type_
    #         _description_
    #     """
    #
    #     def clear_stack(top_node: str, stack: list):
    #         for node in stack[:]:
    #             if self.has_the_child_node(top_node, node):
    #                 stack.remove(node)
    #
    #     def get_node_childs(node_) -> list[str]:
    #         if node_ in self._adj:
    #             child_dict = self._adj[node_]
    #             child_list = []
    #             for node, line_weight in child_dict.items():
    #                 child_list.append((node, line_weight["weight"]))
    #             sorted(child_list, key=lambda x: x[1], reverse=False)
    #             return [x[0] for x in child_list]
    #         else:
    #             return []
    #
    #     def check_node_depends(node_: str) -> True:
    #         depends_nodes = self._pred.get(node_, [])
    #         for de_node in depends_nodes:
    #             if not (de_node in self.node_map
    #                     and self.node_map[de_node]["status"] == 1):
    #                 return False
    #         return True
    #
    #     def normal_next_by_pre_stack():
    #         for pre_node in self.pre_deal_node_stack[:]:
    #             if check_node_depends(pre_node):
    #                 self.pre_deal_node_stack.remove(pre_node)
    #                 return pre_node
    #         return None
    #
    #     def normal_next():
    #         if self.backtrack_stack:
    #             next_node = self.backtrack_stack.pop(0)
    #             adjust_stack(next_node)
    #             refrash_node_map(next_node)
    #             return next_node
    #
    #         next_node = normal_next_by_pre_stack()
    #         if not next_node:
    #             for wait_child_node in self.wait_check_child_stack:
    #                 for child_node in get_node_childs(wait_child_node):
    #                     if child_node not in self.pre_deal_node_stack:
    #                         if child_node not in self.node_map or self.node_map[
    #                             child_node]["status"] != 1:
    #                             self.pre_deal_node_stack.append(child_node)
    #             self.wait_check_child_stack = []
    #             next_node = normal_next_by_pre_stack()
    #         return next_node
    #
    #     def push_trackback_node_stack(fail_node_):
    #         """depends failed node, push trackback nodes in backtrack stack.
    #         """
    #         if trackback_self:
    #             if fail_node_ not in self.node_map:
    #                 self.node_map.update(
    #                     {fail_node_: dict(status=2, deep=1, record=None)})
    #                 return fail_node_
    #             else:
    #                 if self.node_map[fail_node_]["deep"] >= max_trackback_deep:
    #                     return -1
    #                 elif self.node_map[fail_node_]["status"] == 4:
    #                     self.node_map[fail_node_]["status"] == 2
    #                     pass
    #                 else:
    #                     self.node_map[fail_node_]["deep"] += 1
    #                     self.node_map[fail_node_]["status"] = 4
    #                     return fail_node_
    #         depends_nodes = self._pred.get(fail_node_, [])
    #         if not depends_nodes:
    #             return -1
    #
    #         # check trackback limit.
    #         for pre_back_node in depends_nodes:
    #             if pre_back_node not in self.node_map:
    #                 return None
    #             else:
    #                 if self.node_map[pre_back_node][
    #                     "deep"] >= max_trackback_deep:
    #                     return -1
    #
    #         self.backtrack_stack += depends_nodes
    #         if self.backtrack_stack:
    #             return self.backtrack_stack.pop(0)
    #         else:
    #             return None
    #
    #     def adjust_stack(pre_execute_node: str):
    #         """
    #         refrash stack
    #         """
    #         # refrash backtrack stack
    #
    #         clear_stack(pre_execute_node, self.backtrack_stack)
    #         clear_stack(pre_execute_node, self.pre_deal_node_stack)
    #         clear_stack(pre_execute_node, self.wait_check_child_stack)
    #
    #     def refrash_node_map(next_node):
    #         # refrash_node_map_status
    #         ctx_rollback_stack = []
    #         ctx_rollback_stack.append(next_node)
    #         temp_bfs_stack = []
    #         temp_node = next_node
    #         while True:
    #             if not temp_node and not temp_bfs_stack:
    #                 break
    #             if not temp_node:
    #                 temp_node = temp_bfs_stack.pop(-1)
    #
    #             child_node = get_node_childs(temp_node)
    #             temp_node = None
    #             for temp_child_node in child_node:
    #                 if temp_child_node in self.node_map:
    #                     if self.node_map[temp_child_node]["status"] != 2:
    #                         self.node_map[temp_child_node]["status"] = -1
    #                     ctx_rollback_stack.append(temp_child_node)
    #                     temp_bfs_stack.append(temp_child_node)
    #
    #         # old traverse child node which node has exeucted!, if new function use, will remove.
    #         # for node in self.node_map:
    #         #     if self.has_the_child_node(next_node, node):
    #         #         self.node_map[node]["status"] = -1
    #         #         recover node context.
    #
    #         #  change deep.
    #         if next_node in self.node_map:
    #             self.node_map[next_node]["deep"] += 1
    #         else:
    #             self.node_map.update(
    #                 {next_node: dict(status=status, deep=1, record=None)})
    #
    #         # refrash node context.
    #         for _node in reversed(ctx_rollback_stack):
    #             if self.node_map[_node]["record"] is not None:
    #                 self.rollback_ctx(self.node_map[_node]["record"])
    #             self.node_map[_node]["record"] = None
    #
    #     if self.start_node is None:
    #         return None
    #
    #     while True:
    #
    #         if status == 1:
    #             self.wait_check_child_stack.append(node)
    #             yield normal_next()
    #         elif status == 2:
    #             if not is_backtrack:
    #                 yield -1
    #             else:
    #                 next_node = push_trackback_node_stack(node)
    #                 if not next_node or next_node in [-1]:
    #                     yield next_node
    #                 else:
    #                     adjust_stack(next_node)
    #                     refrash_node_map(next_node)
    #                     yield next_node
    #
    #         elif status == 0:
    #             if self.execute_path and node != self.execute_path:
    #                 yield node
    #             else:
    #                 yield -1
    #         elif status == -1:
    #             yield -1
    #         else:
    #             yield -1
