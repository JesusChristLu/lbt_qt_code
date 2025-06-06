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
BaseDag class.
"""

from copy import deepcopy
from typing import List, Dict, Union
from pyQCat.structures import QDict
from pyqcat_visage.exceptions import DagError, DagInvalid
from pyqcat_visage.execute.experiment.exp_map import deal_exp_options
from pyqcat_visage.options import RunOptions


Dict = Union[Dict, QDict]

class BaseDag(RunOptions):
    """Directed Acyclic Graph base class."""

    def __init__(
        self,
        _id: str = None,
        name: str = None,
        official: bool = None,
        adj: Dict = None,
        node_params: Dict = None,
        execute_params: Dict = None,
    ):
        """Initial dag object.

        Args:
            _id (str): The dag id of database save id field.
            name (str): The dag name.
            official (bool): The dag is not official, or custom define.
            adj (dict): Topology structure of the dag.
            node_params (dict): Parameters of dag node to create `Node` object.
            execute_params (dict): Run the dag some parameters.

        """
        self._id = _id
        self.name = name
        self.official = official

        self._adj = adj if isinstance(adj, dict) else {}
        self._degree = {}
        self._in_degree = {}
        self._out_degree = {}
        self._pred = {}
        self._succ = {}
        self._nodes = []
        self._edges = []
        self._root_nodes = None
        self._tail_nodes = None

        self._node_params = node_params if isinstance(node_params, dict) else {}
        self._execute_params = (
            execute_params if isinstance(execute_params, dict) else {}
        )

        self._refresh_args()
        if len(self._nodes) > 0:
            self.validate()

        # Run dag middle operate options
        self._run_options = self._default_run_options()

    def __repr__(self) -> str:
        """Return description."""
        return f"<{self.__class__.__name__} {self.name}>"

    def _refresh_args(self):
        """Refresh dag base args."""
        self._degree.clear()
        self._in_degree.clear()
        self._out_degree.clear()
        self._pred.clear()
        self._succ.clear()
        self._nodes.clear()
        self._edges.clear()

        # update base args
        for key, value in self._adj.items():
            self._nodes.append(key)

            if self._degree.get(key) is None:
                self._degree[key] = 0
            if self._in_degree.get(key) is None:
                self._in_degree[key] = 0
            if self._out_degree.get(key) is None:
                self._out_degree[key] = 0

            if self._pred.get(key) is None:
                self._pred[key] = []
            if self._succ.get(key) is None:
                self._succ[key] = []

            if value and isinstance(value, dict):
                self._out_degree[key] += len(value)
                for sub_key, sub_value in value.items():
                    self._edges.append(f"{key}-{sub_key}")
                    self._degree[key] += 1

                    if self._degree.get(sub_key) is None:
                        self._degree[sub_key] = 1
                    else:
                        self._degree[sub_key] += 1

                    if not self._in_degree.get(sub_key):
                        self._in_degree[sub_key] = 1
                    else:
                        self._in_degree[sub_key] += 1

                    self._succ[key].append(sub_key)

                    if self._pred.get(sub_key) is None:
                        self._pred[sub_key] = [key]
                    else:
                        self._pred[sub_key].append(key)

        # update node_params
        for node in self._nodes:
            if node not in self._node_params.keys():
                self._node_params[node] = {}

        node_params_keys = list(self._node_params.keys())
        for params_key in node_params_keys:
            if params_key not in self._nodes:
                del self._node_params[params_key]

        self._root_nodes = [k for k, v in self._in_degree.items() if v == 0]
        self._tail_nodes = [k for k, v in self._out_degree.items() if v == 0]

    @property
    def id(self) -> str:
        """Return dag id."""
        return self._id

    @property
    def adj(self) -> Dict:
        """Return dag topological structure dict."""
        return self._adj

    @property
    def nodes(self) -> List:
        """Return dag all nodes."""
        return self._nodes

    @property
    def edges(self) -> List:
        """Return dag all edges."""
        return self._edges

    @property
    def root_node(self):
        """Get the root node of the DAG."""
        return self._root_nodes[0]

    @property
    def tail_nodes(self):
        """Get the tail nodes of the DAG."""
        return self._tail_nodes

    def add_node(self, node: str, **kwargs):
        """Add node.

        Args:
            node (str): The node name will add.
            kwargs: The attribute of node.

        """
        if node not in self._adj.keys():
            self._adj[node] = {}
            self._node_params[node] = kwargs
            self._refresh_args()
        else:
            self._node_params[node].update(kwargs)

    def remove_node(self, node: str):
        """Remove node.

        Args:
            node (str): The node name will remove.

        """
        if node in self._nodes:
            new_adj = deepcopy(self._adj)
            for key, value in new_adj.items():
                if key == node:
                    del self._adj[node]
                elif value and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key == node:
                            del self._adj[key][sub_key]
            self._refresh_args()

    def add_edge(self, u_of_edge: str, v_of_edge: str, **kwargs):
        """Add edge.

        Args:
            u_of_edge (str): The front node name of edge.
            v_of_edge (str): The tail node name of edge.
            kwargs: The attribute of edge.

        """
        if u_of_edge not in self._adj.keys():
            self._adj.update({u_of_edge: {v_of_edge: kwargs}})
        else:
            u_dict = self._adj.get(u_of_edge)
            u_dict.update({v_of_edge: kwargs})
        if v_of_edge not in self._adj.keys():
            self._adj.update({v_of_edge: {}})

        self._refresh_args()

    def remove_edge(self, u_of_edge: str, v_of_edge: str):
        """Remove edge.

        Args:
            u_of_edge (str): The front node name of edge.
            v_of_edge (str): The tail node name of edge.

        """
        try:
            del self._adj[u_of_edge][v_of_edge]
        except Exception as err:
            pass
        else:
            self._refresh_args()

    @classmethod
    def from_dict(cls, data: Dict) -> "BaseDag":
        """From dict data create a dag object. data like `dag_history.json`.

        Args:
            data (dict): Create dag data.

        Returns:
            BaseDag: The dag object.
        """

        # def transform_exp(exp):
        #     def trans_options(options):
        #         new_options = {}
        #         for k, v in options.items():
        #             if isinstance(v, list) and isinstance(v[-1], bool):
        #                 new_options[k] = v[0]
        #             elif isinstance(v, Dict):
        #                 if "describe" in v:
        #                     new_options[k] = v["describe"][0]
        #                 else:
        #                     new_options[k] = trans_options(v)
        #             else:
        #                 new_options[k] = v
        #
        #         return new_options
        #
        #     exp = deepcopy(exp)
        #     exp_options = deal_exp_options(
        #         trans_options(exp["exp_params"]["experiment_options"])
        #     )
        #     ana_options = deal_exp_options(
        #         trans_options(exp["exp_params"]["analysis_options"])
        #     )
        #     context_options = deal_exp_options(
        #         trans_options(exp["exp_params"]["context_options"])
        #     )
        #     exp["exp_params"]["experiment_options"] = exp_options
        #     exp["exp_params"]["analysis_options"] = ana_options
        #     exp["exp_params"]["context_options"] = context_options
        #     return exp

        node_params = data.get("node_params")
        # new_node_params = {}
        # for key, exp_data in node_params.items():
        #     new_node_params[key] = transform_exp(exp_data)

        kwargs = {
            "_id": data.get("_id"),
            "name": data.get("dag_name"),
            "official": data.get("official"),
            "adj": data.get("node_edges"),
            "node_params": node_params,
            "execute_params": data.get("execute_params"),
        }
        dag_obj = cls(**kwargs)
        return dag_obj

    def to_dict(self) -> Dict:
        """To dict data.

        Returns:
            dict: The dag object some main information to dict.
        """
        data = {
            "dag_name": self.name,
            "official": self.official,
            "node_edges": self._adj,
            "node_params": self._node_params,
            "execute_params": self._execute_params,
        }
        return data

    def validate(self):
        """Validate dag.

        Raises:
            DagInvalid: Valid the digraph error.

        """
        self.isolate_check()
        self.acyclic_check()

    def successors(self, node: str) -> List:
        """Get successors of the set node.

        Args:
            node (str): The set node name.

        Returns:
            list: Successor node list.

        Raises:
            DagError: If node not exist in the digraph nodes.

        """
        if node in self._nodes:
            successors = self._succ[node]
        else:
            successors = []
        return successors

    def predecessors(self, node: str) -> List:
        """Get predecessors of the set node.

        Args:
            node (str): The set node name.

        Returns:
            list: Predecessor node list.

        Raises:
            DagError: If node not exist in the digraph nodes.

        """
        if node in self._nodes:
            predecessors = self._pred[node]
        else:
            raise DagError(f"The node {node} not in the digraph!")
        return predecessors

    def clear(self):
        """Clear dag."""
        # # clear base information or not
        # self._id = None
        # self.name = None
        # self.official = None

        self._adj.clear()
        self._refresh_args()
        self._run_options = self._default_run_options()

    def bfs(self, start_node: str = None) -> List:
        """Breadth First Search.

        Args:
            start_node (str): Start node name.

        Returns:
            list: This search node result.
        """
        if start_node is None:
            start_node = [k for k, v in self._in_degree.items() if v == 0][0]

        queue = []
        visited = []
        queue.insert(0, start_node)
        visited.append(start_node)

        result = []
        while queue:
            node = queue.pop()
            for sub_node in self._succ[node]:
                if sub_node not in visited:
                    queue.insert(0, sub_node)
                    visited.append(sub_node)
            result.append(node)

        return result

    def dfs(self, start_node: str = None) -> List:
        """Depth First Search.

        Args:
            start_node (str): Start node name.

        Returns:
            list: This search node result.
        """
        if start_node is None:
            start_node = [k for k, v in self._in_degree.items() if v == 0][0]

        stack = []
        visited = []
        stack.append(start_node)
        visited.append(start_node)

        result = []
        while stack:
            node = stack.pop()
            for sub_node in self._succ[node]:
                if sub_node not in visited:
                    stack.append(sub_node)
                    visited.append(sub_node)
            result.append(node)

        return result

    def weight_search(self, start_node: str = None) -> List:
        """Weight First Search.

        Args:
            start_node (str): Start node name.

        Returns:
            list: This search node result.
        """
        if start_node is None:
            start_node = [k for k, v in self._in_degree.items() if v == 0][0]

        queue = []
        visited = []
        queue.insert(0, start_node)
        visited.append(start_node)

        result = []
        while queue:
            node = queue.pop()
            sub_nodes = self._succ[node]
            if len(sub_nodes) == 1:
                sub_node = sub_nodes[0]
                if sub_node not in visited:
                    queue.insert(0, sub_node)
                    visited.append(sub_node)
            else:
                node_edges_dict = self._adj[node]
                weight_list = []
                for sub_node in sub_nodes:
                    edge_dict = node_edges_dict[sub_node]
                    if edge_dict and isinstance(edge_dict, dict):
                        weight = edge_dict.get("weight")
                    else:
                        weight = 0
                    weight_list.append(weight)

                if weight_list:
                    ret_list = list(zip(sub_nodes, weight_list))
                    ret_list.sort(key=lambda x: x[1], reverse=True)
                    for sub_node, weight in ret_list:
                        if sub_node not in visited:
                            queue.insert(0, sub_node)
                            visited.append(sub_node)

            result.append(node)

        return result

    def run(self):
        """Run dag, loop execute nodes logic."""

    def isolate_check(self):
        """Check if the graph has isolated node."""
        zero_degree_nodes = [k for k, v in self._degree.items() if v == 0]
        if len(self._nodes) > 1 and len(zero_degree_nodes) > 0:
            raise DagInvalid(
                graph_name=self.name,
                err_msg=f"There are isolated nodes! "
                f"isolated nodes: {zero_degree_nodes}",
            )

        zero_in_degree_nodes = [k for k, v in self._in_degree.items() if v == 0]
        if len(zero_in_degree_nodes) != 1:
            raise DagInvalid(
                graph_name=self.name,
                err_msg=f"the dag in degree {len(zero_in_degree_nodes)} is non-uniqueness. ",
            )

    def acyclic_check(self):
        """Check if the graph is acyclic."""
        zero_in_degree_nodes = [k for k, v in self._in_degree.items() if v == 0]

        if len(zero_in_degree_nodes) < 1:
            raise DagInvalid(graph_name=self.name, err_msg="No node in_degree is 0!")

        # judge circle
        in_degree_map = deepcopy(self._in_degree)
        while zero_in_degree_nodes:
            node = zero_in_degree_nodes.pop()
            for edge in self._edges:
                u_node, v_node = edge.split("-")
                if u_node == node:
                    in_degree_map[v_node] -= 1
                    if in_degree_map[v_node] == 0:
                        zero_in_degree_nodes.append(v_node)
            del in_degree_map[node]

        if in_degree_map:
            circle_nodes = list(in_degree_map.keys())
            raise DagInvalid(
                graph_name=self.name,
                err_msg=f"The digraph has circle! " f"circle_nodes: {circle_nodes}",
            )

    def dfs_get_paths(
        self,
        start_node: str,
        end_nodes: List[str],
        path: List = None,
        paths: List = None,
    ) -> List[List[str]]:
        """Depth-First Search to get full path.

        Args:
            start_node (str): The node to start search.
            end_nodes (list): The node list to end search.
            path (list): Subpaths obtained by depth-first traversal.
            paths (list): Depth-first traversal of all paths obtained.

        Returns:
            paths: Depth-first traversal of all paths obtained
        """
        #  Store current path and all paths.
        if path is None:
            path = []
        if paths is None:
            paths = []

        # Add the starting node to the current path.
        path.append(start_node)
        if start_node in end_nodes:
            # When the target node is encountered, add the current
            # path to the path list.
            paths.append(path[:])
        else:
            for neighbor in self.successors(start_node):
                # Avoid forming loops.
                if neighbor not in path:
                    # Recursive search.
                    self.dfs_get_paths(neighbor, end_nodes, path, paths)

        # Backtrack to the previous node and pop the current node
        # out of the current path.
        path.pop()

        return paths
