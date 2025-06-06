# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/10
# __author:       Lang Zhu
# __corporation:  OriginQuantum
"""
Dag parser backtrack and plot.
"""

import json
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import matplotlib.path as mpath
import networkx
import networkx as nx
import numpy as np

from pyqcat_visage.md.parser.plot_util import get_ax, plot_to_bytes


class DagTheme(Enum):
    """Dag execute img theme.

    Parameters
    ----------
    Enum : _type_
        _description_
    """
    Light = {
        "background": "write",
        "point": {
            "marker":
                mpath.Path([[-0.8, -0.3], [-0.8, 0.3], [0.8, 0.3], [0.8, -0.3]]),
            "size":
                2000,
            0:
                "#bfbfbf",
            1:
                "#93b87b",
            2:
                "#b20808",
            3:
                "#806e68",
            4:
                "#e98888",
            9:
                "#806eff",
        },
        # solid, dashed, dashdot, dotted
        "line": {
            0: ("#a5a5a5", "solid", 1),
            1: ("#ff392b", "dashdot", 1),
            2: ("#320101", "dashdot", 1),
            3: ("#242f6e", "dashdot", 1),
            4: ("#651510", "dashdot", 1),
        },
        "arrow": {
            0: "arc3, rad=0",
            1: "arc3, rad=0.2",
            2: "arc3, rad=0.4",
            3: "arc3, rad=0.6",
            4: "arc3, rad=0.8",
        }
    }


class DagGPoint:
    """Dag Gragh point.
    This data class  is used to store dag graph node information.
    Returns
    -------
    _type_
        _description_
    """
    __slots__ = [
        "name", "parent", "stats", "child", "deepin", "color", "count"
    ]

    def __init__(self,
                 name: str,
                 parent: Union[str, 'DagGPoint', List] = None,
                 child: Union[str, 'DagGPoint', List] = None,
                 color: str = "#FFFFFF"):
        self.name: str = name
        self.parent: list[str] = self.check_node(parent)
        self.child: list[str] = self.check_node(child)
        self.color: str = color
        self.stats: int = 0
        self.deepin: int = 0
        self.count: int = 0

    def check_node(self, node_info: Union[str, 'DagGPoint',
                                          List]) -> List[str]:
        if isinstance(node_info, str):
            return [node_info]
        elif isinstance(node_info, DagGPoint):
            return [node_info.name]
        elif isinstance(node_info, list):
            res = []
            for x in node_info:
                res += self.check_node(x)
            return res
        else:
            return []

    def __json_encoder__(self):
        return {
            "name": self.name,
            "parent": self.parent,
            "child": self.child,
            "color": self.color,
            "stats": self.stats,
            "deepin": self.deepin,
            "count": self.count,
        }

    def __str__(self):
        return json.dumps(self.__json_encoder__())

    def __repr__(self):
        return self.__str__()


class DagGBacktrack:
    """
    Dag Gragh Backtrack class.
    This data class  is used to store dag backtarck information.
    """

    def __init__(self, name: Union[str, int]):
        self.name = str(name)
        self.path: list = []
        self.result: list = []
        self.stats: bool = False
        self.child_backtrack = {}
        self.father_backtrack: str = ""
        self.reason: str = ""
        self.straight_back_length: int = 1
        self.edges = []

    def add_edge(self, edge: Union[List, Tuple]):
        """add edge to backtrack instrance.

        Parameters
        ----------
        edge : Union[List, Tuple]
            edge info, such as [start_node, end_node, stats, deepin]
        """
        if isinstance(edge, (list, tuple)) and len(edge) >= 3:
            if edge[0] in self.path and edge[1] in self.path:
                self.edges.append(edge)

    def add_path(self, node_name: str, node_detail: Dict):
        """add backtrack path to backtrack instance.

        Parameters
        ----------
        node_name : str
            path node name.
        node_detail : Dict
            node reuslt.
        """
        self.path.append(node_name)
        self.result.append(node_detail)

    def __json_encoder__(self):
        return {
            "name": self.name,
            "path": self.path,
            "result": self.result,
            "child_backtrack": self.child_backtrack,
            "father_backtrack": self.father_backtrack,
            "stats": self.stats,
            "reason": self.reason,
            "straight_back_length": self.straight_back_length,
            "edges": self.edges,
        }

    def __str__(self):
        return json.dumps(self.__json_encoder__())

    def __repr__(self):
        return self.__str__()


class Dag:
    max_deepin = 4
    """
    Dag analysis and plot execute plot class. 
    """

    def __init__(self,
                 nodes: List,
                 pos: Dict = None,
                 edges: Dict = None,
                 execute_path: List[str] = None,
                 node_result: List[Tuple] = None) -> None:
        """dag init.

        Parameters
        ----------
        nodes : List
            the dag graph node list.
        pos : Dict, optional
            the dag graph location pos, by default None
        edges : Dict, optional
            the dag edges, show with dict., by default None
        execute_path : List[str], optional
            DAG diagram execution path, by default None
        node_result : List[Tuple], optional
            DAG graph node execution result, by default None
        """
        self.nodes = nodes
        self.edges = edges
        self.execute_path = execute_path
        self.node_result: list = node_result

        self.g = nx.DiGraph()
        self.deepin = 0
        self.pos = pos

        self.g_edges = []
        self.g_nodes = {}
        self.g_backtrack = {}
        self._theme = DagTheme.Light
        self.status = "success"
        self.img = {}
        self.node_mapping = {}
        self.use_node_mapping = 1
        self.max_node_mapping = 1

    @classmethod
    def load_from_parser(cls,
                         nodes: Dict,
                         edges,
                         execute_path,
                         node_result: Dict = None):
        """create dag by param dict and deal param. useally use to dagparser.

        Parameters
        ----------
        nodes : List
            the dag graph node list.
        edges : Dict, optional
            the dag edges, show with dict., by default None
        execute_path : List[str], optional
            DAG diagram execution path, by default None
        node_result : List[Tuple], optional
            DAG graph node execution result, by default None

        Returns
        -------
        Dag
            the dag instance.

        Raises
        ------
        ValueError
            _description_
        """
        node_list = list(nodes.keys())
        pos = cls._deal_pos(nodes)
        edge_dict = {key: list(edges[key].keys()) for key in edges}
        if node_result is not None:
            node_result = deepcopy(node_result)
        else:
            raise ValueError("no node result.")
        results = []
        for node in execute_path:
            details = node_result[node].pop(0) if len(
                node_result[node]) > 0 else None
            results.append((node, details))
        return cls(node_list, pos, edge_dict, execute_path, results)

    @staticmethod
    def _deal_pos(nodes: Dict) -> Dict:
        """create pos by nodes

        Parameters
        ----------
        nodes : Dict
            _description_

        Returns
        -------
        list
            _description_
        """
        pos = {}
        for node_name, info in nodes.items():
            if not info.get("location", []):
                return {}
            info["location"][1] = - info["location"][1]
            pos.update({node_name: np.array(info["location"])})
        return pos

    @property
    def theme(self):
        """the dag exeucte plot theme.

        Returns
        -------
        str 
            the theme name
        """
        return self._theme.name

    @theme.setter
    def theme(self, value):
        if hasattr(DagTheme, value):
            self._theme = getattr(DagTheme, value)

    def __json_encode__(self):
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "execute_path": self.execute_path,
            "node_result": self.node_result,
            "pos": {key: self.pos[key].tolist()
                    for key in self.pos}
        }

    def __iter__(self):
        for x in self.node_result:
            yield x

    def _g_theme_p_(self, theme_type: str, param: Union[str, int]):
        """the function to get dag theme param.

        Parameters
        ----------
        theme_type : str
            the theme param type such as point, line, arrow etc.
        param : str
            the param which need.

        Returns
        -------
        Union[str, list]
            theme info.
        """
        try:
            return self._theme.value[theme_type][param]
        except:
            return None

    def parser(self):
        """the dag parsing and plot execute img.
        process:
        
        init graph -> parsing dag backtrack and deal node and edge style -> plot execute img  -> plot backtrack img.
        """
        self._init_g()
        self.parse_traceback()

        #     dag main img

        self.img["main"] = self.draw_graph_img(self.g, self.pos, self.g_nodes,
                                               self.g_edges)

        #     deal backtrack
        for key, back in self.g_backtrack.items():
            nodes = {}
            for node in set(back.path):
                point = DagGPoint(name=node)
                if back.child_backtrack and node in back.child_backtrack.values(
                ):
                    point.stats = 4
                else:
                    point.stats = self.g_nodes[node].stats
                nodes.update({node: point})
            # nodes = {node: DagGPoint(name=node, color=self._g_theme_p_("point", 3)) for node in set(back.path)}
            self.img[key] = self.draw_graph_img(self.g, self.pos, nodes,
                                                back.edges)

    def _init_g(self):
        """init networkx graph pos.
        """
        self.g.add_nodes_from(self.nodes)
        if not self.pos:
            self.pos = nx.kamada_kawai_layout(self.g)

    def _get_g_nodes_initial(self):
        """use to parsing dag backtrack init networkx graph nodes.

        Returns
        -------
        dict
        nodes init info dict.
        """
        g_nodes = {}
        for node in self.nodes:
            g_nodes.update({node: DagGPoint(name=node)})

        for node, child_node in self.edges.items():
            if node in g_nodes:
                g_nodes[node].child = child_node

            for parent_node in child_node:
                if parent_node in g_nodes:
                    g_nodes[parent_node].parent.append(node)

        return g_nodes

    def parse_traceback(self):
        """
        Parse dag backtracking and generate dag execution path graph point and edge style functions.
        The first thing you need to do is make sure that the DAG graph is correctly structured 
        (that is, it is indeed directed acyclic graph, no check is done here).
        Traverse the DAG execution path, and judge whether there is backtracking 
        on the path according to the graph structure of dag. If the backtracking is recorded, 
        draw the edge of the execution path and draw the color for the node during the traversal.
        """

        def add_edge_to_backtrack(edge, backstack: List, backtrack_dict_):
            for back_block in backstack:
                back_block_id = back_block[0]
                backtrack_dict_[back_block_id].add_edge(edge)

        def add_path_to_backtrack(path_name, path_detail, backstack: List,
                                  backtrack_dict_):
            # for key in backstack:
            #     key = key[0]
            #     backtrack_dict_[key].add_path(path_name, path_detail)
            backtrack_dict_[backtrack_stack[-1][0]].add_path(
                path_name, path_detail)

        def add_child_backtrack(child_name, child_first_node: str,
                                backstack: List, backtrack_dict_):
            for key in backstack[:-1]:
                key = key[0]
                backtrack_dict_[key].child_backtrack.update(
                    {child_name: child_first_node})

        self.g_nodes = self._get_g_nodes_initial()
        self.g_edges = []
        last_node: str = None
        last_nodes = None
        backtrack_stack = []
        backtrack_dict: Dict = {}
        max_bk_node = 0

        for nodes in self:
            node, node_detail = nodes
            if node not in self.g_nodes:
                print("dag execute path get error node")
                continue
            if last_node is None:
                self.g_nodes[node].stats = 1
                self.g_nodes[node].count += 1
                last_node = node
                last_nodes = nodes
                continue

            if node in self.g_nodes[last_node].child:
                # success edge
                # deal node
                if self.g_nodes[node].stats <= 1:
                    self.g_nodes[node].stats = 1
                else:
                    self.g_nodes[node].stats = 3
                # deal edges
                edge = [last_node, node, 0, 0]
                if edge not in self.g_edges:
                    self.g_edges.append(edge)

                # deal backtrack
                if backtrack_stack:
                    add_path_to_backtrack(node, node_detail, backtrack_stack,
                                          backtrack_dict)
                    add_edge_to_backtrack(edge,
                                          backtrack_stack,
                                          backtrack_dict_=backtrack_dict)
                    if node == backtrack_stack[-1][1]:
                        backtrack_dict[backtrack_stack[-1][0]].stats = 1
                        backtrack_stack.pop(-1)

            elif node in self.g_nodes[last_node].parent or node == last_node:
                # traceback
                # deal node
                self.g_nodes[node].stats = 2
                self.g_nodes[node].deepin += 1

                # deal backtrack
                if not backtrack_stack:
                    # no backtrack add new
                    max_bk_node += 1
                    backtrack_stack.append([max_bk_node, last_node])
                    backtrack_dict.update({
                        backtrack_stack[-1][0]:
                            DagGBacktrack(name=backtrack_stack[-1][0])
                    })
                    backtrack_dict[backtrack_stack[-1][0]].add_path(
                        *last_nodes)
                    backtrack_dict[backtrack_stack[-1][0]].add_path(*nodes)
                elif self.g_nodes[last_node].stats in [2]:
                    # exist backtrack, and straight back
                    backtrack_dict[backtrack_stack[-1][0]].add_path(*nodes)
                elif self.g_nodes[last_node].stats in [1, 3]:
                    # exist backtrack, and happen child backtrack
                    max_bk_node += 1
                    backtrack_stack.append([max_bk_node, last_node])
                    backtrack_dict.update({
                        backtrack_stack[-1][0]:
                            DagGBacktrack(name=backtrack_stack[-1][0])
                    })
                    backtrack_dict[backtrack_stack[-1][0]].add_path(
                        *last_nodes)
                    backtrack_dict[backtrack_stack[-1][0]].add_path(*nodes)
                    backtrack_dict[backtrack_stack[-1]
                    [0]].father_backtrack = backtrack_dict[
                        backtrack_stack[-2][0]].name
                    add_child_backtrack(
                        backtrack_dict[backtrack_stack[-1][0]].name, last_node,
                        backtrack_stack, backtrack_dict)
                else:
                    print("warning node backtrack")

                # deal edge
                self.g_edges.append([
                    last_node, node, self.g_nodes[node].deepin,
                    backtrack_stack[-1][0]
                ])
                backtrack_dict[backtrack_stack[-1][0]].add_edge(
                    [last_node, node, 1, backtrack_stack[-1][0]])
            # elif len(
            #         set(self.g_nodes[node].parent)
            #         & set(self.g_nodes[last_node].parent)) != 0:
            else:
                # near node in the same child node.
                # deal node
                self.g_nodes[node].stats = self.g_nodes[last_node].stats

                # deal edges
                edge = [last_node, node, 0, 0]

                if self.g_nodes[node].stats in [2]:
                    # node
                    self.g_nodes[node].deepin += 1

                edge2 = deepcopy(edge)
                if backtrack_stack and self.g_nodes[node].stats not in [2]:
                    edge2[3] = 1
                add_edge_to_backtrack(edge2, backtrack_stack, backtrack_dict)
                # backtrack
                if backtrack_stack:
                    add_path_to_backtrack(node, node_detail, backtrack_stack,
                                          backtrack_dict)
                    edge = [
                        last_node, node, self.g_nodes[node].deepin,
                        backtrack_stack[-1][0]
                    ]
                    if node == backtrack_stack[-1][1]:
                        backtrack_dict[backtrack_stack[-1][0]].stats = 1
                        backtrack_stack.pop(-1)

                if edge not in self.g_edges:
                    self.g_edges.append(edge)
            # else:
            #     print("warning node")
            #     self.g_nodes[node].stats = 9
            self.g_nodes[node].count += 1
            last_node = node
            last_nodes = nodes

        self.g_backtrack = backtrack_dict

    def draw_graph_img(self,
                       graph: networkx.Graph,
                       pos: Dict,
                       nodes: List[DagGPoint],
                       edges: List[Tuple],
                       ax=None,
                       img_type: str = "png") -> bytes:
        """draw graph img

        Parameters
        ----------
        graph : networkx.Graph
            the networkx graph instrance, the graph which need plot.
        pos : Dict
            the networkx pos.
        nodes : List[DagGPoint]
            need plot nodes info .
        edges : List[Tuple]
            need redraw edges.
        ax : _type_, optional
            matplotlib Axes, by default None, will create by default with `get_ax` funx.
        img_type : str, optional
            the save img type, by default "png"

        Returns
        -------
        bytes
            the graph img bytes.
        """
        # deal ax
        if not ax:
            ax = get_ax((6, 3))

        # deal node size and node marker and node label
        node_label = {node: node for node in graph.nodes}
        if len(self.nodes) >= self.max_node_mapping and self.use_node_mapping:
            node_size = 300
            node_shape = "o"
            for index, node in enumerate(node_label):
                node_label[node] = index
            self.node_mapping = node_label
        else:
            node_size = self._g_theme_p_("point", "size")
            node_shape = self._g_theme_p_("point", "marker")
            for node in node_label:
                node_label[node] = node_label[node].split("Node")[0]

        # draw nodes
        nx.draw(graph,
                pos,
                with_labels=False,
                ax=ax,
                node_size=node_size,
                node_shape=node_shape,
                node_color=self._g_theme_p_("point", 0))
        for x in nodes:
            params = dict(
                pos=pos,
                nodelist=[x],
                node_size=node_size,
                node_shape=node_shape,
                # node_shape = self._g_theme_p_("point", "marker"),
                # node_shape=f"${x}$",
                node_color=self._g_theme_p_("point", nodes[x].stats),
                alpha=0.9,
                ax=ax)
            nx.draw_networkx_nodes(graph, **params)

        # draw edges
        for edge in edges:
            if edge[2] > 1:
                style_ = 1
            else:
                style_ = edge[2]
            # line_style = self._g_theme_p_("line", edge[2])
            # connectionstyle = self._g_theme_p_("arrow", edge[2])
            line_style = self._g_theme_p_("line", style_)
            connectionstyle = self._g_theme_p_("arrow", style_)
            params = dict(
                pos=pos,
                edgelist=[tuple(edge[:2])],
                width=1,
                edge_color=line_style[0],
                style=line_style[1],
                alpha=line_style[2],
                arrowsize=10,
                connectionstyle=connectionstyle,
                # node_size=self._g_theme_p_("point", "size"),
                node_size=node_size,
                label=str(edge[3]),
                # node_shape=self._g_theme_p_("point", "marker"),
                arrows="-|>",
                ax=ax)
            nx.draw_networkx_edges(graph, **params)

        # draw labels

        nx.draw_networkx_labels(graph,
                                pos=pos,
                                labels=node_label,
                                font_size=4,
                                font_color="k",
                                bbox=None,
                                ax=ax)

        return plot_to_bytes(ax.get_figure(), img_type)
