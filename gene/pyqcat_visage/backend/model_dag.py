# -*- coding: utf-8 -*-

# This code is part of pyQCat Visage.
#
# Copyright (c) 2022-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/22
# __author:       YangChao Zhao

from copy import deepcopy
from typing import Dict, List, Union
from pyQCat.structures import QDict
from pyqcat_visage.backend.experiment import VisageExperiment
from pyqcat_visage.execute.dag.base_dag import BaseDag
from pyqcat_visage.execute.tools import register_dag

Dict = Union[Dict, QDict]


class ModelDag(BaseDag):

    def register(self):
        """Register dag to data center DagHistory."""
        data = self.to_dict()
        dag_history_id = register_dag(data)
        self._id = dag_history_id

    def add_node(self, node: str, exp: "VisageExperiment"):
        """Add node.

        Args:
            node (str): The node name will add.
            exp: The VisageExperiment obj.

        """
        if node not in self._adj.keys():
            self._adj[node] = {}
            self._node_params[node] = exp
            self._refresh_args()
        else:
            self._node_params[node] = exp

    @property
    def node_params(self):
        """Get the node parameters"""
        return self._node_params

    @classmethod
    def from_dict(cls, data: Dict, experiments: "QDict" = None) -> BaseDag:
        def _transform_visage_experiment(param_dict, exps) -> "VisageExperiment":
            experiment_options = param_dict.get("exp_params").get("experiment_options")
            analysis_options = param_dict.get("exp_params").get("analysis_options")
            context_options = param_dict.get("exp_params").get("context_options")
            parallel_options = param_dict.get("exp_params").get("parallel_options")
            exp_name = param_dict.get("exp_name")

            normal_exp = None
            for module_exps in exps.values():
                if exp_name in module_exps:
                    normal_exp = deepcopy(module_exps.get(exp_name))
                    break

            if normal_exp:
                normal_exp.tab = "dag"
                normal_exp.port_pos = param_dict.get("port_pos") or param_dict.get("exp_params").get("port_pos")
                normal_exp.location = param_dict.get("location") or param_dict.get("exp_params").get("location")
                normal_exp.role = param_dict.get("role") or param_dict.get("exp_params").get("role")
                normal_exp.adjust_params = param_dict.get("adjust_params") or param_dict.get("exp_params").get("adjust_params")

                normal_exp.model_exp_options.update(experiment_options)
                normal_exp.model_ana_options.update(analysis_options)
                normal_exp.context_options.update(context_options)
                normal_exp.parallel_options.update(parallel_options)
            else:
                raise ValueError(f"Standard experiment set have not {exp_name}")

            return normal_exp

        flash_node_params = data.get("node_params")
        dag_node_params = {}

        for exp_id, _param_dict in flash_node_params.items():

            if not isinstance(_param_dict, QDict):
                _param_dict = QDict(**_param_dict)

            if experiments:
                visage_experiment = _transform_visage_experiment(_param_dict, experiments)
            else:
                visage_experiment = VisageExperiment.from_dict(_param_dict, "dag")

            visage_experiment.gid = exp_id.split("_")[-1]
            dag_node_params[exp_id] = visage_experiment

        return cls(**{
            "_id": data.get("_id"),
            "name": data.get("dag_name"),
            "official": data.get("official"),
            "adj": data.get("node_edges"),
            "node_params": dag_node_params,
            "execute_params": data.get("execute_params"),
        })

    def to_dict(self, parallel_mode=False) -> Dict:
        """To dict data.

        Returns:
            dict: The dag object some main information to dict.
        """
        dag_node_params = self._node_params
        flash_node_params = {}

        for key, vis_exp in dag_node_params.items():
            flash_node_params[key] = vis_exp.to_flash_dag_dict(parallel_mode)
        data = {
            "parallel_mode": parallel_mode,
            "dag_name": self.name,
            "official": self.official,
            "node_edges": self._adj,
            "node_params": flash_node_params,
            "execute_params": self._execute_params,
        }

        return data

    def to_save_dag(self):
        dag_node_params = self._node_params
        flash_node_params = {}

        for key, vis_exp in dag_node_params.items():
            flash_node_params[key] = vis_exp.to_save_dict()
        data = {
            "dag_name": self.name,
            "official": self.official,
            "node_edges": self._adj,
            "node_params": flash_node_params,
            "execute_params": self._execute_params,
        }

        return data

    def subgraph(self, root_node: str, tail_nodes: List[str]) -> "ModelDag":
        """Get the subgraph with the specified start and end nodes.

        Args:
            root_node (str): Start node of the subgraph.
            tail_nodes (list): End node of the subgraph.

        Returns:
            The new subgraph.
        """
        paths = self.dfs_get_paths(start_node=root_node, end_nodes=tail_nodes)
        new_graph = ModelDag(name=f"{self.name}_sub", official=self.official)
        for path in paths:
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                # Add the nodes in the path to the new graph.
                if source not in new_graph.nodes:
                    new_graph.add_node(source, self._node_params[source])
                if target not in new_graph.nodes:
                    new_graph.add_node(target, self._node_params[target])
                if f"{source}-{target}" not in new_graph.edges:
                    # Add the edges between adjacent nodes in the path to the new graph
                    weight = self._adj[source][target]["weight"]
                    new_graph.add_edge(source, target, weight=weight)
        return new_graph

    def to_run_exp_dict(self, parallel_mode=False):
        exp_params = {}
        for exp in self.node_params.values():
            experiment_options, analysis_options, context_options = exp.get_flash_options(parallel_mode)

            exp_name = exp.name
            index = 1
            while exp_name in exp_params:
                exp_name = f"{exp.name}_{index}"
                index += 1

            exp_params[exp_name] = {
                "experiment_options": experiment_options,
                "analysis_options": analysis_options,
                "context_options": context_options
            }

        return exp_params
