# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/18
# __author:       Lang Zhu
# __corporation:  OriginQuantum
"""
Base Parser.
"""
from copy import deepcopy
from typing import AnyStr, ByteString, Dict, Union

from pyqcat_visage.md.generator import DagGenerator
from pyqcat_visage.md.generator.format_module import TitleFormat
from pyqcat_visage.md.parser.parser import Parser, SaveType
from pyqcat_visage.md.parser.parser_experiment import ExperimentParser
from pyqcat_visage.md.parser.plot_dag import Dag


class DagParser(Parser):
    """Dag Parser.
    Parser visage dag record.
    """

    def __init__(self,
                 id: str = None,
                 source_text: Union[Dict, AnyStr, ByteString] = None,
                 load_type: SaveType = SaveType.local,
                 level: int = 1) -> None:
        super().__init__(id, source_text, load_type, level)
        self.dag: Dag = None
        self.node_dict = {}

    def default_options(self):
        opt = super().default_options()
        opt.is_converter = True
        return opt

    def query_source_text(self):
        """get source text from courier by invoker.

        Returns
        -------
        dict
            the dag record dict.
        """
        dag_record_res = self.db.query_dag_record(dag_id=self.id)
        if dag_record_res.get("code", 600) != 200:
            print("get dag record failed., detail:\n", dag_record_res)
            return None
        else:
            return dag_record_res["data"]

    def _special_pretreatment(self):
        if not self.id:
            self.id = self.source_text.get("id")
        if not (self.source_text.get("node_params", None)
                and self.source_text.get("node_edges", None)):
            raise ValueError("the source text format error, can't parser!!!")

        self.node_dict.update(self.source_text["node_params"])
        self.dag = Dag.load_from_parser(
            nodes=self.node_dict,
            edges=self.source_text["node_edges"],
            execute_path=self.source_text.get("execute_nodes", []),
            node_result=self.source_text.get("node_result"))

    def parsing(self):
        """
        parser dag record. create dag generator.
        process:
        title -> envrion show -> instrument -> dag execute params -> qubits show -> execute img and list ->
        show dag backtrack details -> nodes details.
        """
        self.dag.parser()
        self.generator = DagGenerator(level=self.level)
        self.generator.option = self.generator_options

        # env parser
        self.generator.env.id = self.id
        self.generator.env.executor = self.source_text.get("username")
        self.generator.env.runtime_start = self.source_text.get("create_time")
        self.generator.env.runtime_end = self.source_text.get("end_time")

        self.generator.env.version = self.source_text.get("version")
        self.generator.env.sample = self.source_text.get("sample")
        self.generator.env.chiller = self.source_text.get("env_name")
        self.generator.env.file_path = self.source_text.get(
            "file_path")  # todo

        # instruction
        self.generator.instruction = {
            "name": self.source_text.get("dag_name"),
            "description": self.source_text.get("dag_desc"),
            "official": self.source_text.get("official")
        }

        # body
        ## dag execute params
        self.generator.dag_execute_params = self.source_text.get(
            "execute_params")

        ## dag qubits
        self.generator.bit_params_pre = self._get_bits_pre_info()
        self.generator.bit_params_suf = self._get_bits_suf_info()
        self.generator.working_volt = self.source_text.get("conf_work")
        ## dag map
        self.generator.dag_node_map = self.dag.node_mapping

        ## dag execute path
        self.generator.dag_execute_img = self.dag.img

        ## dag backtack details
        self.generator.dag_backtrack = self.dag.g_backtrack

        # result
        self.generator.result = self._get_dag_results()

        # analysis
        self.analysis_nodes_list()

    def _get_bits_pre_info(self) -> Dict:
        """get dag represents the bit pre-execution parameter.

        Returns
        -------
        dict
            bit params maybe more then one.
        """
        return self.source_text.get("conf_pre")

    def _get_bits_suf_info(self) -> Dict:
        """get dag represents the bit after parameter

        Returns
        -------
        dict
            bit params.
        """
        return self.source_text.get("conf_suf")

    def _get_dag_results(self) -> Dict:
        """
        get dag results and translate to generator format.
        todo
        """

        if self.dag.g_backtrack:
            des = "happen_backtrack"
        else:
            des = "no_backtrack"

        result = {"status": self.dag.status, "description": des}
        return result

    def deal_params_change(self, old_params_dict, new_params_dict):
        """deal params change, if new param is different with old params, use `old -> new` replace old dict value.

        Parameters
        ----------
        old_params_dict : dict
            old params dict
        new_params_dict : _type_
            new params dict

        Returns
        -------
        dict
            the params dict, such as:
            {
                "t1" : "100 -> 200"
                "t2" : 200
            }
        """
        # for key in old_params_dict:
        #     old_params_dict[key] = ""
        for key in new_params_dict:
            if isinstance(new_params_dict[key],
                          (list, tuple)) and len(new_params_dict[key]) >= 2:
                new_params_dict[
                    key] = f"{new_params_dict[key][0]} -> {new_params_dict[key][1]}"

        old_params_dict.update(new_params_dict)
        return old_params_dict

    def analysis_nodes_list(self):
        """
        analysis node details list.
        """

        def _recursive_update_dict(old_dict, new_dict):
            for key in new_dict:
                if key in old_dict:
                    if isinstance(new_dict[key], dict) and isinstance(
                            old_dict[key], dict):
                        _recursive_update_dict(old_dict[key], new_dict[key])
                    else:
                        if isinstance(new_dict[key], list):
                            old_dict[key] = new_dict[key][1]
                        else:
                            old_dict[key] = new_dict[key]
                else:
                    if isinstance(new_dict[key], list):
                        old_dict[key] = new_dict[key][1]
                    else:
                        old_dict[key] = new_dict[key]

        node_generator_list = []
        node_counts = {}
        node_params: Dict = deepcopy(self.source_text.get("node_params"))
        qubit_params: Dict = deepcopy(self.source_text.get("conf_pre"))
        for index, node in enumerate(self.dag):
            node_name, node_details = node
            # node counts
            if node_name not in node_counts:
                node_counts.update({node_name: 1})
            else:
                node_counts[node_name] += 1

            # deal node source_text
            id = node_details.pop("exp_id")
            qubits_up = node_details.pop("conf_params")
            exps_up = node_details.pop("exp_params")
            node_details.update({"id": id})

            node_parser = ExperimentParser(source_text=node_details,
                                           level=self.level + 1)
            node_parser.generator_options = self.generator_options
            node_parser.generator_options.show_envrion = False
            node_parser.generator_options.separation_img = False
            if self.generator_options.detail == "simple":
                node_parser.generator_options.is_time_schedule = False
            else:
                node_parser.generator_options.is_time_schedule = True
            # node params
            if node_params.get(node_name):
                if "validator" in node_params[node_name]["exp_params"].get(
                        "experiment_options", {}):
                    node_params[node_name]["exp_params"]["experiment_options"].pop(
                        "validator")
                if "validator" in node_params[node_name]["exp_params"].get(
                        "analysis_options", {}):
                    node_params[node_name]["exp_params"][
                        "analysis_options"].pop("validator")
                node_params[node_name]["exp_params"]
                node_parser.option.exp_params = deepcopy(
                    node_params[node_name]["exp_params"])
                if exps_up:
                    node_parser.option.update_params = exps_up
                    node_params[node_name]["exp_params"]["experiment_options"].update(
                        exps_up.get("experiment_options", {}))
                    node_params[node_name]["exp_params"][
                        "analysis_options"].update(
                        exps_up.get("analysis_options", {}))

            if qubit_params:
                node_parser.option.bits = deepcopy(qubit_params)
                node_parser.option.update_bits = qubits_up
                if qubits_up:
                    _recursive_update_dict(qubit_params, qubits_up)
            # node name
            self.generator.dag_execute_list.append(
                TitleFormat(title=node_name,
                            title_jump_id=id,
                            title_id=f"{node_name}_{index}"))
            node_parser.option.title_jump_id = f"{node_name}_{index}"
            node_name = f"{node_name} - {node_counts[node_name]}"
            node_parser.exp_name = node_name
            node_parser.exp_order_id = index
            node_parser.parser()

            node_generator_list.append(node_parser.generator)

        self.generator.nodes_generator_list = node_generator_list
