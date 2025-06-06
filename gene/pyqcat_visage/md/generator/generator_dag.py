# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/20
# __author:       Lang Zhu
# __corporation:  OriginQuantum
"""
Dag Generator.
"""
from typing import List, Union, Dict

from pyqcat_visage.md.generator.format_module import QubitFormat, TitleFormat
from pyqcat_visage.md.generator.generator import BaseGenerator, Generator


class BacktrackGenerator(BaseGenerator):

    def __init__(self,
                 level: int = 1,
                 name: str = None,
                 path: List = None,
                 result: List = None,
                 child_backtrack: Dict = None,
                 father_backtrack: str = None,
                 stats: int = 0,
                 reason: str = None,
                 straight_back_length: int = 1,
                 **kwargs):
        super().__init__(level)
        self.name = str(name)
        self.path = path
        self.node_result = result
        self.child_backtrack = child_backtrack or {}
        self.father_backtrack = father_backtrack
        self.stats = "success" if str(stats) == "1" else "fail"
        self.reason = reason
        self.straight_back_length = straight_back_length
        self.backtrack_img = b""

    def generate_title(self):
        self.title = self._language_("backtrack") + "_{}".format(self.name)
        self.title.title_id = f"backtrack_{self.name}"

        super().generate_title()

    def generate_body(self):
        """generate backtrack body
        It has the following modules:
        instruction table
        child_backtrack child_backtrack
        backtrack execute img
        backtrack path
        
        """
        runtime_start = self.node_result[0].get("create_time")
        runtime_end = self.node_result[-1].get("create_time")

        instruction_table_data = {
            "backtrack_stats":
                self.stats,
            "runtime_start":
                runtime_start,
            "runtime_end":
                runtime_end,
            "reason":
                self.reason,
            "straight_back_length":
                self.straight_back_length,
            "exist_backtrack_inside":
                True if self.child_backtrack else False,
            "parent_backtrack":
                self.father_backtrack
                if self.father_backtrack else "no_parent_backtrack"
        }

        instruction_table_data = {
            self._language_(key): self._language_(value)
            for key, value in instruction_table_data.items()
        }

        self.add_to_md(
            self.tools.inset_table(table_data=instruction_table_data))

        if self.child_backtrack:
            child_table = [[
                self.tools.internal_jump(name,
                                         link_to_id=f"backtrack_{name}",
                                         end_newline=False), start_node
            ] for name, start_node in self.child_backtrack.items()]
            child_table_metre = ["backtrack", "start_node"]
            self.add_to_md(
                self._language_("backtrack_child_details").format(
                    len(self.child_backtrack)))
            self.link_break()
            self.add_to_md(
                self.tools.inset_table(table_data=child_table,
                                       metre=child_table_metre))
            self.link_break()

        if self.backtrack_img:
            self.add_to_md(
                self.tools.title("backtrack_execute_img", self.level + 1))
            img_id = "backtrack" + self.name + runtime_start + ".png"
            self.add_to_source(
                self.tools.inset_img_base64(img_id, self.backtrack_img, "png"))
            self.add_to_md(
                self.tools.inset_img_link(self.name + "backtrack.png", img_id))

        if self.path and self.node_result and len(self.path) == len(
                self.node_result):
            backtrack_path = [
                self.tools.internal_jump(
                    self.path[x], link_to_id=self.node_result[x].get("id"))
                for x in range(len(self.path))
            ]

            self.add_to_md(self.tools.title("path", self.level + 1))
            self.add_to_md(self.tools.list_block(backtrack_path))


class DagGenerator(Generator):
    """visage Dag generator

    """

    def __init__(self, level: int = 1):
        super().__init__(level)
        self.dag_execute_params = {}
        self.dag_execute_img: Dict = {}
        self.bit_params_pre: Dict = None
        self.bit_params_suf: Dict = None
        self.working_volt: list = None
        self.dag_node_map: Dict = None
        self.nodes_generator_list: list[Generator] = []
        self.dag_execute_list: list[Union[dict, TitleFormat]] = []
        self.dag_backtrack: Dict = None

    def generate_title(self):
        title = self._language_("dag_report", title=True)

        if self.title.title:
            title = f"{self.title.title} {title}"
        self.title.title = title

        super().generate_title()

    def generate_body(self):
        """
        dag boby.
        """
        # dag setting params
        self.deal_dag_execute_params()

        # qubits info
        self.deal_qubits_info()

    def generate_result(self):
        # result
        self.add_to_md("\n")
        super().generate_result()

        # traceback details

    def generate_analysis(self):

        # dag execute path img.
        self.deal_dag_execute_img()

        # dag nodes list

        self.deal_dag_node_relationship_map()

        # dag execute report list.
        self.deal_dag_execute_path()

        # dag backtrack details.
        self.deal_dag_backtrack_details()

        # nodes details
        self.deal_nodes_detail_list()

    def deal_dag_execute_params(self):
        """
        add dag execute params to dag report.
        """
        if self.dag_execute_params:
            dag_execute_params_title: str = self.tools.title(
                self._language_("execute_params", True), self.level + 1)
            self.dag_execute_params = {
                self._language_(key):
                    self._language_(str(self.dag_execute_params[key]))
                for key in self.dag_execute_params
            }
            dag_execute_params_body = self.tools.inset_table(
                self.dag_execute_params)

            self.add_to_md(dag_execute_params_title)
            self.add_to_md(dag_execute_params_body)

    def deal_dag_node_relationship_map(self):
        """
        add dag node relationship map to md.
        """
        if self.dag_node_map:
            node_map_title = self.tools.title(self._language_("node_map", True), self.level + 1)
            node_map_table = self.tools.inset_table(table_data=self.dag_node_map)
            self.add_to_md(node_map_title)
            self.add_to_md(node_map_table)

    def deal_qubits_info(self) -> None:
        """
        add qubits params and working dc info add to dag report.
        """
        if self.bit_params_pre or self.bit_params_suf:
            bits_details_table_body = ""
            bits_details_table_title = ""

            if self.bit_params_suf and self.bit_params_pre:
                bits_details_table_title = self.tools.title(
                    self._language_("bit_params"), self.level + 1)
            elif self.bit_params_pre:
                bits_details_table_title = self.tools.title(
                    self._language_("bit_params_before"), self.level + 1)
            else:
                bits_details_table_title = self.tools.title(
                    self._language_("bit_params_before"), self.level + 1)
            if self.bit_params_pre:
                for x in self.bit_params_pre:
                    bits_details_table_body += self.tools.title(
                        self._language_(x), self.level + 2)
                    bits_details_table_body = bits_details_table_body + "\n\n" + QubitFormat(
                        **self.bit_params_pre[x]).md(
                        language=self._language_,
                        new_params=self.bit_params_suf.get(x)) + "\n\n"
            else:
                for x in self.bit_params_pre:
                    bits_details_table_body += self.tools.title(
                        self._language_(x), self.level + 2)
                    bits_details_table_body = bits_details_table_body + "\n\n" + QubitFormat(
                        **self.bit_params_pre[x]).md(
                        language=self._language_) + "\n\n"
            self.add_to_md(bits_details_table_title)
            self.add_to_md(bits_details_table_body)

        if self.working_volt:
            working_volt_title = self.tools.title(
                self._language_("working_dc"), self.level + 1)
            working_volt_body = self.tools.inset_table(
                table_data=self.working_volt,
                metre=[
                    self._language_("bit"),
                    self._language_("value")
                ])

            self.add_to_md(working_volt_title)
            self.add_to_md(working_volt_body)

    def deal_nodes_detail_list(self) -> None:
        """
        add nodes details list to dag report.
        """
        title = self.tools.title(self._language_("node_list"), self.level)
        md_doc_list = []
        source_doc_list = []
        for node_generator in self.nodes_generator_list:
            md_doc_list.append(node_generator.markdown)
            md_doc_list.append("\n\n")
            source_doc_list.append(node_generator._md_resource)
            source_doc_list.append("\n\n")

        self.add_to_md(title, True)
        self.add_to_md("".join(md_doc_list))
        self.add_to_source("".join(source_doc_list))

    def deal_dag_execute_img(self):
        """deal dag execute img add to md and resource.
        """
        if self.dag_execute_img and self.dag_execute_img.get("main",
                                                             None) is not None:
            dag_img_title = self.tools.title(
                self._language_("dag_execute_img"), self.level + 1)
            dag_img_body = self.tools.inset_img_link("dag_execute_img",
                                                     "dag_execute.png", True)
            self.add_to_md(dag_img_title)
            self.add_to_md(dag_img_body)
            self.add_to_source(
                self.tools.inset_img_base64("dag_execute.png",
                                            self.dag_execute_img["main"],
                                            "png"))

    def deal_dag_execute_path(self) -> None:
        """
        show dag execute node path.
        """
        node_list_md = []
        if self.dag_execute_list and isinstance(self.dag_execute_list, list):
            for index, node in enumerate(self.dag_execute_list):
                if not isinstance(node, (dict, TitleFormat)):
                    print("dag execute node type not support, drop it.")
                    continue
                if isinstance(node, dict):
                    node = TitleFormat(**node)
                node_list_md.append(
                    self.tools.internal_jump(click_text=node.title,
                                             link_id=node.title_id,
                                             link_to_id=node.title_jump_id))
        self.add_to_md(
            self.tools.title(self._language_("dag_execute_path", True),
                             self.level + 1))
        self.add_to_md(self.tools.list_block(node_list_md))

    def deal_dag_backtrack_details(self) -> None:
        """
        deal daga backtrack details.
        if dag has backtrack, and show backtrack details, will show all backtrack scene.
        """
        if self.dag_backtrack:
            backtrack_title = self.tools.title(
                self._language_("backtrack_details"), self.level + 1)
            self.add_to_md(backtrack_title, False)
            md_list = []
            resource_list = []
            backtrack_list = []
            for name, value in self.dag_backtrack.items():
                backtrack_generator = BacktrackGenerator(
                    level=self.level + 2, **value.__json_encoder__())
                backtrack_generator.option = self.option
                backtrack_generator.option.separation_img = False
                backtrack_generator.backtrack_img = self.dag_execute_img.get(
                    name, b"")
                backtrack_generator.execute()
                md_list.append(backtrack_generator.markdown)
                resource_list.append(backtrack_generator.resource)
                backtrack_list.append(
                    self.tools.internal_jump(
                        f"backtrack-{backtrack_generator.name} : {self._language_(backtrack_generator.stats)}",
                        link_to_id=backtrack_generator.title.title_id))

            self.add_to_md(self.tools.list_block(backtrack_list))
            self.add_to_md("".join(md_list))
            self.add_to_source("".join(resource_list))
