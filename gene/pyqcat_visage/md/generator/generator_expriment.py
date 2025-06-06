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
Experiment Generator.
"""
from typing import Union, Dict
from pyqcat_visage.md.generator.format_module import QubitFormat
from pyqcat_visage.md.generator.generator import Generator
from pyqcat_visage.md.generator.mdutil import strong_str


class ExperimentGenerator(Generator):
    """Experiment Generator.
    experiment must in monster and the reuslt save with courier.
    if the experiment is composite experiment, will not show the child experiment detail and result.
    
    Parameters
    ----------
    Generator 
    """

    def __init__(self, level: int = 1):
        super().__init__(level)

        self.result_plot = None
        self.schedule_plot = []
        self.exp_params: Dict = {}
        self.update_params: Dict = {}
        self.bits: Dict = {}
        self.update_bits = {}
        self.instruction = {
            "experimental_status": "unknown",
            "experiment_id": "",
            "execution_sequence_number": 1
        }

    def default_options(self):
        options = super().default_options()
        options.set_validator('is_time_schedule', bool)
        options.set_validator('update_params', bool)
        options.set_validator('update_bits', bool)
        options.set_validator('bits_type', ["qubit", "coupler"])
        options.is_time_schedule = False
        options.update_params = True
        options.update_bits = True
        options.bits_type = "qubit"
        if self.level > 1:
            options.separation_img = False
            options.show_envrion = False
        return options

    def generate_title(self):
        if not self.title.title:
            title = self.__class__.__name__
            base_title = "{} {}".format(self._language_("experiment"),
                                        self._language_("report"))
            if title != "ExperimentGenerator":
                title = title.split("Generator")[0] + " " + base_title
            else:
                title = base_title
            self.title.title = title

        return super().generate_title()

    def generate_body(self):
        """
        Experiment generate body.
        """

        self.deal_exp_params()

        self.deal_bits_params()

    def generate_analysis(self):

        self.deal_experiment_img()

    def deal_exp_params(self) -> None:
        """
        deal experiment params. 
        Check exp option and analysis option separately for parameter updates, 
        modify the display dictionary if they are updated, and finally convert the dictionary to md table doc.
        """
        if self.exp_params:
            exp_options = self.exp_params.get("experiment_options")
            analysis_options = self.exp_params.get("analysis_options")
            if not exp_options and not analysis_options:
                return None
            self.add_to_md(
                self.tools.title(self._language_("execute_params", True),
                                 self.level + 1))
            if exp_options:
                self.add_to_md(
                    self.tools.title(self._language_("experiment_options", True),
                                     self.level + 2))
                for x in self.update_params.get("experiment_options", {}):
                    exp_options.update({
                        x:
                            strong_str(
                                f"{exp_options.get(x)} -> {self.update_params['experiment_options'][x]}"
                            )
                    })
                self.add_to_md(
                    self.tools.inset_table(exp_options,
                                           metre=[
                                               self._language_("params"),
                                               self._language_("value")
                                           ]))
            if analysis_options:
                self.add_to_md(
                    self.tools.title(self._language_("analysis_options", True),
                                     self.level + 2))
                for x in self.update_params.get("analysis_options", {}):
                    analysis_options.update({
                        x:
                            strong_str(
                                f"{analysis_options.get(x)} -> {self.update_params['analysis_options'][x]}"
                            )
                    })

                self.add_to_md(
                    self.tools.inset_table(analysis_options,
                                           metre=[
                                               self._language_("params"),
                                               self._language_("value")
                                           ]))

    def deal_bits_params(self) -> None:
        """Check the bit parameter. If a new parameter is found, it will be marked and converted to md table.
        """

        def _recursive_update_dict(old_dict, new_dict):
            for key in new_dict:
                if key in old_dict:
                    if isinstance(new_dict[key], dict) and isinstance(
                            old_dict[key], dict):
                        _recursive_update_dict(old_dict[key], new_dict[key])
                    elif isinstance(new_dict[key], list):
                        old_dict[key] = self.tools.strong_str(
                            f"{old_dict[key]} ->{new_dict[key][1]}")
                else:
                    old_dict[key] = self.tools.strong_str(
                        f"{new_dict[key][0]} ->{new_dict[key][1]}")

        if self.bits:
            self.add_to_md(
                self.tools.title(self._language_("bit_params", True),
                                 self.level + 1))
            for bit_name, bit in self.bits.items():
                if not bit_name.startswith("q") or bit_name.startswith("c"):
                    continue
                if self.update_bits and bit_name in self.update_bits:
                    update_bit = self.update_bits[bit_name]
                    _recursive_update_dict(bit, update_bit)

                bit_table = QubitFormat(**bit).md(
                    trans_func=self.tools.inset_table,
                    language=self._language_)
                self.add_to_md(bit_table, True)

    def deal_experiment_img(self) -> None:
        """
        deal experiment img, depends the option, add result and schedule img to markdown doc.
        """
        if self.result_plot:
            self.add_to_md(
                self.tools.title(self._language_("results_plot"),
                                 self.level + 1))
            self.add_to_md(
                self.tools.inset_img_link("results_plot", self.result_plot))

        if self.option.is_time_schedule and self.schedule_plot and isinstance(
                self.schedule_plot, list):
            self.add_to_md(
                self.tools.title(self._language_("schedule_plot"),
                                 self.level + 1))
            for index, key in enumerate(self.schedule_plot):
                self.add_to_md(
                    self.tools.inset_img_link(f"schedule_plot_{index}", key))
