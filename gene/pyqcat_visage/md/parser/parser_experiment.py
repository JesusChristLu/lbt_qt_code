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
Experiemnt Parser.
"""
import os.path
from typing import AnyStr, ByteString, Dict, List, Union

from pyqcat_visage.md.generator import ExperimentGenerator
from pyqcat_visage.md.parser.parser import Parser, SaveType


class ExperimentParser(Parser):
    """monster  experiment parser.

    Parameters
    ----------
    Parser : _type_
        Parser parent class.
    """

    def __init__(self,
                 id: str = None,
                 source_text: Union[Dict, AnyStr, ByteString] = None,
                 load_type: SaveType = SaveType.local,
                 level: int = 1) -> None:
        super().__init__(id, source_text, load_type, level)
        self.exp_type: str = None
        self.exp_name: str = None
        self.exp_order_id: int = 1
        self.img_resoure_dict = {}

    def default_options(self):
        """experiment default options.
        add options:
        title_jump_id: use to jump to md doc another way.
        exp_params: experiment params
        update_params: experiment update params.
        bits: experiment bits params
        update_bits: experiment update bits params.
        Returns
        -------
        _type_
            _description_
        """
        opt = super().default_options()
        opt.title_jump_id = None
        opt.exp_params = {}
        opt.update_params = {}
        opt.update_bits = {}
        opt.bits = {}
        return opt

    def _special_pretreatment(self):
        """get experiment execute record from courier by invoker.
        """
        if not self.id:
            self.id = self.source_text.get("id")
        res = self.db.query_exp_record(experiment_id=self.id)
        if type(res) == dict and res.get("code", None) == 200:
            if res.get("data", {}) and isinstance(res["data"], dict):
                exp_execute_dict = res["data"]
                exp_execute_dict.pop("create_time")
                exp_execute_dict.pop("exp_id")
                self.source_text.update(exp_execute_dict)

    def parsing(self):
        """parser experiment source text, usually get source text by dag parser.
        parsing process:
        title -> instrument -> params -> qubits -> result -> result img.
        """
        self.exp_type = self.source_text.get("exp_type")
        self.generator = self.create_generator_obj()
        self.generator.option = self.generator_options
        # parser title
        self.generator.title.title = self.exp_name
        self.generator.title.title_id = self.id
        if self.option.title_jump_id:
            self.generator.title.title_jump_id = self.option.title_jump_id

        # parser instrument
        self.generator.instruction = {
            "experimental_status": self.source_text.get("status", None),
            "experiment_name": self.source_text.get("exp_name"),
            "experiment_id": self.id,
            "experiment_type": self.source_text.get("exp_type", None),
            "execution_sequence_number": self.exp_order_id,
        }

        # record params
        # if self.option.show_exp_params:
        self.generator.exp_params = self.option.exp_params
        self.generator.update_params = self.option.update_params
        self.generator.update_bits = self.option.update_bits
        self.generator.bits = self.option.bits

        # parser result
        self.generator.result = self.source_text.get("result")
        if self.source_text.get("extra"):
            self.generator.result = {
                "file_path":
                    self.generator.tools.url_link(
                        self.source_text["extra"].get("file_path"))
            }
            if self.source_text["extra"].get("result"):
                result_img_id = f"{self.id}result.png"
                self.img_resoure_dict.update(
                    {result_img_id: self.source_text["extra"]["result"]})
                self.generator.result_plot = result_img_id
            if self.source_text["extra"].get("schedule"):
                if isinstance(self.source_text["extra"]["schedule"], list):
                    for index, key in enumerate(
                            self.source_text["extra"]["schedule"]):
                        schedule_img_id = f"{self.id}schedule{index}.png"
                        self.img_resoure_dict.update({schedule_img_id: key})
                        self.generator.schedule_plot.append(schedule_img_id)
                elif isinstance(self.source_text["extra"]["schedule"], str):
                    schedule_img_id = f"{self.id}schedule.png"
                    self.img_resoure_dict.update({schedule_img_id: self.source_text["extra"]["schedule"]})
                    self.generator.schedule_plot.append(schedule_img_id)

    def load_response(self):
        """add experiment img resource. usually include result and schedule.
        """
        for x in self.img_resoure_dict:
            # todo change read local file way.
            file_path = self.img_resoure_dict[x]
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    img_body = f.read()
                    self.generator.add_to_source(
                        self.generator.tools.inset_img_base64(
                            x, img_body, "png"))

    def create_generator_obj(self) -> ExperimentGenerator:
        return ExperimentGenerator(level=self.level)


class RabiParser(ExperimentParser):
    pass
