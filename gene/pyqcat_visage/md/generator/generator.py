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
Base Generator.
"""
from abc import abstractmethod
from uuid import uuid4

from typing import Union, Dict
from pyqcat_visage.md.util import Options
from pyqcat_visage.md.generator import language, mdutil
from pyqcat_visage.md.generator.format_module import EnvrionFormat, TitleFormat


class BaseGenerator:
    """Base Generator.
    The generator is used to generate markdown documents with pre-processed content in a certain format, 
    and can control the generation mode and processing logic.
    """
    tools = mdutil

    def __init__(self, level: int = 1):
        """generator init.

        Parameters
        ----------
        level : int, optional
            the markdown top title level, by default 1
            
        """
        self._level = level
        self._markdown_doc = ""
        self._md_resource = ""
        self._title: TitleFormat = TitleFormat()
        self._options: Options = self.default_options()

    @staticmethod
    def default_options() -> Options:
        """default options.

        Returns:
            Options: the Options of Generator.
        """
        opt = Options(title_jump_id=None)
        opt.set_validator("language", ["cn", "en"])
        opt.set_validator("separation_img", bool)
        opt.set_validator("detail", ["simple", "normal", "detail"])
        opt.language = "en"
        opt.separation_img = True
        opt.show_envrion = True
        opt.detail = "simple"
        return opt

    @property
    def option(self) -> Options:
        """gengerator option

        Returns
        -------
        Options
        """
        return self._options

    @option.setter
    def option(self, opt: Options):
        if isinstance(opt, Options):
            self._options.update(**opt)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        if isinstance(value, int) and 1 <= value <= 4:
            self._level = value
        else:
            print("level error.")

    @property
    def title(self) -> TitleFormat:
        """gengertor md doc title, the top-title.

        Returns
        -------
        TitleFormat
            the title format instance.
        """
        return self._title

    @title.setter
    def title(self, value):
        if isinstance(value, str):
            self._title.title = value
        elif isinstance(value, dict):
            self._title.load(**value)
        elif isinstance(value, TitleFormat):
            self._title = value

    @property
    def markdown(self) -> str:
        """
        Export the markdown document.
        :return: str
        """
        if self._markdown_doc is None:
            self.execute()
        return self._markdown_doc

    @property
    def resource(self):
        """the generator markdown resource.

        Returns
        -------
        str
            the resource str, usually img base64string.
        """
        return self._md_resource

    def link_break(self):
        """insert space romd doc.
        """
        self._markdown_doc += "\n"

    def _language_(self, origin_str: str, title: bool = False) -> str:
        """language deal.

        Parameters
        ----------
        origin_str : str
            the source str will save in markdown
        format_type : str, optional
            _description_, by default ""

        Returns
        -------
        str
            _description_
        """
        if not isinstance(origin_str, str):
            origin_str = str(origin_str)
        origin_str = origin_str.lower()
        if hasattr(language, origin_str):
            origin_str = getattr(language, origin_str)[self.option.language]

        if title:
            return origin_str.title()
        return origin_str

    def add_to_md(self, doc: str, keep_source: bool = False):
        """
        add doc to self markdown_doc.
        if not doc endswith "\n", will add "\n" to the end.

        Args:
            keep_source: (bool) keep doc source text, default is false.
        """
        if not doc.endswith("\n") and not keep_source:
            doc += "\n"
        self._markdown_doc += doc

    def add_to_source(self, doc: str):
        """add img and other source to source doc.

        Parameters
        ----------
        doc : str
            img base64 string or other source.
        """
        self._md_resource += doc
        self._md_resource += "\n"
        # self._md_resource += "\n"

    def generate_title(self):
        """generator title to md str and add to md doc.
        """

        self.add_to_md(
            self.title.md(trans_func=self.tools.title, level=self.level))
        self.link_break()

    @abstractmethod
    def generate_body(self):
        """generator md doc body.
        """

    def execute(self):
        """Generator execute funtion.
        generator process:
        title -> body -> separation resource.
        """

        self.generate_title()
        self.generate_body()
        if self.option.separation_img:
            self.add_to_md("\n")
            self.add_to_md("\n")
            self.add_to_md(self._md_resource, True)


class Generator(BaseGenerator):
    """
    Generator class.
    Inherits from BaseGenerator and refines the generation process/
    """

    def __init__(self, level: int = 1):
        super().__init__(level)
        self._instruction: Dict = {}
        self._result: Dict = {}
        self.env: EnvrionFormat = EnvrionFormat()

    @property
    def instruction(self):
        return self._instruction

    @instruction.setter
    def instruction(self, value: Dict):
        if isinstance(value, dict):
            self._instruction.update(value)
        elif value is None:
            return
        else:
            print("instruction must dict.")

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value: Dict):
        if isinstance(value, dict):
            self._result.update(value)
        elif value is None:
            return
        else:
            print("result must dict.")

    @classmethod
    def load(cls, generator_dict: Dict, level: int = 1) -> None:
        generator = cls(level)
        generator.title = generator_dict.pop("title", None)
        generator.instruction = generator_dict.pop("instruction", None)
        generator.result = generator_dict.pop("result", None)
        generator._load(**generator_dict)
        return generator

    @abstractmethod
    def _load(self, **kwargs) -> None:
        pass

    def generate_title(self):
        if self._markdown_doc is None:
            self._markdown_doc = ""

        if self.level > 1 and not self._title.title_id:
            self._title.title_id = str(uuid4())

        if self._title.title_jump_id is None and self.option.title_jump_id:
            self._title.title_jump_id = self.option.title_juimp_id

        self.add_to_md(
            self.title.md(trans_func=self.tools.title, level=self.level))
        self.link_break()

    def generate_before_instruction(self):
        """generate before instruction, default show run enviroment.
        """

        if not self.option.show_envrion:
            return

        if not isinstance(self.env, EnvrionFormat):
            if isinstance(self.env, dict):
                env = EnvrionFormat()
                env.load(**self.env)
                self.env = env
            else:
                print("no envrion to show")
                return

        self.add_to_md(
            self.tools.title(self._language_("environment", title=True),
                             self.level + 1))

        self.add_to_md(
            self.env.md(trans_func=self.tools.inset_table,
                        language=self._language_))

    def generate_instruction(self):
        """generator instruction, default genrate md strong line  to show instrument.
        """
        self.add_to_md(
            self.tools.title(self._language_("Instruction", True),
                             self.level + 1))
        for key, value in self.instruction.items():
            self.add_to_md(
                self.tools.strong_line(strong_msg=self._language_(key),
                                       text=self._language_(str(value))))
            self.link_break()

    def generate_result(self):
        """generator result, default genrate md table  to show result.
        """
        self.add_to_md(
            self.tools.title(self._language_("result", True), self.level + 1))
        for key, value in self.result.items():
            self.add_to_md(
                self.tools.strong_line(strong_msg=self._language_(key),
                                       text=self._language_(str(value))))
            self.link_break()

    def execute(self) -> None:
        """
        execute generate.
        the generate order:
        title -> special instruction -> instruction -> body -> result -> analysis.
        """
        self.generate_title()
        self.generate_before_instruction()
        self.generate_instruction()
        self.generate_body()
        self.generate_result()
        self.generate_analysis()
        if self.option.separation_img:
            self.add_to_md("\n")
            self.add_to_md("\n")
            self.add_to_md(self._md_resource, True)

    @abstractmethod
    def generate_analysis(self):
        """generate analysis, default None.
        """
