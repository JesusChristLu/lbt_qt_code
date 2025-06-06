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

import json
from abc import abstractmethod
from typing import AnyStr, ByteString, Dict, Union

from pyQCat.invoker import DataCenter, Invoker
from pyQCat.types import SaveType

from pyqcat_visage.md.converter import Converter
from pyqcat_visage.md.generator import Generator
from pyqcat_visage.md.util import Options


class Parser:
    """
    Parser is used to customize the parsing scheme for the markdown text to be generated.
    It can connect courier and local files, query, load and integrate data, 
    and convert it into the format required by the generator.
    """

    def __init__(
        self,
        id: str = None,
        source_text: Union[Dict, AnyStr, ByteString] = None,
        load_type: SaveType = SaveType.local,
        level: int = 1,
    ) -> None:
        """Parser init.

        Parameters
        ----------
        id : str, optional
            the parser object id, by default None
        source_text : Union[Dict, AnyStr, ByteString], optional
            the parser source text, by default None
        load_type : SaveType, optional
            , by default SaveType.local
        level : int, optional
            the markdown doc Top-level directory level, by default 1
            If the document needs to be embedded in other documents, adjust the title level to the required level, 
            such as level 2 or level 3, and no less than level 4 is recommended.
        """
        self.id = id
        self.source_text = source_text
        self.load_type = load_type
        self.db: DataCenter = None
        self.generator: Generator = None
        self.converter_obj: Converter = None
        self.level = level

        self._options: Options = self.default_options()
        self._generator_options = Generator.default_options()
        self._converter_options = Converter.default_options()

    @staticmethod
    def default_options():
        """parser default options.
        
        is_converter: Whether to create a converter, True create, default false.
        Returns
        -------
        _type_
            _description_
        """
        option = Options()
        option.set_validator("is_converter", bool)
        option.set_validator("load_account", bool)
        option.is_converter = False
        option.load_account = False
        return option

    @property
    def option(self) -> Options:
        return self._options

    @option.setter
    def option(self, opt: Options):
        if isinstance(opt, Options):
            self._options.update(**opt)

    @property
    def converter_options(self) -> Options:
        """converter options

        Returns
        -------
        Options
            _description_
        """
        return self._converter_options

    @converter_options.setter
    def converter_options(self, opt: Union[Options, dict]):
        if isinstance(opt, (Options, dict)):
            self._converter_options.update(**opt)

    @property
    def generator_options(self) -> Options:
        """generator options

        Returns
        -------
        Options
            _description_
        """
        return self._generator_options

    @generator_options.setter
    def generator_options(self, opt: Union[Options, dict]):
        if isinstance(opt, (Options, dict)):
            self._generator_options.update(**opt)

    def init_db(self,
                account_name: str = None,
                account_passwd: str = None,
                account_deal: bool = True):
        if account_deal:
            if account_name is None:
                Invoker.load_account()
            else:
                Invoker.verify_account(account_name, account_passwd)
        self.db = DataCenter()

    def pretreatment_parser_text(self):
        """
        Pretreatment parser text.If both id and parser_text are empty, parser_text pre-preparation cannot be completed.
        The existence of both id and parser text is subject to parser text,and id will be discarded or modified.
        """
        if self.source_text is None:
            if self.id is None:
                raise ValueError("No id or sourece_text and parser.")
            else:
                self.source_text = self.query_source_text()
                if self.source_text is None:
                    raise ValueError(
                        f"can't get sourece text by id: {self.id}.")

        if isinstance(self.source_text, (str, bytes)):
            self.source_text = json.loads(self.source_text)

        if isinstance(self.source_text, dict):
            print("pretreatment_parser_text succ")
        else:
            raise TypeError(
                f"source text {type(self.source_text)}type error, must dict or str and bytes could laods to dict"
            )

    def _special_pretreatment(self):
        """Additional parsing flow in addition to the standard parser text flow.

        Returns
        -------
        None
        """

    @abstractmethod
    def parsing(self):
        """Analytic body function

        Returns
        -------
        None
        """

    @abstractmethod
    def load_response(self):
        """Load resources, more different projects can be customized to load different resources, 
        most of the image resources.
        """

    def execute_generator(self) -> str:
        """Execute the generator, if it exists.

        Returns
        -------
        str
            Returns the generated markdown text.
        """
        if self.generator is not None:
            self.generator.execute()

    def execute_converter(self):
        """By default, the markdown converter will generate md text.
        If you need pdf and html to perform this method,
        """
        self.converter_obj.execute(self.generator.markdown)

    @abstractmethod
    def query_source_text(self):
        """query parser text from courier.

        Returns
        -------
        dict
        """
        return None

    def parser(self) -> None:
        """parser execute function.
        parser process:

        Query and verify the text to be transferred ->; Special audit work ->;
        parsing process ->; Load resources ->; Execution generator ->; conversion
        """
        if self.db is None:
            self.init_db(account_deal=False)

        self.pretreatment_parser_text()
        self._special_pretreatment()
        self.parsing()
        self.load_response()
        self.execute_generator()
        if self.option.is_converter:
            if not self.converter_obj:
                self.converter_obj = Converter()
                self.converter_obj.option = self.converter_options
            self.execute_converter()
