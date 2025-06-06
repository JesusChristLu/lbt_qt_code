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
Format module.
"""
from typing import Union, Dict
from abc import abstractmethod
from pyqcat_visage.md.generator.mdutil import title, internal_jump, list_block, inset_table, strong_str
from copy import deepcopy


def format_default_func(*args, **kwargs):
    print(f"format_default_func:\nargs:{args}\nkwargs:{kwargs}")


class Formatter:
    """
    Generator formatter module base calss.
    """
    trans_func = format_default_func

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        trans_func = kwargs.get("trans_func", None)
        if trans_func is not None:
            setattr(cls, "trans_func", trans_func)

    @abstractmethod
    def load(self, **kwargs):
        """load params.
        """
        for key in kwargs:
            if key in self.__dict__:
                setattr(self, key, kwargs[key])

    @abstractmethod
    def md(self, trans_func=None, **kwargs) -> str:
        """generaor markdown doc, return markdown str.

        Parameters
        ----------
        trans_func : _type_, optional
            use trans_func tran class prarms to md doc, by default None

        Returns
        -------
        str
            md doc.
        """
        return ""

    @abstractmethod
    def __json_encode__(self):
        return deepcopy(self.__dict__)


class TitleFormat(Formatter, trans_func=title):
    """deal title format.

    Parameters
    ----------
    trans_func : _type_, optional
        trans title mdutil func, by default title
    """

    def __init__(self,
                 title: str = "",
                 title_id: str = None,
                 title_jump_id: str = None,
                 **kwargs) -> None:
        self.title = title
        self.title_id = title_id
        self.title_jump_id = title_jump_id

    def md(self, trans_func=None, level: int = 1, **kwargs) -> str:
        if trans_func is None:
            trans_func = self.trans_func

        jump_title = internal_jump(click_text=self.title,
                                   link_to_id=self.title_jump_id,
                                   link_id=self.title_id)
        return trans_func(title=jump_title, level=level)


class EnvrionFormat(Formatter, trans_func=inset_table):
    """format envrtion to md table.
    usually table size: 2 * 4

    Parameters
    ----------
    trans_func : _type_, optional
        trans table mdutil, by default inset_table
    """

    def __init__(self,
                 id: str = "",
                 executor: str = "",
                 runtime_start: str = None,
                 runtime_end: str = None,
                 sample: str = None,
                 chiller: str = None,
                 version: str = None,
                 file_path: str = None) -> None:
        self.id = id
        self.executor = executor
        self.runtime_start = runtime_start
        self.runtime_end = runtime_end

        self.sample = sample
        self.chiller = chiller
        self.version = version
        self.file_path = file_path

    def md(self, trans_func=None, language=None, **kwargs) -> str:
        """
        """

        user_t_metra = ["id", "executor", "runtime_start", "runtime_end"]

        user_t_data = [
            self.id, self.executor, self.runtime_start, self.runtime_end
        ]

        sample_t_metra = ["sample", "chiller", "version", "file_path"]
        sample_t_data = [
            self.sample, self.chiller, self.version, self.file_path
        ]

        if language:
            user_t_metra = [language(x) for x in user_t_metra]
            sample_t_metra = [language(x) for x in sample_t_metra]
            user_t_data = [language(str(x)) for x in user_t_data]
            sample_t_data = [language(str(x)) for x in sample_t_data]

        md_doc = "\n"
        md_doc += trans_func([user_t_data], user_t_metra)

        md_doc += "\n"
        md_doc += trans_func([sample_t_data], sample_t_metra)
        md_doc += "\n"

        return md_doc


class BitFormat(Formatter, trans_func=inset_table):
    """
    Format Bit.
    """
    pass


class QubitFormat(BitFormat, trans_func=inset_table):
    """Qubit Format md table.

    Parameters
    ----------
    trans_func : _type_, optional
        trans table mdutil,, by default inset_table
    """

    def __init__(self, parameters: Dict = None, **normal_params):
        self.normal_params = normal_params
        if parameters:
            self.XYwave = parameters.pop("XYwave")
            self.Zwave = parameters.pop("Zwave")
            self.Mwave = parameters.pop("Mwave")
            self.union_readout = parameters.pop("union_readout")
            self.readout_point = parameters.pop("readout_point")
            self.normal_params.update(parameters)

    def md(self,
           trans_func=inset_table,
           language=None,
           new_params: Dict = None,
           **kwargs) -> str:

        old_dict = self.deal_qubits_params_to_single_dict()

        if new_params:
            new_dict = QubitFormat(
                **new_params).deal_qubits_params_to_single_dict()
            metra = ["param", "value_before", "value_after"]
            table_data = []
            for bit_param in new_dict:
                if new_dict[bit_param] != old_dict.get(bit_param):
                    table_data.append([
                        strong_str(bit_param),
                        strong_str(old_dict.get(bit_param),
                                   bold=False,
                                   ltalic=True),
                        strong_str(new_dict[bit_param])
                    ])
                else:
                    table_data.append([
                        bit_param,
                        old_dict.get(bit_param), new_dict[bit_param]
                    ])

            for key in old_dict:
                if key not in new_dict:
                    table_data.append([key, old_dict[key], old_dict[key]])
        else:
            metra = ["param", "value"]
            table_data = old_dict
        if language:
            metra = [language(x) for x in metra]
        return trans_func(table_data, metra)

    def deal_qubits_params_to_single_dict(self):
        """
        Convert the qubits params nested dictionary to a single-layer dictionary.
        Returns
        -------
        dict
            the single-layer dict.
        """
        new_dict = {}
        for key in self.__dict__:
            if key == "normal_params":
                for param, pa_val in self.normal_params.items():
                    new_dict.update({param: pa_val})
            elif self.__dict__[key] and isinstance(self.__dict__[key], dict):
                for param, pa_val in self.__dict__[key].items():
                    new_dict.update({key + "." + param: pa_val})

        return new_dict
