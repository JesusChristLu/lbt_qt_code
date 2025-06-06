# -*- coding: utf-8 -*-
import os
import time

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/24
# __author:       Lang Zhu
# __corporation:  OriginQuantum
from pyqcat_visage.md.parser import parser_dict, Parser
from typing import Union
from pyQCat.log import pyqlog


def execute(id: str,
            id_type: str = "dag",
            theme: str = "white",
            save_type: str = "pdf",
            language: str = "cn",
            report_detail: str = "detail",
            **kwargs):
    """    markdown module calls the port, passes the corresponding ID, and the type of the experiment report,
    can generate the corresponding experiment report. With the theme, save file type, lab report detail level,
    language and other options.

    Parameters
    ----------
    id : str
        the need generator report id, maybe dag, experiment or other.
    id_type : str, optional
        the report type, support dag and experiment, by default "dag"
    theme : str, optional
        the report theme, support light, dark, by default "light"
    save_type : str, optional
        report file save type. support markdown,pdf, html, by default "pdf"
        When the md file is too large, some readers may not be able to open and render it, such as typora.

    Returns
    -------
    _type_
        _description_
    """
    if id_type not in parser_dict:
        return

    exe_parser: Parser = parser_dict[id_type](id=id)
    exe_parser.converter_options.theme = theme
    exe_parser.generator_options.language = language
    exe_parser.generator_options.detail = report_detail
    exe_parser.converter_options.hold_html = True if save_type == "html" else False
    exe_parser.converter_options.hold_pdf = True if save_type == "pdf" else False
    exe_parser.parser()
    if save_type == "pdf":
        return exe_parser.converter_obj.doc_pdf
    elif save_type == "html":
        return exe_parser.converter_obj.doc_html
    elif save_type == "md":
        return exe_parser.generator.markdown


def save_report(report_doc: Union[str, bytes], file_name: str = None, save_type: str = "pdf", file_path: str = None,
                **kwargs):
    def save_bytes(file_path_: str, file_doc: bytes):

        with open(file_path_, "wb+") as f:
            f.write(file_doc)

    def save_str(file_path_: str, file_doc: str):
        with open(file_path_, "w+", encoding="utf-8") as f:
            f.write(file_doc)

    if not file_path:
        pyqlog.warning("report save without file_path")
        return

    if not file_name or not isinstance(file_name, str):
        file_name = f"{time.time()}.{save_type}"
    else:
        if not file_name.endswith(save_type):
            file_name += f".{save_type}"
    if not file_path.endswith("report"):
        file_path = os.path.join(file_path, "report")

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path = os.path.join(file_path, file_name)

    save_execute = {
        "pdf": save_bytes,
        "md": save_str,
        "html": save_str
    }

    save_execute[save_type](file_path, report_doc)
    return file_path
