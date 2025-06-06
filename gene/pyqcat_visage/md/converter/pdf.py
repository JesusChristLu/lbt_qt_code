# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from typing import Union

import pdfkit
from PySide6.QtCore import QMarginsF, QUrl
from PySide6.QtGui import QPageLayout, QPageSize
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/12
# __author:       Lang Zhu
# __corporation:  OriginQuantum

PATH_WK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wkhtmltox")
PATH_CSS = os.path.join(PATH_WK, "css/demo.css")
PATH_EXE = os.path.join(PATH_WK, "bin/wkhtmltopdf.exe")
kit_options = {
    'encoding': 'UTF-8',
    "page-offset": 1,
    "footer-right": "[page] / [topage]",
    "footer-left": "[subsection]",
    "footer-spacing": 3,
    "footer-line": True,
    "footer-font-size": 9
}


def html_to_pdf(html_doc: str):
    """
    converter html to pdf use pdfkit.
    
    Args:
        html_doc(str): must be a content of html file
    """
    config = pdfkit.configuration(wkhtmltopdf=PATH_EXE)
    res = pdfkit.from_string(html_doc,
                             configuration=config,
                             options=kit_options,
                             verbose=False)
    return res


def html_to_pdf2(html_path: Union[str, Path]):
    """
    converter html to pdf use pyside6 QWebEngineView.
    Args:
        html_path: must be a filepath, not content of file.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    loader = QWebEngineView()
    loader.load(QUrl.fromLocalFile(html_path))
    layout = QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait,
                         QMarginsF(0, 0, 0, 0))

    new_file_path = str(html_path).replace("html", "pdf")

    def print_finished():
        page = loader.page()
        app.exit()

    def print_to_pdf(finished):
        page = loader.page()
        page.setBackgroundColor("red")
        page.selectedText()
        page.printToPdf(new_file_path, layout)

    loader.page().pdfPrintingFinished.connect(print_finished)
    loader.loadFinished.connect(print_to_pdf)
    app.exec_()
    with open(new_file_path, "rb") as fp:
        pdf_bytes = fp.read()
    os.remove(new_file_path)
    return pdf_bytes
