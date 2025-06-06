# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/24
# __author:       Lang Zhu
# __corporation:  OriginQuantum

import os
from pathlib import Path

from pyqcat_visage.md.util import Options
from pyqcat_visage.md.converter.extensions import BootStrapExtension
from pyqcat_visage.md.converter.markdown import converter_md_to_html, html_add_theme
from pyqcat_visage.md.converter.pdf import html_to_pdf, html_to_pdf2


class Converter:
    """
    Converter, is used to convert markdown to html and pdf.
    """

    def __init__(self):
        self.doc_pdf: bytes = None
        self.doc_html: str = None
        self._options: Options = self.default_options()

    @staticmethod
    def default_options():
        """the converter options.
        
        theme: the style of html and pdf. default is "white.
        hold_html: Whether to save html, default save.
        hold_pdf: Whether to save pdf, default save.

        Returns
        -------
        Options
        """
        option = Options()
        option.set_validator("hold_html", bool)
        option.set_validator("hold_pdf", bool)
        option.theme = "white"
        option.hold_html = True
        option.hold_pdf = True
        return option

    @property
    def option(self) -> Options:
        return self._options

    @option.setter
    def option(self, opt: Options):
        if isinstance(opt, Options):
            self._options.update(**opt)

    def execute(self, doc_md: str, extensions: list = None):
        """Converter execute funtion.

        Parameters
        ----------
        doc_md : str
            Markdown document to be converted.
        extensions : list, optional
            the markdown package extensions, by default None
            if none will add such exetensions-in.
            [
            'markdown.extensions.toc', 'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            BootStrapExtension()
        ]
        
        """
        extensions = [
            'markdown.extensions.toc', 'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            BootStrapExtension()
        ] or extensions
        temp_html = ""
        if self.option.hold_pdf:
            temp_html = html_add_theme(
                converter_md_to_html(doc_md, extensions), self.option.theme)
            self.doc_pdf = self.trans_pdf_by_pdfkit(temp_html)

        if self.option.hold_html:
            if temp_html:
                self.doc_html = temp_html
            else:
                self.doc_html = html_add_theme(
                    converter_md_to_html(doc_md, extensions),
                    self.option.theme)

    @staticmethod
    def trans_pdf_by_pyside(doc_html: str) -> bytes:
        """translate pdf by pyside6 Qwebengine function.
        This approach does not introduce too many packages and has more topic choices, 
        with the only drawback being that directories cannot be generated.
        Parameters
        ----------
        doc_html : str
            html doc with theme.

        Returns
        -------
        bytes
            the pdf bytes.
        """
        temp_path = "temp.html"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(doc_html)
        new_path = Path(temp_path).absolute()
        doc_pdf = html_to_pdf2(new_path)
        os.remove(temp_path)
        return doc_pdf

    @staticmethod
    def trans_pdf_by_pdfkit(doc_html: str) -> bytes:
        """translate pdf by pdfkit function.
        This way you can generate directories, but it requires a lot of packages to install and is cumbersome.

        Parameters
        ----------
        doc_html : str
            html doc with theme.

        Returns
        -------
        bytes
            the pdf bytes.
        """
        return html_to_pdf(doc_html)
