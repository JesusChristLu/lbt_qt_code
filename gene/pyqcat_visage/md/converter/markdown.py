# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os
# __date:         2022/10/12
# __author:       Lang Zhu
# __corporation:  OriginQuantum
from pathlib import Path
from typing import Union

import markdown

from pyQCat.structures import Cached

abs_path = Path(__file__).absolute().parent


def converter_md_to_html(md_doc: str, extensions: list = None):
    """
    converter markdown doc string to html string.
    Parameters
    ----------
    md_doc : str
        the markdown doc str.
    extensions : list, optional
        the markdown package extensions, by default None
        some extensions can be optimized for the converted html  style, or compatible with more markdown syntax.
        such as table, toc.

    Returns
    -------
    str
        the html str, which without html head, just boby moduel.
    """
    if extensions is not None:
        return markdown.markdown(md_doc,
                                 output_format='html5',
                                 extensions=extensions)
    else:
        return markdown.markdown(md_doc, output_format='html5')


class HtmlStyle(metaclass=Cached):
    """html doc add head and style class.

    Parameters
    ----------
    metaclass : _type_, optional
        _description_, by default Cached
    """

    def __init__(self):
        self.style_path = abs_path.joinpath("style")
        self.themes_path = self.style_path.joinpath("themes")
        self.themes = [
            x.split(".")[0] for x in os.walk(self.themes_path).__next__()[-1]
        ]
        self._init_cache_file()

    def _get_style_content(self, style: Union[str, int] = "white") -> str:
        """generate css content"""
        if isinstance(style, int):
            index = style % len(self.themes)
            style_name = self.themes[index]
        else:
            if style not in self.themes:
                style_name = self.themes[0]
            else:
                style_name = style
        return self.base_style + self.style_data[style_name]

    def _init_cache_file(self):
        data = {}
        for theme in self.themes:
            with open(self.themes_path.joinpath(f"{theme}.css"),
                      "r",
                      encoding="utf-8") as fp:
                data[theme] = fp.read().replace("\n", "").replace("  ", " ")
        base_file = "base"
        with open(self.style_path.joinpath(f"{base_file}/{base_file}.css"),
                  "r",
                  encoding="utf-8") as fp:
            self.base_style = fp.read().replace("\n", "").replace("  ", " ")
        with open(self.style_path.joinpath("head.html"), "r",
                  encoding="utf-8") as fp:
            self.head_html = fp.read()
        self.style_data = data

    def html(self,
             html_doc: str,
             style: Union[str, int] = "themeable-light") -> str:
        """generate new html with css theme"""
        return self.head_html.format(self._get_style_content(style), html_doc)


html_style = HtmlStyle()

theme_list = html_style.themes


def html_add_theme(html_doc: str, theme: str = None) -> str:
    """add html theme with html doc.

    Parameters
    ----------
    html_doc : str
        html doc which the markdown genreator and just body.
    theme : str, optional
        the html style, by default None

    Returns
    -------
    str
        theml style with theme.
    """
    if theme:
        return html_style.html(html_doc, theme)
    else:
        return html_style.html(html_doc)
