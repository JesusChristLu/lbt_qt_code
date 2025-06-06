# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/21
# __author:       XuYao


import os
import json
import shutil
from pathlib import Path
from typing import Union, Dict
from PySide6.QtCore import QDir
from jinja2 import Template
from pyQCat.structures import Cached

abs_path = Path(__file__).absolute().parent.parent
home_path = Path.home()


class CustomTheme(metaclass=Cached):

    def __init__(self):
        style_path = abs_path.joinpath("styles")
        self.themes_path = style_path.joinpath("themes")
        self.theme_template = Template(self._get_file_content(style_path.joinpath("qss_template")))
        self.themes = [x.split(".")[0] for x in os.listdir(self.themes_path)]
        self.new_icon_path = home_path.joinpath(".pyqcat/theme_icons")
        self.icon_path = abs_path.joinpath("_imgs")
        self.add_icon_path()

    def add_icon_path(self):
        try:
            QDir.addSearchPath("icon", self.new_icon_path)
        except:
            QDir.addSearchPath("icon", self.icon_path)
        self._format_theme_icon({}, False)

    def _format_theme_icon(self, theme_dict: Dict, all: bool = True):
        if all:
            shutil.rmtree(self.new_icon_path, ignore_errors=True)
        os.makedirs(self.new_icon_path, exist_ok=True)
        background = theme_dict.get("FirstBackGround", "#0000ff")
        font_color = theme_dict.get("TreeLineColor", "#ff0000")
        theme_color = theme_dict.get("ThemeColor", "#00ff00")
        for icon in os.listdir(self.icon_path):
            if not icon.endswith('.svg'):
                continue
            if not all:
                if not icon.startswith("window-"):
                    continue
            with open(self.icon_path.joinpath(icon), 'r') as fp:
                icon_content = fp.read()
            new_icon = icon_content.replace("#0000ff", background). \
                replace("#ff0000", font_color).replace("#00ff00", theme_color)
            with open(self.new_icon_path.joinpath(icon), "w") as fp:
                fp.write(new_icon)

    @staticmethod
    def _get_file_content(filepath: Path):
        with open(filepath, "r", encoding="utf-8") as fp:
            return fp.read()

    def _format_theme(self, theme_filename: str) -> str:
        theme_content = json.loads(self._get_file_content(self.themes_path.joinpath(theme_filename)))
        self._format_theme_icon(theme_content)
        qss_file = self.theme_template.render(**theme_content)
        return qss_file

    def qss(self, theme: str):
        theme_name = theme if theme in self.themes else self.themes[0]
        theme_filename = theme_name + ".json"
        return self._format_theme(theme_filename)


if __name__ == '__main__':
    themes = CustomTheme()
    print(themes.qss("visage_dark"))
