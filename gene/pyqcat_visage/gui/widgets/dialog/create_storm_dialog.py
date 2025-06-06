# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/10/18
# __author:       Hou XuYao

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from pyQCat.structures import QDict
from .create_storm_dialog_ui import Ui_Dialog
from ..title_window import TitleDialog


class CreateStormDialog(TitleDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = Ui_Dialog()
        self._ui.setupUi(self)

        # self.setWindowFlags(Qt.WindowType.MSWindowsFixedSizeDialogHint)
        # self.show()

    @property
    def ui(self):
        return self._ui

    def get_input(self):
        sample = self._ui.SampleText.text()
        env_name = self._ui.EnvText.text()
        return sample, env_name
