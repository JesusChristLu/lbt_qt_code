# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/02/28
# __author:       Hou XuYao
from typing import Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from pyQCat.structures import QDict
from .copy_workspace_dialog_ui import Ui_Dialog
from ..title_window import TitleDialog


class CopyWorkSpaceDialog(TitleDialog):
    def __init__(self, parent: "WorkSpaceManageWindow" = None,
                 user_list: List = None,
                 sample_cache: Dict = None):
        super().__init__(parent)
        self._ui = Ui_Dialog()
        self._ui.setupUi(self)
        self.parent = parent
        self.set_readonly(True)
        self.sample_cache = {}
        self.user_list = {}

        # self.setWindowFlags(Qt.WindowType.MSWindowsFixedSizeDialogHint)
        # self.show()

    def init_query_data(self):
        self.sample_cache = self.parent.cache_chip_sample_data
        self.user_list = self.parent.user_list
        self.ui.UserText.clear()
        self.ui.UserText.addItems(self.user_list)
        self.ui.UserText_2.clear()
        self.ui.UserText_2.addItems(self.user_list)
        self.ui.SampleText.clear()
        sample_list = list(self.sample_cache.keys())
        self.ui.SampleText.addItems(sample_list)

    @property
    def ui(self):
        return self._ui

    def set_readonly(self, readonly: bool):
        # self._ui.UserText.setDisabled(readonly)
        self._ui.SampleText_2.setDisabled(readonly)
        self._ui.EnvText_2.setDisabled(readonly)

    def get_input(self):
        from_user = self._ui.UserText.currentText()
        from_sample = self._ui.SampleText.currentText()
        from_env = self._ui.EnvText.currentText()
        to_user = self._ui.UserText_2.currentText()
        return from_user, from_sample, from_env, to_user

    def user_change(self):
        if not self.ui.UserText_2.currentText():
            self.ui.UserText_2.setCurrentText(self.parent.gui.backend.username)

    def sample_change(self):
        sample = self._ui.SampleText.currentText()
        env_list = self.sample_cache.get(sample, [])
        self._ui.EnvText.clear()
        self._ui.EnvText.addItems(env_list)
        self.ui.SampleText_2.setCurrentText(self.ui.SampleText.currentText())

    def env_change(self):
        self.ui.EnvText_2.setCurrentText(self.ui.EnvText.currentText())
