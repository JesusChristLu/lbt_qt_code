# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/02/28
# __author:       Hou XuYao

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from pyQCat.structures import QDict
from .create_chimera_dialog_ui import Ui_Dialog
from ..title_window import TitleDialog


class CreateChipDialog(TitleDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = Ui_Dialog()
        self._ui.setupUi(self)

        # self.setWindowFlags(Qt.WindowType.MSWindowsFixedSizeDialogHint)
        # self.show()
        self._ui.DebugText.addItems(["0", "1"])
        self._ui.WindowSizeText.addItems([str(i) for i in range(1, 101)])
        self._ui.WindowSizeText.setCurrentText("10")
        self._ui.AlertDisText.addItems([str(i) for i in range(1, 6)])
        self._ui.AlertDisText.setCurrentText("1")
        self._ui.SecureDisText.addItems([str(i) for i in range(1, 6)])
        self._ui.SecureDisText.setCurrentText("2")

    @property
    def ui(self):
        return self._ui

    def get_input(self):
        sample = self._ui.SampleText.text()
        env_name = self._ui.EnvText.text()
        inst_ip = self._ui.InstIpText.text()
        inst_port = self._ui.InstPortText.text()
        groups = self._ui.GroupText.currentText()
        core_num = int(self._ui.CoreNumText.currentText())
        debug = int(self._ui.DebugText.currentText())
        window_size = int(self._ui.WindowSizeText.currentText())
        alert_dis = int(self._ui.AlertDisText.currentText())
        secure_dis = int(self._ui.SecureDisText.currentText())
        return (
            sample,
            env_name,
            inst_ip,
            inst_port,
            groups,
            core_num,
            debug,
            window_size,
            alert_dis,
            secure_dis,
        )
