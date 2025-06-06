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
from .create_workspace_dialog_ui import Ui_Dialog
from ..title_window import TitleDialog


class CreateWorkSpaceDialog(TitleDialog):
    def __init__(self, parent: "WorkSpaceManageWindow" = None):
        super().__init__(parent)
        self._ui = Ui_Dialog()
        self._ui.setupUi(self)
        self.parent = parent

        # self.setWindowFlags(Qt.WindowType.MSWindowsFixedSizeDialogHint)
        # self.show()

    @property
    def ui(self):
        return self._ui

    def set_readonly(self, readonly: bool):
        self._ui.UserText.setDisabled(readonly)
        self._ui.SampleText.setDisabled(readonly)
        self._ui.EnvText.setDisabled(readonly)

    def get_input(self):
        username = self._ui.UserText.currentText()
        sample = self._ui.SampleText.currentText()
        env_name = self._ui.EnvText.currentText()
        qubit_names = self._ui.qubitText.text()
        config_names = self._ui.configText.text()
        extra_bits = self._ui.extraBitText.text()
        extra_bits = extra_bits.replace("，", ",")
        extra_bits = extra_bits.split(",")
        for bit in extra_bits:
            if bit not in qubit_names:
                qubit_names.append(bit)
        extra_configs = self._ui.extraConfigText.text()
        extra_configs = extra_configs.replace("，", ",")
        extra_configs = extra_configs.split(",")
        for conf in extra_configs:
            if conf not in config_names:
                config_names.append(conf)
        self.set_readonly(False)
        return username, sample, env_name, qubit_names, config_names

    def user_change(self):
        # username = self._ui.UserText.currentText()
        sample_list = list(self.parent.cache_chip_sample_data.keys())
        self._ui.SampleText.clear()
        self._ui.SampleText.addItems(sample_list)

    def sample_change(self):
        sample = self._ui.SampleText.currentText()
        env_list = self.parent.cache_chip_sample_data.get(sample, [])
        self._ui.EnvText.clear()
        self._ui.EnvText.addItems(env_list)

    def env_change(self):
        sample = self._ui.SampleText.currentText()
        env_name = self._ui.EnvText.currentText()
        data = self.parent.query_chip_line_data(sample, env_name)
        bit_names = data["bit_names"] + data.get("pair_names", [])
        conf_names = data["conf_names"]
        if not bit_names:
            conf_names = []
        self._ui.qubitText.set_units(bit_names)
        self._ui.configText.set_units(conf_names)
