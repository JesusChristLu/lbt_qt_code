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
from .change_password_dialog_ui import Ui_Dialog
from ..title_window import TitleDialog


class ChangePasswordDialog(TitleDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = Ui_Dialog()
        self._ui.setupUi(self)

        # self.setWindowFlags(Qt.WindowType.MSWindowsFixedSizeDialogHint)
        # self.show()

    def get_input(self):
        password = self._ui.PasswordText.text()
        password_again = self._ui.PasswordAgainText.text()
        return password, password_again

