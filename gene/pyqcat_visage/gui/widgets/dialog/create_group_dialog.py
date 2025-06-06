# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/08
# __author:       YangChao Zhao

from PySide6.QtCore import Qt

from pyQCat.structures import QDict
from .create_group_dialog_ui import Ui_Dialog
from ..title_window import TitleDialog


class CreateGroupDialog(TitleDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # self.setWindowFlags(Qt.WindowType.MSWindowsFixedSizeDialogHint)
        # self.show()

    def get_input(self):
        group_name = self.ui.name_edit.text()
        description = self.ui.des_edit.text()
        return group_name, description
