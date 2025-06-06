# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/09/25
# __author:       XuYao

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from pyQCat.structures import QDict
from .tips_dialog_ui import Ui_Dialog
from ..title_window import TitleDialog


class TipsDialog(TitleDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.setWindowFlags(Qt.WindowType.MSWindowsFixedSizeDialogHint)
        self.show()

    def set_tips(self, context: str):
        self.ui.contentText.setText(context)
