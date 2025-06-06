# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/01
# __author:       YangChao Zhao

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit

if TYPE_CHECKING:
    from ...main_window import VisageGUI


class FilterLineEdit(QLineEdit):

    def __init__(self, parent: 'VisageGUI'):
        QLineEdit.__init__(self)
        self.parent = parent

    def keyPressEvent(self, event):
        QLineEdit.keyPressEvent(self, event)
        if event.key() == Qt.Key.Key_Return:
            self.parent.backend.filter_str = self.text()
            self.parent.exp_lib_model.refresh()
            self.parent.dag_lib_model.refresh()
