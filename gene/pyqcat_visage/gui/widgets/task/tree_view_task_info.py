# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/08/10
# __author:       XuYao

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.tree_structure import QTreeViewBase


class QTreeViewTaskInfoWidget(QTreeViewBase, PlaceholderTextWidget):

    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        QTreeViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self,
            "Select a Component to look\n\nfrom the Task Info"
        )

    def _define_style(self):
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.expandAll()
