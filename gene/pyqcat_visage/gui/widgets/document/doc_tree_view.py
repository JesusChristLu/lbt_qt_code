# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/16
# __author:       YangChao Zhao

"""
Tree view for Struct Library.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QAbstractItemView

from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.tree_structure import QTreeViewBase


class QTreeViewDocument(QTreeViewBase, PlaceholderTextWidget):
    """Handles editing and displaying a pyqcat-monster experiment object.

    This class extend the `QTreeView`
    """

    def __init__(self, parent: QWidget):
        """
        Inits TreeViewQLibrary
        Args:
            parent (QtWidgets.QWidget): parent widget
        """
        QTreeViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self,
            "Library of experiments.\nClick one to edit it in Options window."
        )

    def _define_style(self):
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
