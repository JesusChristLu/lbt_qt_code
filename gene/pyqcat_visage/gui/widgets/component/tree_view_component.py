# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

from PySide6.QtWidgets import QWidget, QMenu
from PySide6.QtCore import Qt
from pyQCat.invoker import DataCenter

from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.right_click_menu import bind_action
from ..base.tree_structure import QTreeViewBase


class QTreeViewComponentWidget(QTreeViewBase, PlaceholderTextWidget):
    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        QTreeViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self, "Select a Component to edit\n\nfrom the Component Collector"
        )

    def init_right_click_menu(self):
        menu = QMenu(self)

        menu._d = bind_action(menu, "Delete", ":/delete.png")
        menu._d.triggered.connect(self.delete_row)
        menu.addSeparator()
        menu._add_to_space = bind_action(menu, "Add to workspace", ":/add_to_workspace.png")
        menu._add_to_space.triggered.connect(self.add_to_workspace)

        self.right_click_menu = menu

    def delete_row(self):
        indexes = self.selectedIndexes()
        if len(indexes) > 0:
            model = self.model()
            index = indexes[0]
            value = model.data(index)
            model.parent_widget.delete_union_rd(value)

    def add_to_workspace(self):
        indexes = self.selectedIndexes()
        if len(indexes) > 0:
            model = self.model()
            index = indexes[0]
            value = model.data(index)
            db = DataCenter()
            res = db.add_workspace_info(1, bit_attr=value)

    def _define_style(self):
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
