# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/08/1
# __author:       XuYao
from functools import partial

from PySide6.QtWidgets import QTableView, QAbstractItemView, QWidget, QHeaderView, QMenu
from PySide6.QtCore import Signal
from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.right_click_menu import bind_action
from ..base.table_structure import QTableViewBase


class MultiTableView(QTableViewBase, PlaceholderTextWidget):

    task_detail = Signal(dict)
    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        self.ui = parent.parent().parent()
        QTableViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(self, "Select sample/env_name to query chips!")

    def _define_style(self):
        # Handling selection dynamically
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QTableView.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def resizeEvent(self, event):
        self.update_column_widths()
        super().resizeEvent(event)

    def update_column_widths(self):
        total_width = self.width()
        if self.model() and self.model().columns_ratio:
            total_weight = sum(self.model().columns_ratio)
            for i, width in enumerate(self.model().columns_ratio):
                new_width = int((width / total_weight) * total_width)
                self.setColumnWidth(i, new_width)
        else:
            self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def view_clicked(self, index):
        """Select a component and set it in the component widget when you left mouse click.

        In the init, we had to connect with self.clicked.connect(self.viewClicked)

        Args:
            index (QModelIndex): The index
        """

        if not index.isValid():
            return
        model = self.model()
        thread = model.tr_by_index(index)
        self.task_detail.emit(thread)

    # def init_right_click_menu(self):
    #     menu = QMenu(self)
    #     menu._change_group = bind_action(menu, "Save", ":/save.png")
    #
    #     menu._change_group.triggered.connect(self.update_chip)
    #     self.right_click_menu = menu
