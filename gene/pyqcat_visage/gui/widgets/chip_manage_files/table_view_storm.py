# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/10/18
# __author:       XuYao
from functools import partial

from PySide6.QtWidgets import QTableView, QAbstractItemView, QWidget, QHeaderView, QMenu

from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.right_click_menu import bind_action
from ..base.table_structure import QTableViewBase


class QTableViewStormWidget(QTableViewBase, PlaceholderTextWidget):
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
        # self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def init_right_click_menu(self):
        menu = QMenu(self)

        menu.delete = bind_action(menu, "Delete", ":/delete.png")
        menu.addSeparator()
        menu.start = bind_action(menu, "Start", ":/run.png")
        menu.stop = bind_action(menu, "Stop", ":/wifi-off-fill.png")

        menu.delete.triggered.connect(self.delete_item)
        menu.start.triggered.connect(partial(self.control_storm, "start"))
        menu.stop.triggered.connect(partial(self.control_storm, "stop"))

        self.right_click_menu = menu

    def get_item_data(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            chip = model.item_from_index(index)
        else:
            chip = {}
        sample = chip.get("sample")
        env_name = chip.get("env_name")
        return sample, env_name, index.row()

    def delete_item(self):
        sample, env_name, *_, index = self.get_item_data()
        if sample and env_name:
            self.model().widget.delete_storm(sample, env_name, index)

    def control_storm(self, option: str):
        sample, env_name, *_ = self.get_item_data()
        if sample and env_name:
            self.model().widget.control_storm_server(sample, env_name, option)

    def update_column_widths(self):
        total_width = self.width()
        if self.model().columns_ratio:
            total_weight = sum(self.model().columns_ratio)
            for i, width in enumerate(self.model().columns_ratio):
                new_width = int((width / total_weight) * total_width)
                self.setColumnWidth(i, new_width)
        else:
            self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_column_widths()
