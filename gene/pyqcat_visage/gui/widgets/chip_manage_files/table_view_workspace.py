# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/08/1
# __author:       XuYao

from PySide6.QtWidgets import QTableView, QAbstractItemView, QWidget, QHeaderView, QMenu

from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.right_click_menu import bind_action
from ..base.table_structure import QTableViewBase


class QTableViewWorkSpaceWidget(QTableViewBase, PlaceholderTextWidget):
    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        self.ui = parent.parent().parent()
        QTableViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self, "Select sample/env_name to query workspace!"
        )

    def _define_style(self):
        # Handling selection dynamically
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QTableView.SelectRows)
        # self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def init_right_click_menu(self):
        menu = QMenu(self)

        menu._change_group = bind_action(menu, "Change", ":/setting.png")
        menu._change_admin = bind_action(menu, "Delete", ":/delete.png")

        menu._change_group.triggered.connect(self.update_workspace)
        menu._change_admin.triggered.connect(self.delete_workspace)

        self.right_click_menu = menu

    def get_space_data(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            space = model.item_from_index(index)
        else:
            space = {}
        username = space.get("username")
        sample = space.get("sample")
        env_name = space.get("env_name")
        qubit_names = space.get("bit_names", "")
        config_names = space.get("conf_names", "")
        return username, sample, env_name, qubit_names, config_names, index.row()

    def update_workspace(self):
        (
            username,
            sample,
            env_name,
            qubit_names,
            config_names,
            index,
        ) = self.get_space_data()
        if sample and env_name:
            self.model().widget.update_workspace(
                username, sample, env_name, qubit_names, config_names
            )

    def delete_workspace(self):
        username, sample, env_name, *_, index = self.get_space_data()
        if sample and env_name:
            self.model().widget.delete_workspace(username, sample, env_name, index)

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
