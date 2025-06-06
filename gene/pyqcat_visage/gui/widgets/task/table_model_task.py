# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/10/25
# __author:       Xw

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, QAbstractTableModel, Qt
from PySide6.QtWidgets import QMessageBox
from loguru import logger

from ..base.table_structure import QTableModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from .task_manage import DagManagerWindow
    from pyqcat_visage.gui.widgets.task.table_view_task import QTableViewTaskWidget


class QTableModelTaskManage(QTableModelBase):
    disable_column = (0, 1)

    def __init__(
            self,
            gui: "VisageGUI",
            parent: "DagManagerWindow" = None,
            table_view: "QTableViewTaskWidget" = None,
    ):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self.columns = [
            "Name",
            "Last Update",
            "ID",
            "Status"
        ]
        self.columns_ratio = [1, 1, 1, 1]
        self.group_name = None
        self.ui = parent

    @property
    def model_data(self):
        return self.widget.tasks

    def flags(self, index: QModelIndex = None):
        """Set the item flags at the given index.

        Args:
            index (QModelIndex): The index

        Returns:
            Qt flags: Flags from Qt
        """
        table_flag = Qt.ItemFlags(QAbstractTableModel.flags(self, index))

        if not index.isValid():
            return table_flag

        column_num = index.column()
        if column_num in self.disable_column:
            return table_flag

        return table_flag | Qt.ItemFlag.ItemIsEditable

    def removeRows(self, row: int, count: int = 1, parent=QModelIndex()):
        """Delete highlighted rows.

        Args:
            row (int): First row to delete.
            count (int): Number of rows to delete.  Defaults to 1.
            parent (QModelIndex): Parent index.
        """
        self.beginRemoveRows(parent, row, row + count - 1)
        for k in range(row + count - 1, row - 1, -1):
            del self.widget.tasks[k]
        self.endRemoveRows()

    def task_from_index(self, index: QModelIndex):
        return self.model_data[index.row()]

    def _display_data(self, index: QModelIndex):
        item = self.task_from_index(index)
        if index.column() == 0:
            return item.get("task_name")
        elif index.column() == 1:
            return item.get("create_time")
        elif index.column() == 2:
            return item.get("id")
        elif index.column() == 3:
            return item.get("status")
