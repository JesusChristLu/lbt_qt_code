# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/08/1
# __author:       XuYao

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, QAbstractTableModel, Qt
from PySide6.QtWidgets import QMessageBox
from loguru import logger

from ..base.table_structure import QTableModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from .threads_view import MultiTableView
    from .multi_thread import MultThreadWindow


class ThreadTableModel(QTableModelBase):
    disable_column = (0, 1, 2, 3)

    def __init__(
            self,
            gui: "VisageGUI",
            parent=None,
            table_view=None,
    ):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self.group_name = None
        self.ui = parent
        self.columns = []
        self.columns_ratio = []
        self._init_columns()

    def _init_columns(self):
        pass

    @property
    def model_data(self):
        return []

    def flags(self, index: QModelIndex = None):
        """Set the item flags at the given index.

        Args:
            index (QModelIndex): The index

        Returns:
            Qt flags: Flags from Qt
        """

        # ItemFlag.ItemNeverHasChildren|ItemIsEnabled|ItemIsSelectable
        table_flag = Qt.ItemFlags(QAbstractTableModel.flags(self, index))

        if not index.isValid():
            return table_flag

        if not self.gui.backend.is_super:
            return table_flag

        column_num = index.column()
        if column_num in self.disable_column:
            return table_flag

        return table_flag | Qt.ItemFlag.ItemIsEditable

    def setData(
            self,
            index: QModelIndex,
            value,
            role: Qt.ItemDataRole = Qt.ItemDataRole.EditRole,
    ) -> bool:
        """Set the LeafNode value and corresponding data entry to value.
        Returns true if successful; otherwise returns false. The dataChanged()
        signal should be emitted if the data was successfully set.

        Args:
            index (QModelIndex): The index
            value: The value
            role (Qt.ItemDataRole): The role of the data.  Defaults to Qt.EditRole.

        Returns:
            bool: True if successful, False otherwise
        """
        if value is None or not index.isValid():
            return False

        elif role == Qt.ItemDataRole.EditRole:
            column = index.column()
            column_name = self.columns[column]
            chip = self.chip_from_index(index)

            try:
                if value == "":
                    return False
                old_value = chip[column_name]
            except ValueError:
                QMessageBox().critical(
                    self.ui, "Error", f"{column_name} type error, your input is {value}"
                )
                return False

            if old_value == value:
                return False

            chip[column_name] = value
            logger.info(
                f"Setting parameter chip:{column_name}: old value={old_value}; new value={value};"
            )
            return True

    def removeRows(self, row: int, count: int = 1, parent=QModelIndex()):
        """Delete highlighted rows.

        Args:
            row (int): First row to delete.
            count (int): Number of rows to delete.  Defaults to 1.
            parent (QModelIndex): Parent index.
        """
        self.beginRemoveRows(parent, row, row + count - 1)
        for k in range(row + count - 1, row - 1, -1):
            del self.widget.chips[k]
        self.endRemoveRows()

    def chip_from_index(self, index: QModelIndex):
        return self.model_data[index.row()]

    def _display_data(self, index: QModelIndex):
        chip = self.chip_from_index(index)
        return chip.get(self.columns[index.column()])

    def tr_by_index(self, index):
        return self.model_data[index.row()]


class MultiThreadModel(ThreadTableModel):

    def _init_columns(self):
        self.columns = [
            "user",
            "task_id",
            "run_time",
            "expected",
        ]
        self.columns_ratio = [2, 4, 2, 2]

    @property
    def model_data(self):
        return self.widget.table_data


class SchedulerListModel(ThreadTableModel):
    def _init_columns(self):
        self.columns = [
            "exp",
            "user",
            "priority",
            "expected",
            "doc_id",
        ]
        self.columns_ratio = [2, 2, 1, 1, 2]

    @property
    def model_data(self):
        return self.widget.scheduler_data


class NormalListModel(SchedulerListModel):

    @property
    def model_data(self):
        return self.widget.normal_data


class LowPriorityListModel(SchedulerListModel):

    @property
    def model_data(self):
        return self.widget.low_priority_data
