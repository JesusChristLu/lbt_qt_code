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
    from ..chimera_manage import ChimeraManagerWindow
    from .table_view_chimera import QTableViewChimeraWidget


class QTableModelChipManage(QTableModelBase):
    disable_column = (0, 1, 2, 5)

    def __init__(
            self,
            gui: "VisageGUI",
            parent: "ChimeraManagerWindow" = None,
            table_view: "QTableViewChimeraWidget" = None,
    ):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self.columns = [
            "sample",
            "env_name",
            "status",
        ]
        self.columns_ratio = [6, 5, 2]
        self.group_name = None
        self.ui = parent

    @property
    def model_data(self):
        return self.widget.data

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
            item = self.item_from_index(index)

            try:
                if value == "":
                    return False
                old_value = item[column_name]
            except ValueError:
                QMessageBox().critical(
                    self.ui, "Error", f"{column_name} type error, your input is {value}"
                )
                return False

            if old_value == value:
                return False

            item[column_name] = value
            logger.info(
                f"Setting parameter item:{column_name}: old value={old_value}; new value={value};"
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
            del self.widget.data[k]
        self.endRemoveRows()

    def item_from_index(self, index: QModelIndex):
        return self.model_data[index.row()]

    def _display_data(self, index: QModelIndex):
        item = self.item_from_index(index)
        return item.get(self.columns[index.column()])
