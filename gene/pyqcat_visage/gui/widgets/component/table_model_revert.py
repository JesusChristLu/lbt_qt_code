# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2024/01/15
# __author:       XuYao

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex

from ..base.table_structure import QTableModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from ..revert_bits import RevertBitWindow
    from .table_view_revert import QTableViewRevertWidget


class QTableModelRevertBit(QTableModelBase):
    def __init__(
        self,
        gui: "VisageGUI",
        parent: "RevertBitWindow" = None,
        table_view: "QTableViewRevertWidget" = None,
    ):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self.columns = ["name", "create_time"]
        # self.columns_ratio = [1, 3]
        # self.disable_column = list(range(len(self.columns)))

    @property
    def model_data(self):
        return self.widget.data

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
