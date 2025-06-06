# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/04/26
# __author:       YangChao Zhao


from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, QAbstractTableModel, Qt

from ..base.table_structure import QTableModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from .chip_create_widget import ChipCreateWindow
    from PySide6.QtWidgets import QTableView


class QTableModelChip(QTableModelBase):

    def __init__(
            self, gui: 'VisageGUI',
            parent: 'ChipCreateWindow' = None,
            table_view: 'QTableView' = None
    ):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self.columns = []

    @property
    def model_data(self):
        """
        chips: {
            "row-0": {
                "col-0": 1,
                "col-1": 2,
                "col-2": 3,
                "col-3": 4,
            },
            "row-1": {
                "col-0": 5,
                "col-1": 6,
                "col-2": 7,
                "col-3": 8,
            },
            "row-2": {
                "col-0": 9,
                "col-1": 10,
                "col-2": 12,
                "col-3": 13,
            }
        }
        """
        return self.widget.chips

    def _display_data(self, index: QModelIndex):
        row = index.row()
        column = index.column()

        row_name = f"row-{row}"
        col_name = f'col-{column}'

        return str(self.model_data.get(row_name).get(col_name))

    def flags(self, index: QModelIndex = None):
        """Set the item flags at the given index.

        Args:
            index (QModelIndex): The index

        Returns:
            Qt flags: Flags from Qt
        """

        # ItemFlag.ItemNeverHasChildren|ItemIsEnabled|ItemIsSelectable
        table_flag = QAbstractTableModel.flags(self, index)
        return table_flag | Qt.ItemFlag.ItemIsEditable

    def setData(self,
                index: QModelIndex,
                value,
                role: Qt.ItemDataRole = Qt.ItemDataRole.EditRole) -> bool:
        if not value or not index.isValid():
            return False

        elif role == Qt.ItemDataRole.EditRole:
            row_name = f'row-{index.row()}'
            col_name = f'col-{index.column()}'

            value = int(value)
            old_value = self.model_data[row_name][col_name]

            if old_value == value:
                return False

            self.model_data[row_name][col_name] = value
            return True
