# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/16
# __author:       YangChao Zhao

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, Qt, QAbstractTableModel
from PySide6.QtWidgets import QMessageBox
from loguru import logger

from ..base.table_structure import QTableModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from .table_view_channel import QTableViewChannelWidget
    from ..context_window import ContextEditWindow


class QTableModelChannel(QTableModelBase):

    def __init__(
            self, gui: 'VisageGUI',
            parent: 'ContextEditWindow' = None,
            table_view: 'QTableViewChannelWidget' = None
    ):
        super().__init__(gui=gui, parent=parent, table_view=table_view)

        self.ui = parent
        self.columns = [
            'name', 'xy_channel', 'z_dc_channel', 'z_flux_channel',
            'readout_channel', 'probe_bit', 'drive_bit', "bus", "m_lo", "xy_lo"
        ]

    @property
    def model_data(self):
        return self.backend.view_channels

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

        if not self.gui.backend.is_super and \
                not self.gui.backend.is_admin:
            return table_flag

        column_num = index.column()
        bit_info, _ = self.bit_from_index(index)

        if bit_info.startswith('q'):
            if column_num in [0, 5, 6]:
                return table_flag
        else:
            if column_num in [0, 1, 4, 7, 8, 9]:
                return table_flag

        return table_flag | Qt.ItemFlag.ItemIsEditable

    def setData(self,
                index: QModelIndex,
                value,
                role: Qt.ItemDataRole = Qt.ItemDataRole.EditRole) -> bool:
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
            bit_name, _ = self.bit_from_index(index)

            try:
                if value != '-':
                    value = int(value)
                    if value < 0:
                        return False
                old_value = self.gui.backend.view_channels[bit_name].get(column_name)
            except ValueError:
                QMessageBox().critical(self.ui, 'Error', f'{column_name} type error, your input is {value}')
                return False

            if old_value == value:
                return False

            self.gui.backend.view_channels[bit_name][column_name] = value
            logger.info(
                f'Setting parameter {bit_name}: old value={old_value}; new value={value};'
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
        lst = list(self.gui.backend.view_channels.keys())
        for k in range(row + count - 1, row - 1, -1):
            del self.gui.backend.view_channels[lst[k]]
        self.endRemoveRows()

    def bit_from_index(self, index: QModelIndex):
        return list(self.backend.view_channels.items())[index.row()]

    def _display_data(self, index: QModelIndex):
        column = index.column()
        component_name, component = self.bit_from_index(index)

        if column == 0:
            return str(component_name)
        else:
            column_name = self.columns[column]
            return component.get(column_name, '-')
