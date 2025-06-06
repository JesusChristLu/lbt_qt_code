# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/26
# __author:       YangChao Zhao


from typing import List

import numpy as np
from loguru import logger
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QBrush


class QTableModelDat(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_widget = parent
        self.input_data = np.array([])
        self.x_labels = None
        self.y_labels = None
        self.name = None

    def rowCount(self, parent=QModelIndex()):
        if isinstance(self.input_data, List):
            return len(self.input_data)
        else:
            return self.input_data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        if isinstance(self.input_data, List):
            return len(self.input_data[0])
        else:
            shape = self.input_data.shape
            return 0 if len(shape) == 1 else shape[1]

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            if self.x_labels:
                return self.x_labels[section]
            elif section == 0:
                return "x"
            else:
                return f"y-{section}"
        else:
            if self.y_labels:
                return self.y_labels[section]
            else:
                return str(section + 1)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return float(self.input_data[index.row()][index.column()])
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)
        elif role == Qt.ItemDataRole.BackgroundRole:
            v = self.data(index)
            if v != 0:
                return QBrush(Qt.red)

        return None

    def flags(self, index):
        std_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if self.x_labels:
            return std_flags | Qt.ItemFlag.ItemIsEditable
        return std_flags

    def refresh(self):
        self.modelReset.emit()

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
        if value is None or not index.isValid() or value == "":
            return False

        elif role == Qt.ItemDataRole.EditRole:
            column = index.column()
            row = index.row()

            pre_data = self.input_data[row][column]
            self.input_data[row][column] = float(value)
            logger.log(
                "UPDATE",
                f"Change {self.x_labels[row]}-{self.y_labels[column]} {pre_data} to {value}",
            )
            return True
