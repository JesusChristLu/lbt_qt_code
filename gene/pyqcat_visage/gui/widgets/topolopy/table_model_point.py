# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/10/26
# __author:       YangChao Zhao


from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, Qt

from ..base.table_structure import QTableModelBase
from loguru import logger

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from .table_view_point import QPointTableView
    from .context_sidebar import ContextSideBar


class QTableModelPoint(QTableModelBase):
    def __init__(
        self,
        gui: "VisageGUI",
        parent: "ContextSideBar" = None,
        table_view: "QPointTableView" = None,
    ):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self.columns = ["Name", "Value"]

    @property
    def model_data(self):
        return self.backend.context_builder.global_options.custom_points

    def component_from_index(self, index: QModelIndex):
        return list(self.backend.view_context.items())[index.row()]

    def _display_data(self, index: QModelIndex):
        row = index.row()
        column = index.column()

        component_name = list(self.model_data.keys())[row]
        component = self.model_data.get(component_name)

        if column == 0:
            return str(component_name)
        elif column == 1:
            return str(component)

    def flags(self, index: QModelIndex = None):
        column_num = index.column()
        if column_num == 0:
            return (
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )

        if column_num == 1:
            return (
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsEditable
                | Qt.ItemFlag.ItemIsSelectable
            )

    def setData(
        self,
        index: QModelIndex,
        value,
        role: Qt.ItemDataRole = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if value is None or not index.isValid():
            return False

        elif role == Qt.ItemDataRole.EditRole:
            bit = list(self.model_data.keys())[index.row()]

            try:
                value = float(value)

                if not (-1 < value < 1) and not 4000 < value < 8000:
                    logger.error(f"point set error, [-1, 1] or [4000, 8000], but your input is {value}")
                    return False

                old_value = (
                    self.gui.backend.context_builder.global_options.custom_points[bit]
                )
            except ValueError:
                logger.error(f"{bit} set error, your input is {value}")
                return False

            if old_value == value:
                return False

            self.gui.backend.context_builder.global_options.custom_points[bit] = value
            logger.info(
                f"Setting parameter {bit}: old value={old_value}; new value={value};"
            )
            return True

    def bit_from_index(self, index: QModelIndex = None):
        if index and index.isValid():
            return list(self.model_data.keys())[index.row()]
