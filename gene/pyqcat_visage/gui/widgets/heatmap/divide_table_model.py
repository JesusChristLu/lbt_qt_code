# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/02/17
# __author:       YangChao Zhao


from PySide6.QtCore import QModelIndex, QAbstractTableModel, Qt

from pyQCat.types import StandardContext
from ..base.table_structure import QTableModelBase


class QTableModelDivide(QTableModelBase):
    def __init__(self, gui, parent, table_view):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self.columns = ["name", "freq", "IF", "channel", "lo1", "lo2", "gap"]
        self._editable_index = [3]

    @property
    def model_data(self):
        return self.widget.group

    def _display_data(self, index: QModelIndex):
        row = index.row()
        column = index.column()

        bit_name = list(self.model_data.keys())[row]
        infos = self.model_data.get(bit_name)

        if column == 0:
            return bit_name
        else:
            key = self.columns[column]
            return str(infos.get(key))

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

        if index.column() in self._editable_index:
            return table_flag | Qt.ItemFlag.ItemIsEditable

        return table_flag

    def bit_from_index(self, index: QModelIndex):
        return list(self.model_data.items())[index.row()]


class QTableModelAmpDivide(QTableModelDivide):
    def __init__(self, gui, parent, table_view):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self.columns = [
            "name",
            "power",
            "amp",
            "channel",
            "sample delay",
            "sample width",
            "baseband freq",
        ]
        self._editable_index = [2, 4, 5, 6]

    @property
    def model_data(self):
        return self.widget.bus_group


class QTableModelParallelDivide(QTableModelDivide):

    def __init__(self, gui, parent, table_view):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self._editable_index = []
        self.columns = ["name", "xy_lo", "xy_gap", "m_lo", "m_gap", "bus", "probe_power"]

    @property
    def model_data(self):
        context_group_name = self.widget.ui.context_group.currentText()
        if context_group_name == StandardContext.QC.value:
            self.columns = ["name", "xy_lo", "xy_gap", "m_lo", "m_gap", "bus", "probe_power"]
        elif context_group_name == StandardContext.CC.value:
            self.columns = ["name", "idle_point", "dc_min", "dc_max", "z_dc_channel", "z_flux_channel"]
        else:
            self.columns = ["name", "qc", "ql", "qh"]

        if not self.widget.parallel_group.values():
            self.columns = []
        return self.widget.parallel_group
