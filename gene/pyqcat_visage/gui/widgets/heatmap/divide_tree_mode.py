# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/12/08
# __author:       YangChao Zhao

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QTreeView

from pyQCat.tools.allocation import *
from pyQCat.types import StandardContext
from pyqcat_visage.gui.widgets.base.tree_structure import QTreeModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from ..heatmap_widget import HeatMapWindow
    from .struct_tree_view import StructTreeView


class DivideTreeModel(QTreeModelBase):

    def __init__(self, parent: 'HeatMapWindow', gui: 'VisageGUI', view: 'StructTreeView'):
        """Editable table with drop-down rows for component options. Organized
        as a tree model where child nodes are more specific properties of a
        given parent node.

        Args:
            gui (MetalGUI): The main user interface
            view (QTreeView): View corresponding to a tree structure
        """
        super().__init__(parent, gui, view)
        self.headers = ['Field', 'Value']

        self._config_data = {}
        self._load_config_data()

        self.load()

    @property
    def ui(self):
        return self._gui.ui

    @property
    def backend(self):
        return self._gui.backend

    @property
    def data_dict(self):
        return self._config_data

    def load(self):
        """Builds a tree from a dictionary (self.data_dict)"""
        self.headers = [f'Divide Field', 'Value']
        super().load()

    def flags(self, index: QModelIndex):
        """Determine how user may interact with each cell in the table.

        Returns:
            list: List of flags
        """
        control_mode = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled

        if index.column() == 1:
            control_mode |= Qt.ItemFlag.ItemIsEditable

        return control_mode

    def _load_config_data(self):
        self._config_data = {
            StandardContext.QC.value: ParallelAllocationQC.view_options(),
            StandardContext.CC.value: ParallelAllocationCC.view_options(),
            StandardContext.CGC.value: ParallelAllocationCGC.view_options(),
            "IntermediateFreqAllocation": IntermediateFreqAllocation.view_options(),
            "ReadoutAmpAllocation": ReadoutAmpAllocation.view_options()
        }
