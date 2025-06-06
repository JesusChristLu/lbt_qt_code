# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/28
# __author:       YangChao Zhao

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QTreeView

from pyqcat_visage.gui.widgets.base.tree_structure import QTreeModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from ..heatmap_widget import HeatMapWindow
    from .struct_tree_view import StructTreeView


class StructTreeModel(QTreeModelBase):

    def __init__(self, parent: 'HeatMapWindow', gui: 'VisageGUI', view: 'StructTreeView', style: str):
        """Editable table with drop-down rows for component options. Organized
        as a tree model where child nodes are more specific properties of a
        given parent node.

        Args:
            gui (MetalGUI): The main user interface
            view (QTreeView): View corresponding to a tree structure
        """
        super().__init__(parent, gui, view)
        self.style = style
        self.headers = [f'{style} Struct', 'Unit']
        self.load()

    @property
    def ui(self):
        return self._gui.ui

    @property
    def backend(self):
        return self._gui.backend

    @property
    def data_dict(self):
        return self.backend._heatmap_struct.get(self.style)

    def set_style(self, style: str):
        self.style = style
        self.load()

    def load(self):
        """Builds a tree from a dictionary (self.data_dict)"""
        self.headers = [f'{self.style} Struct', 'Unit']
        super().load()

    def flags(self, index: QModelIndex):
        """Determine how user may interact with each cell in the table.

        Returns:
            list: List of flags
        """
        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
