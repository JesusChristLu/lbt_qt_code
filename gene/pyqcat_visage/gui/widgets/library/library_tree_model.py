# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QTreeView, QWidget

from pyqcat_visage.gui.widgets.base.tree_structure import BranchNode, QTreeModelBase

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class LibraryTreeModel(QTreeModelBase):
    """Tree model for component option menu. Overrides rowCount method to
    include placeholder text, and data method to include 3rd column for parsed
    values.
    """

    def __init__(self, parent: QWidget, gui: 'VisageGUI', view: QTreeView, style: str):
        """Editable table with drop-down rows for component options. Organized
        as a tree model where child nodes are more specific properties of a
        given parent node.

        Args:
            gui (MetalGUI): The main user interface
            view (QTreeView): View corresponding to a tree structure
        """
        super().__init__(parent, gui, view)

        self.stype = style
        self.headers = [f'{style} List']
        self.refresh()

    @property
    def ui(self):
        return self._gui.ui

    @property
    def backend(self):
        return self._gui.backend

    @property
    def data_dict(self):
        return self.backend.get_library(self.stype)

    def load(self):
        """Builds a tree from a dictionary (self.data_dict)"""
        if self.stype.lower() not in ['experiments', 'dags']:
            return
        super().load()

    def data(self, index: QModelIndex, role: Qt.ItemDataRole = Qt.ItemDataRole.DisplayRole):
        """Gets the node data.

        Args:
            index (QModelIndex): Index to get data for
            role (Qt.ItemDataRole): The role.  Defaults to Qt.DisplayRole.

        Returns:
            object: Fetched data
        """
        if not index.isValid():
            return None

        # Bold the first
        if (role == Qt.ItemDataRole.FontRole) and (index.column() == 0):
            font = QFont()
            font.setBold(True)
            return font

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                node = self.node_from_index(index)
                if node:
                    # the first column is either a leaf key or a branch
                    # the second column is always a leaf value or for a branch is ''.
                    if isinstance(node, BranchNode):
                        # Handle a branch (which is a nested sub dictionary, which can be expanded)
                        return node.name
                    # We have a leaf
                    else:
                        return node.label

        return None

    def flags(self, index: QModelIndex):
        """Determine how user may interact with each cell in the table.

        Returns:
            list: List of flags
        """
        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
