# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/08/10
# __author:       XuYao

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QTreeView, QWidget

from pyqcat_visage.gui.widgets.base.tree_structure import QTreeModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from ..context_window import ContextEditWindow


class QTreeModelSpaceNote(QTreeModelBase):
    def __init__(
        self,
        parent: "WorkSpaceHisWindow",
        gui: "VisageGUI",
        view: "QTreeViewSpaceNoteWidget",
    ):
        """Editable table with drop-down rows for component options. Organized
        as a tree model where child nodes are more specific properties of a
        given parent node.

        Args:
            parent (QWidget): The parent widget
            gui (MetalGUI): The main user interface
            view (QTreeView): View corresponding to a tree structure
        """
        super().__init__(parent, gui, view)
        self.load()

    @property
    def data_dict(self):
        return self.space

    @property
    def space(self):
        return self._parent_widget.space

    @space.setter
    def space(self, space):
        self._parent_widget._space = space

    def flags(self, index: QModelIndex):
        """Determine how user may interact with each cell in the table.

        Returns:
            list: List of flags
        """
        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
