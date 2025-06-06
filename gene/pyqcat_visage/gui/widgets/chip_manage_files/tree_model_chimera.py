# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2024/01/25
# __author:       XuYao

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QTreeView, QWidget

from pyqcat_visage.gui.widgets.base.tree_structure import QTreeModelBase, LeafNode, parse_param_from_str, BranchNode

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from ..context_window import ContextEditWindow


class QTreeModelChimera(QTreeModelBase):
    def __init__(
        self,
        parent: "ChimeraManagerWindow",
        gui: "VisageGUI",
        view: "QTreeViewChimeraWidget",
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
        return self.item

    @property
    def item(self):
        return self._parent_widget.item

    @item.setter
    def item(self, item):
        self._parent_widget._item = item
