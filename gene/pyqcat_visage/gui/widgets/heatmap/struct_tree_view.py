# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/28
# __author:       YangChao Zhao

"""
Tree view for Struct Library.
"""

from PySide6 import QtGui
from PySide6.QtCore import QModelIndex, Signal
from PySide6.QtWidgets import QHeaderView

from pyqcat_visage.gui.widgets.base.tree_structure import LeafNode, QTreeViewBase, BranchNode


class StructTreeView(QTreeViewBase):
    """Handles editing and displaying a pyqcat-monster experiment object.

    This class extend the `QTreeView`
    """
    choose_struct_signal = Signal(str)

    def _define_style(self):
        self.header().setSectionResizeMode(QHeaderView.Stretch)
        self.setToolTip('Select a key to notify heatmap to refresh')

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """Overrides inherited mousePressEvent to emit appropriate filepath signals
         based on which columns were clicked, and to allow user to clear any selections
        by clicking off the displayed tree.

        Args:
            event (QtGui.QMouseEvent): QMouseEvent triggered by user
        """
        index = self.indexAt(event.pos())

        if index.row() == -1:
            self.clearSelection()
            self.setCurrentIndex(QModelIndex())
            return super().mousePressEvent(event)

        model = self.model()

        node = model.node_from_index(index)

        key = ""

        if isinstance(node, LeafNode):
            while True:

                if isinstance(node, LeafNode):
                    des = node.label
                    key = des
                    node = node.parent
                elif isinstance(node, BranchNode):
                    des = node.name
                    if des == "":
                        break
                    key = f'{des}.{key}'
                    node = node.parent
                else:
                    break

            self.choose_struct_signal.emit(key)

        return super().mousePressEvent(event)
