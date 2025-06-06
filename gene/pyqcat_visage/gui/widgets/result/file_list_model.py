# -*- coding: utf-8 -*-
# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/01/08
# __author:       YangChao Zhao

from typing import Union

from PySide6.QtCore import QModelIndex, Qt, QAbstractListModel
from PySide6.QtGui import (QIcon)
from PySide6.QtWidgets import QWidget

from pyqcat_visage.config import GUI_CONFIG


class FileNode:

    def __init__(self, name: str, parent=None):
        self.name = name
        self.parent = parent
        self.path = []

        self._build_path()

    def has_parent(self):
        return True if self.parent else False

    def _build_path(self):
        self.path = [self.name]

        node = self
        while node.parent is not None:
            node = node.parent
            self.path.append(node.name)

        self.path.reverse()


class FolderFileNode(FileNode):

    def __init__(self, name: str = "", parent=None):
        super().__init__(name, parent)

        self.children = []

    def __len__(self):
        return len(self.children)

    def child_at_row(self, row: int):
        if 0 <= row < len(self.children):
            return self.children[row]

    def row_of_child(self, child):
        for i, child_node in enumerate(self.children):
            if child == child_node:
                return i
        return -1

    def insert_child(self, child: FileNode):
        if child.parent != self:
            child.parent = self
            child._build_path()
        if child not in self.children:
            self.children.append(child)

    def child_names(self):
        child_names = []
        for child in self.children:
            child_names.append(child.name)
        return child_names


class DocumentFileNode(FileNode):

    def __init__(self, name: str, parent: FolderFileNode):
        super().__init__(name, parent)


class QFileListModel(QAbstractListModel):

    def __init__(self, parent: QWidget):
        """Editable table with drop-down rows for component options. Organized
        as a tree model where child nodes are more specific properties of a
        given parent node.

        Args:
            parent (QWidget): The parent widget
        """
        super().__init__(parent=parent)

        self.root: FolderFileNode = FolderFileNode()

    def refresh(self):
        self.beginResetModel()

        # Emit a signal since the model's internal state
        # (e.g. persistent model indexes) has been invalidated.
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = None) -> int:
        return len(self.root)

    def flags(self, index):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def data(self, index, role: Qt.ItemDataRole = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            node = self.node_from_index(index)
            return node.name

        if role == Qt.ItemDataRole.DecorationRole:
            node = self.node_from_index(index)
            if isinstance(node, FolderFileNode):
                return QIcon(GUI_CONFIG.file_icon.folder)
            elif node.name.endswith('.png'):
                return QIcon(GUI_CONFIG.file_icon.png)
            elif node.name.endswith('.txt'):
                return QIcon(GUI_CONFIG.file_icon.txt)
            elif node.name.endswith('.dat'):
                return QIcon(GUI_CONFIG.file_icon.dat)
            elif node.name.endswith('.log'):
                return QIcon(GUI_CONFIG.file_icon.log)
            elif node.name.endswith('.json'):
                return QIcon(GUI_CONFIG.file_icon.json)
        return None

    def node_from_index(self, index: QModelIndex) -> Union[FolderFileNode, DocumentFileNode]:
        if index.isValid():
            return index.internalPointer()
        return self.root

    def index(self, row: int, column: int = 0, parent: QModelIndex = None):
        if len(self.root) > row:
            return self.createIndex(row, 0, self.root.child_at_row(row))
