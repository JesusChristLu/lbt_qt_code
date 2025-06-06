# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QIcon, QFont
from loguru import logger

from pyqcat_visage.gui.widgets.base.tree_structure import (
    QTreeModelBase,
    BranchNode,
    LeafNode,
    parse_param_from_str,
)

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from .tree_view_component import QTreeViewComponentWidget
    from ..component_window import ComponentEditWindow


class QTreeModelComponent(QTreeModelBase):
    def __init__(
        self,
        parent: "ComponentEditWindow",
        gui: "VisageGUI",
        view: "QTreeViewComponentWidget",
    ):
        """Editable table with drop-down rows for component options. Organized
        as a tree model where child nodes are more specific properties of a
        given parent node.

        Args:
            parent (ComponentEditWindow): The parent widget
            gui (VisageGUI): The main user interface
            view (QTreeViewComponentWidget): View corresponding to a tree structure
        """
        super().__init__(parent, gui, view)
        self.load()

    @property
    def data_dict(self):
        return self.component.view_data if self.component else {}

    @property
    def component(self):
        return self._parent_widget.component

    @component.setter
    def component(self, component):
        self._parent_widget._component = component

    def flags(self, index: QModelIndex):
        """Determine how user may interact with each cell in the table.

        Returns:
            list: List of flags

        """
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled

        # dcm and chip not support edit
        if self.component.style == "bin" or self.component.name in [
            "chip_line_connect.json",
            # "hardware_offset.json"
        ]:
            return flags

        return super().flags(index)

    def data(
        self, index: QModelIndex, role: Qt.ItemDataRole = Qt.ItemDataRole.DisplayRole
    ):
        if not index.isValid():
            return None

        # The data in a form suitable for editing in an editor. (QString)
        if role == Qt.ItemDataRole.EditRole:
            return self.data(index, Qt.ItemDataRole.DisplayRole)

        # Bold the first
        if role == Qt.ItemDataRole.FontRole and index.column() == 0:
            font = QFont()
            font.setBold(True)
            return font

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        if role == Qt.ItemDataRole.DisplayRole:
            node = self.node_from_index(index)
            if node:
                # the first column is either a leaf key or a branch
                # the second column is always a leaf value or for a branch is ''.
                if isinstance(node, BranchNode):
                    # Handle a branch (which is a nested sub dictionary, which can be expanded)
                    if index.column() == 0:
                        return node.name
                    return ""
                # We have a leaf
                elif index.column() == 0:
                    return str(node.label)  # key
                elif index.column() == 1:
                    return str(node.value)  # value
                else:
                    return None

        if role == Qt.ItemDataRole.DecorationRole:
            node = self.node_from_index(index)
            if isinstance(node, LeafNode):
                key = ".".join(node.path)
                if key in self.component.edit_records:
                    return QIcon(":/sync.png")

        return None

    def setData(
        self,
        index: QModelIndex,
        value: Any,
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

        if not index.isValid():
            return False

        elif role == Qt.ItemDataRole.EditRole:

            if index.column() == 1:
                node = self.node_from_index(index)

                if isinstance(node, LeafNode):
                    value = str(value)  # new value
                    old_value = node.value  # option value

                    if old_value == value:
                        return False

                    # Set the value of an option when the new value is different
                    else:
                        dic = self.data_dict  # option dict
                        lbl = node.label  # option key

                        # logger.info(
                        #     f"Setting parameter {lbl:>10s}: old value={old_value}; new value={value};"
                        # )

                        #################################################
                        # Parse value if not str
                        # Somewhat legacy code for extended handling of non string options
                        # These days we tend to have all options be strings, so not so relevant, but keep here for now
                        # to allow extended use in te future
                        # if not isinstance(old_value, str):
                        processed_value, used_ast = parse_param_from_str(value)
                        # logger.info(
                        #     f"  Used paring:  Old value type={type(old_value)}; "
                        #     f"New value type={type(processed_value)};"
                        #     f"  New value={processed_value};"
                        #     f"; Used ast={used_ast}"
                        # )
                        value = processed_value
                        #################################################

                        if node.path:  # if nested option
                            if self.component.style in ["qubit", "coupler"]:
                                self.component.edit_records.append(".".join(node.path))
                            for x in node.path[:-1]:
                                dic = dic[x]
                            dic[node.path[-1]] = value
                        else:  # if top-level option
                            if self.component.style in ["qubit", "coupler"]:
                                self.component.edit_records.append(lbl)
                            dic[lbl] = value

                        # if self.optionstype == 'component':
                        #     self.component.rebuild()
                        #     self.gui.refresh()
                        return True
        return False
