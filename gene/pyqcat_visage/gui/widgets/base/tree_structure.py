# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/31
# __author:       HanQing Shi, YangChao Zhao

"""Dict tree base."""

import ast
from typing import Union, TYPE_CHECKING, Any, Dict

from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt
from PySide6.QtGui import QFont, QContextMenuEvent
from PySide6.QtWidgets import QWidget, QTreeView
from loguru import logger

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI

KEY, NODE = range(2)


# List of children given as [(child_name_0, child_node_0), (child_name_1, child_node_1), ...]
# Where child_name_i corresponds to KEY and child_node_i corresponds to NODE


def get_nested_dict_item(dic, key_list: list):
    """Get a nested dictionary item. If key_list is empty, return dic itself.

    Args:
        dic (dict): Dictionary of items
        key_list (list): List of keys

    Returns:
        dict: Nested dictionary

    .. code-block:: python
        :linenos:

        myDict = Dict(
            aa=Dict(
                x1={'dda':34},
                y1='Y',
                z='10um'
            ),
            bb=Dict(
                x2=5,
                y2='YYY sdg',
                z='100um'
            )
        )
        key_list = ['aa', 'x1', 'dda']

        get_nested_dict_item(myDict, key_list)

        returns 34
    """
    if key_list:
        for k in key_list:
            dic = dic[k]
    return dic


class BranchNode:
    """A BranchNode object has a nonzero number of child nodes. These child
    nodes can be either BranchNodes or LeafNodes.

    It is uniquely defined by its name, parent node, and list of
    children. The list of children consists of tuple pairs of the form
    (node_name_i, node_i), where the former is the name of the child node
    and the latter is the child node itself. KEY (=0) and NODE (=1)
    identify their respective positions within each tuple pair.
    """

    def __init__(self, name: str, parent=None, data: Dict = None, vali_data: Dict = None):
        """
        Args:
            name (str): Name of this branch
            parent ([type]): The parent.  Defaults to None.
            data (dict): Node data.  Defaults to None.
            vali_data (dict): Data type.  Defaults to None.
        """
        super(BranchNode, self).__init__()
        self.name = name
        self.parent = parent
        self.children = []
        self._data = data  # dictionary containing the actual data
        self._vali_data = vali_data  # dictionary containing the actual data type

    def __len__(self):
        """Gets the number of children.

        Returns:
            int: The number of children this node has
        """
        return len(self.children)

    @property
    def data(self):
        return self._data

    @property
    def vali_data(self):
        return self._vali_data

    def provide_name(self):
        """Gets the name.

        Returns:
            str: The nodes name
        """
        return self.name  # identifier for BranchNode

    def child_at_row(self, row: int):
        """Gets the child at the given row.

        Args:
            row (int): The row

        Returns:
            Node: The node at the row
        """
        if 0 <= row < len(self.children):
            return self.children[row][NODE]

    def row_of_child(self, child):
        """Gets the row of the given child.

        Args:
            child (Node): The child

        Returns:
            int: Row of the given child.  -1 is returned if the child is not found.
        """
        for i, (_, child_node) in enumerate(self.children):
            if child == child_node:
                return i
        return -1

    def child_with_key(self, key):
        """Gets the child with the given key.

        Args:
            key (str): The key

        Returns:
            Node: The child with the same name as the given key.
            None is returned if the child is not found
        """
        for child_name, child_node in self.children:
            if key == child_name:
                return child_node
        return None

    def insert_child(self, child):
        """Insert the given child.

        Args:
            child (Node): The child
        """
        child.parent = self
        self.children.append((child.provide_name(), child))

    def has_leaves(self):
        """Do I have leaves?

        Returns:
            bool: True if there are leaves, False otherwise
        """
        if not self.children:
            return False
        return isinstance(self.children[KEY][NODE], LeafNode)


class LeafNode:
    """A LeafNode object has no children but consists of a key-value pair,
    denoted by label and value, respectively.

    It is uniquely identified by its root-to-leaf path, which is a list
    of keys whose positions denote their nesting depth (shallow to
    deep).
    """

    def __init__(self, label: str, parent: BranchNode = None, path=None):
        """
        Args:
            label (str): Label for the leaf node
            parent (BranchNode): The parent.  Defaults to None.
            path (list): Node path.  Defaults to None.
        """
        super(LeafNode, self).__init__()
        self.path = path or []
        self.parent = parent
        self.label = label

    @property
    def value(self):
        """Returns the value."""
        _v = get_nested_dict_item(self.parent.data, self.path)

        if isinstance(_v, list) and len(_v) > 0 and isinstance(_v[-1], bool):
            return _v[0]

        return _v

    @property
    def vali_data(self):
        """Validate type only in top branch node, is a dict."""
        # vts = ['str', False]
        #
        # # list type
        # if self.label == 'style':
        #     return [['qarange', 'normal'], False]
        #
        # if self.parent and self.parent.vali_data:
        #     vt = None
        #     vali_data = self.parent.vali_data
        #     if isinstance(vali_data, dict):
        #         vt = vali_data.get(self.label)
        #
        #     if vt:
        #         vts = vt

        _v = get_nested_dict_item(self.parent.data, self.path)

        if _v and isinstance(_v, list) and isinstance(_v[-1], bool):
            return _v[1]

        return None

    def provide_name(self):
        """Get the label.

        Returns:
            str: The label of the leaf node - this is *not* the value.
        """
        return self.label  # identifier for LeafNode (note: NOT value!)


class QTreeModelBase(QAbstractItemModel):

    def __init__(self, parent: QWidget, gui: 'VisageGUI', view: QTreeView):
        """Editable table with drop-down rows for component options. Organized
        as a tree model where child nodes are more specific properties of a
        given parent node.

        Args:
            parent (QWidget): The parent widget
            gui (VisageGUI): The main user interface
            view (QTreeView): View corresponding to a tree structure
        """
        super().__init__(parent=parent)

        self._gui = gui
        self._view = view
        self._parent_widget = parent

        self._row_count = 0

        self.root = BranchNode('')
        self.headers = ['Name', 'Value']
        self.paths = []

    @property
    def data_dict(self):
        return {}

    @property
    def parent_widget(self):
        return self._parent_widget

    def refresh(self, expand: bool = True):
        """Force refresh.

        Completely rebuild the model and tree.
        """
        self.load()
        if expand:
            self._view.expandAll()

    def load(self):
        """ Builds a tree from a dictionary (self.data_dict) """
        # is data dict is none, return directly
        if not self.data_dict:
            return

        self.beginResetModel()

        # Set the data dict reference of the root node. The root node doesn't have a name.
        self.root._data = self.data_dict

        # Clear existing tree paths if any
        self.paths.clear()
        self.root.children.clear()

        # Construct the paths -> sets self.paths
        self.get_paths(self.data_dict, [])

        for path in self.paths:
            root = self.root
            branch = None
            # Combine final label and value for leaf node,
            # so stop at 2nd to last element of each path
            for key in path[:-2]:
                # Look for child node with the name 'key'. If it's not found, create a new branch.
                branch = root.child_with_key(key)
                if not branch:
                    branch = BranchNode(key, data=self.data_dict)
                    root.insert_child(branch)
                root = branch
            # If a LeafNode resides in the outermost dictionary, the above for loop is bypassed.
            # [Note: This happens when the root-to-leaf path length is just 2.]
            # In this case, add the LeafNode right below the master root.
            # Otherwise, add the LeafNode below the final branch.
            if not branch:
                root.insert_child(LeafNode(path[-2], root, path=path[:-1]))
            else:
                branch.insert_child(LeafNode(path[-2], branch, path=path[:-1]))

        # Emit a signal since the model's internal state
        # (e.g. persistent model indexes) has been invalidated.
        self.endResetModel()

    def get_paths(self, cur_dict: Dict, cur_path: list):
        """Recursively finds and saves all root-to-leaf paths in model."""
        for k, v in cur_dict.items():
            if isinstance(v, dict):
                self.get_paths(v, cur_path + [k])
            else:
                self.paths.append(cur_path + [k, v])

    def node_from_index(self, index: QModelIndex) -> Union[BranchNode, LeafNode]:
        if index.isValid():
            # The internal pointer will return the leaf or branch node under the given parent.
            return index.internalPointer()
        return self.root

    def rowCount(self, parent: QModelIndex = None) -> int:
        """Get the number of rows.

        Args:
            parent (QModelIndex): The parent

        Returns:
            int: The number of rows
        """
        node = self.node_from_index(parent)
        if node is None or isinstance(node, LeafNode):
            return 0

        return len(node)

    def columnCount(self, parent: QModelIndex = None) -> int:
        """Get the number of columns.

        Args:
            parent (QModelIndex): The parent

        Returns:
            int: The number of columns
        """
        return len(self.headers)

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
                    return ''
                # We have a leaf
                elif index.column() == 0:
                    return str(node.label)  # key
                elif index.column() == 1:
                    return str(node.value)  # value
                else:
                    return None

        return None

    def setData(
            self,
            index: QModelIndex,
            value: Any,
            role: Qt.ItemDataRole = Qt.ItemDataRole.EditRole
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
                        #     f'Setting parameter {lbl:>10s}: old value={old_value}; new value={value};'
                        # )

                        #################################################
                        # Parse value if not str
                        # Somewhat legacy code for extended handling of non string options
                        # These days we tend to have all options be strings, so not so relevant, but keep here for now
                        # to allow extended use in te future
                        # if not isinstance(old_value, str):
                        processed_value, used_ast = parse_param_from_str(
                            value)
                        # logger.info(f'  Used paring:  Old value type={type(old_value)}; '
                        #             f'New value type={type(processed_value)};'
                        #             f'  New value={processed_value};'
                        #             f'; Used ast={used_ast}')
                        value = processed_value
                        #################################################

                        if node.path:  # if nested option
                            for x in node.path[:-1]:
                                dic = dic[x]
                            dic[node.path[-1]] = value
                        else:  # if top-level option
                            dic[lbl] = value

                        # if self.optionstype == 'component':
                        #     self.component.rebuild()
                        #     self.gui.refresh()
                        return True
        return False

    def headerData(
            self,
            section: int,
            orientation: Qt.Orientation,
            role=Qt.ItemDataRole.DisplayRole
    ):
        """Set the headers to be displayed.

        Args:
            section (int): Section number
            orientation (Qt.Orientation): The orientation
            role (Qt.ItemDataRole): The role

        Returns:
            object: Header data
        """
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                if 0 <= section < len(self.headers):
                    return self.headers[section]

            elif role == Qt.ItemDataRole.FontRole:
                font = QFont()
                font.setBold(True)
                return font

        return None

    def index(self, row: int, column: int, parent: QModelIndex = None):
        """Return my index.

        Args:
            row (int): The row
            column (int): The column
            parent (QModelIndex): The parent

        Returns:
            QModelIndex: internal index
        """
        assert self.root
        branch = self.node_from_index(parent)
        assert branch is not None
        # The third argument is the internal index.
        return self.createIndex(row, column, branch.child_at_row(row))

    def parent(self, child: QModelIndex = None):
        """Gets the parent index of the given node.

        Args:
            child (node): The child

        Returns:
            int: The index
        """
        node = self.node_from_index(child)

        if node is None:
            return QModelIndex()

        parent = node.parent
        if parent is None:
            return QModelIndex()

        grandparent = parent.parent
        if grandparent is None:
            return QModelIndex()

        row = grandparent.row_of_child(parent)
        assert row != -1

        return self.createIndex(row, 0, parent)

    def flags(self, index: QModelIndex):
        """Determine how user may interact with each cell in the table.

        Returns:
            list: List of flags
        """
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled

        if index.column() == 1:
            node = self.node_from_index(index)
            if isinstance(node, LeafNode):
                return flags | Qt.ItemFlag.ItemIsEditable

        return flags


class QTreeViewBase(QTreeView):

    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        QTreeView.__init__(self, parent)

        self.right_click_menu = None

        # This signal is emitted when the item specified by index is expanded.
        self.expanded.connect(self.resize_on_expand)

        self._define_style()

    def autoresize_columns(self, max_width: int = 200):
        """Resize columns to contents with maximum

        Args:
            max_width (int): Maximum window width. Defaults to 200.
        """
        # For TreeView: resizeColumnToContents
        # For TableView: resizeColumnsToContents

        columns = self.model().columnCount(None)
        for i in range(columns):
            self.resizeColumnToContents(i)
            width = self.columnWidth(i)
            if width > max_width:
                self.setColumnWidth(i, max_width)

    def resize_on_expand(self):
        """Resize when exposed."""
        self.resizeColumnToContents(0)

    def _define_style(self):
        pass

    def init_right_click_menu(self):
        pass

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Create options for drop-down context menu.

        Args:
            event (QContextMenuEvent): The event
        """
        if not self.right_click_menu:
            self.init_right_click_menu()

        if self.right_click_menu:
            self.right_click_menu.action = self.right_click_menu.exec_(
                self.mapToGlobal(event.pos())
            )


def parse_param_from_str(text):
    """Attempt to parse a value from a string using ast.

    Args:
        text (str): String to parse

    Return:
        tuple: value, used_ast

    Raises:
        Exception: An error occurred
    """
    text = str(text).strip()
    value = text
    used_ast = False

    try:
        # crude way to handle list and values
        value = ast.literal_eval(text)
        used_ast = True
    except Exception as e:
        # pass
        logger.warning(f'use ast eval {text} error, because {e}!')

    return value, used_ast
