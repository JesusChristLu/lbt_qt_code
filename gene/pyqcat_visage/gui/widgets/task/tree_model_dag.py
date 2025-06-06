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

# from pyqcat_visage.gui.widgets.base.tree_structure import QTreeModelBase
from pyQCat.tools import qarange

from pyqcat_visage.gui.widgets.options_edit_widget import QTreeModelOptions, OptionsEditWidget

from PySide6.QtWidgets import QTreeView, QFormLayout, QLineEdit, QDateTimeEdit

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from ..system_config import SystemConfigWindow
from pyqcat_visage.gui.widgets.base.tree_structure import (
    LeafNode,
    BranchNode,
    parse_param_from_str,
)
from PySide6.QtCore import QModelIndex, Qt
from pyqcat_visage.gui.widgets.options.style import STYLE, VALI_TYPE, OptionType
from PySide6.QtGui import QFont
from typing import Any, Dict
from loguru import logger


class QTreeModelDag(QTreeModelOptions):

    def __init__(self, parent: "OptionsEditWidget", gui: "VisageGUI", view: QTreeView, name: str):
        """Editable table with drop-down rows for component options. Organized
        as a tree model where child nodes are more specific properties of a
        given parent node.

        Args:
            parent (SystemConfigWindow): The parent widget
            gui (VisageGUI): The main user interface
            view (QTreeViewConfig): View corresponding to a tree structure
        """
        super().__init__(parent, gui, view, name)
        self._conf = {"interval": [10, "int", True],
                      "unit": ["s", ["s", "min", "h"], True],
                      "priority": [1, "int", True],
                      "repeat": [5, "int", True],
                      "is_calibration": [False, "bool", True]
                      }

    @property
    def data_dict(self):
        return self._conf

    @data_dict.setter
    def data_dict(self, value):
        self._conf = value

    def get_tree_paths(self, cur_dict: Dict, cur_path: list, is_full: False):
        """Recursively finds and saves all root-to-leaf paths in model."""
        for k, v in cur_dict.items():
            if isinstance(v, dict):
                self.get_tree_paths(v, cur_path + [k], is_full)
            else:
                self.paths.append(cur_path + [k, v])

    def load(self):
        """Builds a tree from a dictionary (self.data_dict)"""
        # if (self.optionstype == 'experiment') and (not self.component):
        if not self._conf:
            return
        # if isinstance(self._conf, object):
        #     form_layout = QFormLayout()
        #
        #     edit_execution_time = QDateTimeEdit()
        #     form_layout.addRow("Execution Time:", edit_execution_time)
        #     print(233)
        #     return
        self.beginResetModel()

        # Set the data dict reference of the root node. The root node doesn't have a name.
        # data_dict 子类中实现
        self.root._data = self.data_dict
        # self.root._vali_data = self.vali_data_dict

        # Clear existing tree paths if any
        self.paths.clear()
        self.root.children.clear()

        # Construct the paths -> sets self.paths
        self.get_tree_paths(self.data_dict, [], False)

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

    def data(
            self, index: QModelIndex, role: Qt.ItemDataRole = Qt.ItemDataRole.DisplayRole
    ):
        """Gets the node data.

        Args:
            index (QModelIndex): Index to get data for
            role (Qt.ItemDataRole): The role..  Defaults to Qt.DisplayRole.

        Returns:
            object: fetched data
        """
        if not index.isValid():
            return None

        if self.data_dict is None:
            return None

        # The data in a form suitable for editing in an editor. (QString)
        if role == Qt.ItemDataRole.EditRole:
            return self.data(index, Qt.ItemDataRole.DisplayRole)

        # Bold the first
        if (role == Qt.ItemDataRole.FontRole) and (index.column() == 0):
            font = QFont()
            font.setBold(True)
            return font

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        if role == Qt.ItemDataRole.DisplayRole:
            node = self.node_from_index(index)
            if node:

                # the first column is either a leaf key or a branch
                # the second column is always a leaf value or for a branch is ''.
                if isinstance(node, BranchNode):
                    # Handle a branch (which is a nested sub dictionary, which can be expanded)
                    if index.column() == 0:
                        return node.name
                    elif index.column() == 1:
                        if node.children[-1][0] == "describe":
                            return node.children[-1][1].value
                        else:
                            return ""
                    else:
                        return ""

                # Leaf None Case
                elif index.column() == 0:
                    return str(node.label)
                elif index.column() == 1:
                    return str(node.value)
                else:
                    logger.error(f"It is impossible to have three columns!")
                    return ""

        if role == VALI_TYPE:
            node = self.node_from_index(index)
            if isinstance(node, LeafNode):
                return node.vali_data
            else:
                return None
                # if isinstance(node, LeafNode):
                #     if node.vali_data:
                #         return node.vali_data[0]

        if role == Qt.CheckStateRole:
            if index.column() == 1:
                node = self.node_from_index(index)
                if node:
                    if isinstance(node, LeafNode):
                        if node.vali_data == "bool" or isinstance(node.value, bool):
                            return node.value
                        # if node.vali_data:
                        #     if node.vali_data[0] == 'bool':
                        #         return node.value

        if role == STYLE:
            v_type = self.data(index, VALI_TYPE)
            if v_type:
                if v_type == "bool":
                    return OptionType.check_box
                elif v_type == "str" and v_type.startswith("spin"):
                    return OptionType.spin_box
                elif isinstance(v_type, list):
                    return OptionType.com_box
                else:
                    return OptionType.line_editor

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
                    value_type = node.vali_data

                    if old_value == value:
                        return False

                    # Set the value of an option when the new value is different
                    else:
                        dic = self.data_dict  # option dict
                        lbl = node.label  # option key

                        logger.debug(
                            f"Setting {self.optionstype} option: {lbl:^20s} < "
                            f"old value={old_value}, new value={value} >"
                        )

                        # Parse value if not str ##############################
                        # Somewhat legacy code for extended handling of non string options
                        # These days we tend to have all options be strings, so not so relevant, but keep here for now
                        # to allow extended use in te future
                        # if not isinstance(old_value, str):
                        processed_value, used_ast = parse_param_from_str(value)
                        logger.debug(
                            f"  Used paring:  Old value type={type(old_value)}; "
                            f"New value type={type(processed_value)};"
                            f"  New value={processed_value};"
                            f"; Used ast={used_ast}"
                        )
                        value = processed_value

                        #################################################
                        # int type auto transform
                        if value_type == "int":
                            value = int(value)

                        # spin or check type transform
                        elif isinstance(value_type, list):
                            if isinstance(value_type[0], int):
                                value = int(value)
                            elif isinstance(value_type[0], float):
                                value = float(value)

                        if node.path:
                            # update model data
                            for x in node.path[:-1]:
                                dic = dic[x]

                            if isinstance(dic[node.path[-1]], list) and dic[node.path[-1]] and isinstance(
                                dic[node.path[-1]][-1], bool
                            ):
                                dic[node.path[-1]][0] = value
                            else:
                                dic[node.path[-1]] = value

                            # list type auto refresh
                            if node.parent and node.label in [
                                "start",
                                "end",
                                "step",
                                "style",
                                "details",
                            ]:

                                start_node = node.parent.child_with_key("start")
                                end_node = node.parent.child_with_key("end")
                                step_node = node.parent.child_with_key("step")
                                style_node = node.parent.child_with_key("style")
                                details_node = node.parent.child_with_key("details")
                                describe_node = node.parent.child_with_key("describe")

                                if style_node.value == "normal":
                                    describe = f"normal | {details_node.value}"
                                else:
                                    try:
                                        details = qarange(
                                            float(start_node.value),
                                            float(end_node.value),
                                            float(step_node.value),
                                        )
                                        dic = self.data_dict
                                        for x in details_node.path[:-1]:
                                            dic = dic[x]
                                        dic[details_node.path[-1]][0] = details
                                    except Exception as e:
                                        logger.debug(
                                            f"{e}, qarange error, please check input!"
                                        )
                                        return False
                                    describe = f"qarange | ({start_node.value}, {end_node.value}, {step_node.value})"

                                dic = self.data_dict
                                for x in describe_node.path[:-1]:
                                    dic = dic[x]
                                dic[describe_node.path[-1]][0] = describe

                                # auto same options

                            else:
                                pre_dict = self.data_dict
                                for x in node.path[:-2]:
                                    pre_dict = pre_dict[x]

                        else:  # if top-level option
                            dic[lbl][0] = value
                        return True

        elif role == Qt.ItemDataRole.CheckStateRole:
            if index.column() == 1:
                node = self.node_from_index(index)

                if isinstance(node, LeafNode):
                    old_value = node.value  # option value

                    if not isinstance(old_value, bool):
                        return False

                    # Set the value of an option when the new value is different
                    else:
                        dic = self.data_dict  # option dict
                        lbl = node.label  # option key

                        if node.path:  # if nested option
                            for x in node.path[:-1]:
                                dic = dic[x]

                            if isinstance(dic[node.path[-1]], list) and isinstance(
                                dic[node.path[-1]][-1], bool
                            ):
                                dic[node.path[-1]][0] = not old_value
                            else:
                                dic[node.path[-1]] = not old_value

                        logger.debug(
                            f"Setting {self.optionstype} option: {lbl:^20s} < "
                            f"old value={old_value}, new value={dic[lbl]} >"
                        )
                        return True

        return False
