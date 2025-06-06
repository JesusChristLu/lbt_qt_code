# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

"""
Tree view for Experiment Library.
"""

from PySide6 import QtCore, QtGui
from PySide6.QtCore import QModelIndex, Signal
from PySide6.QtWidgets import QMenu, QWidget, QTreeView
from pyqcat_visage.gui.widgets.base.right_click_menu import bind_action

from pyqcat_visage.backend.experiment import VisageExperiment
from pyqcat_visage.gui.widgets.base.tree_structure import LeafNode
from pyqcat_visage.gui.widgets.task.create_task_dialog import CreateTaskDialog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class EXPLibraryTreeView(QTreeView):
    """Handles editing and displaying a pyqcat-monster experiment object.

    This class extend the `QTreeView`
    """

    choose_exp_signal = Signal(VisageExperiment)

    def __init__(self, parent: QWidget, gui: "VisageGUI"):
        super().__init__(parent)
        self.gui = gui
        self.init_right_click_menu()
        self.exp_name = None
        self.dialog_flag = False

    def _define_style(self):
        self.setDragEnabled(True)

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

        if (
            event.button() == QtCore.Qt.RightButton
        ):  # Check if right mouse button is clicked
            if index.isValid():
                if isinstance(node, LeafNode):
                    self.exp_name = node.value.name
                    self.right_click_menu.exec(event.globalPos())

        if isinstance(node, LeafNode) and self.dialog_flag is False:
            drag = QtGui.QDrag(self)
            mime_data = QtCore.QMimeData()

            mime_data.setText(f"{node.parent.name}|{node.label}")
            drag.setMimeData(mime_data)
            drag.exec()
            self.choose_exp_signal.emit(node.value)
            model.ui.exp_label.setText(f'| EXP({node.label}) ')
        self.dialog_flag = False
        return super().mousePressEvent(event)

    def init_right_click_menu(self):
        menu = QMenu(self)

        menu.create_task = bind_action(menu, "create_task", ": /tool.png")
        menu.create_task.triggered.connect(self.create_task)
        self.right_click_menu = menu

    def create_task(self):
        self.dialog_flag = True
        dialog = CreateTaskDialog(gui=self.gui, sub_type="exp")
        task_dialog = dialog.exec()
        if task_dialog == 1:
            task_name, policy_type, policy_opt = dialog.get_input()
            self.gui.dag_manage.create_task(
                task_name, policy_type, policy_opt, self.exp_name, "exp"
            )
            self.gui.dag_manage.refresh(True)
            self.gui.dag_manage.show()
