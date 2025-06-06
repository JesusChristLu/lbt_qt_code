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
from copy import deepcopy

from PySide6 import QtGui
from PySide6.QtCore import QModelIndex, Signal, QPoint
from PySide6.QtWidgets import QMenu, QMessageBox, QInputDialog, QWidget
from loguru import logger

from pyqcat_visage.backend.model_dag import ModelDag
from pyqcat_visage.gui.widgets.base.tree_structure import LeafNode
from ..base.right_click_menu import bind_action
from ..base.tree_structure import QTreeViewBase
from pyqcat_visage.gui.widgets.task.create_task_dialog import CreateTaskDialog

from typing import Union, TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class DAGLibraryTreeView(QTreeViewBase):
    """Handles editing and displaying a pyqcat-monster experiment object.

    This class extend the `QTreeView`
    """

    choose_dag_signal = Signal(ModelDag)

    def __init__(self, parent: QWidget, gui: "VisageGUI"):
        super().__init__(parent)
        self.gui = gui
        self.dag_name = None

    def init_right_click_menu(self):
        menu = QMenu(self)

        menu._delete_dag = bind_action(menu, 'Delete', u":/delete.png")
        menu._copy_dag = bind_action(menu, 'Copy', u":/copy.png")
        menu._add_dag = bind_action(menu, 'Add', u":/DAG.png")
        menu._create_task = bind_action(menu, "create_task", u":/tool.png")

        menu._create_task.triggered.connect(self.create_task)
        menu._delete_dag.triggered.connect(self.delete_dag)
        menu._copy_dag.triggered.connect(self.copy_dag)
        menu._add_dag.triggered.connect(self.add_dag)

        self.right_click_menu = menu

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
        if isinstance(node, LeafNode):
            self.choose_dag_signal.emit(node.value)
            self.dag_name = node.label
        return super().mousePressEvent(event)

    def get_position(self, clickedIndex: QPoint):
        """Obtain location of clicked cell in form of row name and number.

        Args:
            clickedIndex (QPoint): The QPoint of the click

        Returns:
            tuple: name, index
        """
        index = self.indexAt(clickedIndex)
        name = 'demo'
        if index.isValid():
            return name, index
        return None, None

    def delete_dag(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            node = model.node_from_index(index)
            dag_name = node.label

            if not model.backend.dags[dag_name].official:
                res = model.backend.db.delete_dag(dag_name)
                if res["code"] == 200:
                    model.backend.dags.pop(dag_name)
                    model.refresh()
                    logger.debug(f'{dag_name} delete success!')
            else:
                QMessageBox().critical(model.parent_widget, 'Error', f'{dag_name} is official!')
        else:
            QMessageBox().critical(model.parent_widget, 'Error', f'Please right click a dag!')

    def copy_dag(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            node = model.node_from_index(index)
            dag_name = node.label

            dag = deepcopy(model.backend.dags.get(dag_name))
            dag.official = False
            dag_name, ok = QInputDialog.getText(
                self, "Dag Name", "Please input new dag name:"
            )
            if ok and dag_name:
                if dag_name in model.backend.dags:
                    QMessageBox().critical(model.parent_widget, 'Error',
                                           f'Dag {dag_name} is existed!')
                else:
                    model.backend.dags[dag_name] = dag
                    dag.name = dag_name
                    model.load()
                    QMessageBox().about(model.parent_widget, 'Ok',
                                        f'add {dag_name} success!')
        else:
            QMessageBox().critical(model.window, 'Error', f'Please right click a dag!')

    def add_dag(self):
        """Add a new empty DAG."""
        model = self.model()
        dag_dict = model.backend.dags
        dag_name, ok = QInputDialog.getText(
            model.parent_widget, "Dag Name", "Please input new dag name:")
        if ok:
            dag = ModelDag(name=dag_name, official=False)
            if dag_name in dag_dict:
                QMessageBox().critical(model.parent_widget, 'FAILED',
                                       f'Dag {dag_name} is existed!')
            else:
                dag_dict[dag_name] = dag
                dag.name = dag_name
                model.load()

    def create_task(self):
        dialog = CreateTaskDialog(gui=self.gui)
        task_dialog = dialog.exec()
        if task_dialog == 1:
            task_name, policy_type, policy_opt = dialog.get_input()
            self.gui.dag_manage.create_task(task_name, policy_type, policy_opt, self.dag_name)
            self.gui.dag_manage.refresh(True)
            self.gui.dag_manage.show()

