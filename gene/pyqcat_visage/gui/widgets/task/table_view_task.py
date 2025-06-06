# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/10/25
# __author:       xw

from PySide6.QtWidgets import QTableView, QAbstractItemView, QWidget, QHeaderView, QMenu
from PySide6.QtCore import Signal, QModelIndex
from ..base.right_click_menu import bind_action
from ..base.table_structure import QTableViewBase
from pyqcat_visage.gui.widgets.task.create_task_dialog import CreateTaskDialog
from ..base.placeholder_text_widget import PlaceholderTextWidget


class QTableViewTaskWidget(QTableViewBase, PlaceholderTextWidget):
    choose_task_signal = Signal(dict)

    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        self.his_index = None
        self.ui = parent.parent().parent()
        QTableViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self, "Select task_name/task_id to query task note!"
        )

    @property
    def gui(self):
        """Returns the GUI."""
        return self.model().gui

    @property
    def backend(self):
        return self.gui.backend

    def _define_style(self):
        # Handling selection dynamically
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QTableView.SelectRows)
        # self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def init_right_click_menu(self):
        menu = QMenu(self)

        menu.delete = bind_action(menu, "Delete", ":/delete.png")
        menu.update = bind_action(menu, "Update", ":/tool.png")
        menu.start = bind_action(menu, "Start", ":/run.png")
        menu.stop = bind_action(menu, "Stop", ":/wifi-off-fill.png")
        menu.delete.triggered.connect(self.delete_task)
        menu.update.triggered.connect(self.update_task)
        menu.start.triggered.connect(self.task_start)
        menu.stop.triggered.connect(self.task_stop)

        self.right_click_menu = menu

    def get_task_data(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            task = model.task_from_index(index)
            return task, index.row()
        else:
            task = {}
            return task, None

    def delete_task(self):
        task, index = self.get_task_data()
        if index is not None:
            self.model().widget.delete_task(task, index)

    def update_task(self):
        task, *_ = self.get_task_data()
        sub_name = task.get("sub_name")
        sub_type = task.get("sub_type")
        dialog = CreateTaskDialog(gui=self.gui, sub_type=sub_type)
        dialog.ui.lineEdit.setText(task.get("task_name"))
        policy_type = task["policy"].get("type")
        dialog.ui.DagPolicyText.setCurrentText(policy_type)
        dialog.update_task(task)
        task_dialog = dialog.exec()
        if task_dialog == 1:
            task_name, policy_type, policy_opt = dialog.get_input()
            self.model().widget.update_task(task_name=task_name,
                                            task_id=task.get("id"),
                                            enable=task.get("enable"),
                                            policy_type=policy_type,
                                            policy_opt=policy_opt,
                                            sub_name=sub_name,
                                            sub_type=sub_type)

    def task_start(self):
        tasks, *_ = self.get_task_data()
        tasks.update({"enable": True})
        self.gui.backend.db.update_custom_task(tasks)
        polict_type = tasks["policy"].get("type")
        if polict_type == "repeat":
            self.backend.run_sweep_dag(tasks)
        else:
            task_id = tasks.get("id")
            self.backend.remove_one_cron(cron_id=task_id)
            self.backend.run_cron(tasks)

    def task_stop(self):
        tasks, *_ = self.get_task_data()
        task_id = tasks.get("id")
        tasks.update({"enable": False})
        self.gui.backend.db.update_custom_task(tasks)
        self.backend.remove_one_cron(cron_id=task_id)

    def update_column_widths(self):
        total_width = self.width()
        if self.model().columns_ratio:
            total_weight = sum(self.model().columns_ratio)
            for i, width in enumerate(self.model().columns_ratio):
                new_width = int((width / total_weight) * total_width)
                self.setColumnWidth(i, new_width)
        else:
            self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_column_widths()

    def view_clicked(self, index: QModelIndex):
        """Select a component and set it in the component widget when you left mouse click.

        In the init, we had to connect with self.clicked.connect(self.viewClicked)

        Args:
            index (QModelIndex): The index
        """

        self.his_index = index

        if self.gui is None or not index.isValid():
            return

        model = self.model()
        task_info = model.task_from_index(index)
        if task_info:
            self.choose_task_signal.emit(task_info)

    def refresh_view(self):
        if self.his_index:
            self.view_clicked(self.his_index)
            