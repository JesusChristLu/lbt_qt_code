# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/08/1
# __author:       XuYao
from functools import partial

from PySide6.QtWidgets import QTableView, QAbstractItemView, QWidget, QHeaderView, QMenu
from PySide6.QtCore import Signal, QModelIndex
from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.right_click_menu import bind_action
from ..base.table_structure import QTableViewBase


class QTableViewChimeraWidget(QTableViewBase, PlaceholderTextWidget):
    choose_item_signal = Signal(dict)

    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        self.ui = parent.parent().parent()
        QTableViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(self, "Select sample/env_name to query chips!")

    def _define_style(self):
        # Handling selection dynamically
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QTableView.SelectRows)
        # self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    @property
    def backend(self):
        """Returns the design."""
        return self.model().backend

    @property
    def gui(self):
        """Returns the GUI."""
        return self.model().gui

    def init_right_click_menu(self):
        menu = QMenu(self)

        menu.save = bind_action(menu, "Save", ":/save.png")
        menu.delete = bind_action(menu, "Delete", ":/delete.png")
        menu.addSeparator()
        menu.reconnect = bind_action(menu, "Reconnect", ":/loading.png")
        menu.start = bind_action(menu, "Start", ":/run.png")
        menu.stop = bind_action(menu, "Stop", ":/wifi-off-fill.png")
        menu.restart = bind_action(menu, "Restart", ":/reset.png")

        menu.save.triggered.connect(self.update_chip)
        menu.delete.triggered.connect(self.delete_chip)
        menu.reconnect.triggered.connect(partial(self.control_qs, "reconnect"))
        menu.start.triggered.connect(partial(self.control_qs, "start"))
        menu.stop.triggered.connect(partial(self.control_qs, "stop"))
        menu.restart.triggered.connect(partial(self.control_qs, "restart"))

        self.right_click_menu = menu

    def get_chip_data(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            chip = model.item_from_index(index)
        else:
            chip = {}
        sample = chip.get("sample")
        env_name = chip.get("env_name")
        inst_ip = chip.get("inst_ip")
        inst_port = chip.get("inst_port")
        groups = chip.get("groups")
        core_num = chip.get("core_num", "1")
        md5_id = chip.get("md5_id", "")
        debug = chip.get("debug", "0")
        window_size = chip.get("window_size", "10")
        alert_dis = chip.get("alert_dis", "1")
        secure_dis = chip.get("secure_dis", "2")
        return (
            sample,
            env_name,
            inst_ip,
            inst_port,
            groups,
            core_num,
            debug,
            window_size,
            alert_dis,
            secure_dis,
            md5_id,
            index.row(),
        )

    def update_chip(self):
        (
            sample,
            env_name,
            inst_ip,
            inst_port,
            groups,
            core_num,
            debug,
            window_size,
            alert_dis,
            secure_dis,
            *_,
        ) = self.get_chip_data()
        if sample and env_name:
            self.model().widget.save_chip(
                sample, env_name, inst_ip, inst_port, groups, core_num,
                debug, window_size, alert_dis, secure_dis
            )

    def delete_chip(self):
        sample, env_name, *_, index = self.get_chip_data()
        if sample and env_name:
            self.model().widget.delete_chip(sample, env_name, index)

    def control_qs(self, option: str):
        sample, env_name, *_ = self.get_chip_data()
        if sample and env_name:
            self.model().widget.control_qs_server(sample, env_name, option)

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
        item = model.item_from_index(index)
        if item:
            self.choose_item_signal.emit(item)

    def refresh_view(self):
        if self.his_index:
            self.view_clicked(self.his_index)


