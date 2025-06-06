# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/08/10
# __author:       XuYao
from PySide6.QtCore import Signal, QModelIndex
from PySide6.QtWidgets import QTableView, QAbstractItemView, QWidget, QHeaderView, QMenu
from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.table_structure import QTableViewBase


class QTableViewWorkSpaceNoteWidget(QTableViewBase, PlaceholderTextWidget):
    choose_space_signal = Signal(dict)

    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        self.ui = parent.parent().parent()
        QTableViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self, "Select sample/env_name to query workspace note!"
        )

    @property
    def backend(self):
        """Returns the design."""
        return self.model().backend

    @property
    def gui(self):
        """Returns the GUI."""
        return self.model().gui

    def _define_style(self):
        # Handling selection dynamically
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QTableView.SelectRows)

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

    def init_right_click_menu(self):
        # menu = QMenu(self)
        #
        #
        # self.right_click_menu = menu
        pass

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
        space = model.item_from_index(index)
        if space:
            self.choose_space_signal.emit(space["change"])

    def refresh_view(self):
        if self.his_index:
            self.view_clicked(self.his_index)
