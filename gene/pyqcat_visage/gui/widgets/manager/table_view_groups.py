# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/27
# __author:       YangChao Zhao

from PySide6.QtCore import QModelIndex, Signal
from PySide6.QtWidgets import QTableView, QAbstractItemView, QWidget, QHeaderView

from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.table_structure import QTableViewBase


class QTableViewGroupWidget(QTableViewBase, PlaceholderTextWidget):
    choose_group_signal = Signal(str)

    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        QTableViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self,
            "Click Query All Groups to index all groups in database."
        )

    def _define_style(self):
        # Handling selection dynamically
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QTableView.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def view_clicked(self, index: QModelIndex):
        """Select a component and set it in the component widget when you left mouse click.

        In the init, we had to connect with self.clicked.connect(self.viewClicked)

        Args:
            index (QModelIndex): The index
        """
        if not index.isValid():
            return

        model = self.model()
        group = model.group_from_index(index)
        if group:
            self.choose_group_signal.emit(group.get('name'))
