# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/10/26
# __author:       YangChao Zhao


from PySide6.QtWidgets import QHeaderView, QMenu

from ..base.right_click_menu import bind_action
from ..base.table_structure import QTableViewBase


class QPointTableView(QTableViewBase):

    def _define_style(self):
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def init_right_click_menu(self):
        menu = QMenu(self)

        menu._create = bind_action(menu, 'Create', u":/cpu.png")
        menu._clear = bind_action(menu, 'Clear', u":/save.png")

        menu._create.triggered.connect(self._create)
        menu._clear.triggered.connect(self._clear)

        self.right_click_menu = menu

    def _create(self):
        self.model().widget.create_custom_point()

    def _clear(self):
        indexes = self.selectedIndexes()
        if indexes:
            model = self.model()
            bit_names = [model.bit_from_index(index) for index in indexes]
            self.model().widget.clear_custom_point(bit_names)
