# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/04/26
# __author:       YangChao Zhao

from PySide6.QtWidgets import QHeaderView, QWidget

from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.table_structure import QTableViewBase


class QTableViewChip(QTableViewBase, PlaceholderTextWidget):

    def __init__(self, parent: QWidget):
        QTableViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self,
            "Click Layout to Set Your Chip (Only Super User)!"
        )

    def _define_style(self):
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
