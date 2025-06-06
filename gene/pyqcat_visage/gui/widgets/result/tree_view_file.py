# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/01/08
# __author:       YangChao Zhao

from PySide6.QtWidgets import QAbstractItemView

from pyqcat_visage.gui.widgets.base.tree_structure import QTreeViewBase


class QTreeViewFileSystem(QTreeViewBase):

    def _define_style(self):
        self.setSelectionMode(QAbstractItemView.SingleSelection)
