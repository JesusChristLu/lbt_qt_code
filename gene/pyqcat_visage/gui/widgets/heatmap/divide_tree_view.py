# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/12/08
# __author:       YangChao Zhao

"""
Tree view for Divide Param Struct Library.
"""

from PySide6.QtWidgets import QHeaderView

from pyqcat_visage.gui.widgets.base.tree_structure import QTreeViewBase


class DivideTreeView(QTreeViewBase):
    """Handles editing and displaying a pyqcat-monster experiment object.

    This class extend the `QTreeView`
    """
    def _define_style(self):
        self.header().setSectionResizeMode(QHeaderView.Stretch)
        self.setToolTip('Select a key to notify heatmap to refresh')
