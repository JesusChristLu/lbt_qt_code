# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

from typing import TYPE_CHECKING

from pyqcat_visage.gui.widgets.base.tree_structure import QTreeModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from .tree_view_config import QTreeViewConfig
    from ..system_config import SystemConfigWindow


class QTreeModelConfig(QTreeModelBase):

    def __init__(self, parent: 'SystemConfigWindow', gui: 'VisageGUI', view: 'QTreeViewConfig'):
        """Editable table with drop-down rows for component options. Organized
        as a tree model where child nodes are more specific properties of a
        given parent node.

        Args:
            parent (SystemConfigWindow): The parent widget
            gui (VisageGUI): The main user interface
            view (QTreeViewConfig): View corresponding to a tree structure
        """
        super().__init__(parent, gui, view)
        self.load()

    @property
    def data_dict(self):
        return self._gui.backend.config
