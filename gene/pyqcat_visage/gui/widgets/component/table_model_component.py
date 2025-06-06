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

from PySide6.QtCore import QModelIndex

from ..base.table_structure import QTableModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from .table_view_component import QTableViewComponentWidget
    from ..component_window import ComponentEditWindow


class QTableModelComponent(QTableModelBase):

    def __init__(
            self, gui: 'VisageGUI',
            parent: 'ComponentEditWindow' = None,
            table_view: 'QTableViewComponentWidget' = None
    ):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self.columns = ['Name', 'Last Update', "ID"]

    @property
    def model_data(self):
        return self.backend.components

    def component_from_index(self, index: QModelIndex):
        # return list(self.model_data.values())[index.row()]
        return self.model_data[index.row()]

    def _display_data(self, index: QModelIndex):
        # component_name = list(self.model_data.keys())[index.row()]
        # if index.column() == 0:
        #     return component_name
        # elif index.column() == 1:
        #     return self.backend.components[component_name].update_time

        component = self.component_from_index(index)
        if index.column() == 0:
            return component.name
        elif index.column() == 1:
            return component.update_time
        elif index.column() == 2:
            return component.qid
