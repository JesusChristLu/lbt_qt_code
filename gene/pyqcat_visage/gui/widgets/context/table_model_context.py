# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/16
# __author:       YangChao Zhao


from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex

from ..base.table_structure import QTableModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from .table_view_context import QTableViewContextWidget
    from ..context_window import ContextEditWindow


class QTableModelContext(QTableModelBase):

    def __init__(
            self, gui: 'VisageGUI',
            parent: 'ContextEditWindow' = None,
            table_view: 'QTableViewContextWidget' = None
    ):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self.columns = ['Name', 'Describe']

    @property
    def model_data(self):
        return self.backend.view_context

    def component_from_index(self, index: QModelIndex):
        return list(self.backend.view_context.items())[index.row()]

    def _display_data(self, index: QModelIndex):
        row = index.row()
        column = index.column()

        component_name = list(self.model_data.keys())[row]
        component = self.model_data.get(component_name)

        if column == 0:
            return str(component_name)
        elif column == 1:
            if component_name == 'crosstalk':
                shape = len(component.get('infos'))
                return f'Matrix: {shape} × {shape}'
            elif component_name == 'working_dc':
                return f'Channel Total {len(component)}'
            elif component_name == 'env_bits':
                env_bits = component.get('env_bits')
                if len(env_bits) > 6:
                    return f'{env_bits[0]} - {env_bits[-1]} | total {len(env_bits)}'
                return str(env_bits)
            else:
                return str(component)
