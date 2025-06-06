# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/27
# __author:       YangChao Zhao

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex

from ..base.table_structure import QTableModelBase

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from ..user_operation import UserManagerWindow
    from .table_view_users import QTableViewUserWidget


class QTableModelUser(QTableModelBase):
    def __init__(
        self,
        gui: "VisageGUI",
        parent: "UserManagerWindow" = None,
        table_view: "QTableViewUserWidget" = None,
    ):
        super().__init__(gui=gui, parent=parent, table_view=table_view)
        self.columns = ["username", "groups", "is_super", "is_admin", "white", "black"]
        self.group_name = None

    @property
    def model_data(self):
        return self.widget.users.get(self.group_name)

    def user_from_index(self, index: QModelIndex):
        return self.model_data[index.row()]

    def _display_data(self, index: QModelIndex):
        user = self.user_from_index(index)
        return user.get(self.columns[index.column()])
