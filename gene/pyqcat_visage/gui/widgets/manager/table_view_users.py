# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/27
# __author:       YangChao Zhao

from PySide6.QtWidgets import QTableView, QAbstractItemView, QWidget, QHeaderView, QMenu

from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.right_click_menu import bind_action, bind_menu_action
from ..base.table_structure import QTableViewBase


class QTableViewUserWidget(QTableViewBase, PlaceholderTextWidget):
    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        self.ui = parent.parent().parent()
        QTableViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self, "Click A Groups to index all user in this group."
        )

    def _define_style(self):
        # Handling selection dynamically
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QTableView.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def init_right_click_menu(self):
        menu = QMenu(self)

        menu._change_group = bind_action(menu, "Change Group", ":/build.png")
        menu._change_admin = bind_action(menu, "Add/Cancel Admin", ":/config.png")

        menu_white_list = QMenu(self)
        menu_white_list.setTitle("WhiteList")
        menu_white_list.add_white = bind_action(menu_white_list, " add whiteList")
        menu_white_list.delete_white = bind_action(menu_white_list, " del whiteList")

        menu_white_list.add_white.triggered.connect(self.add_white_list)
        menu_white_list.delete_white.triggered.connect(self.remove_white_list)
        bind_menu_action(menu, menu_white_list)

        menu_black_list = QMenu(self)
        menu_black_list.setTitle("BlackList")
        menu_black_list.add_black = bind_action(menu_black_list, " add blackList")
        menu_black_list.delete_black = bind_action(menu_black_list, " del blackList")

        menu_black_list.add_black.triggered.connect(self.add_black_list)
        menu_black_list.delete_black.triggered.connect(self.remove_black_list)
        bind_menu_action(menu, menu_black_list)

        menu._change_group.triggered.connect(self.change_user_group)
        menu._change_admin.triggered.connect(self.change_user_admin)

        self.right_click_menu = menu

    def change_user_group(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            user = model.user_from_index(index)
            username = user.get("username")
            model.widget.change_user_group(username)

    def change_user_admin(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            user = model.user_from_index(index)
            model.widget.change_user_admin(user)

    def add_white_list(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            user = model.user_from_index(index)
            username = user.get("username")
            model.widget.add_white_list(username)

    def remove_white_list(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            user = model.user_from_index(index)
            username = user.get("username")
            model.widget.remove_white_list(username)

    def add_black_list(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            user = model.user_from_index(index)
            username = user.get("username")
            model.widget.add_black_list(username)

    def remove_black_list(self):
        index = self.selectedIndexes()
        model = self.model()
        if index:
            index = index[0]
            user = model.user_from_index(index)
            username = user.get("username")
            model.widget.remove_black_list(username)
