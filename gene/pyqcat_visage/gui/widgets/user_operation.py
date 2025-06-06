# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/27
# __author:       YangChao Zhao

from typing import TYPE_CHECKING, Dict
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QMessageBox

from pyqcat_visage.gui.widgets.dialog.change_password_dialog import ChangePasswordDialog
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.gui.user_manage_ui import Ui_MainWindow
from pyqcat_visage.gui.widgets.manager.table_model_groups import QTableModelGroup
from pyqcat_visage.gui.widgets.manager.table_model_users import QTableModelUser
from pyqcat_visage.gui.widgets.dialog.create_group_dialog import CreateGroupDialog

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class UserManagerWindow(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent=None):
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.gui = gui

        self.cur_group = None
        self.group_list = []
        self.users = {}

        self._setup_table()

        self._ui.table_view_group.choose_group_signal.connect(self._change_group)

        self._create_group_dialog = CreateGroupDialog(parent=self)
        self._change_password_dialog = ChangePasswordDialog(parent=self)

    def _setup_table(self):
        self.table_model_group = QTableModelGroup(
            self.gui, self, self._ui.table_view_group
        )
        self._ui.table_view_group.setModel(self.table_model_group)
        self.table_model_user = QTableModelUser(
            self.gui, self, self._ui.table_view_user
        )
        self._ui.table_view_user.setModel(self.table_model_user)

    @property
    def backend(self):
        return self.gui.backend

    def login(self):
        login_user = self.backend.login_user

        self._ui.username_edit.setText(self.username)
        self._ui.group_edit.setText(self.group_name)
        self._ui.email_edit.setText(login_user.get("email"))
        self._ui.is_admin_check.setChecked(self.is_admin)
        self._ui.is_super_check.setChecked(self.is_super)

        ret_data = self.backend.query_all_groups()
        if ret_data.get("code") == 200:
            self.group_list = ret_data.get("data")
            self.table_model_group.refresh_auto()

    def login_out(self):
        self._ui.username_edit.clear()
        self._ui.group_edit.clear()
        self._ui.email_edit.clear()
        self._ui.is_admin_check.setChecked(False)
        self._ui.is_super_check.setChecked(False)
        self.group_list = []
        self.users = {}
        self.table_model_user.refresh_auto()

    @Slot()
    def query_all_groups(self):
        ret_data = self.backend.query_all_groups()
        if ret_data.get("code") == 200:
            self.group_list = ret_data.get("data")
            self.table_model_group.refresh_auto(check_count=False)

    @Slot()
    def create_group(self):
        self._create_group_dialog.show()
        ret = self._create_group_dialog.exec()

        if int(ret) == 1:
            group_name, description = self._create_group_dialog.get_input()
            ret_data = self.backend.create_group(group_name, description)
            if ret_data.get("code") == 200:
                self.query_all_groups()
            self.handler_ret_data(ret_data)

    def _change_group(self, group_name: str = None):
        group_name = group_name or self.cur_group
        self.cur_group = group_name
        ret_data = self.backend.query_group_info(group_name)
        if ret_data.get("code") == 200:
            self.query_all_groups()
            self.users[group_name] = ret_data.get("data")
            self.table_model_user.group_name = group_name
            self.table_model_user.refresh_auto(check_count=False)
        self.handler_ret_data(ret_data)

    def change_user_group(self, target_user: str):
        name, ok = self.ask_items(
            "Which group do you want to add?",
            [group.get("name") for group in self.group_list],
        )
        if ok:
            ret_data = self.backend.change_user_group(target_user, name)
            self.handler_ret_data(ret_data)
            if ret_data.get("code") == 200:
                self._change_group()

    def change_user_admin(self, user: Dict):
        group_name = user.get("groups")
        username = user.get("username")
        ret_data = None

        if user.get("is_admin"):
            if self.ask_ok(
                f"Are you sure cancel {username} administrator permissions?",
                "Change Administrator",
            ):
                ret_data = self.backend.change_group_leader(username, group_name, False)
        else:
            if self.ask_ok(
                f"Are you sure add {username} administrator permissions?",
                "Change Administrator",
            ):
                ret_data = self.backend.change_group_leader(username, group_name, True)

        if ret_data and ret_data.get("code") == 200:
            self._change_group()

        self.handler_ret_data(ret_data)

    def change_password(self):
        self._change_password_dialog.show()
        ret = self._change_password_dialog.exec()
        if int(ret) == 1:
            password, password_again = self._change_password_dialog.get_input()
            if not password:
                QMessageBox().information(self, "Warning", "pls input new password!")
                self.change_password()
                return
            if password != password_again:
                QMessageBox().information(
                    self, "Warning", "The two passwords are inconsistent!"
                )
                self.change_password()
                return
            ret_data = self.backend.change_password(password)
            self.handler_ret_data(ret_data, show_suc=True)
            if ret_data and ret_data.get("code") != 200:
                self.change_password()

    def add_white_list(self, username: str):
        res = self.backend.db.add_white_list(username)
        self.handler_ret_data(res)

    def remove_white_list(self, username: str):
        res = self.backend.db.remove_white_list(username)
        self.handler_ret_data(res)

    def add_black_list(self, username: str):
        res = self.backend.db.add_black_list(username)
        self.handler_ret_data(res)

    def remove_black_list(self, username: str):
        res = self.backend.db.remove_black_list(username)
        self.handler_ret_data(res)
