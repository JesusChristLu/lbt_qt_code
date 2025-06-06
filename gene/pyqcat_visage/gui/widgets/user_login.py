# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/14
# __author:       HanQing Shi, YangChao Zhao

import os
import pickle
from collections import OrderedDict
from typing import TYPE_CHECKING

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QMessageBox
from loguru import logger
from pyQCat.invoker import DataCenter
from pyQCat import get_version

from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.gui.tools.utilies import slot_catch_exception
from pyqcat_visage.gui.user_login_ui import Ui_MainWindow
from pyqcat_visage.gui.widgets.dialog.tips_dialog import TipsDialog
from pyqcat_visage.gui.widgets.title_window import TitleWindow

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class UserLoginWindow(TitleWindow):

    def __init__(self, gui: 'VisageGUI', parent=None):
        super().__init__(parent)
        self.gui = gui
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self._ui.stackedWidget.setCurrentIndex(0)
        self.group_list = ["normal"]

        # load his env
        his_env = self.gui.backend.get_his_env()
        if his_env.invoker_addr:
            addr = his_env.invoker_addr.split(':')
            ip = addr[1][2:]
            port = addr[-1]
            self._ui.ip_edit.setText(ip)
            self._ui.port_edit.setText(port)

        # load cache user
        self.user_map = {}
        if os.path.exists(GUI_CONFIG.cache_user_bin):
            with open(GUI_CONFIG.cache_user_bin, 'rb') as f:
                user_map_bin = f.read()
            user_map = pickle.loads(user_map_bin)
            users = list(user_map.keys())

            self._ui.user_name_box.addItems(users)
            self._ui.user_name_box.setCurrentIndex(len(users) - 1)
            self._ui.pwd_linedit_login.setText(user_map.get(users[0]))

            self.user_map = user_map

    def load_group_names(self):
        db = DataCenter()
        ret_data = db.query_group_names()
        print(ret_data)
        if ret_data.get("code") == 200:
            group_list = ret_data["data"]
            if "super" in group_list:
                group_list.remove("super")
            if group_list:
                self.group_list = group_list
        self._ui.GroupComboBox.clear()
        self._ui.GroupComboBox.addItems(self.group_list)

    def auto_login(self):
        if not self.test_connect():
            self._ui.stackedWidget.setCurrentIndex(0)
            self.show()
            return

        self.take_connect()

        self._load_cache_user()

        if not self.login_to_system():
            self._ui.stackedWidget.setCurrentIndex(1)
            self.show()
            return

        logger.log('FLOW', f'The system is starting, please wait...')
        self.gui.main_window.setEnabled(False)

    @Slot()
    def create_account(self):
        username = self._ui.user_name_lineEdit_register.text()
        group = self._ui.GroupComboBox.currentText()
        password = self._ui.pwd_lineEdit_register.text()
        rep_pwd = self._ui.rpwd_lineEdit.text()
        email = self._ui.mail_lineEdit_register.text()
        if not group:
            QMessageBox.warning(self, "Fail", "pls select group first!")
            return
        response = self.gui.backend.register(username, password, rep_pwd, email, group)

        if response.get('code') == 200:
            QMessageBox().information(self, 'Success', f'Register {username} success!')
            self._ui.user_name_box.addItems([username])
            count = self._ui.user_name_box.count()
            self._ui.user_name_box.setCurrentIndex(count - 1)
            self._ui.pwd_linedit_login.setText(password)
            self.user_map.update({username: password})
            self._ui.stackedWidget.setCurrentIndex(1)
        else:
            QMessageBox().critical(self, 'Error', response.get('msg'))

    @Slot()
    def test_connect(self):
        ip = self._ui.ip_edit.text()
        port = self._ui.port_edit.text()
        if ip and port:
            invoker_addr = f"tcp://{ip}:{port}"
            self._ui.state_label.setText('Wait..')
            self._ui.state_label.setStyleSheet("QLabel[objectName='state_label']{color: #6ddf6d}")
            self._ui.state_label.repaint()
            if self.gui.backend.test_connect(ip, port):
                self.gui.backend.invoker_addr = invoker_addr
                self._ui.state_label.setText('SUC')
                self._ui.state_label.setStyleSheet("QLabel[objectName='state_label']{color: green}")
                return True
            else:
                self._ui.state_label.setText('FAIL')
                self._ui.state_label.setStyleSheet("QLabel[objectName='state_label']{color: red}")

    @Slot()
    def take_connect(self):
        if self.test_connect():
            ip = self._ui.ip_edit.text()
            port = self._ui.port_edit.text()
            self.gui.backend.set_env(ip, port)
            self._ui.stackedWidget.setCurrentIndex(1)
        else:
            QMessageBox().critical(self, 'Error', f'Connect Failed!')

    @slot_catch_exception()
    def login_to_system(self):
        is_remember = self._ui.is_remember_box.isChecked()
        username = self._ui.user_name_box.currentText()
        password = self._ui.pwd_linedit_login.text()
        check_res = self.gui.backend.db.check_user_version(username, get_version())
        if check_res.get("code") >= 400:
            tip = TipsDialog()
            tip.set_tips(check_res.get("msg"))
            ret = tip.exec()
            if int(ret) != 1:
                return False

        response = self.gui.backend.login(username, password)
        if username == "":
            username = response.data.username

        if response.get('code') == 200:
            self.gui.backend.user_state = True
            cache_user_path = os.path.join(GUI_CONFIG.cache_user_path, username)
            if not os.path.exists(cache_user_path):
                os.makedirs(cache_user_path)
            self.gui.backend.cache_user_path = cache_user_path
            if is_remember:
                self._remember(username, password)
            self.gui.backend.context_builder.username = username
            self.gui.backend.init_config()
            self.gui.init_visage()
            self.close_()
            self.gui.main_window.show()
            return True
        else:
            self.handler_ret_data(response)

    @Slot()
    def show_create_account(self):
        self.load_group_names()
        self._ui.stackedWidget.setCurrentIndex(2)

    @Slot()
    def back_up_page(self):
        index = self._ui.stackedWidget.currentIndex()
        if index == 3:
            self._ui.stackedWidget.setCurrentIndex(1)
        else:
            self._ui.stackedWidget.setCurrentIndex(index - 1)

    @Slot(int)
    def choose_user(self, index):
        if index != -1 and self.user_map:
            username = self._ui.user_name_box.itemText(index)
            password = self.user_map.get(username)
            self._ui.pwd_linedit_login.setText(password)

    @staticmethod
    def _remember(username: str, password: str):
        if not os.path.exists(GUI_CONFIG.cache_user_bin):
            user_map = OrderedDict()
            user_map[username] = password
        else:
            with open(GUI_CONFIG.cache_user_bin, 'rb') as f:
                user_map_bin = f.read()
            user_map = pickle.loads(user_map_bin)
            if username in user_map:
                user_map.pop(username)
            user_map[username] = password

        user_map_bin = pickle.dumps(user_map)
        with open(GUI_CONFIG.cache_user_bin, 'wb') as f:
            f.write(user_map_bin)

    def _load_cache_user(self):
        if os.path.exists(GUI_CONFIG.cache_user_bin):
            with open(GUI_CONFIG.cache_user_bin, 'rb') as f:
                user_map_bin = f.read()
            user_map = pickle.loads(user_map_bin)
            if len(user_map) > 0:
                name, pwd = list(user_map.items())[-1]
                self._ui.user_name_box.setCurrentText(name)
                self._ui.pwd_linedit_login.setText(pwd)

    @Slot()
    def forget_password_link(self):
        self._ui.user_name_lineEdit_find.setText(self._ui.user_name_box.currentText())
        self._ui.stackedWidget.setCurrentIndex(3)

    @Slot()
    def find_account(self):
        username = self._ui.user_name_lineEdit_find.text()
        if not username:
            QMessageBox().critical(self, 'Error', f'username is empty!')
            return False

        pre_pwd = self._ui.pre_pwd_edit.text()
        new_pwd = self._ui.new_pwd_edit.text()
        email = self._ui.mail_lineEdit_find.text()

        res = self.gui.backend.find_account(username, pre_pwd=pre_pwd, new_pwd=new_pwd, email=email)

        if res.get('code') == 200:
            QMessageBox().about(self, 'Success', f'Find Success!')
            self._ui.user_name_box.setCurrentText(username)
            self._ui.pwd_linedit_login.setText(new_pwd)
            self._ui.stackedWidget.setCurrentIndex(1)
        else:
            QMessageBox().critical(self, 'Error', res.get('msg'))

    def login_out(self, courier_exit: bool = False):
        self.gui.backend.login_out(courier_exit)
        self._ui.stackedWidget.setCurrentIndex(0 if courier_exit else 1)
        self.show()
