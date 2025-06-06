# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/10/18
# __author:       XuYao
import re
from typing import TYPE_CHECKING, Dict, List
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QMessageBox

from pyqcat_visage.gui.widgets.dialog.create_storm_dialog import CreateStormDialog
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.gui.storm_manage_ui import Ui_MainWindow

from pyqcat_visage.gui.widgets.chip_manage_files.table_model_storm import QTableModelStormManage

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class StormManagerWindow(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent=None):
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.gui = gui

        self.cur_group = None
        self.data = []
        self.cache_sample_data = {}

        self._setup_table()

        self._create_storm_dialog = CreateStormDialog(parent=self)

    @property
    def ui(self):
        return self._ui

    def reset_window_layout(self):
        if self.is_super or self.is_admin:
            self.ui.actionCreateStom.setVisible(True)
            self.ui.tableStormView.right_click_menu.start.setVisible(True)
            self.ui.tableStormView.right_click_menu.stop.setVisible(True)
            self.ui.tableStormView.right_click_menu.delete.setVisible(True)
        else:
            self.ui.actionCreateStom.setVisible(False)
            self.ui.tableStormView.right_click_menu.start.setVisible(False)
            self.ui.tableStormView.right_click_menu.stop.setVisible(False)
            self.ui.tableStormView.right_click_menu.delete.setVisible(False)

    def _setup_table(self):
        self.table_model_chip = QTableModelStormManage(
            self.gui, self, self._ui.tableStormView
        )
        self._ui.tableStormView.setModel(self.table_model_chip)

    def load_sample_cache(self):
        ret_data = self.gui.backend.db.query_storm_sample_data()
        if ret_data.get("code") == 200:
            self.cache_sample_data = ret_data["data"]

    def load_all_data(self):
        self.clear_cache_data()
        self.load_sample_cache()
        sample_list = list(self.cache_sample_data.keys())
        self.ui.SampleContent.addItems(sample_list)
        self.ui.EnvContent.addItems(self.cache_sample_data.get("", []))

    def clear_cache_data(self):
        self.ui.SampleContent.clear()

    def create_storm(self):
        self._create_storm_dialog.show()
        ret = self._create_storm_dialog.exec()
        if int(ret) == 1:
            (
                sample,
                env_name
            ) = self._create_storm_dialog.get_input()
            if not sample:
                QMessageBox().information(self, "Warning", "pls input sample!")
                self.create_storm()
                return
            if not env_name:
                QMessageBox().information(self, "Warning", "pls input env_name!")
                self.create_storm()
                return
            ret_data = self.gui.backend.db.update_storm(sample, env_name)
            self.handler_ret_data(ret_data, show_suc=True)
            if ret_data and ret_data.get("code") != 200:
                self.create_storm()

    def query_storm(self):
        sample = self._ui.SampleContent.currentText()
        env_name = self._ui.EnvContent.currentText()
        ret_data = self.gui.backend.db.query_storm_list(sample, env_name)
        if ret_data.get("code") in (200, 404):
            self.data = ret_data["data"]
            # print(f"query_chip refresh auto..{self.chips}")
            self.table_model_chip.refresh_auto(False)

    def save_storm(self, sample: str, env_name: str):
        ret_data = self.gui.backend.db.update_storm(sample, env_name)
        self.handler_ret_data(ret_data)

    def delete_storm(self, sample: str, env_name: str, index: int):
        if self.ask_ok(
            "Are you sure to <strong style='color:red'>delete</strong> the storm? "
            "This operation will not be recoverable.",
            "Visage Message",
        ):
            ret_data = self.gui.backend.db.delete_storm(sample, env_name)
            self.handler_ret_data(ret_data)
            if ret_data.get("code") == 200:
                self.table_model_chip.removeRows(index)

    def control_storm_server(self, sample: str, env_name: str, option: str):
        if self.ask_ok(
            f"Are you sure to <strong style='color:red'>{option}</strong> Storm-Server ? "
            "This operation will not be recoverable.",
            f"{option} Storm-Server",
        ):
            ret_data = self.gui.backend.db.control_storm(sample, env_name, option)
            self.handler_ret_data(ret_data)

    def refresh(self):
        self.cache_sample_data = {}
        self.load_all_data()

    @staticmethod
    def _get_chip_name(sample: str, env_name: str):
        return f"{sample}_|_{env_name}"

    def sample_change(self):
        sample = self.ui.SampleContent.currentText()
        env_list = self.cache_sample_data.get(sample, [])
        self.ui.EnvContent.clear()
        self.ui.EnvContent.addItems(env_list)
