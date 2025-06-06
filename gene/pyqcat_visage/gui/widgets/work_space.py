# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/10/18
# __author:       XuYao

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox

from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.gui.work_sapce_ui import Ui_MainWindow

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class WorkSpaceWindow(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent=None):
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.gui = gui

        self.bit_names = []
        self.conf_names = []
        self.bit_range = []
        self.conf_range = []
        self.point_names = []
        self.point_range = []
        self.attr_names = []
        self.attr_range = []
        self.auto_pull = 0
        self.auto_push = 0

    @property
    def ui(self):
        return self._ui

    def load_all_data(self):
        ret_data = self.gui.backend.db.workspace_info()
        if ret_data.get("code") == 200:
            self.bit_names = ret_data["data"].get("bit_names", [])
            self.conf_names = ret_data["data"].get("conf_names", [])
            self.bit_range = ret_data["data"].get("bit_range", [])
            self.conf_range = ret_data["data"].get("conf_range", [])
            self.point_range = ret_data["data"].get("point_range", [])
            self.point_names = ret_data["data"].get("point_names", [])
            self.auto_pull = ret_data["data"].get("auto_pull")
            self.auto_push = ret_data["data"].get("auto_push")
            self.attr_range = ret_data["data"].get("attr_range", [])
            self.attr_names = ret_data["data"].get("attr_names", [])
            self.ui.QubitBox.set_units(self.bit_range)
            self.ui.QubitBox.setCurrentText(self.bit_names)
            self.ui.ConfigBox.set_units(self.conf_range)
            self.ui.ConfigBox.setCurrentText(self.conf_names)
            self.ui.PointBox.set_units(self.point_range)
            self.ui.PointBox.setCurrentText(self.point_names)
            self.ui.AttrBox.set_units(self.attr_range)
            self.ui.AttrBox.setCurrentText(self.attr_names)

            self.ui.autoPullCheck.setChecked(bool(self.auto_pull))
            self.ui.autoPushCheck.setChecked(bool(self.auto_push))
            self.gui.add_space_action(auto_pull=self.auto_pull, auto_push=self.auto_push)
        return ret_data

    def change_auto_option(self):
        auto_pull = int(self.ui.autoPullCheck.isChecked())
        auto_push = int(self.ui.autoPushCheck.isChecked())

        ret_data = self.gui.backend.db.workspace_set_auto(auto_push=auto_push,
                                                          auto_pull=auto_pull)
        self.handler_ret_data(ret_data)
        if ret_data.get("code") == 200:
            if auto_push != self.auto_push:
                self.auto_push = auto_push
            if auto_pull != self.auto_pull:
                self.auto_pull = auto_pull
            self.gui.add_space_action(auto_pull=auto_pull, auto_push=auto_push)

    def pull_data(self):
        self.ui.pullButton.setDisabled(True)
        if self.ask_ok(
                "Are you sure to <strong style='color:red'>PULL</strong> the online workspace? "
                "which overwrites the local workspace.", "WorkSpace Message",
        ):
            ret_data = self.gui.backend.db.workspace_pull_push(True)
            self.handler_ret_data(ret_data, show_suc=True)
        self.ui.pullButton.setDisabled(False)
        self.gui.ui.auto_new_pig.hide()

    def push_data(self):
        self.ui.pushButton.setDisabled(True)
        if self.ask_ok(
                "Are you sure to <strong style='color:red'>PUSH</strong> the local workspace?",
                "WorkSpace Message",
        ):
            ret_data = self.gui.backend.db.workspace_pull_push(False)
            self.handler_ret_data(ret_data, show_suc=True)
        self.ui.pushButton.setDisabled(False)

    def refresh_query_data(self):
        ret_data = self.load_all_data()
        self.gui.refresh_workspace_cache(ret_data)

    def save_space_conf(self):
        bit_names = self.ui.QubitBox.currentText()
        conf_names = self.ui.ConfigBox.currentText()
        point_label = self.ui.PointBox.currentText()
        attr_names = self.ui.AttrBox.currentText()
        if not bit_names and not conf_names:
            QMessageBox.warning(self, "Warning", "pls select qubits or configs!")
            return
        if not point_label:
            QMessageBox.warning(self, "Warning", "pls select point_label!")
            return
        rest_data = self.gui.backend.db.save_workspace_info(bit_names, conf_names, point_label, attr_names)
        if self.gui.backend.config.system.point_label in point_label:
            self.gui.ui.tabTopology.topology_view.workspace_change.emit(bit_names)
        self.handler_ret_data(rest_data, show_suc=True)


