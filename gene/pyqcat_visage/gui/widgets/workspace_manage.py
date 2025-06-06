# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/7/31
# __author:       XuYao

from typing import TYPE_CHECKING, Dict, List

from pyqcat_visage.gui.widgets.dialog.copy_workspace_dialog import CopyWorkSpaceDialog
from pyqcat_visage.gui.widgets.dialog.create_workspace_dialog import (
    CreateWorkSpaceDialog,
)
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.gui.workspace_manage_ui import Ui_MainWindow
from pyqcat_visage.gui.widgets.chip_manage_files.table_model_workspace import (
    QTableModelWorkSpace,
)

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class WorkSpaceManageWindow(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent=None):
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.gui = gui

        self.cur_group = None
        self.chips = []
        self.data = []
        self.group_list = []
        self.user_list = []
        self.cache_chip_sample_data = {}
        self.cache_chip_line_data = {}
        # self.refresh_user_info()

        self._setup_table()

        self._create_workspace_dialog = CreateWorkSpaceDialog(parent=self)
        self._copy_workspace_dialog = CopyWorkSpaceDialog(parent=self)

    @property
    def ui(self):
        return self._ui

    def _setup_table(self):
        self.table_model_workspace = QTableModelWorkSpace(
            self.gui, self, self._ui.tableWorkSpaceView
        )
        self._ui.tableWorkSpaceView.setModel(self.table_model_workspace)

    def load_sample_cache(self):
        ret_data = self.gui.backend.db.query_chip_sample_data()
        if ret_data.get("code") == 200:
            self.cache_chip_sample_data = ret_data["data"]

    def load_user_data(self):
        if self.is_super:
            ret_data = self.gui.backend.db.query_usernames()
        else:
            ret_data = self.gui.backend.db.query_usernames(self.groups)
        if ret_data.get("code") == 200:
            self.user_list = [""] + ret_data["data"]
            self.ui.workUserContent.addItems(self.user_list)
            self._create_workspace_dialog.ui.UserText.addItems(self.user_list)
        self.load_sample_cache()

    def load_all_data(self):
        self.clear_cache_data()
        self.load_user_data()
        sample_list = list(self.cache_chip_sample_data.keys())
        self.ui.workSampleContent.addItems(sample_list)
        self.ui.workEnvContent.addItems(self.cache_chip_sample_data.get("", []))

    def clear_cache_data(self):
        self.user_list.clear()
        self.group_list.clear()
        self.ui.workUserContent.clear()
        self.ui.workSampleContent.clear()
        self._create_workspace_dialog.ui.UserText.clear()

    def query_workspace(self):
        username = self._ui.workUserContent.currentText()
        sample = self._ui.workSampleContent.currentText()
        env_name = self._ui.workEnvContent.currentText()
        ret_data = self.gui.backend.db.query_workspace(username, sample, env_name)
        if ret_data.get("code") in (200, 404):
            self.data = ret_data["data"]
            for space in self.data:
                space["bit_names"] = ",".join(space.get("bit_names", []))
                space["conf_names"] = ",".join(space.get("conf_names", []))
            self.table_model_workspace.refresh_auto(False)

    def disable_workspace_option(self, option: bool):
        self._create_workspace_dialog.ui.UserText.setDisabled(option)
        self._create_workspace_dialog.ui.SampleText.setDisabled(option)
        self._create_workspace_dialog.ui.EnvText.setDisabled(option)

    def creat_work_space(self):
        self._create_workspace_dialog.setWindowTitle("Create WorkSpace")
        self.disable_workspace_option(False)
        self._create_workspace_dialog.ui.UserText.setCurrentText("")
        self._create_workspace_dialog.ui.SampleText.clear()
        self._create_workspace_dialog.ui.EnvText.clear()
        self._create_workspace_dialog.ui.qubitText.clear()
        self._create_workspace_dialog.ui.configText.clear()
        self._create_workspace_dialog.ui.extraBitText.clear()
        self._create_workspace_dialog.ui.extraConfigText.clear()
        while True:
            self._create_workspace_dialog.show()
            ret = self._create_workspace_dialog.exec()
            if int(ret) == 1:
                res = self._save_workspace()
                if res == 200:
                    break
                continue
            else:
                break

    def _save_workspace(self):
        (
            username,
            sample,
            env_name,
            qubit_names,
            config_names,
        ) = self._create_workspace_dialog.get_input()
        ret_data = self.gui.backend.db.update_workspace(
            username, sample, env_name, qubit_names, config_names
        )
        self.handler_ret_data(ret_data, show_suc=True)
        code = ret_data.get("code")
        if code == 200:
            self._create_workspace_dialog.ui.extraBitText.clear()
            self._create_workspace_dialog.ui.extraConfigText.clear()
            for space in self.data:
                if (
                    space["username"] == username
                    and space["sample"] == sample
                    and space["env_name"] == env_name
                ):
                    space["bit_names"] = ",".join(qubit_names).strip(",")
                    space["conf_names"] = ",".join(config_names).strip(",")
                    break
        return code

    def update_workspace(
        self,
        username: str,
        sample: str,
        env_name: str,
        qubit_names: str,
        config_names: str,
    ):
        self.disable_workspace_option(True)
        self._create_workspace_dialog.setWindowTitle("Update WorkSpace")
        self._create_workspace_dialog.ui.UserText.setCurrentText(username)
        self._create_workspace_dialog.ui.SampleText.setCurrentText(sample)
        self._create_workspace_dialog.ui.EnvText.setCurrentText(env_name)
        qubits = self._create_workspace_dialog.ui.qubitText.multi_select
        qubit_flag = False
        config_flag = False
        for name in qubit_names.strip(" ").split(","):
            if name not in qubits:
                qubits.append(name)
                qubit_flag = True
        configs = self._create_workspace_dialog.ui.configText.multi_select
        for conf in config_names.strip(" ").split(","):
            if conf not in configs:
                configs.append(conf)
                config_flag = True
        if qubit_flag:
            self._create_workspace_dialog.ui.qubitText.loadItems()
        if config_flag:
            self._create_workspace_dialog.ui.configText.loadItems()
        self._create_workspace_dialog.ui.qubitText.setCurrentText(qubit_names)
        self._create_workspace_dialog.ui.configText.setCurrentText(config_names)
        self._create_workspace_dialog.ui.extraBitText.clear()
        self._create_workspace_dialog.ui.extraConfigText.clear()
        while True:
            self._create_workspace_dialog.show()
            ret = self._create_workspace_dialog.exec()
            if int(ret) == 1:
                res = self._save_workspace()
                if res == 200:
                    break
                continue
            else:
                break

    def delete_workspace(self, username: str, sample: str, env_name: str, index: int):
        if self.ask_ok(
            "Are you sure to <strong style='color:red'>delete</strong> the workspace? "
            "This operation will not be recoverable.",
            "Visage Message",
        ):
            ret_data = self.gui.backend.db.delete_workspace(username, sample, env_name)
            self.handler_ret_data(ret_data)
            if ret_data.get("code") == 200:
                self.table_model_workspace.removeRows(index)

    def refresh(self):
        self.cache_chip_sample_data = {}
        self.cache_chip_line_data = {}
        self.load_all_data()

    @staticmethod
    def _get_chip_name(sample: str, env_name: str):
        return f"{sample}_|_{env_name}"

    def query_chip_line_data(self, sample: str, env_name: str):
        chip_name = self._get_chip_name(sample, env_name)
        data = self.cache_chip_line_data.get(chip_name)
        if not data:
            ret_data = self.gui.backend.db.query_chip_line_space(sample, env_name)
            if ret_data.get("code") == 200:
                data = ret_data["data"]
                qubits = data.get("bit_names")
                file_format = data.get("file_format")
                conf_names = []
                for file_str in file_format:
                    for bit in qubits:
                        conf_names.append(file_str.format(bit))
                conf_names += data.get("file", [])
                data["conf_names"] = conf_names
                self.cache_chip_line_data[chip_name] = data
        return data

    def space_sample_change(self):
        sample = self.ui.workSampleContent.currentText()
        env_list = self.cache_chip_sample_data.get(sample, [])
        self.ui.workEnvContent.clear()
        self.ui.workEnvContent.addItems(env_list)

    def copy_space(self):
        self._copy_workspace_dialog.init_query_data()
        while True:
            self._copy_workspace_dialog.show()
            ret = self._copy_workspace_dialog.exec()
            if int(ret) == 1:
                res = self.gui.backend.db.copy_workspace(*self._copy_workspace_dialog.get_input())
                self.handler_ret_data(res, show_suc=True)
            else:
                break
