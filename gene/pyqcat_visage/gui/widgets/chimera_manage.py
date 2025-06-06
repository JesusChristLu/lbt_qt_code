# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/7/31
# __author:       XuYao
import re
from typing import TYPE_CHECKING, Dict, List, Union
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QMessageBox

from pyqcat_visage.gui.widgets.chip_manage_files.tree_model_chimera import QTreeModelChimera
from pyqcat_visage.gui.widgets.component.tree_delegate_options import QOptionsDelegate
from pyqcat_visage.gui.widgets.dialog.create_chimera_dialog import CreateChipDialog
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.gui.chimera_manage_ui import Ui_MainWindow

from pyqcat_visage.gui.widgets.chip_manage_files.table_model_chimera import (
    QTableModelChipManage,
)

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class ChimeraManagerWindow(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent=None):
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.gui = gui

        self.cur_group = None
        self.data = []
        self._item = {}
        self.workspaces = []
        self.groups_list = []
        self.user_list = []
        self.cache_chip_sample_data = {}
        self.cache_chip_line_data = {}
        self.core_num_select = list(str(i) for i in range(1, 6))
        # self.refresh_user_info()
        self.editor_keys = ["inst_ip", "inst_port", "debug", "core_num",
                            "window_size", "alert_dis", "secure_dis", "home_path"]

        self._setup_model()

        self._create_chip_dialog = CreateChipDialog(parent=self)
        self._create_chip_dialog.ui.CoreNumText.addItems(self.core_num_select)
        self.ui.tableChipView.choose_item_signal.connect(
            self._edit_item
        )

    @property
    def item(self):
        return self._item

    def _edit_item(self, item: Dict):
        self.set_item(item)
        self.ui.tableChipView.hide_placeholder_text()

    @property
    def ui(self):
        return self._ui

    def _reload_panel(self, visible: bool = True):
        self.ui.actionCreateChip.setVisible(visible)
        self.ui.tableChipView.right_click_menu.start.setVisible(visible)
        self.ui.tableChipView.right_click_menu.stop.setVisible(visible)
        self.ui.tableChipView.right_click_menu.restart.setVisible(visible)
        self.ui.tableChipView.right_click_menu.save.setVisible(visible)
        self.ui.tableChipView.right_click_menu.delete.setVisible(visible)
        if visible:
            self._create_chip_dialog.ui.GroupText.show()
            self._ui.checkBoxShow.show()
            # self.ui.group_widget.show()
        else:
            self._create_chip_dialog.ui.GroupText.hide()
            self._ui.checkBoxShow.hide()
            # self.ui.group_widget.hide()

    def reset_window_layout(self):
        self._ui.checkBoxShow.setChecked(False)
        if self.is_super:
            self.load_group_data()
        self._reload_panel(self.is_super)

    def _setup_model(self):
        self.table_model_chip = QTableModelChipManage(
            self.gui, self, self._ui.tableChipView
        )
        self._ui.tableChipView.setModel(self.table_model_chip)

        self.context_tree_model = QTreeModelChimera(
            self, self.gui, self.ui.TreeChimeraView
        )
        self.ui.TreeChimeraView.setModel(self.context_tree_model)
        self.tree_model_delegate = QOptionsDelegate(self, editor_keys=self.editor_keys)
        self.ui.TreeChimeraView.setItemDelegate(self.tree_model_delegate)

    def load_sample_cache(self):
        ret_data = self.gui.backend.db.query_chip_sample_data()
        if ret_data.get("code") == 200:
            self.cache_chip_sample_data = ret_data["data"]

    def load_user_data(self):
        if self.is_super:
            ret_data = self.gui.backend.db.query_usernames()
        else:
            ret_data = self.gui.backend.db.query_usernames(self.group_name)
        if ret_data.get("code") == 200:
            self.user_list = [""] + ret_data["data"]
        self.load_sample_cache()

    def load_group_data(self):
        if self.is_super:
            ret_data = self.gui.backend.db.query_group_names()
            if ret_data.get("code") == 200:
                self.groups_list = [""] + ret_data["data"]
                # self.ui.chipGroupContent.addItems(self.groups_list)
                self._create_chip_dialog.ui.GroupText.addItems(self.groups_list)

    def load_all_data(self):
        self.clear_cache_data()
        self.load_user_data()

        sample_list = list(self.cache_chip_sample_data.keys())
        # self.ui.chipSampleContent.addItems(sample_list)
        # self.ui.chipEnvContent.addItems(self.cache_chip_sample_data.get("", []))

    def clear_cache_data(self):
        self.user_list.clear()
        self.groups_list.clear()
        # self.ui.chipGroupContent.clear()
        # self.ui.chipSampleContent.clear()
        self._create_chip_dialog.ui.GroupText.clear()

    def create_chip(self):
        self._create_chip_dialog.show()
        ret = self._create_chip_dialog.exec()
        if int(ret) == 1:
            (
                sample,
                env_name,
                inst_ip,
                inst_port,
                groups,
                core_num,
                debug,
                window_size,
                alert_dis,
                secure_dis,
            ) = self._create_chip_dialog.get_input()
            if not sample:
                QMessageBox().warning(self, "Save Fail", "pls input sample!")
                self.create_chip()
                return
            if not env_name:
                QMessageBox().warning(self, "Save Fail", "pls input env_name!")
                self.create_chip()
                return
            if not inst_ip:
                QMessageBox().warning(self, "Save Fail", "pls input inst_ip!")
                self.create_chip()
                return
            if not inst_port:
                QMessageBox().warning(self, "Save Fail", "pls input inst_port!")
                self.create_chip()
                return
            if not core_num:
                QMessageBox().warning(self, "Save Fail", "pls input core_num!")
                self.create_chip()
                return
            if core_num > window_size:
                QMessageBox().warning(
                    self, "Save Fail", "core_num must be < window_size!"
                )
                self.create_chip()
                return
            if alert_dis > secure_dis:
                QMessageBox().information(
                    self, "Warning", "alert_dis must be < secure_dis!"
                )
                self.create_chip()
                return

            if not self.is_super:
                groups = self.group_name
            else:
                if not groups:
                    QMessageBox().information(self, "Warning", "pls input groups!")
                    self.create_chip()
                    return
            ret_data = self.gui.backend.db.update_chip(
                sample,
                env_name,
                inst_ip,
                inst_port,
                groups,
                core_num,
                debug,
                window_size,
                alert_dis,
                secure_dis,
            )
            self.handler_ret_data(ret_data, show_suc=True)
            if ret_data and ret_data.get("code") != 200:
                self.create_chip()

    def query_chip(self):
        # sample = self._ui.chipSampleContent.currentText()
        # env_name = self._ui.chipEnvContent.currentText()
        # groups = self._ui.chipGroupContent.currentText()
        show_all = self._ui.checkBoxShow.isChecked()
        ret_data = self.gui.backend.db.query_chip(show_all=show_all)
        if ret_data.get("code") in (200, 404, 204):
            self.data = ret_data["data"]
            # print(f"query_chip refresh auto..{self.data}")
            self.table_model_chip.refresh_auto(False)

    def save_chip(
            self,
            sample: str,
            env_name,
            inst_ip,
            inst_port,
            groups: str = None,
            core_num: Union[str, int] = "1",
            debug: Union[str, int] = "0",
            window_size: Union[str, int] = "10",
            alert_dis: Union[str, int] = "1",
            secure_dis: Union[str, int] = "2",
    ):
        if self.is_super:
            groups = None
        if not re.match(r"^[1-5]$", str(core_num)):
            QMessageBox().warning(
                self, "Save Fail", "core_num must be in randint(1, 5)"
            )
            return
        if int(core_num) > int(window_size):
            QMessageBox().warning(
                self, "Save Fail", "core_num must be < window_size!"
            )
            return
        if int(alert_dis) > int(secure_dis):
            QMessageBox().warning(
                self, "Save Fail", "alert_dis must be < secure_dis!"
            )
            return
        if not 1 <= int(window_size) <= 100:
            QMessageBox().warning(
                self, "Save Fail", "window_size must in randint(1, 100)!"
            )
            return
        if not 1 <= int(alert_dis) <= 5:
            QMessageBox().warning(
                self, "Save Fail", "alert_dis must in randint(1, 5)!"
            )
            return
        if not 1 <= int(secure_dis) <= 5:
            QMessageBox().warning(
                self, "Save Fail", "secure_dis must in randint(1, 5)!"
            )
            return
        ret_data = self.gui.backend.db.update_chip(
            sample,
            env_name,
            inst_ip,
            inst_port,
            groups=groups,
            core_num=int(core_num),
            debug=int(debug),
            window_size=int(window_size),
            alert_dis=int(alert_dis),
            secure_dis=int(secure_dis),
        )
        self.handler_ret_data(ret_data)

    def delete_chip(self, sample: str, env_name: str, index: int):
        if self.ask_ok(
                "Are you sure to <strong style='color:red'>delete</strong> the chip? "
                "This operation will not be recoverable.",
                "Visage Message",
        ):
            ret_data = self.gui.backend.db.delete_chip(sample, env_name)
            self.handler_ret_data(ret_data)
            if ret_data.get("code") == 200:
                self.table_model_chip.removeRows(index)

    def control_qs_server(self, sample: str, env_name: str, option: str):
        if self.ask_ok(
                f"Are you sure to <strong style='color:red'>{option}</strong> QS-Server ? "
                "This operation will not be recoverable.",
                f"{option} QS-Server",
        ):
            ret_data = self.gui.backend.db.control_qs_server(sample, env_name, option)
            self.handler_ret_data(ret_data)

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

    def chip_sample_change(self):
        # sample = self.ui.chipSampleContent.currentText()
        # env_list = self.cache_chip_sample_data.get(sample, [])
        # self.ui.chipEnvContent.clear()
        # self.ui.chipEnvContent.addItems(env_list)
        pass

    def force_refresh(self):
        """Force refresh."""
        self.context_tree_model.load()

    def set_item(self, item=None):
        self._item = item

        if item is None:
            self.force_refresh()
            return

        # Labels
        # ) from {space.__class__.__module__}
        # label_text = f"{space.data_dict.name} | {space.data_dict.class_name}"
        # self.ui.labelComponentName.setText(label_text)
        # self.ui.labelComponentName.setCursorPosition(0)  # Move to left
        # self.setWindowTitle(label_text)
        # self.parent().setWindowTitle(label_text)

        self.force_refresh()

        self.ui.TreeChimeraView.autoresize_columns()  # resize columns

        data_dict = item
        if not isinstance(item, dict):
            data_dict = item.to_dict()

        # self.ui.textEdit.setText(json.dumps(data_dict, indent=4))

    def _look_item(self, item):
        self.set_item(item)
        self.ui.TreeChimeraView.hide_placeholder_text()
