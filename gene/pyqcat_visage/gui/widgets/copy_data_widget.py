import re
from typing import TYPE_CHECKING

from PySide6.QtCore import Slot, QTime, QTimer
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QMessageBox, QApplication

from pyqcat_visage.gui.widgets.dialog.tips_dialog import TipsDialog
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.gui.copy_data_ui import Ui_MainWindow

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class CopyDataWindow(TitleWindow):
    """Heatmap window."""

    def __init__(self, gui: 'VisageGUI', parent=None):
        self.gui = gui
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.data_dict = {}
        self._ui.CopyTips.hide()

    def init_first_data(self):
        self.update_user()
        # self.update_sample()
        # self.update_env_name()
        # self.update_point_label()
        self.change_local()
        self.default_local_env()
        self.update_element_configs()

    def copy_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self._ui.OtherNameText_2.text())

    def reset_window_layout(self):
        if not self.is_super:
            self._ui.SyncButton.hide()
        else:
            self._ui.SyncButton.show()

    def show(self) -> None:
        self.init_first_data()
        # self._ui.OtherNameText_2.setContextMenuPolicy(Qt.ActionsContextMenu)
        # copy_action = self._ui.OtherNameText_2.addAction("copy text")
        # copy_action.triggered.connect(self.copy_text)
        super().show()

    def query_usernames(self):
        key = f"{self.username}//users"
        names = self.data_dict.get(key)
        if names is None:
            ret_data = self.gui.backend.query_username_list()
            if ret_data and ret_data.get("code") == 200:
                self.data_dict[key] = ret_data["data"]
            else:
                self.data_dict[key] = []
            names = self.data_dict[key]
        return names

    def query_samples(self, username: str):
        key = f"{self.username}//{username}//samples"
        samples = self.data_dict.get(key)
        if samples is None:
            ret_data = self.gui.backend.query_sample_list(username)
            if ret_data and ret_data.get("code") == 200:
                self.data_dict[key] = ret_data["data"]
            else:
                self.data_dict[key] = []
            samples = self.data_dict[key]
        return samples

    def query_env_names(self, username: str, sample: str):
        key = f"{self.username}//{username}|{sample}//env_name"
        env_names = self.data_dict.get(key)
        if env_names is None:
            ret_data = self.gui.backend.query_env_name_list(username, sample)
            if ret_data and ret_data.get("code") == 200:
                self.data_dict[key] = ret_data["data"]
            else:
                self.data_dict[key] = []
            env_names = self.data_dict[key]
        return env_names

    def query_point_labels(self, username: str, sample: str, env_name: str):
        key = f"{self.username}//{username}|{sample}|{env_name}//point_label"
        point_labels = self.data_dict.get(key)
        if point_labels is None:
            ret_data = self.gui.backend.query_point_label_list(username,
                                                               sample,
                                                               env_name)
            if ret_data and ret_data.get("code") == 200:
                self.data_dict[key] = ret_data["data"]
            else:
                self.data_dict[key] = []
            point_labels = self.data_dict[key]
        return point_labels

    def query_element_names(self, username: str, sample: str,
                            env_name: str, point_label: str):
        key = f"{self.username}//{username}|{sample}|{env_name}|" \
              f"{point_label}//element_names"
        bits = self.data_dict.get(key)
        if bits is None:
            ret_data = self.gui.backend.query_other_user_data(username,
                                                              "BaseQubit",
                                                              sample,
                                                              env_name,
                                                              point_label)
            if ret_data and ret_data.get("code") == 200:
                ret_data["data"].sort(key=lambda x: int(re.findall("[0-9]+", x)[0]))
                self.data_dict[key] = ret_data["data"]
            else:
                self.data_dict[key] = []
            bits = self.data_dict[key]
        return bits

    def reset_window(self):
        self.data_dict = {}
        # self._ui.OtherNameText.clear()
        # self._ui.SampleText.clear()
        # self._ui.EnvNameText.clear()
        # self._ui.PointLabelText.clear()
        # self._ui.ElementNamesText.clear()
        self.init_first_data()

    def update_user(self):
        self._ui.OtherNameText.clear()
        # self._ui.SampleText.clear()
        # self._ui.EnvNameText.clear()
        # self._ui.PointLabelText.clear()
        # self._ui.ElementNamesText.clear()
        data = self.query_usernames()
        self._ui.OtherNameText.addItems(data)

    def update_sample(self):
        self._ui.SampleText.clear()
        # self._ui.EnvNameText.clear()
        # self._ui.PointLabelText.clear()
        # self._ui.ElementNamesText.clear()
        data = self.query_samples(self.get_from_input_env()[0])
        self._ui.SampleText.addItems(data)
        self.sync_sample()

    def update_env_name(self):
        self._ui.EnvNameText.clear()
        # self._ui.PointLabelText.clear()
        # self._ui.ElementNamesText.clear()
        data = self.query_env_names(*self.get_from_input_env()[:2])
        self._ui.EnvNameText.addItems(data)
        self.sync_sample()
        self.sync_env_name()

    def update_point_label(self):
        self._ui.PointLabelText.clear()
        # self._ui.ElementNamesText.clear()
        data = self.query_point_labels(*self.get_from_input_env()[:3])
        self._ui.PointLabelText.addItems(data)
        self.sync_point_label()

    def update_element_names(self):
        self._ui.ElementNamesText.clear()
        data = self.query_element_names(*self.get_from_input_env())
        self._ui.ElementNamesText.multi_select = data
        self._ui.ElementNamesText.loadItems()

    def update_element_configs(self):
        key = f"element_config_names"
        data = self.data_dict.get(key)
        if not data:
            res = self.gui.backend.query_conf_type_list()
            if res.get("code") == 200:
                self._ui.ElementConfigsText.multi_select = res["data"]
                self._ui.ElementConfigsText.loadItems()

    def get_from_input_env(self):
        return self._ui.OtherNameText.currentText(), \
               self._ui.SampleText.currentText(), \
               self._ui.EnvNameText.currentText(), \
               self._ui.PointLabelText.currentText()

    def get_to_input_env(self):
        return self._ui.OtherNameText_2.text(), \
               self._ui.SampleText_2.text(), \
               self._ui.EnvNameText_2.text(), \
               self._ui.PointLabelText_2.text()

    def default_local_env(self):
        self._ui.OtherNameText_2.setText(self.gui.backend.login_user["username"])
        self._ui.SampleText_2.setText(self.gui.backend.config.system.sample)
        self._ui.EnvNameText_2.setText(self.gui.backend.config.system.env_name)
        self._ui.PointLabelText_2.setText(self.gui.backend.config.system.point_label)
        self.sync_sample()
        self.sync_env_name()
        self.sync_point_label()

    def permission_some_label(self, state: bool, init_user: bool = True):
        # if init_user:
        #     self._ui.OtherNameText_2.setEnabled(state)
        # else:
        #     self._ui.OtherNameText_2.setEnabled(init_user)
        self._ui.OtherNameText_2.setEnabled(False)
        self._ui.SampleText_2.setEnabled(state)
        self._ui.EnvNameText_2.setEnabled(state)
        self._ui.PointLabelText_2.setEnabled(state)

    def change_local(self):
        if self.is_super:
            if self._ui.LocalBox.isChecked():
                self.permission_some_label(False, init_user=False)
                self.default_local_env()
            else:
                self.permission_some_label(True)
        else:
            self.permission_some_label(False)
            self.default_local_env()

    def user_local_state(self):
        """
            normal user: unchecked LocalBox and normal user
        Do not edit sample,env name, or point label
        Returns:
            bool: if True: shield else Open
        """
        if not self._ui.LocalBox.isChecked() and not self.is_super:
            return True
        else:
            return False

    def sync_sample(self):
        if self.user_local_state():
            self._ui.SampleText_2.setText(self._ui.SampleText.currentText())

    def sync_env_name(self):
        if self.user_local_state():
            self._ui.EnvNameText_2.setText(self._ui.EnvNameText.currentText())

    def sync_point_label(self):
        if self.user_local_state():
            self._ui.PointLabelText_2.setText(self._ui.PointLabelText.currentText())

    def sync_env_data(self):
        if self.is_super and not self._ui.LocalBox.isChecked():
            self._ui.SampleText_2.setText(self._ui.SampleText.currentText())
            self._ui.EnvNameText_2.setText(self._ui.EnvNameText.currentText())
            self._ui.PointLabelText_2.setText(self._ui.PointLabelText.currentText())

    def _warning_msg(self, msg):
        QMessageBox().warning(self, "Warning", msg)

    def show_copy_button(self):
        self._ui.CopyButton.setEnabled(True)
        self._ui.CopyTips.hide()

    def copy_other_data(self):
        from_username, from_sample, from_env_name, from_point_label = \
            self.get_from_input_env()
        to_username, to_sample, to_env_name, to_point_label = \
            self.get_to_input_env()
        if not self.is_super and not self._ui.LocalBox.isChecked():
            if from_sample != to_sample or from_env_name != from_env_name:
                self._warning_msg("Normal User| use Local| sample and env must same:\n"
                                  f"{from_sample!r}->{to_sample!r}\n"
                                  f"{from_env_name!r}->{to_env_name!r}")
                return
        if self._ui.LocalBox.isChecked() and to_username != self.username:
            return self._warning_msg("use Local| to-user must be yourself!")
        miss_str = "Missing parameter. pls input "
        if not from_username:
            return self._warning_msg(f"{miss_str}'From-user'")
        if not from_sample:
            return self._warning_msg(f"{miss_str}'From-sample'")
        if not from_env_name:
            return self._warning_msg(f"{miss_str}'From-env_name'")
        if not from_point_label:
            return self._warning_msg(f"{miss_str}'From-point_label'")
        if not to_username:
            return self._warning_msg(f"{miss_str}'To-user'")
        if not to_sample:
            return self._warning_msg(f"{miss_str}'To-sample'")
        if not to_env_name:
            return self._warning_msg(f"{miss_str}'To-env_name'")
        if not to_point_label:
            return self._warning_msg(f"{miss_str}'To-point_label'")
        element_names = self._ui.ElementNamesText.text()
        element_configs = self._ui.ElementConfigsText.text()
        if not element_names:
            element_names = None
            tip = TipsDialog()
            tip.set_tips("Element_names is not selected, which means that all bits will be copied! "
                         "pls choose 'ok' to continue or cancel to reselect!")
            ret = tip.exec()
            if int(ret) != 1:
                return
        local = True if self._ui.LocalBox.isChecked() else False
        copy_qubit = True if self._ui.CopyQubitBox.isChecked() else False
        if local:
            ret_data = self.gui.backend.copy_other_user_data(*self.get_from_input_env(),
                                                             local,
                                                             element_names=element_names,
                                                             element_configs=element_configs,
                                                             copy_qubit=copy_qubit)
        else:
            ret_data = self.gui.backend.copy_other_user_data(*self.get_from_input_env(),
                                                             local,
                                                             *self.get_to_input_env(),
                                                             element_names=element_names,
                                                             element_configs=element_configs,
                                                             copy_qubit=copy_qubit)
        if ret_data.get("code") == 200:
            self._ui.CopyButton.setEnabled(False)
            self._ui.CopyTips.show()
            QTimer(self).singleShot(8000, self.show_copy_button)
        # self.handler_ret_data(ret_data, show_suc=True)

