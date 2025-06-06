# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/17
# __author:       YangChao Zhao

from PySide6.QtWidgets import QFileDialog, QMessageBox
from loguru import logger

from pyQCat.config import PyqcatConfig
from pyQCat.structures import QDict
from pyqcat_visage.gui.system_config_ui import Ui_MainWindow
from pyqcat_visage.gui.tools.utilies import slot_catch_exception
from pyqcat_visage.gui.widgets.component.tree_delegate_options import QOptionsDelegate
from pyqcat_visage.gui.widgets.config.tree_model_config import QTreeModelConfig
from pyqcat_visage.gui.widgets.title_window import TitleWindow


class SystemConfigWindow(TitleWindow):

    def __init__(self, gui, parent=None):
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.gui = gui

        self.tree_model = None
        self.old_sample = None

        self._setup_default_config()

    @property
    def backend(self):
        return self.gui.backend

    def _setup_default_config(self):
        self.tree_model = QTreeModelConfig(self, self.gui, self._ui.tree_view_config)

        self.tree_model_delegate = QOptionsDelegate(self, self.gui)
        self._ui.tree_view_config.setItemDelegate(self.tree_model_delegate)
        self._ui.tree_view_config.setModel(self.tree_model)
        self.tree_model_delegate.point_label_signal.connect(self.change_point_label)
        self._ui.tree_view_config.expandAll()

    def _open_io_file(self, file_name: str):
        try:
            pyqcat_config = PyqcatConfig(file_name)
            self.gui.backend.context_builder.config = QDict(**pyqcat_config.to_dict())
            self.gui.backend.config.system.invoker_addr = self.gui.backend.invoker_addr
            return True
        except Exception as e:
            logger.error(f'load config file failed, because {e}!')
            return False

    @slot_catch_exception()
    def import_config(self):
        cur_path = self.backend.config.system.config_path
        title = "Open File"
        filter_ = "config(*.conf)"
        file_name, flt = QFileDialog.getOpenFileName(self, title, cur_path, filter_)
        self.old_sample = self.gui.backend.config.system.sample
        if file_name == "":
            return

        if not self._open_io_file(file_name):
            QMessageBox.critical(self, "Error", "Open File Failed!")

        self._ui.file_edit.setText(file_name)
        self.tree_model.refresh()

    @slot_catch_exception(process_reject=True)
    def export_config(self):
        cur_path = self.backend.config.system.config_path
        filename, flt = QFileDialog.getSaveFileName(self, "保存文件", cur_path, "config(*.conf)")
        if filename == "":
            QMessageBox.critical(self, "Error", "File name is null!")
            return

        self.gui.backend.save_config(filename)

    @slot_catch_exception(process_reject=True)
    def save_config(self):
        # if self.ask_ok('Are you sure to save config?', 'System Config'):
        try:
            # temp solve: close save config init visage
            self.gui.main_window.setup_close()
            self._ui.pushButton.setEnabled(False)
            self.gui.init_visage(update_config=True)
            self.backend.save_config()
            self.backend.init_execute_context()
            # self.gui.main_window.start_communication_thread()
            ret_data = QDict(code=200)
            self.backend.db.sync_qs_status()
            logger.log("UPDATE", "Save Config Success!")
            self._ui.pushButton.setEnabled(True)
            sample = self.gui.backend.config.system.sample

            if self.old_sample != sample:
                msg_box = QMessageBox(self)
                msg_box.setText("Change the sample and whether to stop the task?")
                msg_box.setWindowTitle("Task Tips ")
                msg_box.addButton(QMessageBox.Ok)
                msg_box.addButton(QMessageBox.Cancel)
                result = msg_box.exec()
                if result == QMessageBox.Ok:
                    ret_data = self.backend.db.query_custom_task()
                    if ret_data.get("code") == 200:
                        for task in ret_data.get("data"):
                            self.backend.remove_one_cron(task.get("id"))
        except Exception as e:
            # solve: After exception, this widget is not enabled
            self._ui.pushButton.setEnabled(True)
            ret_data = QDict(code=600, msg=str(e))

        self.handler_ret_data(ret_data)

    def change_point_label(self, point_label: str):
        self.gui.backend.config.system.point_label = point_label
        self.tree_model.refresh()

    def clear(self):
        self.tree_model_delegate.clear()


