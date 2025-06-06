# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/12
# __author:       YangChao

from typing import TYPE_CHECKING

from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import QFileDialog

from pyQCat.structures import QDict
from pyqcat_visage.gui.run_setting_ui import Ui_MainWindow
from pyqcat_visage.gui.widgets.title_window import TitleWindow

if TYPE_CHECKING:
    from ..main_window import VisageGUI


class RunSettingWidget(TitleWindow):

    def __init__(self, gui: 'VisageGUI', parent=None):
        super().__init__(parent)
        self.gui = gui
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

    @Slot()
    def import_sp(self):
        path = self.gui.backend.config.get('system').get('config_path')
        dirname = QFileDialog.getExistingDirectory(self, "Import File", path)
        if dirname:
            self._ui.simulator_data_path.setText(dirname)
        else:
            self.handler_ret_data(QDict(code=600, msg='No choose correct path'))

    @Slot()
    def ok(self):
        run_options = QDict(
            exp_save_mode=self._ui.exp_save_mode.currentIndex(),
            simulator_data_path=self._ui.simulator_data_path.text(),
            use_simulator=self._ui.use_simulator.isChecked(),
            dag_save_mode=self._ui.dag_save_mode.currentIndex(),
            use_backtrace=self._ui.use_backtrace.isChecked(),
            register_dag=self._ui.register_dag.isChecked(),
            simulator_delay=float(self._ui.simulator_delay.text())
        )
        mongo_options = QDict(
            inst_host=self._ui.inst_host.text(),
            inst_port=self._ui.inst_mongo.value(),
            inst_log=self._ui.inst_log.value()

        )
        self.gui.backend.config.mongo = mongo_options
        self.gui.backend.config.run = run_options
        # self.gui.backend.context_builder.config.run = run_options
        self.check_simulator_for_toolbar_state(run_options.use_simulator)
        self.gui.system_config_widget.tree_model.refresh()
        # self.gui.backend.save_config()
        self.close_()

    def check_simulator_for_toolbar_state(self, use_simulator: bool):
        if use_simulator:
            self.gui.main_window.set_toolbar_state(True)
        else:
            if self.gui.main_window.state_flag:
                self.gui.main_window.set_toolbar_state(True)
            else:
                self.gui.main_window.set_toolbar_state(False)

    @Slot()
    def default(self):
        self._ui.dag_save_mode.setCurrentIndex(1)
        self._ui.exp_save_mode.setCurrentIndex(0)
        self._ui.use_simulator.setCheckState(Qt.CheckState.Unchecked)
        self._ui.use_backtrace.setCheckState(Qt.CheckState.Unchecked)
        self._ui.register_dag.setCheckState(Qt.CheckState.Unchecked)
        self._ui.simulator_data_path.clear()
        self._ui.simulator_delay.clear()

    def refresh(self):
        run_options = self.gui.backend.config.run
        self._ui.exp_save_mode.setCurrentIndex(run_options.exp_save_mode)
        self._ui.dag_save_mode.setCurrentIndex(run_options.dag_save_mode)
        self._ui.use_simulator.setCheckState(Qt.CheckState.Checked if run_options.use_simulator
                                             else Qt.CheckState.Unchecked)
        self.check_simulator_for_toolbar_state(run_options.use_simulator)
        self._ui.use_backtrace.setCheckState(Qt.CheckState.Checked if run_options.use_backtrace
                                             else Qt.CheckState.Unchecked)
        self._ui.register_dag.setCheckState(Qt.CheckState.Checked if run_options.register_dag
                                            else Qt.CheckState.Unchecked)
        self._ui.simulator_data_path.setText(run_options.simulator_data_path)
        value = run_options.simulator_delay if run_options.simulator_delay != {} else 0
        self._ui.simulator_delay.setText(str(value))

        mongo_options = self.gui.backend.config.mongo
        self._ui.inst_host.setText(mongo_options.get("inst_host", None))
        self._ui.inst_mongo.setValue(mongo_options.get("inst_port", 27017))
        self._ui.inst_log.setValue(mongo_options.get("inst_log", 27021))
