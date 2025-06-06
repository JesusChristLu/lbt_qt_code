# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/01
# __author:       Lang Zhu

from typing import TYPE_CHECKING

from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import QFileDialog

from pyQCat.log import logger
from pyQCat.structures import QDict
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.gui.report_edit_window import Ui_MainWindow

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class ReportWindow(TitleWindow):

    def __init__(self, gui: 'VisageGUI', parent=None):
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self.gui = gui
        self._ui.setupUi(self)
        self._init_show_update_info()
        self.temp_file = None

    def _init_show_update_info(self):
        report = self.gui.backend.config.report

        self._ui.lineEdit.setText(report.file_path)
        self._ui.is_report.setCheckState((Qt.CheckState.Checked if report.is_report else Qt.CheckState.Unchecked))
        self._ui.is_report.setCheckState((Qt.CheckState.Checked if report.is_report else Qt.CheckState.Unchecked))
        self._ui.theme.setCurrentText(report.theme)
        self._ui.save_type.setCurrentText(report.save_type)
        self._ui.language.setCurrentText(report.language)
        self._ui.detail.setCurrentText(report.report_detail)

    @Slot()
    def update_report(self):
        report = QDict(
            is_report=True if self._ui.is_report.checkState() == Qt.CheckState.Checked else False,
            id_type="dag",
            theme=self._ui.theme.currentText(),
            system_theme=None,
            save_type=self._ui.save_type.currentText(),
            language=self._ui.language.currentText(),
            report_detail=self._ui.detail.currentText(),
            file_path=self._ui.lineEdit.text(),
        )
        self.gui.backend.config.report = report
        logger.debug(f"set report success.\n{self.gui.backend.config.report}")
        self.close_()

    @Slot()
    def choose_path(self):
        path = self.gui.backend.config.get('system').get('config_path')
        dirname = QFileDialog.getExistingDirectory(self, "Import File", path)
        if dirname:
            self._ui.lineEdit.setText(dirname)
        else:
            self.handler_ret_data(QDict(code=600, msg='No choose correct path'))

    @Slot()
    def cancel(self):
        self._ui.lineEdit.clear()
