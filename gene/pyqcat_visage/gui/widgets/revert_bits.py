# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2024/01/15
# __author:       XuYao
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List

from PySide6.QtCore import QDateTime
from PySide6.QtWidgets import QMessageBox, QDialog

from pyqcat_visage.gui.widgets.component.table_model_revert import QTableModelRevertBit
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.gui.revert_bit_ui import Ui_MainWindow

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class RevertBitWindow(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent=None):
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.gui = gui

        self.data = []
        self._setup_table()

    @property
    def ui(self):
        return self._ui

    def _setup_table(self):
        self.table_model_revert = QTableModelRevertBit(
            self.gui, self, self._ui.tableRevertView
        )
        self._ui.tableRevertView.setModel(self.table_model_revert)

    def show(self):
        time_ = datetime.now()
        self._ui.TimeNodeText.setDateTime(QDateTime(time_))
        return super().show()

    def get_input_time_str(self):
        return self.ui.TimeNodeText.dateTime().toPython().strftime("%Y-%m-%d %H:%M:%S")

    def query_revert_bits(self):
        time_node = self.get_input_time_str()
        res = self.gui.backend.db.query_revert_bits(time_node)
        if res.get("code") == 200:
            self.data = res.get("data", [])
            self.table_model_revert.refresh_auto(False)

    def revert_bits(self):
        if not self.data:
            QMessageBox.warning(self, "Warning", "no data to revert, pls query first!")
            return
        time_node = self.get_input_time_str()
        if self.ask_ok(
                f"Are you sure <strong style='color:red'>Revert {time_node}</strong> "
                f"to local env ?",
                "Revert Warning",
        ):
            res = self.gui.backend.db.revert_more_bits(time_node)
            if res.get("code") != 200:
                QMessageBox.warning(self, "Warning", res.get("msg"))
