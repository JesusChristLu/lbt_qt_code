# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/07/31
# __author:       HanQing Shi


from typing import TYPE_CHECKING
from PySide6.QtCore import Slot, Signal
from PySide6.QtGui import QTextCharFormat, QBrush, QColor, QTextCursor
from PySide6.QtWidgets import QDialog
from pyqcat_visage.gui.diagnose_ui import Ui_Dialog

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI
from pyQCat.errors import BUSAllocatedError, LOAllocatedError


class DiagnoseWindow(QDialog):
    """Show the error messages window."""

    handle_error = Signal(str)

    def __init__(self, gui: "VisageGUI", parent=None):
        self.gui = gui
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.pushButton_link.setStyleSheet(
            """
            color: #FF0033;
            font-family: Titillium;
            font-size: 18px;
            font-weight:bold;
            """
        )
        policy = self.ui.pushButton_link.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.ui.pushButton_link.setSizePolicy(policy)

        self.error = ""

    @property
    def config(self):
        return self.gui.backend.config

    def show(self):
        log_path = self.config.system.log_path or self.config.system.local_root
        self.ui.plainTextEdit.setPlainText(str(self.error)+f"\n\nLog path: {log_path}")
        cursor = self.ui.plainTextEdit.textCursor()
        text_format = QTextCharFormat()
        text_format.setForeground(QBrush(QColor(255, 107, 89, 255)))
        text_format.setFontWeight(2000)
        cursor.setPosition(0)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.setCharFormat(text_format)
        super().show()

    @Slot()
    def go_to(self):
        """Go to handler."""
        if isinstance(self.error, BUSAllocatedError):
            tab = f"AMP_Bus-{self.error.bus_str}"
        elif isinstance(self.error, LOAllocatedError):
            lo_type = "m" if self.error.lo_type == "readout" else "xy"
            lo_num = list(self.error.bit_params.values())[0]["lo"]
            tab = f"LO_{lo_type}-lo-{int(lo_num)}"
        else:
            tab = None
        self.handle_error.emit(tab)
        self.ui.plainTextEdit.clear()
        self.close()

    def check_link(self, link: bool):
        if link:
            self.ui.pushButton_link.show()
        else:
            self.ui.pushButton_link.hide()
