# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/01/08
# __author:       YangChao Zhao

from PySide6.QtWidgets import (
    QWidget,
    QHeaderView,
    QMenu,
    QDialog,
    QFormLayout,
    QLabel,
    QCheckBox,
    QDialogButtonBox,
    QLineEdit,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QContextMenuEvent, QFont

from pyQCat.log import pyqlog
from pyqcat_visage.gui.widgets.base.table_structure import QTableViewBase

from ..base.right_click_menu import bind_action


class QTableViewDat(QTableViewBase):
    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        super().__init__(parent)
        self.clicked.connect(self.view_clicked)
        self._define_style()
        self.right_click_menu = None

    def _define_style(self):
        font = QFont()
        font.setPointSize(13)
        horizontal_header = self.horizontalHeader()
        vertical_header = self.verticalHeader()
        horizontal_header.setSectionResizeMode(QHeaderView.Stretch)
        horizontal_header.setFont(font)
        vertical_header.setFont(font)

    def init_right_click_menu(self):
        menu = QMenu(self)

        menu._add_timing = bind_action(menu, "ADD Line Timing", ":/report.png")
        menu._add_timing.triggered.connect(self.add_timing)

        menu._backtrack = bind_action(menu, "Backtrack", ":/cancel.png")
        menu._backtrack.triggered.connect(self.backtrack)

        menu._refresh = bind_action(menu, "Refresh", ":/refresh.png")
        menu._refresh.triggered.connect(self.refresh)

        menu._unit_state_view = bind_action(menu, "View", ":/refresh.png")
        menu._unit_state_view.triggered.connect(self.unit_state_view)

        menu._unit_state_view = bind_action(menu, "View All", ":/refresh.png")
        menu._unit_state_view.triggered.connect(self.unit_state_view_all)

        menu._import = bind_action(menu, "Import", ":/import.png")
        menu._import.triggered.connect(self.import_hardware_data)

        menu._import = bind_action(menu, "Export", ":/report.png")
        menu._import.triggered.connect(self.export_hardware_data)

        self.right_click_menu = menu

    def add_timing(self):
        model = self.model()

        dialog = QDialog(self.model().main_widget)
        form = QFormLayout(dialog)

        form.addRow(QLabel("Please input args:"))

        # Value1
        unit_row = "Physical Unit: "
        uni_edit = QLineEdit(dialog)
        form.addRow(unit_row, uni_edit)

        # Value2
        delay_raw = "Delay (xy|z) (zc|zp|zd)"
        delay_edit = QLineEdit(dialog)
        form.addRow(delay_raw, delay_edit)

        # extend
        extend_raw = "Protocol (c1|c2|delay)"
        extend_edit = QLineEdit(dialog)
        form.addRow(extend_raw, extend_edit)

        # force
        is_force = "Is Force"
        force_edit = QCheckBox(dialog)
        form.addRow(is_force, force_edit)

        # Add Cancel and OK button
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dialog
        )
        form.addRow(button_box)

        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        # Process when OK button is clicked
        if dialog.exec() == QDialog.Accepted:
            try:
                unit = uni_edit.text()
                delay = delay_edit.text()
                _extend = extend_edit.text()
                force = force_edit.isChecked()

                if unit and delay:
                    if unit.startswith("q"):
                        model.main_widget.hardware_offset_func(
                            "qubit", unit, *delay.strip().split(","), force
                        )
                    elif unit.startswith("c"):
                        model.main_widget.hardware_offset_func(
                            "coupler", unit, *delay.strip().split(","), force
                        )
                else:
                    model.main_widget.hardware_offset_func(
                        "extend", *_extend.strip().split(","), force
                    )

            except Exception as err:
                pyqlog.error(f"Add timing error: {err}")

    def backtrack(self):
        self.model().main_widget.hardware_offset_func("backtrack")

    def refresh(self):
        self.model().main_widget.hardware_offset_func("refresh")

    def unit_state_view(self):
        self.model().main_widget.hardware_offset_func("view")

    def unit_state_view_all(self):
        self.model().main_widget.hardware_offset_func("view_all")

    def import_hardware_data(self):
        self.model().main_widget.hardware_offset_func("import")

    def export_hardware_data(self):
        self.model().main_widget.hardware_offset_func("export")

    def contextMenuEvent(self, event: QContextMenuEvent):
        if not self.right_click_menu:
            # self.init_right_click_menu()
            pass

        if self.right_click_menu:
            self.right_click_menu.action = self.right_click_menu.exec_(
                self.mapToGlobal(event.pos())
            )
