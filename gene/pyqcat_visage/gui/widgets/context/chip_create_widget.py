# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/28
# __author:       HanQing Shi, YangChao Zhao

from PySide6.QtCore import Slot, Signal

from pyQCat.executor import ChipGeneratorBase, ChipLineConnect
from pyQCat.structures import QDict
from .chip_create_ui import Ui_MainWindow
from .table_model_chip_bits import QTableModelChip
from ..title_window import TitleWindow


class ChipCreateWindow(TitleWindow):
    load_chip_topology = Signal(str, str, list)

    def __init__(self, gui, parent=None):
        self.gui = gui
        self.context_widget = parent
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.load_chip_topology.connect(self.gui.ui.tabTopology.load)

        self.chips = {}
        self.chip_bit_model = QTableModelChip(gui, self, self.ui.tableView)
        self.ui.tableView.setModel(self.chip_bit_model)
        self.chip_type_dict = {
            x.chip_type: x for x in ChipGeneratorBase.__subclasses__()
        }
        self.chip_type_dict.update({ChipGeneratorBase.chip_type: ChipGeneratorBase})
        self._ui.chipTypeCom.addItems(list(self.chip_type_dict.keys()))
        self._ui.chipTypeCom.setCurrentText(ChipGeneratorBase.chip_type)

    @property
    def ui(self):
        return self._ui

    @Slot()
    def ok(self):
        row = self._ui.row_edit.text()
        col = self._ui.col_edit.text()
        layout_style = self._ui.layout_style.currentText()
        chip_type = self._ui.chipTypeCom.currentText()

        if chip_type == ChipGeneratorBase.chip_type:
            qubit_names = []
            for v in self.chips.values():
                for _v in v.values():
                    qubit_names.append(f"q{_v}")
            self.load_chip_topology.emit(row, col, qubit_names)
            loc_map = self.gui.ui.tabTopology.chip_topology.bit_map
            chip_generator = ChipGeneratorBase(
                row=row,
                col=col,
                qaio_type=self.gui.backend.config.system.qaio_type,
                layout_style=layout_style,
                loc_map=loc_map,
            )
        else:
            if chip_type in self.chip_type_dict:
                chip_generator = self.chip_type_dict[chip_type]()
            else:
                return QDict(code=600, msg=f"unknown chip type")

        if self.ask_ok("Are you sure to create a new chip structure?", "Chip"):
            ret_data = chip_generator.generator()
            if isinstance(ret_data, ChipLineConnect):
                self.gui.backend.model_channels = ret_data
                self.gui.backend.view_channels = QDict()
                self.gui.backend.view_channels.update(ret_data.qubit_params)
                self.gui.backend.view_channels.update(ret_data.coupler_params)
                ret_data = QDict(code=200)

            self.handler_ret_data(ret_data)

            if ret_data.get("code") == 200:
                self.context_widget.channel_table_model.refresh_auto(check_count=False)
                self.close_()

    @Slot()
    def cancel(self):
        self._ui.row_edit.setText("")
        self._ui.col_edit.setText("")

    @Slot()
    def layout(self):
        row = int(self._ui.row_edit.text())
        col = int(self._ui.col_edit.text())
        self.chip_bit_model.columns = [f"col-{i}" for i in range(col)]

        self.chips.clear()
        di = 0
        for i in range(row):
            self.chips[f"row-{i}"] = {}
            for j in range(col):
                self.chips[f"row-{i}"].update({f"col-{j}": di})
                di += 1

        self.chip_bit_model.refresh_auto()

    @Slot()
    def choose_chip_type(self):
        chip_type = self._ui.chipTypeCom.currentText()
        if chip_type not in self.chip_type_dict:
            print("chip type out of range")
            return
        show_flag = True
        if chip_type != ChipGeneratorBase.chip_type:
            show_flag = False
        self._ui.col_edit.setEnabled(show_flag)
        self._ui.row_edit.setEnabled(show_flag)
        self._ui.layout_button.setVisible(show_flag)
