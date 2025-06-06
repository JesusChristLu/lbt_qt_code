# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/08
# __author:       YangChao Zhao

from PySide6.QtCore import Slot

from pyQCat.structures import QDict
from .context_build_dialog_ui import Ui_Dialog
from ..title_window import TitleDialog
from pyQCat.types import StandardContext
from ....config import GUI_CONFIG


class SQCDialog(TitleDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self._qubit_list = []
        self._coupler_list = []
        self._bit_list = []

        # self.setWindowFlags(Qt.WindowType.MSWindowsFixedSizeDialogHint)
        self._last_name = None
        self._env_change = False

    @property
    def env_bits(self):
        return self._bit_list

    def get_input(self):

        union_bits = []
        union_str = self.ui.union_edit.text()
        if union_str:
            union_bits = eval(union_str)

        physical_units = self.ui.base_qubit_com.currentText()

        if isinstance(physical_units, list) and len(physical_units) == 1:
            physical_units = physical_units[0]

        # if ',' in physical_units:
        #     physical_units = physical_units.split(',')

        return QDict(
            stardard_context=self.ui.context_com.currentText(),
            base_qubit=physical_units,
            use_dcm=self.ui.use_dcm_check.isChecked(),
            crosstalk=self.ui.crosstalk_check.isChecked(),
            ac_switch=self.ui.ac_switch_check.isChecked(),
            readout_type=self.ui.read_com.currentText(),
            union_bits=union_bits,
            working_type=self.ui.work_type_com.currentText(),
            divide_type=self.ui.divide_type_com.currentText(),
            max_point_unit=self.ui.max_qubit_com.currentText()
        )

    def set_qubit_items(self, bit_list: list):
        self._env_change = True
        self._bit_list = bit_list
        self._qubit_list = [bit for bit in bit_list if bit.startswith("q")]
        self._coupler_list = [bit for bit in bit_list if bit.startswith("c")]

        # bug solve: before set env bit must clear combox
        self.ui.base_qubit_com.clear()
        self.ui.max_qubit_com.set_units(self._bit_list)

    def _checkout_qnt(self):
        if self._last_name != StandardContext.NT.value or self._env_change:
            self.ui.base_qubit_com.clear()
            self.ui.base_qubit_com.addItems(self._bit_list)
            self._env_change = False

    def _checkout_sqc(self):
        if self._last_name != StandardContext.QC.value or self._env_change:
            self.ui.base_qubit_com.clear()
            self.ui.base_qubit_com.set_units(self._qubit_list)
            self.ui.read_com.clear()
            self.ui.read_com.addItems(list(GUI_CONFIG.std_context.qubit_calibration.values()))
            self._env_change = False

    def _checkout_scc(self):
        if self._last_name not in [StandardContext.CPC.value, StandardContext.CC.value] or self._env_change:
            self.ui.base_qubit_com.clear()
            self.ui.base_qubit_com.set_units(self._coupler_list)
            self.ui.read_com.clear()
            self.ui.read_com.addItems(list(GUI_CONFIG.std_context.coupler_calibration.values()))
            self._env_change = False

    def _checkout_union_read(self):
        self.ui.union_edit.setEnabled(True)

    def _checkout_pair(self):
        if self._last_name != StandardContext.CGC.value or self._env_change:
            self.ui.base_qubit_com.clear()

            pair_names = []
            for coupler in self._coupler_list:
                q1, q2 = coupler[1:].split('-')
                pair_names.append(f'q{q1}q{q2}')

            self.ui.base_qubit_com.set_units(pair_names)

            self.ui.read_com.clear()
            self.ui.read_com.addItems(list(GUI_CONFIG.std_context.cz_gate_calibration.values()))
            self._env_change = False

    def _checkout_crosstalk(self):
        if self._last_name != StandardContext.CM.value or self._env_change:
            self.ui.base_qubit_com.clear()
            self.ui.base_qubit_com.set_units(self.merge_bits(self._bit_list))
            self._env_change = False

    @Slot(str)
    def format_dialog(self, name: str):
        if name == StandardContext.QC.value:
            self._checkout_sqc()
        elif name in [StandardContext.CPC.value, StandardContext.CC.value]:
            self._checkout_scc()
        elif name == StandardContext.NT.value:
            self._checkout_qnt()
        elif name == StandardContext.URM.value:
            self._checkout_union_read()
        elif name == StandardContext.CM.value:
            self._checkout_crosstalk()
        elif name == StandardContext.CGC.value:
            self._checkout_pair()

        self._last_name = name

    def init_dialog(self):
        self.format_dialog(self.ui.context_com.currentText())

    @staticmethod
    def merge_bits(*args):

        bits = []
        for arg in args:
            bits.extend(arg)

        new_bits = []

        for i in range(len(bits) - 1):
            b1 = bits[i]
            for b2 in bits[i + 1:]:
                new_bits.append(b1+b2)

        return new_bits

    def set_data(self, pd: QDict = None):
        self.ui.context_com.setCurrentText(pd.stardard_context)
        self.init_dialog()
        self.ui.base_qubit_com.set_text(pd.base_qubit)
        self.ui.use_dcm_check.setChecked(pd.use_dcm)
        self.ui.crosstalk_check.setChecked(pd.crosstalk)
        self.ui.ac_switch_check.setChecked(pd.ac_switch)
        self.ui.read_com.setCurrentText(pd.readout_type)
        self.ui.union_edit.setText(','.join([str(i) for i in pd.union_bits]))
        self.ui.work_type_com.setCurrentText(pd.working_type)
        self.ui.divide_type_com.setCurrentText(pd.divide_type)
        self.ui.max_qubit_com.set_text(pd.max_point_unit)
