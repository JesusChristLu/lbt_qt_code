# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/16
# __author:       YangChao Zhao

from PySide6.QtCore import QModelIndex, Signal
from PySide6.QtWidgets import QTableView, QAbstractItemView, QWidget, QMenu, QHeaderView, QMessageBox

from pyQCat.structures import QDict
from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.right_click_menu import bind_action
from ..base.table_structure import QTableViewBase


class QTableViewContextWidget(QTableViewBase, PlaceholderTextWidget):
    choose_component_signal = Signal(QDict)

    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        self.ui = parent.parent().parent().parent()
        QTableViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self,
            "Click Create Context to Generate a Experiment Context!"
        )

    def _define_style(self):
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QTableView.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # noinspection DuplicatedCode
    def init_right_click_menu(self):
        menu = QMenu(self)

        menu._d = bind_action(menu, 'Delete', u":/delete.png")
        menu.addSeparator()
        menu._create_context = bind_action(menu, 'Create Context', u":/create_context.png")
        menu._clear_context = bind_action(menu, 'Clear Context', u":/clear.png")
        menu._reset_context = bind_action(menu, 'Reset Context', u":/refresh.png")
        menu.addSeparator()
        menu._add_inst = bind_action(menu, 'Add Inst', u":/inst.png")
        menu._add_qubit = bind_action(menu, 'Add Qubit', u":/qubit.png")
        menu._add_dcm = bind_action(menu, 'Add Dcm', u":/dcm.png")
        menu._add_crosstalk = bind_action(menu, 'Add Crosstalk', u":/crosstalk.png")
        menu._add_compensate = bind_action(menu, 'Add Compensate', u":/compensate.png")

        menu._d.triggered.connect(self.delete_row)
        menu._add_inst.triggered.connect(self.ui.context_add_inst)
        menu._add_qubit.triggered.connect(self.ui.context_add_qubit)
        menu._add_dcm.triggered.connect(self.ui.context_add_dcm)
        menu._add_crosstalk.triggered.connect(self.ui.context_add_crosstalk)
        menu._add_compensate.triggered.connect(self.ui.context_add_compensates)
        menu._create_context.triggered.connect(self.ui.context_create)
        menu._clear_context.triggered.connect(self.ui.context_clear)
        menu._reset_context.triggered.connect(self.ui.context_reset)

        self.right_click_menu = menu

    @property
    def backend(self):
        """Returns the design."""
        return self.model().backend

    @property
    def gui(self):
        """Returns the GUI."""
        return self.model().gui

    def view_clicked(self, index: QModelIndex):
        if self.gui is None or not index.isValid():
            return

        model = self.model()
        _, component = model.component_from_index(index)
        if component and not isinstance(component, str):
            self.choose_component_signal.emit(component)

    def delete_row(self):
        """Create message box to confirm row deletion."""
        indexes = self.selectedIndexes()
        if len(indexes) > 0:
            index = indexes[0]
            name, _ = self.model().component_from_index(index)
            if name == 'crosstalk':
                self.backend.experiment_context._crosstalk_dict = None
            elif name.startswith('q'):
                for qubit in self.backend.experiment_context.qubits:
                    if qubit.name == name:
                        self.backend.experiment_context.qubits.remove(qubit)
            elif name.startswith('c'):
                for coupler in self.backend.experiment_context.couplers:
                    if coupler.name == name:
                        self.backend.experiment_context.couplers.remove(coupler)
            elif name.startswith('PulseCorrection'):
                for key in list(self.backend.experiment_context.compensates.keys()):
                    if str(self.backend.experiment_context.compensates[key]) == name:
                        self.backend.experiment_context.compensates.pop(key)
            elif name == 'working_dc':
                self.backend.experiment_context._working_dc = None
            elif name.startswith('IQdiscriminator'):
                if isinstance(self.backend.experiment_context.discriminators, list):
                    for v in self.backend.experiment_context.discriminators:
                        if v.name == name:
                            self.backend.experiment_context.discriminators.pop(v)
                else:
                    self.backend.experiment_context.discriminators = None
            self.model().refresh_auto()
        else:
            QMessageBox().critical(self, 'Error', f'Please choose one object!')
