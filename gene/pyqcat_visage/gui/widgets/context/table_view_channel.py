# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/16
# __author:       YangChao Zhao

from PySide6.QtWidgets import QHeaderView, QWidget, QMenu

from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.right_click_menu import bind_action
from ..base.table_structure import QTableViewBase


class QTableViewChannelWidget(QTableViewBase, PlaceholderTextWidget):

    def __init__(self, parent: QWidget):
        self.ui = parent.parent().parent().parent()
        QTableViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self,
            "Click Create Chip Line to Set Your Chip (Only Super User)!"
        )

    def _define_style(self):
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # noinspection DuplicatedCode
    def init_right_click_menu(self):
        menu = QMenu(self)

        menu._chip_layout = bind_action(menu, 'Chip Layout', u":/cpu.png")
        menu._save_chip_line = bind_action(menu, 'Save Chip Line', u":/save.png")
        menu._sync_chip_line = bind_action(menu, 'Sync Chip Line', u":/reset.png")
        menu.addSeparator()
        menu._init_bit = bind_action(menu, 'Init Base Qubit', u":/init_qubit.png")
        menu._init_config = bind_action(menu, 'Init Config Data', u":/init_config.png")
        menu._init_qubit_pair = bind_action(menu, 'Init Qubit Pair', u":/pair.png")
        menu.addSeparator()
        menu._set_env_bit = bind_action(menu, 'Set Env Bit', u":/set_env_bit.png")
        menu._set_working_dc = bind_action(menu, 'Set Working DC', u":/crosstalk.png")
        menu.addSeparator()
        menu._auto_add_channel = bind_action(menu, 'Auto Add Channel', u":/create_context.png")
        menu._auto_com_channel = bind_action(menu, 'Auto Com Channel', u":/clear.png")

        menu._set_env_bit.triggered.connect(self.ui.context_set_env_bit)
        menu._set_working_dc.triggered.connect(self.set_working_dc)
        menu._chip_layout.triggered.connect(self.ui.chip_layout)
        menu._save_chip_line.triggered.connect(self.ui.chip_save_line)
        menu._sync_chip_line.triggered.connect(self.ui.sync_chip_line)
        menu._init_bit.triggered.connect(self._init_bit)
        menu._init_config.triggered.connect(self._init_config)
        menu._init_qubit_pair.triggered.connect(self._init_qubit_pair)
        menu._auto_add_channel.triggered.connect(self._auto_add_channel)
        menu._auto_com_channel.triggered.connect(self._auto_com_channel)

        self.right_click_menu = menu

    def _auto_add_channel(self):
        indexes = self.selectedIndexes()

        res = self._check_indexes(indexes)
        if res:
            module_name, start_channel = res

            for i, index in enumerate(indexes):
                bit_name, bit = self.ui.channel_table_model.bit_from_index(index)
                if bit_name.startswith('c') and module_name in ['xy_channel', 'readout_channel',
                                                                "bus", "m_lo", "xy_lo"]:
                    continue
                bit[module_name] = start_channel + i

    def _auto_com_channel(self):
        indexes = self.selectedIndexes()

        res = self._check_indexes(indexes)
        if res:
            module_name, start_channel = res

            for i, index in enumerate(indexes):
                bit_name, bit = self.ui.channel_table_model.bit_from_index(index)
                if bit_name.startswith('c') and module_name in ['xy_channel', 'readout_channel',
                                                                "bus", "m_lo", "xy_lo"]:
                    continue
                bit[module_name] = start_channel

    def _check_indexes(self, indexes: list):
        if indexes:
            col = [_index.column() for _index in indexes]
            if len(set(col)) == 1 and (0 < col[0] < 5 or 5 < col[0] < 10):
                first_index = indexes[0]
                bit_name, first_bit = self.ui.channel_table_model.bit_from_index(first_index)
                module_name = self.ui.channel_table_model.columns[first_index.column()]
                start_channel = first_bit.get(module_name, 1)
                start_channel = 1 if start_channel == '-' else int(start_channel)
                return module_name, start_channel

    def _init_bit(self):
        indexes = self.selectedIndexes()
        bit_names = []

        if indexes:
            bit_names = [self.ui.channel_table_model.bit_from_index(index)[0] for index in indexes]

        if bit_names:
            self.ui.chip_init_base_qubit(bit_names)

    def _init_config(self):
        indexes = self.selectedIndexes()
        bit_names = []

        if indexes:
            bit_names = [self.ui.channel_table_model.bit_from_index(index)[0] for index in indexes]

        if bit_names:
            self.ui.chip_init_config_data(bit_names)

    def _init_qubit_pair(self):
        indexes = self.selectedIndexes()
        bit_names = []

        if indexes:
            bit_names = [self.ui.channel_table_model.bit_from_index(index)[0] for index in indexes]

        self.ui.create_qubit_pair(bit_names)

    def set_working_dc(self):
        indexes = self.selectedIndexes()
        bit_names = []

        if indexes:
            bit_names = [self.ui.channel_table_model.bit_from_index(index)[0] for index in indexes]

        self.ui.context_set_working_dc(bit_names)
