# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

from enum import Enum

from PySide6.QtCore import Qt

VALUE = Qt.ItemDataRole.UserRole + 1
STYLE = Qt.ItemDataRole.UserRole + 2
VALI_TYPE = Qt.ItemDataRole.UserRole + 3


class OptionType(Enum):
    key = 0
    spin_box = 1
    com_box = 2
    check_box = 3
    line_editor = 4
