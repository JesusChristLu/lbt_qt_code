# -*- coding: utf-8 -*-
# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6 import QtGui
from PySide6.QtWidgets import QStyledItemDelegate, QLineEdit, QComboBox, QStyleOptionViewItem, QWidget, QDoubleSpinBox
from loguru import logger

from .style import OptionType
from .tree_model_options import QTreeModelOptions


class QOptionsDelegate(QStyledItemDelegate):

    choose_exp_signal = Signal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex):
        style = QTreeModelOptions.style(index) or 4

        if not isinstance(style, int):
            style = style.value

        if style == OptionType.com_box.value:
            editor = QComboBox(parent)
            editor.setFrame(False)
            editor.setEditable(False)
            vali_type = QTreeModelOptions.vali_type(index)
            vali_type = [str(v) for v in vali_type]
            editor.addItems(vali_type)

        else:
            editor = QLineEdit(parent)
            editor.setFrame(False)

        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex):
        model = index.model()
        text = model.data(index, Qt.ItemDataRole.EditRole)

        if isinstance(editor, QComboBox):
            editor.setCurrentText(text)
        elif isinstance(editor, QDoubleSpinBox):
            editor.setValue(float(text))
        elif isinstance(editor, QLineEdit):
            editor.setText(str(text))
        else:
            logger.error(f'QOptionsDelegate can not find {editor}')

    def setModelData(self, editor: QWidget, model: QTreeModelOptions, index: QModelIndex):
        text = ""
        emit_signal = False

        # bug fix 2023/04/28: only parallel exp name need to emit signal
        left_index = model.index(index.row(), index.column() - 1, index.parent())
        key = model.data(left_index)

        if isinstance(editor, QComboBox):
            text = editor.currentText()
            emit_signal = self.is_in_clickable_area(editor) and key == "exp_name"
        elif isinstance(editor, QDoubleSpinBox):
            text = editor.value()
        elif isinstance(editor, QLineEdit):
            text = editor.text()

        text = None if text == '' else text

        model.setData(index, text)
        if emit_signal:
            self.choose_exp_signal.emit()

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    @staticmethod
    def is_in_clickable_area(editor: QWidget):
        # Check if the mouse is in the specified area.
        pos = editor.mapFromGlobal(QtGui.QCursor.pos())
        if pos is not None:
            return pos.x() > -90 and pos.y() > 0
        else:
            return False
