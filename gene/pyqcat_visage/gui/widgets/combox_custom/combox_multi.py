# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/08/1
# __author:       XuYao

import copy
import re

from typing import List
from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import QComboBox, QLineEdit, QListWidget, QCheckBox, QListWidgetItem, QStyledItemDelegate, \
    QHBoxLayout, QSpacerItem, QSizePolicy
from pyqcat_visage.config import GUI_CONFIG


# # .widgets.combox_custom.combox_multi

class CustomListWidget(QListWidget):
    def __init__(self):
        super().__init__()

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is not None:
            checkbox = self.itemWidget(item)
            if checkbox is not None:
                checkbox.event(event)
                return True
        return super().mousePressEvent(event)


class CustomCheckBox(QCheckBox):
    def __init__(self):
        super().__init__()

    def event(self, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                # 获取当前复选框的矩形范围
                rect = self.rect()

                # 扩大矩形范围
                expanded_rect = rect.adjusted(-1, -1, 1, 1)

                # 如果鼠标事件位于扩大后的矩形范围内，则切换复选框状态
                if expanded_rect.contains(event.pos()):
                    self.setChecked(not self.isChecked())

                return True  # 消费该事件，阻止传递给默认处理方法
        elif event.type() == QEvent.Type.MouseButtonRelease:
            return True
        return super().event(event)


class QMultiComboBox(QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items = None
        self.row_num = 0
        self.Selectedrow_num = 0
        self.qCheckBox = []
        self.qLineEdit = QLineEdit()
        # self.qLineEdit.setReadOnly(True)
        self.qListWidget = None

        self.qLineEdit.editingFinished.connect(self.update_line_text)
        self.setLineEdit(self.qLineEdit)
        self.multi_select = []
        # self.qLineEdit.textChanged.connect(self.printResults)

    def set_units(self, units: List[str]):
        self.multi_select = units
        self.loadItems()

    def loadItems(self):
        if not self.multi_select:
            self.clear()
            return
        self.items = copy.deepcopy(self.multi_select)
        self.items.insert(0, 'all')
        self.row_num = len(self.items)
        # self.Selectedrow_num = 0
        self.qCheckBox = []
        # self.qLineEdit = QLineEdit()
        # self.qLineEdit.setReadOnly(True)
        self.qListWidget = QListWidget()
        delegate = ListWidgetItemDelegate(self.qListWidget)
        self.qListWidget.setItemDelegate(delegate)
        self.addQCheckBox(0)
        self.qCheckBox[0].stateChanged.connect(self.All)
        for i in range(1, self.row_num):
            self.addQCheckBox(i)
            self.qCheckBox[i].stateChanged.connect(self.showMessage)
        self.setModel(self.qListWidget.model())
        self.setView(self.qListWidget)

    def update_line_text(self):

        if not self.items:
            return
        text_list = self.text()
        select_list = self.Selectlist()
        if not text_list:
            for i, text in enumerate(self.items):
                if self.qCheckBox[i].isChecked():
                    self.qCheckBox[i].setChecked(False)

        for text in text_list:
            if not text:
                continue
            if text in self.items:
                if text not in select_list:
                    index = self.items[:].index(text)
                    if not self.qCheckBox[index].isChecked():
                        self.qCheckBox[index].setChecked(True)
        for i, text_ in enumerate(self.items[1:]):
            if text_ not in text_list:
                if self.qCheckBox[i + 1].isChecked():
                    self.qCheckBox[i + 1].setChecked(False)
        # if text_item_length == len(select_list) == 0:
        #     self.qCheckBox[0].setChecked(False)
        # if text_item_length == len(select_list) == len(self.items) - 1:
        #     self.qCheckBox[0].setChecked(True)

    def text(self):
        text_str = re.sub(r"(\n|\s)+", "", self.qLineEdit.text())
        text_str = re.sub(r"(，)+", ",", text_str).strip(",")
        new_text = []
        for text in text_str.split(","):
            if text:
                new_text.append(text.strip(" "))
        return new_text

    def get_multi_select(self, *args, **kwargs):
        """SELECT data from db and update self._multi_select"""
        # self.qLineEdit.setReadOnly(True)
        pass

    def showPopup(self):
        if not self.items:
            return QComboBox.showPopup(self)
        #  重写showPopup方法，避免下拉框数据多而导致显示不全的问题
        select_list = self.Selectlist()  # 当前选择数据
        self.loadItems()  # 重新添加组件
        for select in select_list:
            index = self.items[:].index(select)
            self.qCheckBox[index].setChecked(True)  # 选中组件
        return QComboBox.showPopup(self)

    def printResults(self):
        list = self.Selectlist()

    def addQCheckBox(self, i):
        self.qCheckBox.append(CustomCheckBox())
        qItem = QListWidgetItem(self.qListWidget)
        self.qCheckBox[i].setText(self.items[i])
        self.qListWidget.setItemWidget(qItem, self.qCheckBox[i])

    def Selectlist(self):
        Outputlist = []
        for i in range(1, self.row_num):
            if self.qCheckBox[i].isChecked() == True:
                Outputlist.append(self.qCheckBox[i].text())
        self.Selectedrow_num = len(Outputlist)
        return Outputlist

    def showMessage(self):
        Outputlist = self.Selectlist()
        # self.qLineEdit.setReadOnly(False)
        self.qLineEdit.clear()
        show = ','.join(Outputlist)

        if self.Selectedrow_num == 0:
            self.qCheckBox[0].setCheckState(Qt.CheckState.Unchecked)
        elif self.Selectedrow_num == self.row_num - 1:
            self.qCheckBox[0].setCheckState(Qt.CheckState.Checked)
        else:
            self.qCheckBox[0].setCheckState(Qt.CheckState.PartiallyChecked)
        self.qLineEdit.setText(show)
        # self.qLineEdit.setReadOnly(True)

    def All(self, state):
        if state == 2:
            for i in range(1, self.row_num):
                self.qCheckBox[i].setChecked(True)
        elif state == 1:
            if self.Selectedrow_num == 0:
                self.qCheckBox[0].setCheckState(Qt.CheckState.Checked)
        elif state == 0:
            self.reset()

    def reset(self):
        for i in range(self.row_num):
            self.qCheckBox[i].setChecked(False)

    def clear(self) -> None:
        super().clear()
        self.items = []
        self.row_num = 0

    def currentText(self):
        text = QComboBox.currentText(self).split(',')
        if text.__len__() == 1:
            if not text[0]:
                return []
        return text

    def set_check(self, text):
        if text in self.multi_select:
            index = self.multi_select.index(text) + 1
            self.qCheckBox[index].setCheckState(Qt.CheckState.Checked)

    def set_text(self, text):
        if text:
            self.reset()
            if isinstance(text, str):
                text = text.split(",")

            super().setCurrentText(','.join(text))
            for t in text:
                self.set_check(t)
        else:
            super().setCurrentText(','.join(text))
            self.reset()
            self.set_check(text)

    def setCurrentText(self, text):
        self.set_text(text)


class ListWidgetItemDelegate(QStyledItemDelegate):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.listWidget = parent

    def sizeHint(self, option, index):
        size = QStyledItemDelegate.sizeHint(self, option, index)
        size.setHeight(GUI_CONFIG.multi_box_row_height)
        return size


if __name__ == '__main__':
    from PySide6 import QtWidgets, QtCore
    import sys

    items = ['Python', 'R', 'Java', 'C++']

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    comboBox1 = QMultiComboBox(Form)
    comboBox1.setGeometry(QtCore.QRect(10, 10, 400, 20))
    comboBox1.setMinimumSize(QtCore.QSize(100, 20))
    # comboBox1.loadItems(items)
    comboBox1.multi_select = items
    comboBox1.loadItems()

    Form.show()
    sys.exit(app.exec_())
