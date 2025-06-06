# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/08/1
# __author:       XuYao

from PySide6.QtCore import Qt, QRect, QSortFilterProxyModel
from PySide6.QtWidgets import QWidget, QComboBox, QLineEdit, QApplication, QCompleter
from PySide6.QtGui import QMouseEvent
import sys

# .widgets.combox_custom.combox_search


class SearchComboBox(QComboBox):
    """combobox with search function"""

    def __init__(self, parent=None, variable: bool = False):
        super().__init__(parent)
        # self.setFocusPolicy(Qt.StrongFocus)
        self.setEditable(True)
        self.variable = variable

        # Add a filter model to filter matches
        self.pFilterModel = QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(
            Qt.CaseInsensitive
        )  # Case insensitivity
        self.pFilterModel.setSourceModel(self.model())

        # Add a QCompleter using the filter model
        self.completer = QCompleter(self.pFilterModel, self)
        # Always display all (filtered) completion results
        self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)  # Case insensitivity
        self.setCompleter(self.completer)

        # Qcombobox Slot function when the text in the edit bar changes
        self.lineEdit().textEdited.connect(self.pFilterModel.setFilterFixedString)
        self.completer.activated.connect(self.on_completer_activated)

    # When selected in the Qcompleter list, select the appropriate subproject from the drop-down list of items
    def on_completer_activated(self, text):
        if text:
            index = self.findText(text)
            self.setCurrentIndex(index)
            # self.activated[str].emit(self.itemText(index))

    # Update the model for filters and completers when the model changes
    def setModel(self, model):
        super().setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)

    # Update model columns for filters and completers when model columns change
    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        super().setModelColumn(column)

    # Responds to the return button event
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Enter & e.key() == Qt.Key_Return:
            text = self.currentText()
            index = self.findText(text, Qt.MatchExactly | Qt.MatchCaseSensitive)
            if self.variable and index < 0:
                new_text = f"{text} (new)"
                self.addItem(new_text, text)
                index = self.findText(new_text, Qt.MatchExactly | Qt.MatchCaseSensitive)
            self.setCurrentIndex(index)
            self.hidePopup()
            super().keyPressEvent(e)
        else:
            super().keyPressEvent(e)

    def text(self):
        if self.variable:
            text = self.currentData()
            if text is None:
                text = self.currentText()
        else:
            text = self.currentText()
        return text


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SearchComboBox()
    l = [
        "",
        "3ewqc",
        "2wqpu",
        "1kjijhm",
        "4kjndw",
        "5ioijb",
        "6eolv",
        "11ofmsw",
    ]
    win.addItems(l)
    win.show()
    sys.exit(app.exec_())
