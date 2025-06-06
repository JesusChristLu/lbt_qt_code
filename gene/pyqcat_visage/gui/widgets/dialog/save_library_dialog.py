# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/12/01
# __author:       YangChao Zhao

from PySide6.QtCore import Qt

from .save_library_dialog_ui import Ui_Dialog
from ..title_window import TitleDialog


class QSaveLibraryDialog(TitleDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.setWindowFlags(Qt.WindowType.MSWindowsFixedSizeDialogHint)
        self.show()

    def get_input(self):
        save_type = self.ui.type_com_box.currentText()
        describe = self.ui.describe_edit.text()
        items = [item.text() for item in self.ui.library_widget.selectedItems()]
        return save_type, describe, items

    def set_collections(self, collections: list):
        self.ui.library_widget.addItems(sorted(collections))
