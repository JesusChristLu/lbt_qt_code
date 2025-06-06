# -*- coding: utf-8 -*-
# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao
import time

from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QStyledItemDelegate,
    QComboBox,
    QStyleOptionViewItem,
    QWidget,
    QLineEdit,
)
from loguru import logger

from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.gui.widgets.combox_custom.combox_search import SearchComboBox


class EditorWrapper:
    def __init__(self, editor):
        self.editor = editor

    def deleteEditor(self):
        # Manually delete custom objects
        self.editor.deleteLater()


class QOptionsDelegate(QStyledItemDelegate):
    choose_type_signal = Signal(tuple)
    point_label_signal = Signal(str)

    def __init__(self, parent: QWidget = None, gui: "VisageGUI" = None, editor_keys: list = None):
        self.gui = gui
        super().__init__(parent)
        self.cache_editor = {}
        self.editor_keys = editor_keys

        self.timestamp = 0
        self.init_time()

    def init_time(self):
        self.timestamp = int(time.time())

    def createEditor(
            self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ):
        if time.time() - self.timestamp > 3600:
            self.clear()
            self.init_time()
        model = index.model()
        left_index = model.index(index.row(), index.column() - 1, index.parent())
        key = left_index.data()
        if key in self.cache_editor:
            editor = self.cache_editor[key].editor
        else:
            if key == "pulse_type":
                editor = QComboBox(parent)
                editor.setFrame(False)
                editor.setEditable(False)
                editor.addItems(GUI_CONFIG.cz_pulse_types)
                editor_wrapper = EditorWrapper(editor)
                self.cache_editor[key] = editor_wrapper
            elif key == "point_label":
                editor = SearchComboBox(parent, True)
                editor.setFrame(False)
                # editor.completer.activated.connect(self.change_labels)
                editor_wrapper = EditorWrapper(editor)
                # editor.setEditable(False)
                if self.gui:
                    sample = self.gui.backend.config.system.sample
                    env_name = self.gui.backend.config.system.env_name
                    username = self.gui.backend.username
                    res = self.gui.backend.db.query_point_label_list(username, sample, env_name)
                    if res.get("code") == 200:
                        labels = res.get("data", [])
                        old_point = self.gui.backend.config.system.point_label
                        if old_point not in labels:
                            editor_wrapper.editor.addItem(f"{old_point} (new)", old_point)
                        for label in labels:
                            editor_wrapper.editor.addItem(label, label)
                    self.cache_editor[key] = editor_wrapper
                    editor = editor_wrapper.editor
            else:
                editor = QLineEdit(parent)
                editor.setMaxLength(100000)

            self.init_time()
        editor.setWindowTitle(key)
        # editor.windowTitle()
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex):
        model = index.model()
        text = model.data(index, Qt.ItemDataRole.EditRole)

        if isinstance(editor, QComboBox):
            editor.setCurrentText(text)
        elif isinstance(editor, QLineEdit):
            editor.setText(str(text))
        else:
            logger.error(f"QOptionsDelegate can not find {editor}")

    def setModelData(self, editor: QWidget, model, index: QModelIndex):
        if self.editor_keys and editor.windowTitle() not in self.editor_keys:
            return
        text = ""
        parent_index = model.parent(index)
        parent_key = model.data(parent_index)
        if isinstance(editor, SearchComboBox):
            text = editor.text()
            pre_text = model.data(index)
            if pre_text != text:
                self.point_label_signal.emit(text)
                return
        elif isinstance(editor, QComboBox):
            text = editor.currentText()
            pre_text = model.data(index)
            if text != pre_text:
                self.choose_type_signal.emit((text, parent_key))
                # bugfix: update pulse type must direct return,
                # continue set model maybe cause index disorder
                return
        elif isinstance(editor, QLineEdit):
            text = editor.text()

        text = None if text == "" else text

        model.setData(index, text)

    def clear(self):
        for _, editor in self.cache_editor.items():
            editor.deleteEditor()
        self.cache_editor.clear()

    def destroyEditor(self, editor, index):
        # Gets a custom object for the wrapper editor
        editor_wrapper = getattr(editor, 'wrapper', None)
        if editor_wrapper:
            # Manually delete custom objects to prevent editor deletion
            editor_wrapper.deleteEditor()
