# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

"""GUI front-end interface for pyqcat-visage in PySide6."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderTextWidget(QWidget):
    """QTableView or Tree with placeholder text if empty.

    This class extends the `QWidget` class.
    """
    __placeholder_text = "The table is empty."

    def __init__(self, placeholder_text: str = None):
        """
        Args:
            placeholder_text (str): Placeholder text..  Defaults to None.
        """
        self._placeholder_text = placeholder_text or self.__placeholder_text

        self._placeholder_label = QLabel(self._placeholder_text, self)
        self.setup_placeholder_label()

    def setup_placeholder_label(self):
        """QComponents will be displayed here when you create them."""
        self.update_placeholder_text()

        if not self.layout():
            layout = QVBoxLayout()
            self.setLayout(layout)

        self.layout().addWidget(self._placeholder_label)

    def update_placeholder_text(self, text=None):
        """Update the placeholder text to the given string.

        Args:
            text (str): New placeholder text.  Defaults to None.
        """
        if text:
            self._placeholder_text = text

        label = self._placeholder_label
        label.setText(self._placeholder_text)

        # Text
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        # transparent
        label.setAutoFillBackground(False)
        label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # color PlaceholderText
        palette = self.palette()
        # This enum value has been introduced in Qt 5.12
        if hasattr(palette, 'PlaceholderText'):
            placeholder_color = palette.PlaceholderText
        else:
            placeholder_color = QPalette.WindowText
        color = palette.color(placeholder_color)
        palette.setColor(QPalette.Text, color)
        palette.setColor(QPalette.Text, color)
        label.setPalette(palette)

    def show_placeholder_text(self):
        """Show the placeholder text."""
        self._placeholder_label.show()

    def hide_placeholder_text(self):
        """Hide the placeholder text."""
        self._placeholder_label.hide()
