# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/21
# __author:       XuYao

import re
import sys
from abc import abstractmethod

from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QApplication,
    QMessageBox,
    QInputDialog,
    QDialog,
    QListWidget,
    QHBoxLayout,
    QDialogButtonBox,
)

from pyQCat.invoker import MULTI_THREAD
from pyqcat_visage.exceptions import PyQCatVisageError
from pyqcat_visage.gui.main_window_layout import TitleGUI
from pyqcat_visage.gui.tools.theme import CustomTheme


class TitleLayout:
    style = None
    title_button_style = """QPushButton[objectName='WindowMin']
                                {
                                image: url(icon:/window-min.svg);
                                text-align:top;
                                background:#6DDF6D;
                                border-radius:5px;
                                border:none;
                                font-size:13px;
                                }
                                QPushButton[objectName='WindowMin']:hover{background:green;}

                                QPushButton[objectName='WindowMax']
                                {
                                image: url(icon:/window-max.svg);
                                background:#F7D674;border-radius:5px;
                                border:none;
                                font-size:13px;
                                }
                                QPushButton[objectName='WindowMax']:hover{background:orange;}

                                QPushButton[objectName='WindowClose']
                                {
                                image: url(icon:/window-close.svg);
                                background-color: #F76677;
                                border-radius:5px;
                                border:none;
                                font-size:13px;
                                }
                                QPushButton[objectName='WindowClose']:hover{background:red;}
                                """
    custom_theme = CustomTheme()
    is_admin = False
    is_super = False
    username = ""
    group_name = ""

    def __init__(self, *args, **kwargs):
        self.new_window = None
        if type(self) == TitleLayout:
            raise TypeError("TitleLayout class may not be instantiated")

    @classmethod
    def init_user(cls, backend: "BaseBackend"):
        cls.is_super = backend.is_super
        cls.is_admin = backend.is_admin
        cls.username = backend.username
        cls.group_name = backend.group_name

    def reset_window_layout(self):
        pass

    # @abstractmethod
    # bugfix: Nuitka/Qt bug https://github.com/Nuitka/Nuitka/issues/1988
    def close_(self):
        if self.new_window:
            if self.new_window.sub_window.__class__.__name__ == "VisageExtension":
                self.new_window.on_WindowClose_clicked()
            else:
                self.new_window.close()
            return True
        return False

    def hide(self):
        if self.new_window:
            self.new_window.hide()

    def set_multi_thread_title(self):
        if MULTI_THREAD != 0 and self.new_window:
            self.new_window._ui.TitleLabel.setText(
                f"{self.windowTitle()}-{MULTI_THREAD}"
            )
        else:
            self.new_window._ui.TitleLabel.setText(f"{self.windowTitle()}")

    # @abstractmethod
    def show(self) -> None:
        if not self.new_window:
            self.new_window = TitleGUI(sub_window=self)
            self.set_multi_thread_title()
        if TitleWindow.style and TitleWindow.style != "default":
            new_style = re.sub(
                "QHeaderView::(up|down)-arrow[^}]*\}", "", TitleWindow.style
            )
            self.new_window.setStyleSheet(new_style + TitleWindow.title_button_style)
        else:
            if TitleWindow.style is None:
                # add path to find icon
                TitleWindow.custom_theme.add_icon_path()
            self.new_window.setStyleSheet(TitleWindow.title_button_style)

        # bug solve:
        # When the window is minimized, it only changes the state of new_window,
        # but the state of the child control does not change. The next time the
        # show method is executed, the state of new_window is not changed from
        # WindowMinimized to WindowNoState
        # bug solve:
        # First, maximize the page, then minimize it, and the page cannot be displayed
        # when opened again
        ws = self.new_window.windowState()
        if ws not in [
            Qt.WindowState.WindowMinimized,
            Qt.WindowState.WindowNoState,
            Qt.WindowState.WindowMaximized,
            Qt.WindowState.WindowFullScreen,
            Qt.WindowState.WindowActive,
        ]:
            self.new_window.setWindowState(Qt.WindowState.WindowMaximized)
        elif ws == Qt.WindowState.WindowMinimized:
            self.new_window.setWindowState(Qt.WindowState.WindowNoState)
        self.reset_window_layout()
        self.new_window.show()
        self.new_window.activateWindow()

    def isMinimized(self):
        if self.new_window:
            return self.new_window.isMinimized()

    def showNormal(self):
        if self.new_window:
            return self.new_window.showNormal()

    def activateWindow(self):
        if self.new_window:
            return self.new_window.activateWindow()

    # @abstractmethod
    def setStyleSheet(self, style):
        """Set a fixed button style"""
        TitleWindow.style = style
        if style == "default":
            new_style = TitleWindow.title_button_style
        else:
            new_style = (
                re.sub("QHeaderView::(up|down)-arrow[^}]*\}", "", style)
                + TitleWindow.title_button_style
            )

        # bug record:
        # |   ERROR    |  ERROR [restore_window_settings]: 'NoneType' object has no attribute 'setStyleSheet'
        if self.new_window:
            self.new_window.setStyleSheet(new_style)

    # @abstractmethod
    # bugfix: Nuitka/Qt bug https://github.com/Nuitka/Nuitka/issues/1988
    def showFullScreen_(self):
        """full screen"""
        if self.new_window:
            self.new_window.showFullScreen()
            return True
        return False

    @abstractmethod
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Esc to exit full screen."""
        key = event.key()
        if key == int(Qt.Key.Key_Escape):
            if self.new_window:
                self.new_window.showNormal()
                return True
        return False

    def handler_ret_data(
        self, ret_data, show_suc: bool = False, describe: str = "Success!"
    ):
        if ret_data:
            code = ret_data.get("code")
            msg = ret_data.get("msg")
            if ret_data.get("code") < 300:
                if show_suc:
                    QMessageBox().information(
                        self.new_window, "Success", f"{msg or describe}"
                    )
            elif ret_data.get("code") == 800:
                QMessageBox().warning(self.new_window, "Warning", msg)
            elif ret_data.get("code") == 601:
                QMessageBox().critical(self.new_window, f"OperationError-{code}", msg)
            elif ret_data.get("code") == 602:
                QMessageBox().critical(self.new_window, f"LogicError-{code}", msg)
            else:
                QMessageBox().critical(self.new_window, f"Error-{code}", msg)

    def handle_error(self, error: PyQCatVisageError):
        if error:
            QMessageBox().critical(
                self.new_window,
                f"{error.__class__.__name__}-{error.code}",
                error.message,
            )

    def ask_ok(self, msg: str, title: str = None):
        title = title or "PyQCat: Quantum Chip Calibration"
        reply = QMessageBox().question(
            self.new_window,
            title,
            msg,
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            return True

    def ask_input(self, title: str, msg: str):
        return QInputDialog.getText(self.new_window, title, msg)

    def ask_items(self, title: str, items: list):
        return QInputDialog.getItem(
            self.new_window, title, "Save Type", items, 0, False
        )

    def ask_mul_items(self, title: str, items: list):
        dialog = MultiSelectDialog(self.new_window, items)
        dialog.setWindowTitle(title)
        if dialog.exec():
            return dialog.selected

    def saveGeometry_(self):
        return self.new_window.saveGeometry()

    def saveState_(self):
        return self.new_window.saveState()

    def restoreGeometry_(self, gem):
        if not self.new_window:
            self.new_window = TitleGUI(sub_window=self)
            self.set_multi_thread_title()
        self.new_window.restoreGeometry(gem)


class TitleWindow(TitleLayout, QMainWindow):
    def __init__(self, *args, parent: QWidget = None, **kwargs):
        super().__init__(parent)
        super(TitleLayout, self).__init__(parent)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if not super(TitleWindow, self).keyPressEvent(event):
            super(TitleLayout, self).keyPressEvent(event)


class TitleDialog(TitleLayout, QDialog):
    def __init__(self, *args, parent: QWidget = None, **kwargs):
        super().__init__(parent)
        super(TitleLayout, self).__init__(parent)

    def show(self) -> None:
        super().show()
        # bugfix: Memory leak caused by Dialog creation
        # self.new_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    def accept(self):
        super(TitleLayout, self).accept()
        self.close_()

    def reject(self):
        super(TitleLayout, self).reject()
        self.close_()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if not super(TitleDialog, self).keyPressEvent(event):
            super(TitleLayout, self).keyPressEvent(event)


class MultiSelectDialog(QDialog):
    def __init__(self, parent, items, selected: set = None):
        super().__init__(parent)

        self.items = items
        self.selected = selected or set()

        self.list_widget = QListWidget()
        self.list_widget.addItems(self.items)
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        if selected:
            for item in selected:
                self.list_widget.item(items.index(item)).setSelected(True)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        layout = QHBoxLayout()
        layout.addWidget(self.list_widget)
        layout.addWidget(button_box)
        self.setLayout(layout)

        # accept button
        button_box.accepted.connect(self.accept)
        # reject
        button_box.rejected.connect(self.reject)

    def accept(self) -> None:
        self.selected = set([item.text() for item in self.list_widget.selectedItems()])
        super().accept()

    def reject(self) -> None:
        self.selected = set()
        super().reject()


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    win = TitleGUI()
    win.show()
    sys.exit(app.exec_())
