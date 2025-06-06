# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/26
# __author:       YangChao Zhao


from typing import TYPE_CHECKING

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QFont, QContextMenuEvent
from PySide6.QtWidgets import QWidget, QTableView
from pyqcat_visage.gui.widgets.base.placeholder_text_widget import PlaceholderTextWidget

if TYPE_CHECKING:
    from ...main_window import VisageGUI
    from ..heatmap_widget import HeatMapWindow


class QTableModelBase(QAbstractTableModel):

    def __init__(
        self, gui: 'VisageGUI',
        parent: "HeatMapWindow" = None,
        table_view: QTableView = None,
    ):
        super().__init__(parent=parent)

        self._tableView = table_view
        self._row_count = -1

        self.gui = gui
        self.widget = parent
        self.columns = ['Name', 'Value']
        self.columns_ratio = [1, 1]

    @property
    def backend(self):
        """Returns the backend."""
        return self.gui.backend

    @property
    def model_data(self):
        return []

    def refresh_auto(self, check_count: bool = True):
        """Automatic refresh, update row count, view, etc."""
        # We could not do if the widget is hidden
        new_count = self.rowCount()

        def model_reset():
            # When a model is reset it should be considered that all
            # information previously retrieved from it is invalid.
            # This includes but is not limited to the rowCount() and
            # columnCount(), flags(), data retrieved through data(), and roleNames().
            # This will lose the current selection.
            self.modelReset.emit()

            # for some reason the horizontal header is hidden even if I call this in init
            self._tableView.horizontalHeader().show()

            # update row count
            self._row_count = new_count

            if not new_count:
                if hasattr(self._tableView, "update_placeholder_text"):
                    self._tableView.update_placeholder_text("No data found, please check the search conditions.")

        # The table model supports two refresh mechanisms. When updating the model
        # with the number of Item records, the row number can be used for refresh
        # detection; For updating the specific information of the model item, you
        # can directly refresh without considering the number of rows
        if check_count and self._row_count != new_count:
            model_reset()
        elif not check_count:
            model_reset()
        else:
            pass

    def update_view(self):
        """Updates the view."""
        if self._tableView:
            self._tableView.resizeColumnsToContents()

    def rowCount(self, parent: QModelIndex = None) -> int:
        """Returns the number of rows.

        Args:
            parent (QModelIndex): Unused.  Defaults to None.

        Returns:
            int: The number of rows
        """
        if self.backend and self.model_data:
            num = len(self.model_data)
            if num == 0:
                if isinstance(self._tableView, PlaceholderTextWidget):
                    self._tableView.show_placeholder_text()
            else:
                if isinstance(self._tableView, PlaceholderTextWidget):
                    self._tableView.hide_placeholder_text()
            return num
        else:
            if isinstance(self._tableView, PlaceholderTextWidget):
                self._tableView.show_placeholder_text()
            return 0

    def columnCount(self, parent: QModelIndex = None) -> int:
        """Returns the number of columns.

        Args:
            parent (QModelIndex): Unused.  Defaults to None.

        Returns:
            int: The number of columns
        """
        return len(self.columns)

    def headerData(self,
                   section,
                   orientation: Qt.Orientation,
                   role=Qt.ItemDataRole.DisplayRole):
        """Set the headers to be displayed.

        Args:
            section (int): Section number
            orientation (Qt orientation): Section orientation
            role (Qt display role): Display role.  Defaults to DisplayRole.

        Returns:
            str: The header data, or None if not found
        """

        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                if section < len(self.columns):
                    return self.columns[section]
            else:
                return section

        elif role == Qt.ItemDataRole.FontRole:
            font = QFont()
            if len(self.columns) > 5:
                font.setPointSize(8)
            font.setBold(True)
            return font

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Depending on the index and role given, return data. If not returning
        data, return None (PySide equivalent of QT "invalid QVariant").

        Returns:
            str: Data depending on the index and role
        """

        if not index.isValid() or not self.backend:
            return

        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_data(index)

        if role == Qt.ItemDataRole.EditRole:
            return self.data(index, Qt.ItemDataRole.DisplayRole)

        # The font used for items rendered with the default delegate. (QFont)
        elif role == Qt.ItemDataRole.FontRole:
            if index.column() == 0:
                font = QFont()
                font.setBold(True)
                return font

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)

    def _display_data(self, index: QModelIndex):
        pass


class QTableViewBase(QTableView):

    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        QTableView.__init__(self, parent)
        self.clicked.connect(self.view_clicked)
        self._define_style()
        self.right_click_menu = None
        self.init_right_click_menu()

    def init_right_click_menu(self):
        pass

    def _define_style(self):
        pass

    def view_clicked(self, index: QModelIndex):
        pass

    def contextMenuEvent(self, event: QContextMenuEvent):
        if not self.right_click_menu:
            self.init_right_click_menu()

        if self.right_click_menu:
            self.right_click_menu.action = self.right_click_menu.exec_(
                self.mapToGlobal(event.pos())
            )
