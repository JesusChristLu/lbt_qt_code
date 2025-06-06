# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QMenu, QAbstractItemView

from pyqcat_visage.gui.widgets.base.placeholder_text_widget import PlaceholderTextWidget
from ..base.right_click_menu import bind_action
from ..base.tree_structure import QTreeViewBase


class QTreeViewOptionsWidget(QTreeViewBase, PlaceholderTextWidget):
    def __init__(self, parent: QWidget):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        QTreeViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self, "Select a Experiment to edit\n\nfrom the Experiment Library"
        )

    def _define_style(self):
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)

    def init_right_click_menu(self):
        menu = QMenu(self)

        menu._full_option = bind_action(menu, "Full Options", ":/full-screen.png")
        menu._main_options = bind_action(menu, "Main Options", ":/collapse.png")
        menu.addSeparator()
        menu._save_to_db = bind_action(menu, "SaveDB", ":/save.png")
        menu._save_to_local = bind_action(menu, "SaveLocal", ":/save_as.png")
        menu.addSeparator()

        menu._full_option.triggered.connect(self._full_option)
        menu._main_options.triggered.connect(self._main_options)
        menu._save_to_db.triggered.connect(self.model().options_widget.save_experiment)
        menu._save_to_local.triggered.connect(
            self.model().options_widget.save_exp_to_local
        )

        self.right_click_menu = menu

    def _full_option(self):
        model = self.model()
        model.gui.force_refresh_options_widgets(is_full=True)

    def _main_options(self):
        model = self.model()
        model.gui.force_refresh_options_widgets(is_full=False)
