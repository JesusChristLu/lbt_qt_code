# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, Signal
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import QTableView, QAbstractItemView, QHeaderView, QMenu

from pyQCat.qubit import BaseQubit
from ..base.placeholder_text_widget import PlaceholderTextWidget
from ..base.right_click_menu import bind_action
from ..base.table_structure import QTableViewBase

if TYPE_CHECKING:
    from ..component_window import ComponentEditWindow


class QTableViewComponentWidget(QTableViewBase, PlaceholderTextWidget):
    choose_component_signal = Signal(BaseQubit)

    class MenuTable:
        IMPORT_ROOM = "import_room"
        IMPORT_IIR = "import_iir"
        IMPORT_FIR = "import_fir"
        IMPORT_RES = "import_res"
        TO_FILE = "to_file"

    def __init__(self, parent: "ComponentEditWindow"):
        """
        Args:
            parent (QtWidgets.QWidget): The parent widget
        """
        self.component_widget = parent
        QTableViewBase.__init__(self, parent)
        PlaceholderTextWidget.__init__(
            self,
            "Click Query All to index all components in database."
        )
        self.his_index = None
        self.q_menu_dict = {"import_room": self.menu_import_room, "import_iir": self.menu_import_iir,
                            "import_fir": self.menu_import_fir, "import_res": self.menu_import_res,
                            "to_file": self.menu_to_file}

        self.select_items = []

    def _define_style(self):
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QTableView.SelectItems)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setAutoFillBackground(True)

    @property
    def backend(self):
        """Returns the design."""
        return self.model().backend

    @property
    def gui(self):
        """Returns the GUI."""
        return self.model().gui

    def view_clicked(self, index: QModelIndex):
        """Select a component and set it in the component widget when you left mouse click.

        In the init, we had to connect with self.clicked.connect(self.viewClicked)

        Args:
            index (QModelIndex): The index
        """

        self.his_index = index

        if self.gui is None or not index.isValid():
            return

        model = self.model()
        component = model.component_from_index(index)
        if component:
            self.choose_component_signal.emit(component)

    def refresh_view(self):
        if self.his_index:
            self.view_clicked(self.his_index)

    def contextMenuEvent(self, event: QContextMenuEvent):
        menu = QMenu(self)
        indexes = self.selectedIndexes()
        component_list = [self.model().component_from_index(x) for x in self.selectedIndexes()]
        self.select_items = []
        if len(component_list) > 0:
            menu_select = [self.MenuTable.TO_FILE]
            if len(component_list) == 1:
                if component_list[0].name == "character.json":
                    menu_select.extend(
                        [self.MenuTable.IMPORT_ROOM, self.MenuTable.IMPORT_IIR, self.MenuTable.IMPORT_FIR])

                elif component_list[0].name.startswith("distortion_"):
                    menu_select.append(self.MenuTable.IMPORT_RES)

            self.select_items = component_list

            for x in menu_select:
                self.q_menu_dict[x](menu)
            menu.exec_(self.mapToGlobal(event.pos()))

    # def contextMenuEvent(self, event: QContextMenuEvent):
    #     if not self.right_click_menu:
    #         self.init_right_click_menu()
    #
    #     if self.right_click_menu:
    #         indexes = self.selectedIndexes()
    #         if len(indexes) > 0:
    #             index = indexes[0]
    #             component = self.model().component_from_index(index)
    #             if component.name == "character.json":
    #                 self.right_click_menu.import_room.setVisible(True)
    #                 self.right_click_menu.import_iir.setVisible(True)
    #                 self.right_click_menu.import_fir.setVisible(True)
    #                 self.right_click_menu.import_response.setVisible(False)
    #             elif component.name.startswith('distortion_'):
    #                 self.right_click_menu.import_room.setVisible(False)
    #                 self.right_click_menu.import_iir.setVisible(False)
    #                 self.right_click_menu.import_fir.setVisible(False)
    #                 self.right_click_menu.import_response.setVisible(True)
    #             else:
    #                 return
    #
    #         self.right_click_menu.action = self.right_click_menu.exec_(
    #             self.mapToGlobal(event.pos())
    #         )

    def menu_import_room(self, menu: QMenu):
        menu.import_room = bind_action(menu, 'Import Room', u":/delete.png")
        menu.import_room.triggered.connect(self.component_widget.import_room_data)

    def menu_import_iir(self, menu: QMenu):
        menu.import_room = bind_action(menu, 'Import IIR', u":/delete.png")
        menu.import_room.triggered.connect(self.component_widget.import_iir_data)

    def menu_import_fir(self, menu: QMenu):
        menu.import_room = bind_action(menu, 'Import FIR', u":/delete.png")
        menu.import_room.triggered.connect(self.component_widget.import_fir_data)

    def menu_import_res(self, menu: QMenu):
        menu.import_room = bind_action(menu, 'Import RES', u":/delete.png")
        menu.import_room.triggered.connect(self.component_widget.import_response_data)


    def menu_to_file(self, menu: QMenu):
        menu.to_file = bind_action(menu, 'to file', u":/delete.png")
        menu.to_file.triggered.connect(self.s_to_file)
        if self.component_widget.his_flag:
            menu.addSeparator()
            menu.revert = bind_action(menu, 'Revert to this', u":/update.png")
            menu.revert.triggered.connect(self.revert_bit)

    def s_to_file(self):
        if self.select_items:
            print(self.select_items)
            self.component_widget.save_to_file(file_list=self.select_items)
            self.select_items = []

    def revert_bit(self):
        if self.select_items:
            item = self.select_items[0]
            if self.component_widget.ask_ok(
                    f"Are you sure to revert <strong style='color:red'>{item.name}</strong>:{item.qid}"
                    f"to local env ? It's going to override local {item.name}",
                    "Revert Bit",
            ):
                ret_data = self.component_widget.backend.db.revert_bit(item.qid)
                if ret_data.get("code") == 200:
                    self.component_widget.backend.context_builder.\
                        refresh_chip_data({item.data["name"]: item.data})
                self.component_widget.handler_ret_data(ret_data, show_suc=True)

    def init_right_click_menu(self):
        pass
        # menu = QMenu(self)
        #
        # menu.import_room = bind_action(menu, 'Import Room', u":/delete.png")
        # menu.import_iir = bind_action(menu, 'Import IIR', u":/delete.png")
        # menu.import_fir = bind_action(menu, 'Import FIR', u":/delete.png")
        # menu.import_response = bind_action(menu, 'Import RES', u":/delete.png")
        #
        # menu.import_room.triggered.connect(self.component_widget.import_room_data)
        # menu.import_iir.triggered.connect(self.component_widget.import_iir_data)
        # menu.import_fir.triggered.connect(self.component_widget.import_fir_data)
        # menu.import_response.triggered.connect(self.component_widget.import_response_data)
        #
        # self.right_click_menu = menu
