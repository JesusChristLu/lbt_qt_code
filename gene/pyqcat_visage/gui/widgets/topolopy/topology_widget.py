# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/29
# __author:       HanQing Shi
"""Chip Topology class."""

from typing import TYPE_CHECKING, Union, List
from pyQCat.structures import QDict

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
)

from PySide6.QtCore import Slot
from .topology_view import TopologyView, TopologyScene


class TopologyWidget(QWidget):
    """Chip Topology widget."""

    def __init__(self, gui: "VisageGUI", parent=None, color_conf: QDict = None):
        super().__init__(parent)
        self.gui = gui
        self._color_conf = color_conf if color_conf else self.gui.graphics_theme

        # layout and splitter
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 2, 2)
        self.setLayout(self._layout)
        self.chip_topology = None

        #  topology widget
        self._scene = TopologyScene()
        self.topology_view = TopologyView(parent=self, scene=self._scene, color_conf=self.color_conf, gui=gui)
        self.topology_view.setScene(self._scene)
        self._layout.addWidget(self.topology_view)

    @property
    def color_conf(self):
        return self._color_conf

    @color_conf.setter
    def color_conf(self, color_conf):
        if color_conf:
            self._color_conf = color_conf
            self.topology_view.color_conf = color_conf

    def init_theme(self, color_conf: QDict = None, rerender: bool = False):
        if color_conf:
            self.color_conf = color_conf
            self.topology_view.init_theme(self.color_conf, rerender=rerender)

        if rerender:
            self.hide()
            self.show()

    def load(self, row: Union[str, int], col: Union[str, int], qubit_names: List = None):
        # self.topology_view.build_context = self.gui._context_widget.build_std_context
        self.chip_topology = self.topology_view.load(row, col, qubit_names)
        self.set_env_bits(self.gui.backend.context_builder.global_options.env_bits)

    @Slot(list)
    def set_env_bits(self, env_list):
        env_list = [x.lower() for x in env_list]
        self.topology_view.set_env_bits(env_list)

    @Slot(list)
    def refresh(self, physical_list):
        self.topology_view.refresh(physical_bits=physical_list)
