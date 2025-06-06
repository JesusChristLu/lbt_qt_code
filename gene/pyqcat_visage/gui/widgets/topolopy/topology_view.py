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
import os
import re
import time
from typing import Union, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QColor, QWheelEvent, QActionGroup
from PySide6.QtWidgets import (
    QMenu,
    QMessageBox,
    QInputDialog,
    QGraphicsView,
    QGraphicsScene,
    QFileDialog,
)
from loguru import logger
from prettytable import PrettyTable

from pyQCat.processor.topology import SquareLattice, LineTopology
from pyQCat.qubit import NAME_PATTERN
from pyQCat.structures import QDict
from pyqcat_visage.gui.widgets.base.right_click_menu import (
    bind_action,
    bind_menu_action,
    bind_check_group_action,
)
from .item import (
    QubitItem,
    CoupleItem,
    CoupleDirectionEnum,
    BasicItem,
    Band,
    diff_points_distance,
    get_title,
)


class TopologyScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self._title_item = None

    @property
    def title_item(self):
        return self._title_item

    @title_item.setter
    def title_item(self, item):
        self._title_item = item
        self.addItem(self._title_item)


class BaseTopologyView(QGraphicsView):
    def __init__(self, parent, scene, color_conf):
        super().__init__(parent)
        self.qubit_dict = {}
        self.couple_dict = {}
        self.color_conf = color_conf
        self.t_scene: TopologyScene = scene

    def base_load(
        self, row: Union[str, int], col: Union[str, int], qubit_names: List = None
    ):
        """Plotting the chip topology on graphics.

        Args:
            row (Union[str, int]): The row of the chip.
            col (Union[str, int]): The column of the chip.
            qubit_names (List[str]): The chip line qubit names.

        steps:
            1. clear scene;
            2. create topology web with row and col.
            3. create qubit items ,add cache & scene.
            4. create coupler items, add cache &scene.
            5. create band
            6. create title.
        """

        self.couple_dict.clear()
        self.qubit_dict.clear()
        self.t_scene.clear()

        row = int(row)
        col = int(col)
        if col == 1:
            chip_topology = LineTopology(row, qubit_names)
        else:
            chip_topology = SquareLattice(row, col, qubit_names)

        # init qubit
        max_point = None
        for coordinate, qubit in chip_topology.bit_text.items():
            qubit_item = QubitItem(qubit, self.color_conf, coordinate)
            self.t_scene.addItem(qubit_item)
            max_point = qubit_item.r_center
            self.qubit_dict.update({qubit_item.name: qubit_item})
        # init couple
        for coordinate, couple in chip_topology.c_map.items():
            if couple in chip_topology.c_vedge.values():
                direction = CoupleDirectionEnum.VERTICAL
            elif couple in chip_topology.c_hedge.values():
                direction = CoupleDirectionEnum.ACROSS
            couple_item = CoupleItem(
                couple, self.color_conf, coordinate, direction=direction
            )
            self.t_scene.addItem(couple_item)
            self.t_scene.addItem(couple_item.back_rect)
            self.couple_dict.update({couple_item.name: couple_item})
        return chip_topology, max_point

    def init_theme(self, color_conf: QDict = None, rerender: bool = False):
        """
        init theme, use to change theme.
        """
        if self.color_conf:
            self.color_conf = color_conf
            for item in self.t_scene.items():
                if isinstance(item, BasicItem):
                    item.init_theme(color_conf=color_conf)

            if self.t_scene.title_item is not None:
                self.t_scene.title_item.setDefaultTextColor(
                    QColor(*self.color_conf.font_color)
                )

    def reset_bits(self):
        for bit in self.qubit_dict.values():
            bit.reset()
        for coupler in self.couple_dict.values():
            coupler.reset()

    def wheelEvent(self, event: QWheelEvent):
        """
        Hold down the Ctrl key and scroll the mouse wheel to zoom in/out.
        """
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            scale_factor = 1.2
            if delta < 0:
                scale_factor = 1.0 / scale_factor
            current_matrix = self.transform()
            self.setTransform(current_matrix.scale(scale_factor, scale_factor))
        else:
            super().wheelEvent(event)


class TopologyView(BaseTopologyView):
    envs_change = Signal(list)
    physical_change = Signal(list)
    workspace_change = Signal(list)
    show_workspace = Signal(bool)
    custom_bits = Signal(list)

    bit_select_mode = ["select all", "just qubit", "just coupler"]
    workspace_mode = ["show", "hidden"]
    workspace_bit_range = []
    workspace_conf_cache = []

    def __init__(self, parent, scene, color_conf, gui):
        super().__init__(parent, scene, color_conf)

        self.band = Band(self.qubit_dict, self.couple_dict)
        self.gui = gui

        self.menu = None

        # menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)

        # cache stack
        self.selected_cache = []
        self.env_cache = []
        self.physical_bit_cache = []
        self.radius = 0
        self.setRenderHint(QPainter.Antialiasing, True)
        self.workspace_bit_cache = []
        self.workspace_change.connect(self.set_workspace_bits)
        self.show_workspace.connect(self.is_show_workspace_bit)

    @property
    def mouse_select_mode(self):
        return self.band.select_mode

    @mouse_select_mode.setter
    def mouse_select_mode(self, value):
        self.band.select_mode = value

    @property
    def backend(self):
        return self.gui.backend

    def load(
        self, row: Union[str, int], col: Union[str, int], qubit_names: List = None
    ):
        """Plotting the chip topology on graphics.

        Args:
            row (Union[str, int]): The row of the chip.
            col (Union[str, int]): The column of the chip.
            qubit_names (List[str]): The chip line qubit names.

        steps:
            1. clear scene;
            2. create topology web with row and col.
            3. create qubit items ,add cache & scene.
            4. create coupler items, add cache &scene.
            5. create band
            6. create title.
        """
        self.t_scene.title_item = None
        self.band.clear()
        chip_topology, max_point = self.base_load(row, col, qubit_names)

        row = int(row)
        col = int(col)
        self.band.band_item = None
        max_point = [
            max_point.x() + QubitItem.R_ELLIPSE,
            max_point.y() + QubitItem.R_ELLIPSE,
        ]
        self.band.max_point = max_point
        self.band.create_band()
        self.t_scene.addItem(self.band.band_item)
        self.t_scene.title_item = get_title(row * col, self.color_conf, self.t_scene)
        self.init_menu()
        return chip_topology

    def refresh(self, physical_bits, env_bits=None):
        """
        refresh show and cache physical bites and env bits.
        """
        if env_bits:
            self.set_env_bits(env_bits, emit=False)
        really_physical_bits = []
        for bit in physical_bits:
            bit = str(bit)
            if bit in self.env_cache:
                really_physical_bits.append(bit)
            elif bit in self.qubit_dict or bit in self.couple_dict:
                continue
            elif (
                bit.count("q") == 2
                or bit.count("c") == 2
                or ("q" in bit and "c" in bit)
            ):
                for x in range(len(bit) - 1):
                    x += 1
                    if bit[x] in ["q", "c"]:
                        bit1 = bit[:x]
                        bit2 = bit[x:]
                        really_physical_bits.append(bit1)
                        really_physical_bits.append(bit2)
                        break

        self.set_physical_bits(really_physical_bits, emit=False)
        self.t_scene.invalidate()

    def select_with_radius(self, bit_name):
        """
        click radius, use with select some items.
        """
        select_bits = []
        if bit_name in self.qubit_dict:
            point = self.qubit_dict[bit_name].r_center
            select_bits.append(bit_name)
            for qubit, item in self.qubit_dict.items():
                if not diff_points_distance(point, item.r_center, self.radius * 50):
                    item.setSelected(True)
                    select_bits.append(qubit)
            if self.mouse_select_mode in [0, 2]:
                for coupler, item in self.couple_dict.items():
                    if not diff_points_distance(point, item.r_center, self.radius * 50):
                        item.setSelected(True)
                        select_bits.append(coupler)
        else:
            select_bits = [bit_name]
        select_bits = list(set(select_bits))
        return select_bits

    def bit(self, bit) -> Union[BasicItem, None]:
        """
        quick set bit without judge is coupler or qubit.
        """
        if bit in self.qubit_dict:
            return self.qubit_dict[bit]
        elif bit in self.couple_dict:
            return self.couple_dict[bit]
        else:
            return BasicItem("error", self.color_conf, None)

    def init_menu(self):
        menu = QMenu(self)
        menu.set_env_bits = bind_action(menu, "set env_bit", ":/set_env_bit.png")
        menu.set_env_bits.triggered.connect(self.set_env_bits)

        menu.remove_envs = bind_action(menu, "remove env_bit", ":/clear.png")
        menu.remove_envs.triggered.connect(self.remove_env_bits)

        menu.set_physical_bits = bind_action(
            menu, "set physical bits", ":/init_qubit.png"
        )
        menu.set_physical_bits.triggered.connect(self.set_physical_bits)

        menu.set_click_radius = bind_action(menu, "click radius", ":/tool.png")
        menu.set_click_radius.triggered.connect(self.set_click_radius)

        menu.set_click_radius = bind_action(menu, "clear bands", ":/tool.png")
        menu.set_click_radius.triggered.connect(self.band.clear)

        mouse_mode_menu = QMenu(self)
        mouse_mode_menu.setTitle("select mode")
        mouse_mode_menu_group = QActionGroup(self)
        mouse_mode_menu_group.setExclusive(True)
        mouse_mode_menu.all = bind_check_group_action(
            mouse_mode_menu, "all", mouse_mode_menu_group, True
        )
        mouse_mode_menu.qubit = bind_check_group_action(
            mouse_mode_menu, "qubit", mouse_mode_menu_group
        )
        mouse_mode_menu.coupler = bind_check_group_action(
            mouse_mode_menu, "coupler", mouse_mode_menu_group
        )
        menu.mouse_select = bind_menu_action(menu, mouse_mode_menu, ":/refresh.png")
        mouse_mode_menu.all.triggered.connect(self.select_mouse_bind_all)
        mouse_mode_menu.qubit.triggered.connect(self.select_mouse_bind_qubit)
        mouse_mode_menu.coupler.triggered.connect(self.select_mouse_bind_coupler)
        menu.addSeparator()
        if self.backend.login_user and (
            self.backend.is_super
            or self.backend.is_admin
        ):
            channel_menu = QMenu(self)
            channel_menu.setTitle("Channel operate")
            channel_menu.set_xy_lo = bind_action(
                channel_menu, "set same xy lo", ":/file-code.png"
            )
            channel_menu.set_m_lo = bind_action(
                channel_menu, "set same m lo", ":/file-code.png"
            )
            channel_menu.set_m_lo = bind_action(
                channel_menu, "set same bus", ":/file-code.png"
            )
            channel_menu.view_lo = bind_action(channel_menu, "view lo", ":/report.png")
            channel_menu.import_inst = bind_action(
                channel_menu, "Import Inst Info", ":/import2.png"
            )
            channel_menu.export_inst = bind_action(
                channel_menu, "Export Inst Info", ":/export.png"
            )
            channel_menu.sync_chip_line = bind_action(
                channel_menu, "Sync To Chip Line", ":/update.png"
            )

            channel_menu.set_xy_lo.triggered.connect(self.set_xy_lo)
            channel_menu.set_m_lo.triggered.connect(self.set_m_lo)
            channel_menu.set_m_lo.triggered.connect(self.set_bus)
            channel_menu.view_lo.triggered.connect(self.view_lo)
            channel_menu.import_inst.triggered.connect(self.import_inst)
            channel_menu.export_inst.triggered.connect(self.export_inst)
            channel_menu.sync_chip_line.triggered.connect(self.sync_chip_line)
            bind_menu_action(menu, channel_menu, ":/file-code.png")

        menu_amp = QMenu(self)
        menu_amp.setTitle("Amp <--> Freq")
        menu_amp.amp_to_freq = bind_action(menu_amp, "Amp to Freq", ":/compensate.png")
        menu_amp.freq_to_amp = bind_action(menu_amp, "Freq to Amp", ":/compensate.png")

        menu_amp.amp_to_freq.triggered.connect(self.amp_to_freq)
        menu_amp.freq_to_amp.triggered.connect(self.freq_to_amp)
        bind_menu_action(menu, menu_amp, ":/compensate.png")

        menu_point = QMenu(self)
        menu_point.setTitle("Point Operate")
        menu_point.view_point = bind_action(
            menu_point, "View Point", ":/compensate.png"
        )
        menu_point.set_custom_point = bind_action(
            menu_point, "Set Custom Point", ":/tool.png"
        )

        menu_point.view_point.triggered.connect(self.view_point)
        menu_point.set_custom_point.triggered.connect(self.set_custom_point)
        bind_menu_action(menu, menu_point, ":/point.png")

        menu.sync_coupler_pq = bind_action(menu, "Sync Coupler Probe Qubit", ":/tool.png")
        menu.sync_coupler_pq.triggered.connect(self.sync_coupler_pq)

        workspace_menu = QMenu(self)
        workspace_menu.setTitle("Workspace Operate")
        workspace_menu.set_workspace = bind_action(
            workspace_menu, "set workspace_bit", ":/add.png"
        )
        workspace_menu.delete_workspace = bind_action(
            workspace_menu, "remove workspace_bit", ":/delete2.png"
        )

        workspace_menu.set_workspace.triggered.connect(self.select_workspace_bits)
        workspace_menu.delete_workspace.triggered.connect(self.delete_workspace_bits)

        menu.addSeparator()
        show_space_menu = QMenu(self)
        show_space_menu.setTitle("Show? Workspace")
        show_space_group = QActionGroup(self)
        show_space_group.setExclusive(True)
        show_space_menu.show_space = bind_check_group_action(
            show_space_menu, "show", show_space_group, True
        )
        show_space_menu.hidden_space = bind_check_group_action(
            show_space_menu, "hidden", show_space_group
        )
        show_space_menu.show_space.triggered.connect(self.show_workspace_select)
        show_space_menu.hidden_space.triggered.connect(self.hidden_workspace_select)

        menu.workspace_menu = bind_menu_action(
            menu, workspace_menu, ":/workspace-operate.png"
        )
        menu.show_space_menu = bind_menu_action(menu, show_space_menu, ":/is-show.png")
        self.menu = menu
        return menu

    def open_menu(self, position):
        """
        open right click menu.
        """
        if not self.menu:
            self.init_menu()
        self.menu.exec_(self.mapToGlobal(position))

    def import_inst(self):
        cur_path = self.backend.config.get("system").get("config_path")
        filename = QFileDialog.getOpenFileName(self, "Import File", cur_path)[0]
        self.backend.import_inst_information(filename)

    def export_inst(self):
        save_name, ok = self.gui.main_window.ask_input(
            "Save Instrument Information", "Please input file name"
        )
        if ok:
            save_name = "inst_info"
            config_path = self.backend.config.get("system").get("config_path")
            dirname = os.path.join(
                config_path,
                "Instrument",
                f'{save_name}-{time.strftime("%Y-%m-%d-%H_%M_%S")}',
            )
            if self.gui.main_window.ask_ok(
                f"Are you sure to save the instrument lo/bus info to file? "
                f"It will be saved in {dirname}",
                "Are u ok?",
            ):
                self.backend.export_inst_information(dirname)

    def remove_env_bits(self):
        """ """
        self.selected_cache = []
        self.set_env_bits([])

    def set_env_bits(self, env_lists=None, emit=True):
        """ """
        env_lists = env_lists or self.selected_cache
        self.clear_bits_selected(env_lists)
        remove_bits = list(set(self.env_cache) - set(env_lists))
        for bit in self.physical_bit_cache:
            self.bit(bit).reset_status()
        self.physical_bit_cache = []
        for bit in remove_bits:
            self.bit(bit).set_envs(False)

        for bit in env_lists:
            self.bit(bit).set_envs(True)

        self.env_cache = [x.lower() for x in env_lists]
        self.selected_cache = []
        if emit:
            self.envs_change.emit(self.env_cache)

    def set_physical_bits(self, physical_bits=None, emit=True):
        physical_bits = physical_bits or self.selected_cache
        self.clear_bits_selected(physical_bits)
        res = set(physical_bits) - set(self.env_cache)
        if res:
            warning_box = QMessageBox()
            warning_box.setText(f"{res} physical bits not in envs")
            warning_box.exec_()
            return

        remove_bits = list(set(self.physical_bit_cache) - set(physical_bits))
        append_bits = list(set(physical_bits) - set(self.physical_bit_cache))
        for bit in remove_bits:
            self.bit(bit).set_physical_bits(False)
        for bit in append_bits:
            self.bit(bit).set_physical_bits(True)

        self.physical_bit_cache = [x.lower() for x in physical_bits]
        self.selected_cache = []
        if emit:
            self.physical_change.emit(self.physical_bit_cache)

    def set_workspace_bits(self, workspace_bits: list = None, force: bool = False):
        if workspace_bits is None:
            workspace_bits = []
        workspace_bits = list(set(workspace_bits) & set(self.workspace_bit_range))
        if force:
            ret_data = self.backend.db.save_workspace_info(
                workspace_bits, self.workspace_conf_cache
            )
        else:
            ret_data = None

        if ret_data is None or ret_data.get("code") == 200:
            remove_bits = list(set(self.workspace_bit_cache) - set(workspace_bits))
            # append_bits = list(set(workspace_bits) - set(self.workspace_bit_cache))
            for bit in remove_bits:
                self.bit(bit).set_font_bits(False)
            # for bit in append_bits:
            #     self.bit(bit).set_font_bits(True)
            for bit in workspace_bits:
                self.bit(bit).set_font_bits(True)
            self.workspace_bit_cache = workspace_bits
        elif ret_data and force:
            self.gui.main_window.handler_ret_data(ret_data)

    def delete_workspace_bits(self):
        ret_data = self.backend.db.delete_workspace_info(0, self.selected_cache)

        if ret_data and ret_data.get("code") == 200:
            for bit in self.selected_cache:
                self.bit(bit).set_font_bits(False)
            for bit in self.selected_cache:
                if bit in self.workspace_bit_cache:
                    self.workspace_bit_cache.remove(bit)
            # self.gui.main_window.handler_ret_data(ret_data)

    def is_show_workspace_bit(self, show: bool = True):
        for bit in self.workspace_bit_cache:
            if show:
                self.bit(bit).set_font_bits(True)
            else:
                self.bit(bit).set_font_bits(False)

    # def emit_change_workspace(self, workspace_bit: list):
    #     self.workspace_change.

    def set_click_radius(self):
        value, ok = QInputDialog.getInt(
            self,
            "Enter the radius: ",
            "Enter an integer:",
            self.radius,
            minValue=0,
            maxValue=4,
        )

        if ok and isinstance(value, int):
            self.radius = value

    def select_mouse_bind_all(self):
        self.mouse_select_mode = 0

    def select_mouse_bind_qubit(self):
        self.mouse_select_mode = 1

    def select_mouse_bind_coupler(self):
        self.mouse_select_mode = 2

    def set_xy_lo(self, physical_bits=None):
        self._set_lo_num(physical_bits=physical_bits)

    def set_m_lo(self, physical_bits=None):
        self._set_lo_num(module="m", physical_bits=physical_bits)

    def set_bus(self, physical_bits=None):
        physical_bits = physical_bits or self.selected_cache
        physical_qubits = []
        for bit in physical_bits:
            if bit.startswith("q"):
                physical_qubits.append(bit)
        value, ok = QInputDialog.getInt(
            self,
            f"Enter the bus num: ",
            "Enter an integer:",
            0,
            minValue=0,
            maxValue=20,
        )
        if ok and isinstance(value, int):
            self.backend.band_bus_num([value, physical_qubits])

    def _set_lo_num(self, module: str = "xy", physical_bits=None):
        physical_bits = physical_bits or self.selected_cache
        physical_qubits = []
        for bit in physical_bits:
            if bit.startswith("q"):
                physical_qubits.append(bit)

        value, ok = QInputDialog.getInt(
            self,
            f"Enter the {module} lo num: ",
            "Enter an integer:",
            0,
            minValue=0,
            maxValue=20,
        )

        if ok and isinstance(value, int):
            self.backend.band_lo_num([module, value, physical_qubits])

    def set_custom_point(self):
        reply = QMessageBox().question(
            self,
            "Dialog",
            "Are u suer to set custom point?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.selected_cache:
                self.custom_bits.emit(self.selected_cache)

    def view_lo(self):
        self.backend.refresh_lo_info(log=True)

    def select_workspace_bits(self):
        # self.workspace_change.emit(self.selected_cache, force: bool = True)
        self.set_workspace_bits(self.selected_cache, True)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            if event.modifiers() == Qt.ControlModifier:
                # self.selected_cache = self.t_scene.selectedItems()
                self.band.start(self.mapToScene(event.pos()))
                pass
            else:
                self.selected_cache = []
                self.band.start(self.mapToScene(event.pos()))
        else:
            for bit in self.selected_cache:
                self.bit(bit).setSelected(True)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        self.band.move(self.mapToScene(event.pos()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            if self.band:
                self.band.end(self.mapToScene(event.pos()))
                self.selected_cache = self.band.select_cache
                self.band.change_bit_select(
                    self.selected_cache, True, self.qubit_dict, self.couple_dict
                )
                self.band.select_cache = []
            if self.t_scene.selectedItems():
                select_item = [x.name for x in self.t_scene.selectedItems()]
                add_item = list(set(select_item) - set(self.selected_cache))
                remove_item = list(set(self.selected_cache) - set(select_item))
                add_bits = []
                if add_item:
                    for bit in add_item:
                        add_bits.extend(self.select_with_radius(bit))
                if remove_item:
                    self.band.change_bit_select(
                        remove_item, False, self.qubit_dict, self.couple_dict
                    )

                self.selected_cache = list(set(select_item) | set(add_bits))
                self.selected_cache = list(set(self.selected_cache) - set(remove_item))

        self.band.clear()

    def set_selected_cache(self, units: List):
        self.selected_cache = units
        self.band.change_bit_select(
            units, True, self.qubit_dict, self.couple_dict
        )
        remove_units = []
        for qn in list(self.qubit_dict.keys()):
            if qn not in units:
                remove_units.append(qn)
        for cn in list(self.couple_dict.keys()):
            if cn not in units:
                remove_units.append(cn)
        self.band.change_bit_select(
            remove_units, False, self.qubit_dict, self.couple_dict
        )

    def clear_bits_selected(self, bit_list):
        for bit in bit_list:
            item = self.bit(bit)
            if item.name != "error":
                item.setSelected(False)

    def amp_to_freq(self, physical_bits=None):
        self._ac_spectrum_transform_tool(physical_bits)

    def freq_to_amp(self, physical_bits=None):
        self._ac_spectrum_transform_tool(physical_bits, 1)

    def show_workspace_select(self):
        self.show_workspace.emit(True)

    def hidden_workspace_select(self):
        self.show_workspace.emit(False)

    def _ac_spectrum_transform_tool(self, physical_bits=None, mode: int = 0):
        physical_bits = physical_bits or self.selected_cache
        if not isinstance(physical_bits, List) or len(physical_bits) != 1:
            return

        physical_bits = physical_bits[0]

        describe = "amp to freq" if mode == 0 else "freq to amp"
        value, ok = QInputDialog.getText(
            self,
            f"Transform {physical_bits} {describe}: ",
            f"Enter an value | {describe}:",
        )
        if ok:
            self.backend.ac_spectrum_transform(physical_bits, float(value), mode)

    def sync_chip_line(self):
        self.gui.context_widget.sync_chip_line()

    def view_point(self):
        if self.selected_cache:
            table = PrettyTable()
            table.field_names = [
                "unit",
                "max point",
                "min point",
                "idle point",
                "readout point",
                "max + idle",
                "max + idle + read",
            ]
            for bit in self.selected_cache:
                unit = self.backend.context_builder.chip_data.get_physical_unit(bit)
                if unit:
                    table.add_row(
                        [
                            bit,
                            unit.dc_max,
                            unit.dc_min,
                            unit.idle_point,
                            unit.readout_point.amp,
                            unit.idle_point + unit.dc_max,
                            unit.idle_point + unit.dc_max + unit.readout_point.amp,
                        ]
                    )
                else:
                    logger.warning(f"No find {bit}!")
            logger.info(f"Working point information as follow:\n{table}")
        else:
            logger.warning(f"Please choose one physical unit!")

    def sync_coupler_pq(self):
        if self.selected_cache:
            for bit in self.selected_cache:
                if re.match(NAME_PATTERN.coupler, bit):
                    coupler = self.backend.context_builder.chip_data.get_physical_unit(bit)
                    if coupler:
                        probe_qubit = self.backend.context_builder.chip_data.get_physical_unit(f"q{coupler.probe_bit}")
                        drive_qubit = self.backend.context_builder.chip_data.get_physical_unit(f"q{coupler.drive_bit}")
                        if (
                            (probe_qubit.inst.bus == drive_qubit.inst.bus) and
                            (probe_qubit.probe_power != drive_qubit.probe_power)
                        ):
                            logger.warning(
                                f"Coupler's probe bit and drive bit have the "
                                f"same read bus, but the probe power is different!"
                            )
                        coupler.probe_freq = probe_qubit.probe_freq
                        coupler.probe_power = probe_qubit.probe_power
                        coupler.probe_drive_freq = probe_qubit.drive_freq
                        coupler.probe_drive_power = probe_qubit.drive_power
                        coupler.probe_XYwave = probe_qubit.XYwave
                        coupler.save_data()
                    else:
                        logger.warning(f"No find {bit}!")
        else:
            logger.warning(f"Please choose coupler!")
