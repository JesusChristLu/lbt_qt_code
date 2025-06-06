# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

import json
import os
import time
from collections import defaultdict
from functools import cmp_to_key
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QHeaderView, QWidget

from pyQCat.executor.structures import ChipConfigField
from pyQCat.hardware_manager import HardwareOffsetManager
from pyQCat.pulse_adjust import get_pulse_default_params
from pyQCat.structures import QDict
from pyQCat.tools.utilities import sort_bit
from pyqcat_visage.backend.component import VisageComponent
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.exceptions import ChipError, OperationError
from pyqcat_visage.gui.component_editor_ui import Ui_MainWindow
from pyqcat_visage.gui.tools.utilies import slot_catch_exception
from pyqcat_visage.gui.widgets.component.table_model_component import (
    QTableModelComponent,
)
from pyqcat_visage.gui.widgets.component.tree_delegate_options import QOptionsDelegate
from pyqcat_visage.gui.widgets.component.tree_model_component import QTreeModelComponent
from pyqcat_visage.gui.widgets.dialog.save_as_dialog import QSaveAsDialog
from pyqcat_visage.gui.widgets.result.table_model_dat import QTableModelDat
from pyqcat_visage.gui.widgets.revert_bits import RevertBitWindow
from pyqcat_visage.gui.widgets.title_window import TitleWindow

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class ComponentEditWindow(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent: QWidget = None):
        self.gui = gui
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.tree_model = None
        self.table_model = None
        self._component = None

        self._setup_components_editor()

        self.ui.table_view_component.choose_component_signal.connect(
            self._edit_component
        )

        self.ui.bit_pic_label.setMaximumSize(100, 100)

        self._history_state = False
        self.his_flag = False
        self._revert = RevertBitWindow(gui, self)

    def set_query_button_action(self, enable: bool):
        self.ui.actionQueryAll.setEnabled(enable)

    def _setup_components_editor(self):
        self.table_model = QTableModelComponent(
            self.gui, self, self.ui.table_view_component
        )
        self.ui.table_view_component.setModel(self.table_model)
        self.tree_model = QTreeModelComponent(
            self, self.gui, self.ui.tree_view_component
        )
        self.tree_model_delegate = QOptionsDelegate(self)
        self.ui.tree_view_component.setModel(self.tree_model)
        self.ui.tree_view_component.setItemDelegate(self.tree_model_delegate)
        self.dat_table_model = QTableModelDat(parent=self)
        self.ui.table_view_dat.setModel(self.dat_table_model)
        self.ui.table_view_dat.setVisible(False)
        self.ui.table_view_dat.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.tree_model_delegate.choose_type_signal.connect(self.change_pulse_type)
        self.ui.tree_view_component.doubleClicked.connect(self.refresh_component_view)

    @property
    def component(self) -> "VisageComponent":
        return self._component

    @property
    def backend(self):
        return self.gui.backend

    @slot_catch_exception(process_reject=True)
    def chip_check(self):
        self.backend.query_all_component()

        shape, ok = self.ask_input("Expect Chip Shape", "Please input chip shape:")

        if ok:
            row, col = shape.split(",")
            components = self.backend.components
            chip_params = components.get("chip_line_connect.json")

            if not chip_params:
                raise ChipError(
                    "chip_line_connect.json doesn't exist, please create it first!"
                )

            chip_data = chip_adapter(chip_params.view_data, int(row), int(col))
            chip_params.data["json"] = chip_data
            self.backend.update_single_config(*chip_params.to_data())

            for qubit_name, qubit_params in chip_data.get("QubitParams").items():
                qubit_component = components.get(qubit_name)
                if qubit_component:
                    bit = qubit_params.get("bit")
                    _row = qubit_params.get("_row")
                    _col = qubit_params.get("_col")
                    qubit_component.data["parameters"]["bit"] = bit
                    qubit_component.data["parameters"]["_row"] = _row
                    qubit_component.data["parameters"]["_col"] = _col
                    qubit_component.data["parameters"]["name"] = qubit_name
                    self.backend.save_component(qubit_name)
                    logger.log(
                        "UPDATE",
                        f"Update {qubit_name}: (bit={bit}) | (row={_row}) | (col={_col})!",
                    )
                else:
                    logger.warning(f"{qubit_name} doesn't exist in the database!")

                dat_name = f"distortion_{qubit_name}.dat"
                bin_name = f"{qubit_name}.bin"

                for name in [dat_name, bin_name]:
                    if name not in components:
                        logger.warning(f"{name} doesn't exist in the database!")

            for coupler_name, coupler_params in chip_data.get("CouplerParams").items():
                dat_name = f"distortion_{coupler_name}.dat"
                bin_name = f"{coupler_name}.bin"

                for name in [coupler_name, dat_name, bin_name]:
                    if name not in components:
                        logger.warning(f"{name} doesn't exist in the database!")

                coupler_component = components.get(coupler_name)
                if coupler_component:
                    bit = coupler_params.get("bit")
                    coupler_component.data["parameters"]["bit"] = bit
                    coupler_component.data["parameters"]["name"] = coupler_name
                    self.backend.save_component(coupler_name)
                    logger.log("UPDATE", f"Update {coupler_name}: (bit={bit})!")

            for name in ["character.json", "crosstalk.json", "instrument.json"]:
                if name not in components:
                    logger.warning(f"{name} doesn't exist in the database!")

    @slot_catch_exception(process_warning=True)
    def query_all(self):
        self.his_flag = False
        self.gui.query_all()

    @slot_catch_exception(process_warning=True)
    def query_one(self):
        self.his_flag = False
        self.backend.query_component(
            self.ui.name_edit.text(),
            self.ui.user_edit.text(),
            self.ui.point_edit.text(),
            self.ui.sample_editor.text(),
            self.ui.env_name_edit.text(),
        )
        self._page_volume_state()
        self.table_model.refresh_auto(check_count=False)

    @slot_catch_exception(process_warning=True)
    def query_history(self):
        self.his_flag = True
        name = self.ui.name_edit.text()
        qid = self.ui.qid_label.text()

        if qid:
            self.backend.query_component_by_id(qid)
        elif name:
            if name != self._history_state:
                self._page_volume_state(name)
            username = self.ui.user_edit.text()
            point_label = self.ui.point_edit.text()
            sample = self.ui.sample_editor.text()
            env_name = self.ui.env_name_edit.text()
            page = self.ui.page_spinBox.value()
            volume = self.ui.volume_spinBox.value()
            self.backend.query_history_component(
                name=name,
                user=username,
                point_label=point_label,
                sample=sample,
                env_name=env_name,
                page=page,
                volume=volume,
            )
        else:
            raise OperationError("Please input component name!")

        self.table_model.refresh_auto(check_count=False)

    @slot_catch_exception(process_warning=True)
    def save_one(self):
        # name = self.ui.name_edit.text()
        ret_data = self.backend.save_component(self.component)
        self.handler_ret_data(ret_data)
        if ret_data.get("code") == 200:
            name = "name" if self.component.data.get("name") else "filename"
            self.gui.backend.context_builder.refresh_chip_data(
                {self.component.data[name]: self.component.data}
            )

    @slot_catch_exception()
    def bit_import(self):
        cur_path = self.backend.config.get("system").get("config_path")

        filenames = QFileDialog.getOpenFileNames(self, "Import File", cur_path)
        ret_data = self.backend.import_components(filenames[0])
        if ret_data.code == 200:
            self.table_model.refresh_auto()
        self.handler_ret_data(ret_data, show_suc=True)

    @slot_catch_exception()
    def save_to_file(self, file_list=None):
        save_name, ok = self.ask_input("Save Component", "Please input file name")
        if ok:
            config_path = self.backend.config.get("system").get("config_path")
            dirname = os.path.join(
                config_path,
                "Component",
                self.backend.config.system.sample,
                self.backend.config.system.point_label,
                save_name,
            )
            if self.ask_ok(
                f"Are you sure to save the components info in the component collector to file? "
                f"It will be saved in {dirname}",
                "Component",
            ):
                self.handler_ret_data(self.backend.save_to_file(dirname, save_list=file_list), show_suc=True)

    @slot_catch_exception()
    def page_change(self, index: int):
        if self._history_state and index:
            self.query_history()

    @slot_catch_exception()
    def volume_change(self, index: int):
        if self._history_state and index:
            self.query_history()

    @slot_catch_exception(process_reject=True)
    def save_as(self):
        dialog = QSaveAsDialog()
        dialog.set_default(self.backend.config)
        ret = dialog.exec()

        if int(ret) == 1:
            point, sample, env_name = dialog.get_input()
            ret_data = self.backend.save_component_as(point, sample, env_name)
            self.handler_ret_data(ret_data, show_suc=True)

    @slot_catch_exception(process_warning=True)
    def refresh(self):
        ret_data = self.backend.refresh_component()
        if ret_data.code == 200:
            self.table_model.refresh_auto()
            self.ui.table_view_component.refresh_view()
            self.backend.context_builder.refresh_chip_data(ret_data.get("data"))
        self.handler_ret_data(ret_data)

    def set_component(self, component: "VisageComponent" = None):
        """Main interface to set the component (by name)

        Args:
            component: Set the component name, if None then clears
        """
        self._component = component

        if component is None:
            # TODO: handle case when name is none: just clear all
            # TODO: handle case where the component is made in jupyter notebook
            self.force_refresh()
            return

        label_text = f"{component.name} | {component.update_time}"
        self.ui.name_edit.setText(component.name)
        self.ui.user_edit.setText(component.username)
        self.ui.sample_editor.setText(component.sample)
        self.ui.point_edit.setText(component.point_label)
        self.ui.env_name_edit.setText(component.env_name)

        img = QPixmap(GUI_CONFIG.component_icon.get(component.style))
        self.ui.bit_pic_label.setPixmap(img)
        self.ui.bit_pic_label.setScaledContents(True)

        self.ui.editor_group.setTitle(label_text)

        if component.name == "crosstalk.json":
            self.ui.tree_view_component.setVisible(False)
            self.ui.table_view_dat.right_click_menu = None
            self.ui.table_view_dat.setVisible(True)
            bit_infos = component.view_data.get("infos")
            ac_crosstalk = component.view_data.get("ac_crosstalk")
            self.dat_table_model.input_data = ac_crosstalk
            self.dat_table_model.x_labels = bit_infos
            self.dat_table_model.y_labels = bit_infos
            self.dat_table_model.refresh()
        elif component.style != "dat":
            self.ui.tree_view_component.setVisible(True)
            self.ui.table_view_dat.setVisible(False)
            if component.style == "qubit_pair":
                self.tree_model.refresh(True)
            else:
                self.tree_model.refresh(False)
            self.ui.tree_view_component.autoresize_columns()  # resize columns
        else:
            self.ui.tree_view_component.setVisible(False)
            self.ui.table_view_dat.right_click_menu = None
            self.ui.table_view_dat.setVisible(True)
            view_data = component.view_data.get("dat")
            self.dat_table_model.input_data = np.array(view_data)
            self.dat_table_model.x_labels = None
            self.dat_table_model.y_labels = None
            self.dat_table_model.refresh()

    def refresh_component_view(self, index):
        def hardware_action():
            self.ui.tree_view_component.setVisible(False)

            # optimize: only for hardware offset component use click menu
            if self.ui.table_view_dat.right_click_menu is None:
                self.ui.table_view_dat.init_right_click_menu()

            self.ui.table_view_dat.setVisible(True)

        if self._component.name == "hardware_offset.json":
            label = self.tree_model.node_from_index(index).label
            if label == "link":
                hardware_action()
                data = self._component.view_data.get("link")
                self.dat_table_model.input_data = data
                self.dat_table_model.name = label
                row = len(data)
                col = len(data[0])
                self.dat_table_model.x_labels = [str(i) for i in range(col)]
                self.dat_table_model.y_labels = [str(i) for i in range(row)]
                self.dat_table_model.refresh()
            elif label == "xy_delay":
                hardware_action()
                data = self._component.view_data.get("xy_delay")
                # self.dat_table_model.input_data = [data]
                self.dat_table_model.input_data = [[d] for d in data]
                self.dat_table_model.name = label
                self.dat_table_model.y_labels = [f"XY-{i}" for i in range(len(data))]
                self.dat_table_model.x_labels = ["Delay"]
                self.dat_table_model.refresh()
            elif label == "z_delay":
                hardware_action()
                data = self._component.view_data.get("z_delay")
                self.dat_table_model.input_data = [[d] for d in data]
                self.dat_table_model.name = label
                self.dat_table_model.y_labels = [f"Z-{i}" for i in range(len(data))]
                self.dat_table_model.x_labels = ["Delay"]
                self.dat_table_model.refresh()
            elif label == "apply":
                flag = self._component.view_data.get("apply")
                self._component.view_data["apply"] = not flag
                self.tree_model.refresh()

    def hardware_offset_func(self, name, *args):
        if self._component.name == "hardware_offset.json":
            manager = HardwareOffsetManager.from_data(self.component.view_data)
            chip_data = self.backend.context_builder.chip_data.cache_config.get(
                ChipConfigField.chip_line
            )

            if name == "backtrack":
                manager.backtrack()
            elif name == "refresh":
                manager.reset()
            elif name == "extend":
                c1, c2, delay, force = args
                if c1[0] != c2[0]:
                    xy = int(c1[2:]) if c1.startswith("xy") else int(c2[2:])
                    z = int(c2[1:]) if c1.startswith("xy") else int(c1[1:])
                    manager.insert_xyz_timing(xy, z, float(delay), force)
                else:
                    manager.insert_z2_timing(
                        int(c1[1:]), int(c2[1:]), float(delay), force
                    )
            elif name == "qubit":
                unit, xyd, zd, force = args
                if chip_data:
                    qubit_params = chip_data.get("QubitParams").get(unit)
                    xy_channel = qubit_params.get("xy_channel")
                    z_flux_channel = qubit_params.get("z_flux_channel")
                    manager.insert_xyz_timing(
                        xy_channel, z_flux_channel, float(xyd) - float(zd), force
                    )
            elif name == "coupler":
                unit, dc, dp, dd, force = args
                if chip_data:
                    params = chip_data.get("CouplerParams").get(unit)
                    z_flux_channel = params.get("z_flux_channel")
                    zp_channel = (
                        chip_data.get("QubitParams")
                        .get(f"q{params.get('probe_bit')}")
                        .get("z_flux_channel")
                    )
                    zd_channel = (
                        chip_data.get("QubitParams")
                        .get(f"q{params.get('drive_bit')}")
                        .get("z_flux_channel")
                    )
                    manager.insert_zz_timing(
                        z_flux_channel,
                        zp_channel,
                        zd_channel,
                        float(dc),
                        float(dp),
                        float(dd),
                        force,
                    )
            elif name == "view":
                self.backend.view_hardware_offset()
            elif name == "view_all":
                self.backend.view_hardware_offset(view_unlink=True)
            elif name == "import":
                filename = QFileDialog.getOpenFileName(
                    self,
                    "Select hardware offset record data `.json`",
                    self.backend.config.system.config_path,
                )[0]

                with open(filename, encoding="utf-8") as fp:
                    record_data = json.load(fp)

                manager.reset()

                insert_state_map = defaultdict(list)

                for unit, record in record_data.items():
                    if unit.startswith("q"):
                        state = manager.insert_xyz_timing(
                            record.get("xy").get("channel"),
                            record.get("z").get("channel"),
                            record.get("xy").get("delay")
                            - record.get("z").get("delay"),
                        )
                        insert_state_map[str(state)].append(unit)
                    elif unit.startswith("c"):
                        if len(record) == 3:
                            state = manager.insert_zz_timing(
                                record.get("zp").get("channel"),
                                record.get("zd").get("channel"),
                                record.get("zc").get("channel"),
                                record.get("zp").get("delay"),
                                record.get("zd").get("delay"),
                                record.get("zc").get("delay"),
                            )
                            insert_state_map[str(state)].append(unit)
                        elif len(record) == 2:
                            channels = []
                            offset = None
                            for v in record.values():
                                channels.append(v.get("channel"))
                                if offset is None:
                                    offset = v.get("delay")
                                else:
                                    offset -= v.get("delay")
                            state = manager.insert_z2_timing(*channels, offset)
                            insert_state_map[str(state)].append(unit)
                        else:
                            logger.error(f"Coupler zz timing insert need 2+ channel!")

                for key, value in insert_state_map.items():
                    logger.log("RESULT", f"Insert state {key}: {value}")

            elif name == "export":
                self.backend.view_hardware_offset(save_records=True)

            self.component.view_data.update(manager.to_data())
            # self.set_component(self._component)
            if self.ui.table_view_dat.isVisible():
                if self.dat_table_model.name == "link":
                    self.dat_table_model.input_data = self.component.view_data.get(
                        "link"
                    )
                elif self.dat_table_model.name == "xy_delay":
                    data = self._component.view_data.get("xy_delay")
                    self.dat_table_model.input_data = [[d] for d in data]
                elif self.dat_table_model.name == "z_delay":
                    data = self._component.view_data.get("z_delay")
                    self.dat_table_model.input_data = [[d] for d in data]
                else:
                    return
                self.dat_table_model.refresh()

    def force_refresh(self):
        """Force refresh."""
        self.tree_model.load()

    def _edit_component(self, component: "VisageComponent"):
        self.set_component(component)
        self.ui.tree_view_component.hide_placeholder_text()

    def _page_volume_state(self, name: str = None):
        self._history_state = name
        self.ui.page_spinBox.setValue(1)

    def import_room_data(self):
        self._import_distortion_sos("Room_temperature_sos_filter")

    def import_iir_data(self):
        self._import_distortion_sos("Low_temperature_IIR_sos_filter")

    def import_fir_data(self):
        self._import_distortion_sos("Low_temperature_FIR_tf_filter")

    @slot_catch_exception(process_warning=True)
    def _import_distortion_sos(self, name: str):
        bits = self.ask_mul_items(
            "Which qubit do you want to import?", self.component.view_data.keys()
        )
        auto_bits = []
        for bit in bits:
            if bit not in self.component.view_data:
                self.handler_ret_data(QDict(code=600, msg=f"No fild {bit}!"))
                continue
            auto_bits.append(bit)

        cur_path = self.backend.config.get("system").get("config_path")
        filenames = QFileDialog.getOpenFileNames(self, "Import File", cur_path)[0]
        if filenames:
            for filename in filenames:
                data = np.loadtxt(filename).tolist()
                for bit in auto_bits:
                    self.component.view_data[bit]["distortion_sos"][name].append(data)
                    self.component.view_data[bit]["distortion_type"] = "sos"
            logger.log("UPDATE", f"Import {auto_bits} {name} success!")

    def import_response_data(self):
        if self.component:
            cur_path = self.backend.config.get("system").get("config_path")
            filename = QFileDialog.getOpenFileName(self, "Import File", cur_path)[0]
            if filename:
                data = np.loadtxt(filename).tolist()
                self.component.view_data["dat"] = data
            self.set_component(self.component)
            logger.log("UPDATE", f"Import {self.component.name} success!")

    def delete_union_rd(self, key: str):
        if self.component.name in GUI_CONFIG.component_sup_delete:
            if self.component.name == "hardware_offset.json":
                manager = HardwareOffsetManager.from_data(self.component.view_data)
                manager.reset()
                self.component.view_data.update(manager.to_data())
                logger.log("UPDATE", f"Reset {self.component.name} success!")
            else:
                ok = self.component.data.get("json").pop(key, None)
                if ok:
                    data = self.component.data
                    self.set_component(VisageComponent(data))

                logger.log("UPDATE", f"Delete {self.component.name} {key} success!")

            self.tree_model.refresh()

    def change_pulse_type(self, pulse_infos: tuple):
        if self.component.style == "qubit_pair":

            def update_params(label: str):
                pulse_type, parent_key = pulse_infos
                pulse_cls, default_params = get_pulse_default_params(pulse_type)
                pre_pulse_params = self.component.view_data["metadata"]["std"][label][
                    "params"
                ]
                default_params["phase"] = pre_pulse_params.get(parent_key).get("phase")
                default_params["amp"] = pre_pulse_params.get(parent_key).get("amp")
                default_params["pulse_type"] = pulse_type
                default_params["freq"] = None
                default_params["ac_branch"] = "right"
                default_params.pop("time")
                default_params.pop("name")
                pre_pulse_params[parent_key] = default_params

            update_params("cz")
            update_params("zz")

            self.tree_model.refresh()

    def refresh_all(self):
        self._page_volume_state()
        self.table_model.refresh_auto(False)

    def show_revert_bits(self):
        self._revert.show()


def qubit_bit_to_coordinate(bit: int, sr: int, sc: int):
    if bit >= sr * sc:
        raise ChipError(f"bit-{bit} error, shape ({sr}, {sc})")

    row = bit // sc
    col = bit % sc

    return row, col


def chip_adapter(chip_data, row, col):
    qubit_count = chip_data.get("QubitCount")
    coupler_count = chip_data.get("CouplerCount")
    qubit_params = chip_data.get("QubitParams")
    coupler_params = chip_data.get("CouplerParams")

    if (
        qubit_count != row * col
        or coupler_count != (col - 1) * row + (row - 1) * col
        or qubit_count != len(qubit_params)
        or coupler_count != len(coupler_params)
    ):
        raise ChipError(f"chip shape error, please check!")

    qubit_names = sorted(list(qubit_params.keys()), key=cmp_to_key(sort_bit))
    for i, qubit_name in enumerate(qubit_names):
        params = qubit_params.get(qubit_name)
        params["bit"] = i
        params["name"] = qubit_name
        _row, _col = qubit_bit_to_coordinate(i, row, col)
        params["_row"] = _row
        params["_col"] = _col

    for i, coupler_name in enumerate(list(coupler_params.keys())):
        params = coupler_params.get(coupler_name)
        params["bit"] = i
        params["name"] = coupler_name

    chip_data["shape"] = [row, col]

    return chip_data
