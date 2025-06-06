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
from typing import TYPE_CHECKING, List

import numpy as np
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QWidget, QInputDialog, QMessageBox, QFileDialog
from loguru import logger

from pyQCat.qubit.qubit_pair import QubitPair, build_cz_gate_struct
from pyQCat.structures import QDict
from pyQCat.types import StandardContext, RespCode
from pyqcat_visage.exceptions import UserPermissionError
from pyqcat_visage.gui.context_config_ui import Ui_MainWindow
from pyqcat_visage.gui.tools.utilies import slot_catch_exception
from pyqcat_visage.gui.widgets.context.chip_create_widget import ChipCreateWindow
from pyqcat_visage.gui.widgets.context.table_model_channel import QTableModelChannel
from pyqcat_visage.gui.widgets.context.table_model_context import QTableModelContext
from pyqcat_visage.gui.widgets.context.tree_model_context import QTreeModelContext
from pyqcat_visage.gui.widgets.dialog.context_dialog import SQCDialog
from pyqcat_visage.gui.widgets.title_window import TitleWindow

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class ContextEditWindow(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent: QWidget = None):
        self.gui = gui
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.channel_table_model = None
        self.context_table_model = None
        self.context_tree_model = None
        self._component = None
        self._std_input = None

        self._setup_context_editor()
        self._setup_actions()

        self.chip_create_widget = ChipCreateWindow(self.gui, self)
        self.context_dialog = SQCDialog()

    def reset_window_layout(self):
        if not self.is_super and not self.is_admin:
            self.ui.action_sync_chip_Line.setVisible(False)
            self.ui.action_chip_layout.setVisible(False)
            self.ui.action_save_line.setVisible(False)
        else:
            self.ui.action_sync_chip_Line.setVisible(True)
            self.ui.action_chip_layout.setVisible(True)
            self.ui.action_save_line.setVisible(True)

    def _setup_context_editor(self):
        self.context_table_model = QTableModelContext(
            self.gui, self, self.ui.table_view_context
        )
        self.ui.table_view_context.setModel(self.context_table_model)

        self.context_tree_model = QTreeModelContext(
            self, self.gui, self.ui.tree_view_context
        )
        self.ui.tree_view_context.setModel(self.context_tree_model)

        self.channel_table_model = QTableModelChannel(
            self.gui, self, self.ui.table_view_channel
        )
        self.ui.table_view_channel.setModel(self.channel_table_model)

        self.ui.table_view_context.choose_component_signal.connect(self._look_component)

    def _setup_actions(self):
        policy_group = QActionGroup(self)
        policy_group.addAction(self.ui.action_maximum)
        policy_group.addAction(self.ui.action_minimum)
        policy_group.setExclusive(True)
        self.ui.action_minimum.setChecked(True)

    # ----------------------------- property -----------------------------------

    @property
    def experiment_context(self):
        return self.gui.backend.experiment_context

    @property
    def component(self):
        return self._component

    @property
    def backend(self):
        return self.gui.backend

    # ----------------------------- Chip Slot -----------------------------------

    @slot_catch_exception()
    def chip_layout(self):
        if self.is_super or self.is_admin:
            self.chip_create_widget.show()
        else:
            raise UserPermissionError(f'Sorry {self.username}, only admin user can edit chip!')

    @slot_catch_exception(process_reject=True)
    def chip_save_line(self):
        login_user = self.backend.login_user
        ret_data = None
        if self.is_super or self.is_admin:
            if self.ask_ok(
                "Are you sure to save chip line? Under the same configuration, "
                "it will overwrite your previous records",
                "Chip",
            ):
                ret_data = self.backend.save_chip_line()
        else:
            raise UserPermissionError(f'Sorry {login_user.get("username")}, only admin user can edit chip!')

        self.handler_ret_data(ret_data, show_suc=True)

    @slot_catch_exception(process_reject=True)
    def sync_chip_line(self):
        if not self.is_admin and not self.is_super:
            QMessageBox().critical(
                self,
                "Warning",
                "No permission to sync local bits channel to chip line, pls contact the administrator!",
            )
            return
        if self.ask_ok(
                "Are you sure to sync local bits channel to chip line?"
                "it will overwrite chip line channel and lo",
                "Sync Chip Line",
        ):
            ret_data = self.backend.db.sync_chip_line()
            self.handler_ret_data(ret_data, show_suc=True)

    @slot_catch_exception(process_reject=True)
    def chip_init_base_qubit(self, bit_names: List = None):
        anno = "-".join([bit_names[0], bit_names[-1]]) if bit_names else "all"
        if self.ask_ok(
            f"Are you sure to init {anno} qubit data from chip line?", "Chip"
        ):
            # ret = self.backend.query_component(name="chip_line_connect.json")
            #
            # if ret.get("code") != 200:
            #     ret_data = QDict(
            #         code=600, msg="No query chip line information, Please save first!"
            #     )
            # else:
            #     # todo
            ret_data = self.backend.initial_base_qubit_data(bit_names)
            if ret_data.get("code") == 200:
                self.backend.refresh_chip_context()
            self.handler_ret_data(ret_data, True, "Init Base Bit")

    def chip_del_init_qubit_data(self, bit_names: List = None, delete: bool = True):
        anno = "-".join([bit_names[0], bit_names[-1]]) if bit_names else "all"
        if self.ask_ok(
            f"Are you sure to <strong style='color:red'>delete</strong> init {anno}"
            f" qubit data from chip line?", "Chip"
        ):
            ret_data = self.backend.initial_base_qubit_data(bit_names, delete)
            if ret_data.get("code") == 200:
                self.backend.refresh_chip_context()
            self.handler_ret_data(ret_data, True, "Init Base Bit")

    @slot_catch_exception(process_reject=True)
    def chip_init_config_data(self, bit_names: List = None, delete: bool = False):

        def _init_config_data_():
            ret = self.backend.query_component(name="chip_line_connect.json")
            if ret.get("code") != 200:
                ret_data = QDict(
                    code=600, msg="No query chip line information, Please save first!"
                )
            else:
                self.gui.component_editor.table_model.refresh_auto()
                ret_data = self.backend.initial_config_data(bit_names, delete)
            if ret_data.get("code") == 200:
                self.backend.refresh_chip_context()
            self.handler_ret_data(ret_data, True, "Init Config Data")

        anno = "-".join([bit_names[0], bit_names[-1]]) if bit_names else "all"
        if not delete:
            if self.ask_ok(
                f"Are you sure to init {anno} config data from chip line?", "Chip"
            ):
                _init_config_data_()
        else:
            _init_config_data_()

    @slot_catch_exception(process_reject=True)
    def chip_del_init_config_data(self, bit_names: List = None, delete: bool = True):
        if self.ask_ok(
                "Are you sure you want to initialize after "
                "<strong style='color:red'>delete</strong> the config data",
                "Del and Init Config Data",
        ):
            self.chip_init_config_data(bit_names=bit_names, delete=delete)

    def create_qubit_pair(self, bit_names):
        ret_data = QDict(code=600, msg="Qubit Pair only support two qubit!")
        bits = [int(bit[1:]) for bit in bit_names if bit.startswith("q")]
        if len(bits) == 2:
            sorted(bits)
            pair_name = f"q{bits[0]}q{bits[1]}"
            ret_data = self.backend.query_component(pair_name)
            if ret_data.get("code") == 200:
                ret_data = QDict(code=600, msg=f"Qubit Pair {pair_name} is exist!")
            else:
                pair = QubitPair(pair_name)
                pair = build_cz_gate_struct(pair)
                ret_data = pair.save_data()
                if ret_data.get("code") == 200:
                    self.gui.backend.context_builder.chip_data.cache_qubit_pair.update({
                        pair_name: pair
                    })
        self.handler_ret_data(ret_data)

    @slot_catch_exception(process_reject=True)
    def chip_init_sos_rt_data(self, bit_names: List = None):
        """Initial `qubit` or `coupler` sos RoomTemperature digital filter data."""
        file_path = QFileDialog.getExistingDirectory(self, "Select sos RT data Path")
        file_list = os.listdir(file_path)
        character_name = "character.json"

        try:
            ctx_manager = self.backend.context_builder
            character_dict = ctx_manager.chip_data.cache_config.get(character_name, {})
            if not ctx_manager.chip_data.cache_qubit or not ctx_manager.chip_data.cache_coupler:
                raise ValueError(f"No cache_qubit or cache_coupler data.")
        except Exception as err:
            self.backend.refresh_chip_context()
            ctx_manager = self.backend.context_builder
            character_dict = ctx_manager.chip_data.cache_config.get(character_name, {})

        qc_map = {}
        qc_map.update(ctx_manager.chip_data.cache_qubit)
        qc_map.update(ctx_manager.chip_data.cache_coupler)
        all_names = list(qc_map.keys())
        select_names = bit_names if bit_names else all_names

        distortion_sos = {
            "Gaussian_filter_order": [],
            "Room_temperature_sos_filter": [],
            "Low_temperature_IIR_sos_filter": [],
            "Low_temperature_FIR_tf_filter": [],
        }
        for name in select_names:
            qc_obj = qc_map.get(name)
            if qc_obj:
                ac_channel = qc_obj.z_flux_channel
                file = f"Channel {ac_channel}_sos_digital_filter_RT.dat"
                if file in file_list:
                    file_name = os.path.join(file_path, file)
                    rt_arr = np.loadtxt(file_name)
                    single_dict = character_dict.get(name, {})
                    single_dict.update({"distortion_type": "sos"})
                    single_sos_dict = single_dict.get("distortion_sos", distortion_sos)
                    single_sos_dict.update(
                        {
                            "Gaussian_filter_order": [7],
                            "Room_temperature_sos_filter": [rt_arr.tolist()]
                        }
                    )
                else:
                    logger.warning(f"Update {name} sos RT data error: Not found file: {file}")
            else:
                logger.warning(f"Update {name} sos RT data error: Maybe {name} isn't Qubit or Coupler")

        try:
            data = self.backend.db.update_single_config(character_name, character_dict)
            if data.get("code") == RespCode.resp_success.value:
                logger.log("UPDATE", f"Save {character_name} to data service success.")
            else:
                logger.warning(
                    f"Save {character_name} to data service error: {data.get('msg')}"
                )
        except Exception as err:
            logger.warning(f"Save {character_name} to data service error: \n{err}")

    # ----------------------------- Context Slot -----------------------------------

    @slot_catch_exception(deprecated=True)
    def context_create(self):
        ret_data = self.backend.create_default_context()
        self.handler_ret_data(ret_data)
        if ret_data and ret_data.get("code") == 200:
            self.context_table_model.refresh_auto()

    @slot_catch_exception(deprecated=True)
    def context_set_env_bit(self):
        indexes = self.ui.table_view_channel.selectedIndexes()
        env_bit_list = []
        for index in indexes:
            bit, _ = self.channel_table_model.bit_from_index(index)
            env_bit_list.append(bit)

        self.gui.ui.tabTopology.set_env_bits(list(set(env_bit_list)))

    @slot_catch_exception(process_warning=True, deprecated=True)
    def context_add_inst(self):
        # self.backend.context_builder.reset()
        ret_data = self.backend.add_inst()
        self.handler_ret_data(ret_data)
        if ret_data and ret_data.get("code") == 200:
            self.context_table_model.refresh_auto()

    @slot_catch_exception(process_warning=True, deprecated=True)
    def context_add_qubit(self):
        # self.backend.context_builder.reset()
        items = self.backend.context_builder.env_bits
        ret_data = None
        if items:
            bit_name, ok = QInputDialog.getItem(
                self, "Add Bit", "Please choose bit name", items, 0, False
            )
            if ok:
                ret_data = self.backend.context_add_bit(bit_name)
        else:
            ret_data = QDict(code=600, msg="Please choose env bits first!")
        self.handler_ret_data(ret_data)
        if ret_data and ret_data.get("code") == 200:
            self.context_table_model.refresh_auto()

    @slot_catch_exception(process_warning=True, deprecated=True)
    def context_add_dcm(self):
        # self.backend.context_builder.reset()
        items = self.backend.context_builder.env_bits
        ret_data = None
        if items:
            bit_name, ok = QInputDialog.getItem(
                self, "Add Dcm", "Please choose dcm name", items, 0, False
            )
            if ok:
                ret_data = self.backend.context_add_dcm(bit_name)
        else:
            ret_data = QDict(code=600, msg="Please choose env bits first!")
        self.handler_ret_data(ret_data)
        if ret_data and ret_data.get("code") == 200:
            self.context_table_model.refresh_auto()

    @slot_catch_exception(process_warning=True, deprecated=True)
    def context_add_crosstalk(self):
        # self.backend.context_builder.reset()
        ret_data = self.backend.context_add_crosstalk()
        self.handler_ret_data(ret_data)
        if ret_data and ret_data.get("code") == 200:
            self.context_table_model.refresh_auto()

    @slot_catch_exception(process_warning=True, deprecated=True)
    def context_add_compensates(self):
        # self.backend.context_builder.reset()
        ret_data = self.backend.context_add_compensates(
            self.ui.action_minimum.isChecked()
        )
        self.handler_ret_data(ret_data)
        if ret_data and ret_data.get("code") == 200:
            self.context_table_model.refresh_auto()

    @slot_catch_exception(deprecated=True)
    def context_max_env(self):
        if self.ask_ok("Are you sure to change maximize compensate?", "Context"):
            self.ui.action_maximum.setChecked(True)

    @slot_catch_exception(deprecated=True)
    def context_min_env(self):
        if self.ask_ok("Are you sure to change minimize compensate?", "Context"):
            self.ui.action_minimum.setChecked(True)

    @slot_catch_exception(deprecated=True)
    def context_clear(self):
        if self.ask_ok("Are you sure you want to empty the context?", "Context"):
            # self.backend.context_builder.reset()
            # ret_data = self.backend.context_builder.clear_context()
            # self.handler_ret_data(ret_data)
            self.context_table_model.refresh_auto()

    @slot_catch_exception(deprecated=True)
    def context_reset(self):
        if self.ask_ok("Are you sure you want to reset the context?", "Context"):
            # self.backend.context_builder.reset()
            # ret_data = self.backend.context_builder.reset_context()
            # self.handler_ret_data(ret_data)
            self.context_table_model.refresh_auto()

    @slot_catch_exception(deprecated=True)
    def context_set_working_dc(self, bit_names):
        self.backend.context_builder.record_working_dc = bit_names
        self.backend.experiment_context.config_working_dc(bit_names)

    # ----------------------------- Standard Slot -----------------------------------

    @slot_catch_exception(process_warning=True, deprecated=True)
    def build_std_context(self):
        items = self.backend.context_builder.env_bits
        ret_data = None
        if items:
            if items != self.context_dialog.env_bits:
                self.context_dialog.set_qubit_items(items)
            self.context_dialog.init_dialog()
            self.context_dialog.show()
            ret = self.context_dialog.exec_()

            if int(ret) == 1:
                self.setEnabled(False)
                ret_data = self.build_response()
                self.setEnabled(True)
        else:
            ret_data = QDict(code=600, msg="Please choose env bits first!")
        self.handler_ret_data(ret_data)
        if ret_data and ret_data.get("code") == 200:
            self.context_table_model.refresh_auto()

    def build_response(self, pd: QDict = None):
        if pd is None:
            pd = self.context_dialog.get_input()
        else:
            self.context_dialog.set_data(pd)

        self._std_input = pd

        if pd.stardard_context == StandardContext.QC.value:
            ret_data = self.backend.build_sqc_context(pd)
        elif pd.stardard_context == StandardContext.CPC.value:
            ret_data = self.backend.build_cpc_context(pd)
        elif pd.stardard_context == StandardContext.NT.value:
            ret_data = self.backend.build_nt_context(pd)
        elif pd.stardard_context == StandardContext.CM.value:
            ret_data = self.backend.build_crosstalk_context(pd)
        elif pd.stardard_context == StandardContext.URM.value:
            ret_data = self.backend.build_union_read_context(pd)
        elif pd.stardard_context == StandardContext.CGC.value:
            ret_data = self.backend.build_cz_calibration_context(pd)
        elif pd.stardard_context == StandardContext.CC.value:
            ret_data = self.backend.build_coupler_cali_context(pd)
        else:
            ret_data = QDict(code=600, msg=f"no {pd.stardard_context}")
        return ret_data

    # ----------------------------- Other method -----------------------------------

    def _look_component(self, component):
        self.set_component(component)
        self.ui.tree_view_context.hide_placeholder_text()

    def set_component(self, component=None):
        self._component = component

        if component is None:
            # TODO: handle case when name is none: just clear all
            # TODO: handle case where the component is made in jupyter notebook
            self.force_refresh()
            return

        # Labels
        # ) from {component.__class__.__module__}
        # label_text = f"{component.data_dict.name} | {component.data_dict.class_name}"
        # self.ui.labelComponentName.setText(label_text)
        # self.ui.labelComponentName.setCursorPosition(0)  # Move to left
        # self.setWindowTitle(label_text)
        # self.parent().setWindowTitle(label_text)

        self.force_refresh()

        self.ui.tree_view_context.autoresize_columns()  # resize columns

        data_dict = component
        if not isinstance(component, dict):
            data_dict = component.to_dict()

        self.ui.textEdit.setText(json.dumps(data_dict, indent=4))

    def force_refresh(self):
        """Force refresh."""
        self.context_tree_model.load()

    def login_out(self):
        self.chip_create_widget.ui.row_edit.clear()
        self.chip_create_widget.ui.col_edit.clear()
        # todo clear topology and heatmap picture

    def cache_context(self):
        """Cache chip context."""
        logger.warning("Caching context using context widget is deprecated.")

    def load_chip_line(self):
        # search chip line
        self.backend.query_chip_line()

        # init chip info
        if self.backend.model_channels and self.backend.model_channels.shape:
            row, col = self.backend.model_channels.shape
            qubit_names = list(self.backend.model_channels.qubit_params.keys())
            self.chip_create_widget.ui.row_edit.setText(str(row))
            self.chip_create_widget.ui.col_edit.setText(str(col))
            self.chip_create_widget.load_chip_topology.emit(
                str(row), str(col), qubit_names
            )

        # refresh model
        self.channel_table_model.refresh_auto()
