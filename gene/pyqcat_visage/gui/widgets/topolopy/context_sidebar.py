# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/06/02
# __author:       Lang Zhu

from typing import List

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QWidget, QMessageBox

from pyQCat.types import StandardContext
from pyqcat_visage.backend.experiment import VisageExperiment
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.gui.tools.utilies import slot_catch_exception
from pyqcat_visage.gui.widgets.title_window import MultiSelectDialog
from .context_sidebat_ui import Ui_Form
from .table_model_point import QTableModelPoint
from loguru import logger


def physical_units_trans(units):
    """
    trans physical units str and list.
    if str trans to list;
    if list trans to str;
    """
    if isinstance(units, str):
        return units.split(",")
    elif isinstance(units, list):
        if len(units) > 1:
            return ",".join(units)
        elif len(units) == 1:
            return units[0]
        else:
            return ""

    return ""


class ContextSideBar(QWidget):
    """
    Context Sidebar Widget. Use to set and show Context params.

    Signal:
        change_physical_unit: use to refresh option editor.
        refresh_topology: use to refresh topology show physical bit.
    """

    change_physical_unit = Signal(VisageExperiment)
    refresh_topology = Signal(list)

    def __init__(self, gui, parent, color_conf=None):
        super().__init__(parent)
        self.gui = gui
        self.refresh_com_ing = False
        self._ui = Ui_Form()
        self._ui.setupUi(self)
        self.color_conf = color_conf
        self.parent().setWindowTitle("Context")
        self.exp_cache = None
        self.physical_com_cache = {
            StandardContext.QC.value: [],
            StandardContext.CGC.value: [],
            StandardContext.CC.value: [],
            StandardContext.NT.value: [],
            StandardContext.URM.value: [],
        }
        self.custom_point_model = QTableModelPoint(
            self.gui, self, self._ui.point_table_view
        )
        self._ui.point_table_view.setModel(self.custom_point_model)
        self._ui.tabWidget.setCurrentIndex(0)
        self.load()

    @property
    def ui(self):
        return self._ui

    @property
    def context_builder(self):
        return self.gui.backend.context_builder

    @property
    def parallel_mode(self):
        return self.gui._parallel_checkbox.isChecked()

    @property
    def check_global(self):
        """
        check context builder global params is changed?
        if change return new global context params, if not return {}
        """
        new_global = {}
        if (
                self._ui.com_work_type.currentText()
                != self.context_builder.global_options.working_type
        ):
            new_global.update({"working_type": self._ui.com_work_type.currentText()})

        if (
                self._ui.com_divide_type.currentText()
                != self.context_builder.global_options.divide_type
        ):
            new_global.update({"divide_type": self._ui.com_divide_type.currentText()})

        if (
                self._ui.com_max_qubit.currentText()
                != self.context_builder.global_options.max_point_unit
        ):
            new_global.update({"max_point_unit": self._ui.com_max_qubit.currentText()})

        if (
                self._ui.crosstalk_check.isChecked()
                != self.context_builder.global_options.crosstalk
        ):
            new_global.update({"crosstalk": self._ui.crosstalk_check.isChecked()})

        if (
                self._ui.online_check.isChecked()
                != self.context_builder.global_options.online
        ):
            new_global.update({"online": self._ui.online_check.isChecked()})

        if (
                self._ui.xy_crosstalk_check.isChecked()
                != self.context_builder.global_options.xy_crosstalk
        ):
            new_global.update({"xy_crosstalk": self._ui.xy_crosstalk_check.isChecked()})

        if (
                self._ui.com_online_qubit.currentText()
                != self.context_builder.global_options.online_unit
        ):
            new_global.update({"online_unit": self._ui.com_online_qubit.currentText()})

        if (
                self._ui.f02_opt_qubit.currentText()
                != self.context_builder.global_options.f12_opt_bits
        ):
            new_global.update({"f12_opt_bits": self._ui.f02_opt_qubit.currentText()})

        if (
                self._ui.custom_point_check.isChecked()
                != self.context_builder.global_options.custom
        ):
            new_global.update({"custom": self._ui.custom_point_check.isChecked()})

        return new_global

    @property
    def current_tab(self):
        """
        return the  tab widget show tab. if default context tab return 0, if experiment return 1.
        """
        return self._ui.tabWidget.currentIndex()

    def load(self):
        """
        load context data to widget, usually use to widget init.
        """
        if self.context_builder:
            gop = self.context_builder.global_options
            if gop.max_point_unit:
                self._ui.com_max_qubit.setCurrentText(",".join(gop.max_point_unit))

            if gop.online_unit:
                self._ui.com_online_qubit.setCurrentText(",".join(gop.online_unit))

            if gop.f12_opt_bits:
                self._ui.f02_opt_qubit.setCurrentText(",".join(gop.f12_opt_bits))

            self._ui.com_work_type.setCurrentText(gop.working_type)
            self._ui.com_divide_type.setCurrentText(gop.divide_type)

            if gop.crosstalk:
                self._ui.crosstalk_check.setChecked(True)
            else:
                self._ui.crosstalk_check.setChecked(False)

            if gop.xy_crosstalk:
                self._ui.xy_crosstalk_check.setChecked(True)
            else:
                self._ui.xy_crosstalk_check.setChecked(False)

            if gop.online:
                self._ui.online_check.setChecked(True)
            else:
                self._ui.online_check.setChecked(False)

            if gop.custom:
                self._ui.custom_point_check.setChecked(True)
            else:
                self._ui.custom_point_check.setChecked(False)

        #     context_type = self._ui.default_context_com.currentText()
        #     self.refresh_com(context_type)
        #     if gop.context_data:
        #         self._ui.default_check.setChecked(
        #             gop.context_data[context_type].get("default", False)
        #         )
        #         self._ui.default_physical_unit_com.setCurrentText(
        #             gop.context_data[context_type].get("physical_unit", "")
        #         )
        #         self._ui.default_read_com.setCurrentText(
        #             gop.context_data[context_type].get("readout_type", "")
        #         )
        #
        # if self.current_tab == 0:
        #     physical_unit = self._ui.default_physical_unit_com.currentText()
        # elif self.current_tab == 1:
        #     physical_unit = self._ui.exp_physical_unit_com.currentText()
        # else:
        #     physical_unit = []
        # self.refresh_topology.emit(physical_unit)

    def _change_tab_com(self, physical_com, read_com, context_type, refresh_type=None):
        """
        change com, use to fresh com select.
        """

        def refresh_default(_physical_com, _read_com):
            context_data = self.gui.backend.context_builder.global_options.context_data
            if context_data:
                physical_units = context_data[context_type]["physical_unit"]
                readout_type = context_data[context_type]["readout_type"]
            else:
                physical_units = ""
                readout_type = ""

            self._ui.default_physical_unit_com.clear()
            self._ui.default_physical_unit_com.set_units(_physical_com)
            self._ui.default_physical_unit_com.setCurrentText(physical_units)
            self._ui.default_read_com.clear()
            if _read_com:
                self._ui.default_read_com.addItems(_read_com)
                self._ui.default_read_com.setCurrentText(readout_type)

            self._ui.default_context_com.setCurrentText(context_type)

        def refresh_exp(_physical_com, _read_com):
            physical_units = self._ui.exp_physical_unit_com.currentText()
            readout_type = self._ui.exp_read_com.currentText()
            if self.exp_cache:
                physical_units = self.exp_cache.context_options.physical_unit[0]
                readout_type = self.exp_cache.context_options.readout_type[0]
            self._ui.exp_physical_unit_com.clear()
            self._ui.exp_physical_unit_com.set_units(_physical_com)
            self._ui.exp_physical_unit_com.setCurrentText(physical_units)
            self._ui.exp_read_com.clear()
            if _read_com:
                self._ui.exp_read_com.addItems(_read_com)
                self._ui.exp_read_com.setCurrentText(readout_type)

        if refresh_type is None:
            refresh_type = self.current_tab
        if refresh_type == 0:
            refresh_default(physical_com, read_com)
        elif refresh_type == 1:
            refresh_exp(physical_com, read_com)
        elif refresh_type == 2:
            refresh_default(physical_com, read_com)
            refresh_exp(physical_com, read_com)

    @Slot(str)
    def refresh_com(self, name: str, refresh_type=None):
        """
        refresh tab widget physical unit and readout type com selected.
        this func use to control select items.
        """
        if self.refresh_com_ing and refresh_type != 2:
            return

        read_com = (
            list(GUI_CONFIG.std_context[name].values())
            if name in GUI_CONFIG.std_context
            else []
        )
        if name in [StandardContext.QC.value]:
            physical_units = self.physical_com_cache[StandardContext.QC.value]
        elif name in [StandardContext.CM.value]:
            physical_units = self.physical_com_cache[StandardContext.URM.value]
        elif name in [StandardContext.CPC.value, StandardContext.CC.value]:
            physical_units = self.physical_com_cache[StandardContext.CC.value]
        elif name == StandardContext.CGC.value:
            physical_units = self.physical_com_cache[StandardContext.CGC.value]
        elif name == StandardContext.NT.value:
            physical_units = self.physical_com_cache[StandardContext.NT.value]
        elif name == StandardContext.URM.value:
            # bugfix: 2024/02/04 URM context can not limit physical unit type
            # physical_units = self.physical_com_cache[StandardContext.QC.value]
            physical_units = []
        else:
            return
        self._change_tab_com(
            physical_units, read_com, context_type=name, refresh_type=refresh_type
        )

    def refresh_experiment_options(self, exp: VisageExperiment):
        """
        refresh experiments options by edit options signal .
        """
        # cache exp
        self.exp_cache = exp
        if not self.exp_cache:
            return
        self.refresh_com_ing = True
        self._ui.exp_context_com.clear()
        self._ui.exp_read_com.clear()
        self._ui.exp_physical_unit_com.clear()

        if self.exp_cache.context_options.name:
            self._ui.exp_context_com.addItems(exp.context_options.name[1])
            context_type = exp.context_options.name[0]
            self._ui.exp_context_com.setCurrentText(context_type)
            self.refresh_com(name=context_type, refresh_type=2)

            self._ui.exp_physical_unit_com.setCurrentText(self.exp_cache.context_options.physical_unit[0])
            self._ui.exp_read_com.setCurrentText(self.exp_cache.context_options.readout_type[0])

            self.refresh_topology.emit(self._ui.exp_physical_unit_com.currentText())
            self._ui.tabWidget.setCurrentWidget(self._ui.exp_tab)
        self.refresh_com_ing = False

    @Slot(list)
    def envs_refresh(self, env_lists):
        """
        change envs will do, signal by topology widget, will update context envs bit
        and refresh physical unit select item cache.

        """
        if env_lists is None:
            self.gui.backend.set_env_bit(env_bit_list=env_lists, set_all=True)
        else:
            self.gui.backend.set_env_bit(env_bit_list=env_lists, set_all=False)

        bit_list = []
        qubit_list = [x for x in env_lists if str(x).startswith("q")]
        coupler_list = [x for x in env_lists if str(x).startswith("c")]
        bit_list.extend(qubit_list)
        bit_list.extend(coupler_list)
        pair_list = [
            "q{}q{}".format(
                x[1:].split("-", maxsplit=1)[0], x[1:].split("-", maxsplit=1)[1]
            )
            for x in coupler_list if x.startswith("c")
        ]
        union_lists = []

        cache = {
            StandardContext.QC.value: qubit_list,
            StandardContext.CGC.value: pair_list,
            StandardContext.CC.value: coupler_list,
            StandardContext.NT.value: env_lists,
            StandardContext.URM.value: union_lists,
        }

        self.physical_com_cache = cache
        self.refresh_com(self._ui.default_context_com.currentText())
        self.refresh_com(self._ui.exp_context_com.currentText())
        max_qubit = self._ui.com_max_qubit.currentText()
        self._ui.com_max_qubit.clear()
        self._ui.com_max_qubit.set_units(bit_list)
        self._ui.com_max_qubit.set_text(max_qubit)

        online_qubit = self._ui.com_online_qubit.currentText()
        self._ui.com_online_qubit.clear()
        self._ui.com_online_qubit.set_units(bit_list)
        self._ui.com_online_qubit.set_text(online_qubit)

        f12_opt_bits = self._ui.f02_opt_qubit.currentText()
        self._ui.f02_opt_qubit.clear()
        self._ui.f02_opt_qubit.set_units(qubit_list)
        self._ui.f02_opt_qubit.set_text(f12_opt_bits)

    @Slot(list)
    def physical_refresh(self, physical_lists):
        """
        refresh physical unit, signal by topology widget, will set com value.
        """
        if physical_lists:
            show_text = ",".join(physical_lists)
        else:
            show_text = ""
        if self.current_tab == 0:
            self._ui.default_physical_unit_com.setCurrentText(show_text)
            # context_type = self._ui.default_context_com.currentText()
            # self.context_builder.context_data[context_type]['physical_unit'] = physical_lists
        elif self.current_tab == 1:
            if self.exp_cache:
                self._ui.exp_physical_unit_com.setCurrentText(show_text)
                self.exp_cache.context_options.physical_unit[0] = show_text
        else:
            return

    def judge_emit_option_refresh(self):
        self.change_physical_unit.emit(self.exp_cache)

    @Slot()
    def update_options_edit(self):
        if not self.exp_cache:
            return
        self.judge_emit_option_refresh()

    @slot_catch_exception()
    def create_custom_point(self, units: List[str] = None):
        bits = []
        if units is None:
            units = []
            units.extend(list(self.context_builder.chip_data.cache_qubit.keys()))
            units.extend(list(self.context_builder.chip_data.cache_coupler.keys()))

            dialog = MultiSelectDialog(self, units)
            dialog.setWindowTitle("Which qubit do you want to import?")

            if dialog.exec():
                bits = dialog.selected
        else:
            bits = units

        for bit in bits:
            if bit not in self.context_builder.global_options.custom_points:
                self.context_builder.global_options.custom_points[bit] = 0

        self.custom_point_model.refresh_auto()

    def clear_custom_point(self, bit_names):
        for bit in bit_names:
            self.context_builder.global_options.custom_points.pop(bit)
        self.custom_point_model.refresh_auto()

    @Slot()
    def update_global(self):
        check_global = self.check_global
        if check_global:
            reply = QMessageBox().question(
                self,
                "context global change",
                "will change context global config,sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

            self.context_builder.set_global_options(**check_global)
            self.gui.backend.cache_context_manager()

    @slot_catch_exception()
    def context_save(self):
        """
        save context to visage experiment and context.
        1. check global;
        2. save context by options edit widget.
        3. send signal to options edit.
        4. send signal to topology refresh physical unit.
        """
        if self.exp_cache:
            self.exp_cache.context_options.readout_type[
                0
            ] = self._ui.exp_read_com.currentText()
            self.exp_cache.context_options.name[
                0
            ] = self._ui.exp_context_com.currentText()
            exp_physical_unit = self._ui.exp_physical_unit_com.currentText()
            self.exp_cache.context_options.physical_unit[0] = physical_units_trans(
                exp_physical_unit
            )
            self.gui.options_window.save_experiment()

            self.judge_emit_option_refresh()
            self.refresh_topology.emit(exp_physical_unit)

    @Slot()
    def default_refresh(self):

        def refresh_dag(model_dag):
            if model_dag.node_params:
                for _node in model_dag.node_params.values():
                    update_visage_exp_context(_node, ctx_type=context_type, rd_type=readout_type, pc_unit=physical_unit)

        def update_visage_exp_context(vis_experiment, ctx_type, rd_type, pc_unit):
            if vis_experiment.context_options and ctx_type in vis_experiment.context_options.name[1]:
                vis_experiment.context_options.name[0] = ctx_type
                vis_experiment.context_options.readout_type[0] = rd_type
                vis_experiment.context_options.physical_unit[0] = pc_unit
                self.gui.options_window.parallel_check(vis_experiment)

        context_type = self._ui.default_context_com.currentText()
        readout_type = self._ui.default_read_com.currentText()
        physical_unit = physical_units_trans(self._ui.default_physical_unit_com.currentText())

        self.context_builder.global_options.context_data[context_type][
            "readout_type"
        ] = readout_type
        self.context_builder.global_options.context_data[context_type][
            "physical_unit"
        ] = physical_unit

        for exp_type, exp_type_dict in self.gui.backend.experiments.items():
            if exp_type_dict:
                for exp_name, v_exp in exp_type_dict.items():
                    update_visage_exp_context(v_exp, ctx_type=context_type, rd_type=readout_type, pc_unit=physical_unit)
                    if exp_type == "BatchExperiment":
                        refresh_dag(v_exp.dag)

        for dag in self.gui.backend.dags.values():
            refresh_dag(dag)

        if self.exp_cache and context_type in self.exp_cache.context_options.name[1]:
            self._ui.exp_physical_unit_com.setCurrentText(physical_unit)
            self._ui.exp_context_com.setCurrentText(context_type)
            self._ui.exp_read_com.setCurrentText(readout_type)
            self.context_save()

        logger.info("default context refresh success.")
        self.gui.backend.cache_context_manager()
