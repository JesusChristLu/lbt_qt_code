# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/28
# __author:       YangChao Zhao, HanQing Shi

"""HeatMap windows."""

import time
from typing import TYPE_CHECKING, Union

from PySide6.QtCore import Slot, Signal, Qt
from PySide6.QtGui import QAction
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QGraphicsScene, QMenu, QInputDialog
from PySide6.QtWidgets import QLabel, QHeaderView, QAbstractItemView
from PySide6.QtWidgets import QTableView
from loguru import logger
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from pyQCat.processor.heatmap import HeatMap, TwoQubitInteractionHeatMap
from pyQCat.qubit import Coupler
from pyQCat.structures import QDict
from pyQCat.tools.allocation import *
from pyQCat.tools.utilities import get_bound_ac_spectrum
from pyQCat.types import StandardContext
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.gui.heatmap_set_value_ui import Ui_SetValueWindow
from pyqcat_visage.gui.heatmap_ui import Ui_MainWindow
from pyqcat_visage.gui.tools.utilies import slot_process_reject, slot_process_warning
from pyqcat_visage.gui.widgets.heatmap.divide_table_model import (
    QTableModelDivide,
    QTableModelAmpDivide,
    QTableModelParallelDivide,
)
from pyqcat_visage.gui.widgets.heatmap.divide_tree_mode import DivideTreeModel
from pyqcat_visage.gui.widgets.heatmap.struct_tree_model import StructTreeModel
from pyqcat_visage.gui.widgets.title_window import TitleWindow

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class HeatMapWindow(TitleWindow):
    """Heatmap window."""

    def __init__(self, gui: "VisageGUI", parent=None):
        self.gui = gui
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self._ui.tabWidget.setCurrentIndex(0)

        self.struct_tree_model = None
        self.divide_table_model = None
        self.divide_config_mode = None

        # 2023/07/27
        # Not a good data structure design for divide tables.
        # Scalability was not considered and the data structure needed to
        # be redesigned for optimization.

        # used for IF divide.
        self.bucket = {}
        self.group = {}

        # used for readout amp/power divide.
        self.bus_buket = {}
        self.bus_group = {}

        # parallel divide group
        self.parallel_bucket = {}
        self.parallel_group = {}

        self.divide_params = QDict()

        self._current_topic = "drive_freq"

        self._extension_ui()

        # initialize view
        # todo instead with QGraphicsItems.
        _, window_h = self.size().toTuple()
        fig_size = (window_h / 100 - 1, window_h / 100 - 1)
        self._fig = Figure(figsize=fig_size)
        self._view = FigureCanvasQTAgg(self._fig)
        self._set_value = SetValueWidget(self)

        self._graphic_scene = QGraphicsScene()

        # connect matplotlib signal
        self._fig.canvas.mpl_connect("pick_event", self._pick_qubit)

        self.display_unit = []

        # ready save qubits
        self._edit_bits = {}
        self._edit_keys = defaultdict(list)
        self.divide_bits = []
        self.divide_amp_bits = []
        self._cmap_theme = "viridis"

        # add mouse select.
        self._ax = None
        self._rand = dict(
            start_location=None, end_location=None, is_press=False, rand_id=0
        )
        self._rand_line_dict = {}
        self._select_bits = []
        self._control_select = False
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self.right_menu)
        self._view.mpl_connect("button_press_event", self.mat_mouse_press)
        self._view.mpl_connect("motion_notify_event", self.mat_mouse_move)
        self._view.mpl_connect("button_release_event", self.mat_mouse_release)
        self._view.mpl_connect("key_press_event", self.mat_press_event)
        self._view.mpl_connect("key_release_event", self.mat_release_event)

    def _extension_ui(self):
        self._ui.info_tree_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._ui.info_table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._ui.parallel_table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._ui.splitter.setStretchFactor(0, 5)
        self._ui.splitter.setStretchFactor(1, 2)
        self.ui.splitter_2.setStretchFactor(0, 5)
        self.ui.splitter_2.setStretchFactor(1, 2)

        self._setup_struct()
        self._setup_status_bar()

        context_names = list(GUI_CONFIG.std_context.keys())
        for ctx_name in context_names:
            self.parallel_bucket[ctx_name] = {}
        self._ui.context_group.addItems(context_names)

    @property
    def current_topic(self):
        """Get the current displayed topic."""
        return self._current_topic

    @property
    def backend(self):
        """Get the backend."""
        return self.gui.backend

    @property
    def qubits(self):
        """Get the system qubit set."""
        return self.backend.context_builder.chip_data.cache_qubit

    @property
    def couplers(self):
        """Get the system coupler set."""
        return self.backend.context_builder.chip_data.cache_coupler

    @property
    def qubit_pairs(self):
        """Get the system qubit pair set."""
        return self.backend.context_builder.chip_data.cache_qubit_pair

    @property
    def edit_keys(self):
        return self._edit_keys

    @property
    def ui(self):
        """Get the ui object."""
        return self._ui

    @property
    def parallel_config(self):
        return self.divide_config_mode.data_dict

    def _setup_status_bar(self):
        self.sample_label = QLabel(f" Sample Name (***) ")
        self._ui.statusbar.addWidget(self.sample_label)

        self.time_label = QLabel(f" Last Refresh Time (***) ")
        self._ui.statusbar.addWidget(self.time_label)

        self.field_label = QLabel(f" Field Name (***) ")
        self._ui.statusbar.addWidget(self.field_label)

    def _setup_struct(self):
        self.struct_tree_model = StructTreeModel(
            self, self.gui, self._ui.structTreeView, "Qubit"
        )
        self._ui.structTreeView.setModel(self.struct_tree_model)
        self._ui.structTreeView.choose_struct_signal.connect(self._show_params)
        self.struct_tree_model.set_style(self._ui.component_combox.currentText())

        self.divide_table_model = QTableModelDivide(
            self.gui, self, self._ui.info_tree_view
        )
        self._ui.info_tree_view.setModel(self.divide_table_model)
        self._ui.info_tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Divide readout amp/power table.
        self.amp_divide_table_model = QTableModelAmpDivide(
            self.gui, self, self._ui.info_table_view
        )
        self._ui.info_table_view.setModel(self.amp_divide_table_model)
        self._ui.info_table_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        validator = QDoubleValidator(-1, 1, 5)
        self._ui.std_amp_edit.setValidator(validator)

        # Divide parallel group table
        self.parallel_divide_mode = QTableModelParallelDivide(
            self.gui, self, self._ui.parallel_table_view
        )
        self._ui.parallel_table_view.setModel(self.parallel_divide_mode)
        self._ui.parallel_table_view.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )

        # Divide parallel config params tree
        self.divide_config_mode = DivideTreeModel(
            self, self.gui, self._ui.parallel_table_view
        )
        self._ui.parallel_tree_view.setModel(self.divide_config_mode)

    def _show_params(self, param: str):
        """Set the current topic."""
        if len(self.qubits) == 0:
            return self.handler_ret_data(
                QDict(code=800, msg=f"Heatmap data is empty, please query first!")
            )
        self._current_topic = param
        self.reload()

    def _pop_set_value_window(self, qubit, value_types: str):
        """open sub window."""
        self._set_value.set_label(qubit, value_types)
        self._set_value.qubit = qubit
        self._set_value.show()

    # ----------------------------- Tool Slot ---------------------------

    @Slot()
    def save_picture(self):
        if self._fig is not None:
            config_path = self.backend.config.get("system").get("config_path")
            dirname = os.path.join(
                config_path, "HeatMap", self._ui.component_combox.currentText()
            )
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            name = f"{self._current_topic}-{time.strftime('%Y-%m-%d-%H_%M_%S')}.png"
            file_path = os.path.join(dirname, name)
            logger.log("UPDATE", f"Save HeatMap Png in {file_path}!")
            self._fig.savefig(file_path)

    @slot_process_reject()
    def tool_save_db(self):
        if self.ask_ok("Are you sure to save heatmap to database?", "HeatMap"):
            for key, qubit in self._edit_bits.items():
                qubit.save_data(update_list=self.edit_keys.get(key))
                self.backend.context_builder.update_records.pop(key, None)
            self._edit_bits.clear()
            self._edit_keys.clear()
            self.reload()

    @Slot()
    def tool_save_local(self):
        map_name, ok = self.ask_input("Save HeatMap", "Please input file name")
        if ok:
            config_path = self.backend.config.get("system").get("config_path")
            dirname = os.path.join(
                config_path,
                "HeatMap",
                f'{map_name}-{time.strftime("%Y-%m-%d-%H_%M_%S")}',
            )
            if self.ask_ok(
                f"Are you sure to save heatmap to local? It will be saved in {dirname}",
                "HeatMap",
            ):
                ret_data = self.backend.save_heatmap_to_local(dirname)
                self.handler_ret_data(ret_data, show_suc=True)

    @slot_process_warning()
    def tool_query(self):
        self.gui.query_all()

    def set_query_button_action(self, enable: bool):
        self._ui.actionQuery.setEnabled(enable)
        self._ui.actionRefresh.setEnabled(enable)

    def refresh_all(self, *args):
        sample = self.backend.config.system.sample
        time_str = self._time_stamp()
        self.sample_label.setText(f" Sample Name ({sample}) ")
        self.time_label.setText(f" Last Refresh Time ({time_str}) ")

        self._load_lo_info()
        self._edit_bits.clear()
        self._edit_keys.clear()

    @Slot()
    def tool_import(self):
        self.handler_ret_data(
            QDict(code=800, msg="Sorry, this feature has not been enabled yet")
        )

    @slot_process_warning()
    def tool_refresh(self):
        self.tool_query()

    @Slot(str)
    def change_component(self, component: str):
        self.struct_tree_model.set_style(component)

    @Slot(str)
    def reload(self):
        """build and re-build heatmap object."""

        def get_attr(qubit, topic):
            topic_list = topic.split(".")
            try:
                freq_max, freq_min = get_bound_ac_spectrum(qubit)
                freq_map = {"freq_max": freq_max, "freq_min": freq_min}
            except:
                freq_map = {"freq_max": 0.0, "freq_min": 0.0}

            res = qubit
            for t in topic_list:
                if topic in ["freq_max", "freq_min"]:
                    return freq_map.get(topic)
                if topic.startswith("dcm"):
                    dcm = self.backend.context_builder.chip_data.cache_dcm.get(
                        f"{qubit.name}_01.bin", QDict()
                    )
                    res = dcm.to_dict().get(topic_list[-1])
                    break
                res = getattr(res, t)

            return res

        annotation_format = ".{}g".format(self._ui.precision_box.value() or 7)
        if self._ui.component_combox.currentIndex() == 0:
            value_map = {
                (qubit,): get_attr(qubit, self._current_topic)
                for qubit in self.qubits.values()
            }
            heatmap = HeatMap(
                value_map,
                show_value=self._ui.show_value_box.isChecked(),
                edit_units=self._edit_bits,
                annotation_format=annotation_format,
            )
            describe = f"Qubit.{self._current_topic}"
        elif self._ui.component_combox.currentIndex() == 1:
            value_map = {}
            for name, coupler in self.couplers.items():
                q0, q1 = [f"q{r}" for r in name[1:].split("-")]
                value_map[(self.qubits.get(q0), self.qubits.get(q1))] = get_attr(
                    coupler, self._current_topic
                )
            heatmap = TwoQubitInteractionHeatMap(
                value_map,
                show_value=self._ui.show_value_box.isChecked(),
                edit_units=self._edit_bits,
                style="Coupler",
                annotation_format=annotation_format,
            )
            describe = f"Coupler.{self._current_topic}"
        else:
            value_map = {}
            pattern = re.compile(r"([a-zA-Z]+)(\d+)")
            for key, pair in self.qubit_pairs.items():
                matches = pattern.findall(key)
                result = [match[0] + match[1] for match in matches]
                q0, q1 = result
                cur_topic = check_qubit_pair_topic(self._current_topic, pair)
                value_map[(self.qubits.get(q0), self.qubits.get(q1))] = get_attr(
                    pair, cur_topic
                )
            heatmap = TwoQubitInteractionHeatMap(
                value_map,
                show_value=self._ui.show_value_box.isChecked(),
                edit_units=self._edit_bits,
                style="QubitPair",
                annotation_format=annotation_format,
            )
            describe = f"QubitPair.{self._current_topic}"

        heatmap.config["collection_options"]["cmap"] = self._cmap_theme

        if value_map:
            # clear canvas.
            self._fig.clear()
            self._ax = self._fig.subplots()
            # get heatmap and draw.
            heatmap.plot(
                self._ax,
                title=f"{self._ui.component_combox.currentText()} {self._current_topic} HeatMap.",
            )
            self.display_unit = heatmap.display_unit
            self._view.draw()
            # update graphics view.
            self._graphic_scene.addWidget(self._view)
            self._ui.graphicsViewHeatmap.setScene(self._graphic_scene)
            self._ui.graphicsViewHeatmap.show()
            self.field_label.setText(f" Field Name ({describe}) ")
        else:
            logger.warning(f"No find any base qubit!")

    def _pick_qubit(self, event):
        """Pick qubit in the canvas."""

        artist = event.artist
        component: Union[Qubit, Coupler, QubitPair, None] = None
        if event.mouseevent.dblclick:
            # if (
            #     self._current_topic.split(".")[-1] in UserForbidden.change_chip_params
            #     and self.backend.login_user
            #     and not (
            #         self.is_super
            #         or self.is_admin
            #     )
            # ):
            #     warning_box = QMessageBox()
            #     warning_box.setText(
            #         "normal user can't change inst, please contact your administrator"
            #     )
            #     warning_box.exec_()
            #     return
            _, ind_array = artist.contains(event.mouseevent)
            ind = self.display_unit[ind_array["ind"][0]]
            if self._ui.component_combox.currentIndex() == 0:
                component = sorted(list(self.qubits.values()))[ind]
            if self._ui.component_combox.currentIndex() == 1:
                for coupler in self.couplers.values():
                    if int(coupler.bit) == int(ind):
                        component = coupler
                        break
            if self._ui.component_combox.currentIndex() == 2:
                component = list(self.qubit_pairs.values())[ind]
        if component is not None:
            topic = check_qubit_pair_topic(self._current_topic, component)
            self._pop_set_value_window(component, topic)
            self._edit_bits[component.name] = component
            self.backend.context_builder.update_records.update({component.name: {}})

    @staticmethod
    def _time_stamp():
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    @Slot(int)
    def show_value_change(self, state: int):
        if self._ui.show_value_box.isChecked():
            self._ui.precision_box.setEnabled(True)
        else:
            self._ui.precision_box.setEnabled(False)
        if self._current_topic:
            self.reload()
            logger.debug(f"HeatMap Refresh {'value' if state else 'key'} State")

    @Slot(int)
    def precision_change(self, value: int):
        if self._current_topic:
            self.reload()
            logger.debug(f"HeatMap Refresh {'precision' if value else 'key'} Value")

    @Slot(str)
    def change_cmap(self, theme: str):
        self._cmap_theme = theme
        if self._current_topic:
            self.reload()

    # ----------------------------- Divide Slot ---------------------------
    def _get_divide_bits(
        self,
        view: QTableView,
        model: Union[QTableModelDivide, QTableModelAmpDivide],
        group: Dict,
    ):
        """Get the bits number which need to be divided."""
        indexes = view.selectedIndexes()
        divide_bits = (
            [model.bit_from_index(index)[0] for index in indexes]
            if indexes
            else list(group.keys())
        )
        self._refresh_divide_params()
        return divide_bits

    @Slot()
    def divide_baseband_freq(self):
        self.divide_bits = self._get_divide_bits(
            self._ui.info_tree_view, self.divide_table_model, self.group
        )
        group_name = self._ui.group_combo.currentText()
        if group_name:
            aller = IntermediateFreqAllocation(self.backend.context_builder.chip_data)
            aller.set_allocation_options(mode=group_name.split("-")[0])
            aller.allocate(self.divide_bits)
            self.change_group(self._ui.group_combo.currentText())
        else:
            pyqlog.warning("Please choose group first!")

    @Slot()
    def save_baseband_freq(self):
        if self.bucket:
            if self.ask_ok("Are you sure to save baseband freq?", "OK"):
                for bit, data in self.group.items():
                    if data.freq and bit in self.divide_bits:
                        qubit = self.qubits.get(bit)
                        qubit.save_data()
                self.divide_bits.clear()

    @Slot()
    def divide_readout_amp(self):
        """Divide amp and power for parallel or union readout."""
        self.divide_amp_bits = self._get_divide_bits(
            self._ui.info_table_view, self.amp_divide_table_model, self.bus_group
        )
        bus_name = self._ui.bus_combo.currentText()
        if bus_name:
            aller = ReadoutAmpAllocation(self.backend.context_builder.chip_data)
            aller.set_allocation_options(standard_amp=float(self._ui.std_amp_edit.text()))
            aller.allocate(self.divide_amp_bits)
            self.change_bus_group(bus_name)
        else:
            pyqlog.warning("Please choose group first!")

    @Slot()
    def save_readout_param(self):
        """Save divided qubits to database."""
        if self.bus_buket and self.ask_ok(
            "Are you sure to save divided results?", "OK"
        ):
            for bit, data in self.bus_group.items():
                if bit in self.divide_amp_bits:
                    qubit = self.qubits.get(bit)
                    qubit.save_data()

            self.divide_amp_bits.clear()

    @Slot(str)
    def change_group(self, group: str):
        if group:
            same_lo_qubits = self.bucket.get(group)

            module = group.split("-")[0]

            temp = {}

            for qubit in same_lo_qubits:
                freq = qubit.drive_freq if module == "xy" else qubit.probe_freq
                bf = (
                    qubit.XYwave.baseband_freq
                    if module == "xy"
                    else qubit.Mwave.baseband_freq
                )
                channel = qubit.xy_channel if module == "xy" else qubit.readout_channel
                gap = qubit.inst.xy_gap if module == "xy" else qubit.inst.m_gap
                (
                    lo1,
                    lo2,
                ) = (
                    "-",
                    "-",
                )
                if freq and bf:
                    lo1 = 8100 if freq < 6000 else 5900
                    lo2 = round(freq + lo1 - bf, 3)

                temp[qubit.name] = QDict(
                    name=qubit.name,
                    freq=freq,
                    IF=bf,
                    channel=channel,
                    lo1=lo1,
                    lo2=lo2,
                    gap=gap,
                )

            self.group = temp
            self.divide_table_model.refresh_auto(False)

    @Slot(str)
    def change_bus_group(self, bus: str):
        """Slot function to replace BUS serial number.

        Args:
            bus (str): bus number.
        """
        if bus:
            same_bus_qubits = self.bus_buket.get(bus)
            temp = {}
            for qubit in same_bus_qubits:
                bf = qubit.Mwave.baseband_freq or "-"
                channel = qubit.readout_channel or "-"
                sample_delay = qubit.sample_delay or "-"
                sample_width = qubit.sample_width or "-"
                power = qubit.probe_power or "-"
                amp = qubit.Mwave.amp or "-"
                temp[qubit.name] = {
                    "name": qubit.name,
                    "sample delay": sample_delay,
                    "sample width": sample_width,
                    "channel": channel,
                    "power": power,
                    "amp": amp,
                    "baseband freq": bf,
                    "index": [],
                }
            self.bus_group = temp
            self.amp_divide_table_model.refresh_auto(False)

    @Slot(str)
    def change_parallel_group(self, group: str):
        self.ui.parallel_set_button.setEnabled(False)
        if group:
            context_group_name = self._ui.context_group.currentText()
            parallel_units = self.parallel_bucket.get(context_group_name).get(group)
            if parallel_units:
                temp = {}
                for qubit in parallel_units:
                    if context_group_name == StandardContext.QC.value:
                        temp[qubit.name] = QDict(
                            name=qubit.name,
                            xy_lo=qubit.inst.xy_lo,
                            xy_gap=qubit.inst.xy_gap,
                            m_lo=qubit.inst.m_lo,
                            m_gap=qubit.inst.m_gap,
                            bus=qubit.inst.bus,
                            probe_power=qubit.probe_power,
                        )
                    elif context_group_name == StandardContext.CGC.value:
                        temp[qubit.name] = QDict(
                            name=qubit.name, qc=qubit.qc, ql=qubit.ql, qh=qubit.qh
                        )
                    elif context_group_name == StandardContext.CC.value:
                        temp[qubit.name] = QDict(
                            name=qubit.name,
                            idle_point=qubit.idle_point,
                            dc_min=qubit.dc_min,
                            dc_max=qubit.dc_max,
                            z_dc_channel=qubit.z_dc_channel,
                            z_flux_channel=qubit.z_flux_channel,
                        )
                self.parallel_group = temp
                self.parallel_divide_mode.refresh_auto(False)
                self.gui.ui.tabTopology.topology_view.set_selected_cache(
                    list(temp.keys())
                )
                self.ui.parallel_set_button.setEnabled(True)
        else:
            self.parallel_group.clear()
            self.parallel_divide_mode.refresh_auto(False)

    @Slot(str)
    def change_parallel_context(self, name: str):
        parallel_ctx_group = self.parallel_bucket.get(name)
        self.ui.parallel_group.clear()
        self.ui.mode_group.clear()
        self.ui.parallel_table_view.reset()
        if parallel_ctx_group:
            self.ui.parallel_group.addItems(list(parallel_ctx_group.keys()))

        if name == StandardContext.QC.value:
            self.ui.mode_group.addItems(ParallelAllocationQC._default_allocation_options()["validator"]["mode"][0])

    def _refresh_divide_params(self):
        self.divide_params.max_gap = self._ui.max_gap_box.value()
        self.divide_params.accuracy = self._ui.accuracy_box.value()
        self.divide_params.goal_gap = self._ui.goal_gap_edit.text()
        self.divide_params.left_lo1 = self._ui.left_lo1_box.value()
        self.divide_params.right_lo1 = self._ui.right_lo1_box.value()
        self.divide_params.expect_if = self._ui.expect_if_box.value()
        self.divide_params.min_freq = self._ui.min_freq_box.value()
        self.divide_params.max_freq = self._ui.max_freq_box.value()
        self.divide_params.mid_freq = self._ui.mid_freq_box.value()

    def _load_lo_info(self):
        self.backend.refresh_lo_info()

        bit_map = {}
        bus_bit_map = {}
        for key, value in self.backend.lo_map.items():
            bit_map[key] = [self.qubits.get(bit) for bit in value]

        for key, value in self.backend.bus_map.items():
            bus_bit_map[key] = [self.qubits.get(bit) for bit in value]

        self.bucket = bit_map
        self.bus_buket = bus_bit_map

        self._ui.group_combo.clear()
        bit_map_list = list(bit_map.keys())
        bit_map_list.sort(key=lambda x: int(x.split("-")[-1]))
        self._ui.group_combo.addItems(bit_map_list)

        self._ui.bus_combo.clear()
        bus_bit_map_list = list(bus_bit_map.keys())
        bus_bit_map_list.sort(key=lambda x: int(x.split("-")[-1]))
        self._ui.bus_combo.addItems(bus_bit_map_list)

    # mouse event
    def mat_mouse_press(self, event):
        if event.inaxes and event.button is MouseButton.LEFT:
            if not self._control_select:
                self.clear_rubber_band()
                self._select_bits = []
            _id = len(self._rand_line_dict) + 1
            self._rand = dict(
                start_location=None, end_location=None, is_press=True, rand_id=_id
            )
            self._rand_line_dict.update({_id: []})
            self._rand["start_location"] = (event.xdata, event.ydata)

    def mat_mouse_move(self, event):
        if event.inaxes:
            if self._rand["is_press"]:
                self._rand["end_location"] = (event.xdata, event.ydata)
                self.draw_rubber_band()
        elif self._rand["is_press"]:
            self.check_select_node()
            self._rand = dict(
                start_location=None, end_location=None, is_press=False, rand_id=0
            )

    def mat_mouse_release(self, event):
        if event.inaxes and event.button is MouseButton.LEFT and self._rand["is_press"]:
            self.check_select_node()
            self._rand = dict(
                start_location=None, end_location=None, is_press=False, rand_id=0
            )

    def mat_press_event(self, event):
        if event.key == "control":
            self._control_select = True

    def mat_release_event(self, event):
        if event.key == "control":
            self._control_select = False

    def check_select_node(self):
        if not (self._rand["end_location"] and self._rand["start_location"]):
            return

        select_x_range = [
            self._rand["start_location"][0],
            self._rand["end_location"][0],
        ]
        select_y_range = [
            self._rand["start_location"][1],
            self._rand["end_location"][1],
        ]
        select_x_range.sort()
        select_y_range.sort()

        def in_select(_key) -> bool:
            if re.match(NAME_PATTERN.coupler, _key):
                sc = _key.split("-")
                probe_q = f"q{sc[0][1:]}"
                drive_q = f"q{sc[1]}"
                x = (self.qubits[probe_q].col + self.qubits[drive_q].col) / 2
                y = (self.qubits[probe_q].row + self.qubits[drive_q].row) / 2
            elif re.match(NAME_PATTERN.qubit, _key):
                x = self.qubits[_key].col
                y = self.qubits[_key].row
            else:
                logger.warning(f"Heatmap no support {_key}")
                return False

            if (
                select_x_range[0] < x < select_x_range[1]
                and select_y_range[0] < y < select_y_range[1]
            ):
                return True
            else:
                return False

        bit_dict = None
        if self._ui.component_combox.currentIndex() == 0:
            bit_dict = self.qubits
        elif self._ui.component_combox.currentIndex() == 1:
            bit_dict = self.couplers
        elif self._ui.component_combox.currentIndex() == 2:
            bit_dict = self.qubit_pairs
        if not bit_dict:
            return
        for key, qubit in bit_dict.items():
            if in_select(key):
                self._select_bits.append(qubit)

    def draw_rubber_band(self):
        start = self._rand["start_location"]
        end = self._rand["end_location"]
        self.remove_pre_rand(self._rand["rand_id"])
        self._rand_line_dict[self._rand["rand_id"]].append(
            self._ax.plot([start[0], start[0]], [start[1], end[1]], c="blue")[0]
        )
        self._rand_line_dict[self._rand["rand_id"]].append(
            self._ax.plot([start[0], end[0]], [start[1], start[1]], c="blue")[0]
        )
        self._rand_line_dict[self._rand["rand_id"]].append(
            self._ax.plot([end[0], end[0]], [end[1], start[1]], c="blue")[0]
        )
        self._rand_line_dict[self._rand["rand_id"]].append(
            self._ax.plot([end[0], start[0]], [end[1], end[1]], c="blue")[0]
        )
        self._view.draw_idle()

    def remove_pre_rand(self, rand_id, draw=True):
        for line in self._rand_line_dict[rand_id]:
            line.remove()
        self._rand_line_dict[rand_id].clear()

        if draw:
            self._view.draw_idle()

    def clear_rubber_band(self):
        for key in self._rand_line_dict:
            self.remove_pre_rand(key, draw=False)

        self._rand_line_dict.clear()
        self._view.draw_idle()

    # menu
    def right_menu(self, point):
        right_menu = QMenu(self)
        right_menu.addAction(QAction("修改", self, triggered=self.change_values))
        right_menu.addAction(
            QAction("设为None", self, triggered=self.change_value_to_none)
        )
        right_menu.exec_(self._view.mapToGlobal(point))

    def change_values(self):
        if self._select_bits:
            # if (
            #     self._current_topic.split(".")[-1] in UserForbidden.change_chip_params
            #     and self.backend.login_user
            #     and not (
            #         self.is_super
            #         or self.is_admin
            #     )
            # ):
            #     warning_box = QMessageBox()
            #     warning_box.setText(
            #         "normal user can't change inst, please contact your administrator"
            #     )
            #     warning_box.exec_()
            #     return

            value, ok = QInputDialog.getDouble(
                self,
                "set bits",
                "set bits",
                decimals=5,
            )
            if ok:
                self.modify_values(value)
            self._select_bits = []

    def change_value_to_none(self):
        if self._select_bits:
            self.modify_values(value=None)

    def modify_values(self, value):
        for bit in self._select_bits:
            key = check_qubit_pair_topic(self._current_topic, bit)
            tem_key = key.split(".")
            set_deep_str(bit, tem_key, value)
            self._edit_bits[bit.name] = bit
            self._edit_keys[bit.name].append(key)
            self.backend.context_builder.update_records.update({bit.name: {}})
        self.reload()
        self._select_bits = []

    def parallel_divide(self):
        parallel_physical_units = self._ui.physical_unit_edit.text()
        ctx_name = self._ui.context_group.currentText()
        if ctx_name in [StandardContext.QC.value, StandardContext.CC.value]:

            if not parallel_physical_units:
                parallel_physical_units = self.backend.context_builder.global_options.env_bits
            elif isinstance(parallel_physical_units, str):
                parallel_physical_units = [v.strip() for v in parallel_physical_units.split(",")]

            if parallel_physical_units:
                if ctx_name == StandardContext.QC.value:
                    aller = ParallelAllocationQC(self.backend.context_builder.chip_data)
                    aller.set_allocation_options(mode=self.ui.mode_group.currentText())
                else:
                    aller = ParallelAllocationCC(self.backend.context_builder.chip_data)
                aller.set_allocation_options(**self.parallel_config.get(ctx_name))
                aller.allocate(parallel_physical_units)

                self.parallel_bucket[ctx_name] = aller.run_options.parallel_obj_map
                self._ui.parallel_group.clear()
                self._ui.parallel_group.addItems(aller.run_options.parallel_group_names)
        elif ctx_name == StandardContext.CGC.value:
            if parallel_physical_units:
                parallel_physical_units = parallel_physical_units.split(",")
            else:
                parallel_physical_units = []
                qubit_pair_map = self.backend.context_builder.chip_data.cache_qubit_pair

                for key, value in qubit_pair_map.items():
                    if (
                        value.ql in self.backend.context_builder.global_options.env_bits
                        and value.qh
                        in self.backend.context_builder.global_options.env_bits
                    ):
                        parallel_physical_units.append(key)
            if parallel_physical_units:
                aller = ParallelAllocationCGC(self.backend.context_builder.chip_data)
                aller.set_allocation_options(**self.parallel_config.get(ctx_name))
                aller.allocate(parallel_physical_units)

                self.parallel_bucket[ctx_name] = aller.run_options.parallel_obj_map
                self._ui.parallel_group.clear()
                self._ui.parallel_group.addItems(aller.run_options.parallel_group_names)
        else:
            logger.warning(
                f"Parallel grouping of {ctx_name} is currently not supported"
            )

    def parallel_set(self):
        indexes = self.ui.parallel_table_view.selectedIndexes()
        unit_index = [i.row() for i in indexes]
        ctx_name = self._ui.context_group.currentText()
        if ctx_name == StandardContext.QC.value:
            if self._parallel_validate(unit_index):
                if len(indexes) > 1:
                    items = list(self.parallel_group.keys())
                    set_units = [items[index.row()] for index in indexes]
                else:
                    set_units = list(self.parallel_group.keys())

                self.gui.context_sidebar.ui.default_context_com.setCurrentText(
                    self._ui.context_group.currentText()
                )
                self.gui.ui.tabTopology.topology_view.set_physical_bits(set_units)
        elif ctx_name == StandardContext.CGC.value:
            self.gui.context_sidebar.ui.default_context_com.setCurrentText(
                self._ui.context_group.currentText()
            )
            set_units = list(self.parallel_group.keys())
            self.gui.ui.tabTopology.topology_view.refresh(set_units)
            self.gui.context_sidebar.ui.default_physical_unit_com.reset()
            for units in set_units:
                self.gui.context_sidebar.ui.default_physical_unit_com.set_check(units)
        else:
            logger.warning(
                f"Parallel grouping of {ctx_name} is currently not supported"
            )

    def _parallel_validate(self, indexes: List = None):
        records = defaultdict(set)
        for i, (key, value) in enumerate(list(self.parallel_group.items())):
            if not indexes or i in indexes:
                records[f"xy-lo-{value.xy_lo}"].add(value.xy_gap)
                records[f"m-lo-{value.m_lo}"].add(value.m_gap)
                records[f"bus-{value.bus}"].add(value.probe_power)

        msg = ""
        for k, v in records.items():
            if len(v) > 1:
                msg += f"{k} parallel validator error, {v}\n"

        if msg:
            logger.warning(f"Set Parallel Error:\n{msg}")
            return False

        return True


class SetValueWidget(TitleWindow):
    """Window to set the value of the selected qubit field."""

    _qubit_value_changed_signal = Signal()

    def __init__(self, heat_map_ui: "HeatMapWindow", parent=None):
        super().__init__(parent)
        self.heat_map_ui = heat_map_ui
        self.ui = Ui_SetValueWindow()
        self.ui.setupUi(self)
        self.qubit = None
        self._qubit_value_changed_signal.connect(self.heat_map_ui.reload)
        self._topic = None

    def set_label(self, qubit: Union[Qubit, Coupler, QubitPair], value_types: str):
        """Set the label text.

        Args:
            qubit (Qubit, Coupler, QubitPair): The qubit object.
            value_types (str): The value to show.
        """
        self._topic = value_types
        label = f"{qubit} {value_types}: "
        self.ui.label.setText(label)
        self.ui.lineEdit.setText(str(get_deep_attr(qubit, value_types)))

    @Slot()
    def change_value(self):
        """Change value slot function."""
        value = self.ui.lineEdit.text()
        if value.lower() == "none":
            value = "0"
        if re.findall("[a-zA-Z]+", value):
            self.ask_ok(f"The parameter:[{value}] cannot be set!", "changeValue")
            return
        keys = self._topic.split(".")
        set_deep_str(self.qubit, keys, value)
        self._qubit_value_changed_signal.emit()
        self.heat_map_ui.edit_keys[self.qubit.name].append(self._topic)
        self.close_()


def get_deep_attr(source, key: Union[str, list]):
    if source is None:
        return None

    if isinstance(key, str):
        key = key.split(".")

    if isinstance(key, list) and len(key) >= 1:
        if len(key) == 1:
            return getattr(source, key[0], None)
        else:
            return get_deep_attr(getattr(source, key[0]), key[1:])
    return None


def set_deep_str(source, param_path: Union[str, list], value, judge_value=True):
    if judge_value:
        if value is not None:
            value = float(value)
            if param_path[-1].endswith("channel") or param_path[-1] in [
                "bus",
                "m_lo",
                "xy_lo",
            ]:
                value = int(value)
            elif param_path[-1] in ["goodness", "tunable"]:
                value = bool(int(value))

    if type(param_path) is list:
        if len(param_path) == 1:
            if hasattr(source, param_path[0]):
                setattr(source, param_path[0], value)
        elif len(param_path) > 1:
            temp_source = getattr(source, param_path[0], None)
            if temp_source:
                set_deep_str(
                    temp_source,
                    param_path=param_path[1:],
                    value=value,
                    judge_value=False,
                )
    elif type(param_path) is str and hasattr(source, param_path):
        setattr(source, param_path, value)


def divide_baseband_freq(fq_dict: Dict, setting: QDict):
    def divide(_fq_dict: Dict):
        """qubit freq divide."""

        for key in list(_fq_dict.keys()):
            fq = _fq_dict.get(key)
            if not (fq and setting.max_freq > fq > setting.min_freq):
                logger.warning(f"{key} freq is {fq} MHz, pop out!")
                _fq_dict.pop(key)

        fq_items = sorted(_fq_dict.items(), key=lambda x: x[1])

        bucket = defaultdict(list)
        index = 0
        mq = None
        is_lower = True if fq_items[0][1] < setting.min_freq else False
        for item in fq_items:
            _, fq = item

            if fq >= setting.min_freq and is_lower is True:
                is_lower = False
                if index > 0:
                    index += 1
                bucket[f"Group-{index}"].append(item)
                mq = fq
                continue

            mq = mq if mq else fq

            if fq - mq < setting.max_gap:
                bucket[f"Group-{index}"].append(item)
                if len(bucket[f"Group-{index}"]) == setting.max_num:
                    index += 1
                    mq = None
            else:
                index += 1
                bucket[f"Group-{index}"].append(item)
                mq = fq

        for key, val in bucket.items():
            logger.info(f"{key} | count ({len(val)}) | {val}")

        return bucket

    def cal_base_freq(lo1: float, lo2: float, fq: float):
        """formula: fq - baseband_freq = lo2 - lo1"""
        return np.round(fq + lo1 - lo2, setting.accuracy)

    def work(_fq_dict: Dict, _group: str = "Group"):
        """divide baseband freq by lo1, lo2 and qubit freq."""
        freq_list = list(_fq_dict.values())

        max_freq, min_freq = max(freq_list), min(freq_list)

        if max_freq - min_freq > setting.max_gap:
            raise ValueError("Max Gap larger 500 MHz!")

        if max_freq >= setting.min_freq > min_freq:
            raise ValueError("Max freq larger 6 GHz, but min freq lower 6 GHz")

        lo1 = setting.left_lo1 if max_freq < setting.min_freq else setting.right_lo1
        mid_freq = np.round((min_freq + max_freq) / 2, setting.accuracy)
        lo2 = np.round(mid_freq - setting.expect_if + lo1, setting.accuracy)

        base_freq_dict = {}
        for key, val in _fq_dict.items():
            baseband_freq = cal_base_freq(lo1, lo2, val)
            base_freq_dict[key] = {
                "fq": val,
                "baseband_freq": baseband_freq,
                "lo1": lo1,
                "lo2": lo2,
                "group": _group,
            }

        return base_freq_dict

    pre_bucket = divide(fq_dict)
    new_bucket = {}
    for group, value in list(pre_bucket.items()):
        new_val = work(dict(value), group)
        new_bucket[group] = new_val
    return new_bucket


def check_qubit_pair_topic(topic: str, pair):
    if isinstance(pair, QubitPair):
        if "qc." in topic:
            topic = topic.replace("qc", pair.qc)
        elif "qh." in topic:
            topic = topic.replace("qh", pair.qh)
        elif "ql." in topic:
            topic = topic.replace("ql", pair.ql)

    return topic
