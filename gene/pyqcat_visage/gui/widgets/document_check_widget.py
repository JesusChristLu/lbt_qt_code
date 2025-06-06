# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/16
# __author:       YangChao Zhao

"""HeatMap windows."""

import pickle
import struct
from copy import deepcopy
from datetime import datetime
from threading import Thread
from typing import TYPE_CHECKING, Optional, Union, List, Dict

import numpy as np
from PySide6.QtCharts import QChart, QValueAxis, QLineSeries
from PySide6.QtCore import QTimer, Slot, Qt, Signal
from PySide6.QtGui import QPen, QPainter
from PySide6.QtWidgets import QLabel, QWidget
from bson import SON, ObjectId

from pyQCat.database.ODM import ExperimentDoc
from pyQCat.pulse import compile_pulse, PulseComponent, ac_crosstalk_correct
from pyQCat.structures import QDict
from pyQCat.tools import connect_server, disconnect_server
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.gui.document_check_ui import Ui_MainWindow
from pyqcat_visage.gui.widgets.document.doc_tree_model import DocTreeModel
from pyqcat_visage.gui.widgets.title_window import TitleWindow

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI

EXP_DICT = {}


class QMyChart(QChart):
    callout_press = Signal(str)


class DocumentCheckWindow(TitleWindow):
    """Experimental Data Package Retrieval Component.

    It can be used to:

        1. Retrieve the timing chart of the XY, Z, and M lines in the experiment;
        2. Retrieve the hardware parameters of the XY, Z, and M line channel output in the experiment;
        3. Check the scanning status of the experiment, including changes in waveform or hardware parameters;
        4. Review experimental registration related information, including user, experimental name, experimental
        environment, experimental options, analysis options, etc;
    """

    def __init__(self, gui: "VisageGUI", parent: QWidget = None):
        """ Init DocumentCheckWindow.

        Args:
            gui (VisageGUi): Used to request experimental information from the database.
            parent (QWidget): Suggest inheriting from TitleWindow.
        """
        super().__init__(parent)

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self._ui.splitter.setStretchFactor(0, 1)
        self._ui.splitter.setStretchFactor(1, 5)

        self.gui = gui

        self.exp_dict = {}
        self.model_data = {}
        self.context_data = {}
        self.sweep_dict = {}
        self.exp_id = None

        self._parts = None
        self._search_thread = None

        self._setup_model()
        self._setup_status_bar()

        # Indicates that the engine should antialias edges of primitives if possible.
        self._ui.schedule_chart_view.setRenderHint(QPainter.Antialiasing)

        # Set the mouse pointer to a cross star
        self._ui.schedule_chart_view.setCursor(Qt.CursorShape.CrossCursor)

        # mouseMove signal/slot
        self._ui.schedule_chart_view.mouseMove.connect(self.do_chart_view_mouse_move)

        # config schedule line color
        self.__color_line = GUI_CONFIG.document_schedule.color_lines

        self._init_chart()

        # 2000ms interval refresh
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh)

    def do_chart_view_mouse_move(self, point):
        self._ui.schedule_chart_view.chart().mapToValue(point)

    def _init_chart(self):
        self.chart = QMyChart()
        self.chart.setAcceptHoverEvents(True)
        self._ui.schedule_chart_view.setChart(self.chart)
        self._ui.schedule_chart_view.init_callout()

        # create axis x
        self.__axis_x = QValueAxis()
        self.__axis_x.setTitleText(GUI_CONFIG.document_schedule.x_label)
        self.__axis_x.setRange(0, 10)
        self.__axis_x.setTickCount(GUI_CONFIG.document_schedule.x_tick_count)
        self.__axis_x.setLabelFormat("%.3f")
        self.chart.addAxis(self.__axis_x, Qt.AlignBottom)

        # create axis y
        self.__axis_y = QValueAxis()
        self.__axis_y.setTitleText(GUI_CONFIG.document_schedule.y_label)
        self.__axis_y.setRange(-1, 1)
        self.__axis_y.setTickCount(GUI_CONFIG.document_schedule.y_tick_count)
        self.__axis_y.setLabelFormat("%.3f")
        self.chart.addAxis(self.__axis_y, Qt.AlignLeft)

        self.chart.callout_press.connect(self._ui.schedule_chart_view.del_callout)

    def _setup_model(self):
        self.doc_tree_model = DocTreeModel(self, self.gui, self._ui.treeView)
        self._ui.treeView.setModel(self.doc_tree_model)

    def _setup_status_bar(self):
        self.user_label = QLabel(f" User (***) ")
        self._ui.statusbar.addWidget(self.user_label)

        self.type_label = QLabel(f" Type (***) ")
        self._ui.statusbar.addWidget(self.type_label)

        self.name_label = QLabel(f" Name (***) ")
        self._ui.statusbar.addWidget(self.name_label)

        self.time_label = QLabel(f" Time (***) ")
        self._ui.statusbar.addWidget(self.time_label)

        self.fake_label = QLabel(" Pulse (***) ")
        self._ui.statusbar.addWidget(self.fake_label)

    def draw_raw_wave(self, waves, sample_rates, names, set_range):
        self.chart.removeAllSeries()

        for i, sample_rate in enumerate(sample_rates):
            series_wave = QLineSeries()
            pen = QPen(self.__color_line[i % len(self.__color_line)])
            pen.setWidth(GUI_CONFIG.document_schedule.pen_width)
            series_wave.setPen(pen)
            series_wave.setName(names[i])

            # bugfix 2023/03/31 by YangChao Zhao: The sampling rate cannot be taken as a round operation,
            # as a sampling rate of 1.2 leads to inaccurate single sampling cycles. If the
            # round operation is used, if the Z-line timing is too long, it will cause time
            # accuracy distortion
            vx = 0
            step = 1 / sample_rate
            wave = waves[i]

            point_count = len(wave)
            for j in range(point_count):
                value = wave[j]
                series_wave.append(vx, value)
                vx = vx + step

            if set_range:
                if i == 0:
                    self.__axis_x.setRange(0, vx - step)
                else:
                    cur_max = self.__axis_x.max()
                    self.__axis_x.setRange(0, max(vx - step, cur_max))

            self.chart.addSeries(series_wave)
            series_wave.attachAxis(self.__axis_x)
            series_wave.attachAxis(self.__axis_y)
            series_wave.hovered.connect(self._ui.schedule_chart_view.tooltip)
            series_wave.clicked.connect(self._ui.schedule_chart_view.keep_callout)

    @Slot()
    def query(self):
        self._ui.actionQuery.setEnabled(False)
        exp_id = self._ui.exp_id_edit.text().strip()
        ret_data = self.gui.backend.db.query_exp_record(experiment_id=exp_id)
        if ret_data.get("code") == 200:
            self._ui.treeView.hide_placeholder_text()
            self.context_data = ret_data.get("data")
            self.user_label.setText(f' User ({self.context_data.get("username")}) ')
            self.type_label.setText(f' Type ({self.context_data.get("exp_type")}) ')
            self.name_label.setText(f' Name ({self.context_data.get("exp_name")}) ')
            self.time_label.setText(
                f' Time ({str(self.context_data.get("create_time"))}) '
            )
            if self.context_data.get("exp_type") == "composite":
                self.model_data = self.context_data
                self.doc_tree_model.refresh(expand=False)
                self._ui.actionQuery.setEnabled(True)
            else:
                # to check single exp
                if not self._search_thread or not self._search_thread.is_alive():
                    global EXP_DICT
                    EXP_DICT = {}
                    self.timer.start()
                    self._search_thread = Thread(
                        target=query_exp_dict,
                        args=(
                            exp_id,
                            self.gui.backend.config.mongo.inst_host,
                            self.gui.backend.config.mongo.inst_port,
                            self.context_data.get("extra")
                            .get("context")
                            .get("crosstalk"),
                            self._ui.show_delay.isChecked(),
                        ),
                    )
                    self._search_thread.setDaemon(True)
                    self._search_thread.start()
        else:
            self._ui.actionQuery.setEnabled(True)
            self.handler_ret_data(ret_data)

    def refresh(self):
        global EXP_DICT

        def format_sweep_control():
            sweep_dict = {}
            sweep_control = self.exp_dict.get("sweep_control")
            for key, value in sweep_control.items():
                if "waveform" in key:
                    new_value = deepcopy(value)
                    new_value.pop("waveform")
                    sweep_dict[key] = new_value
                else:
                    sweep_dict[key] = value
            self.sweep_dict = sweep_dict

        if EXP_DICT:
            self._ui.treeView.hide_placeholder_text()
            self._ui.actionQuery.setEnabled(True)
            self.timer.stop()
            if EXP_DICT.get("error") is not None:
                self.handler_ret_data(QDict(code=800, msg=EXP_DICT.get("error")))
                self.exp_dict = {}
                self._init_selector()
                self.model_data = self.context_data
            else:
                self.exp_dict = deepcopy(EXP_DICT)
                fake_pulse = self.exp_dict.get("fake_pulse")
                describe = "Fake" if fake_pulse else "Actual"
                self.fake_label.setText(f" Pulse ({describe}) ")
                format_sweep_control()
                self._init_selector()
                self.model_data = self.exp_dict

    def _init_selector(self):
        if self.exp_dict:
            self._ui.module_com.clear()
            loops = len(
                list(self.exp_dict.get("sweep_control").values())[0].get("points")
            )
            loop_items = [str(i) for i in range(loops)]
            self._ui.loop_com.clear()
            self._ui.loop_com.addItems(loop_items)
            self._ui.module_com.addItems(
                [
                    "XY_control",
                    "Z_flux_control",
                    "Read_out_control",
                    "Z_dc_control",
                    "sweep_control",
                    "Context",
                ]
            )
            combo_items = []
            for module in ["XY_control", "Z_flux_control", "Read_out_control"]:
                act_module = self.exp_dict.get("measure_aio").get(module)
                for m in act_module.values():
                    combo_items.append(f"{module}-{m.get('channel')}")
            self._ui.combo_com.set_units(combo_items)
        else:
            self._ui.module_com.clear()
            self._ui.loop_com.clear()
            self._ui.channel_com.clear()
            self._ui.combo_com.clear()
            self._ui.module_com.addItem("Context")

    @Slot()
    def enlarge(self):
        self._ui.schedule_chart_view.chart().zoom(1.2)

    @Slot()
    def narrow(self):
        self._ui.schedule_chart_view.chart().zoom(0.8)

    @Slot()
    def reset(self):
        self._ui.schedule_chart_view.chart().zoomReset()

        # bug fix: when y_axis change, reset failed
        # min_v = self.__axis_y.min()
        # max_v = self.__axis_y.max()
        # if max_v != 1.0 or min_v != -1.0:
        self.__axis_y.setRange(-1, 1)
        self._schedule_plot()

    @Slot(str)
    def change_module(self, name: str):
        if not self._ui.fix_canvas.isChecked():
            self.reset()

        if name == "sweep_control":
            self.model_data = self.sweep_dict
            self.doc_tree_model.refresh(expand=False)
        elif name == "Z_dc_control":
            self.model_data = self.exp_dict.get("measure_aio").get(name)
            self.doc_tree_model.refresh(expand=False)
        elif name == "Context":
            self.model_data = self.context_data
            self.doc_tree_model.refresh(expand=False)
        elif name:
            measure_aio = self.exp_dict.get("measure_aio")
            module = measure_aio.get(name)
            self._ui.channel_com.clear()
            self._ui.channel_com.addItems(
                [str(m.get("channel")) for m in module.values()]
            )
            self._ui.channel_com.setCurrentIndex(0)

    @Slot(str)
    def change_channel(self, name: str):
        if not self._ui.fix_canvas.isChecked():
            self.reset()

        self._schedule_plot(parts=[(None, name)])

    @Slot(str)
    def change_loop(self, name: str):
        if not self._ui.fix_canvas.isChecked():
            self.reset()

        self._schedule_plot(loop=name, set_range=False)

    @Slot()
    def compare_pulse(self):
        if not self._ui.fix_canvas.isChecked():
            self.reset()

        items = self._ui.combo_com.currentText()
        if len(items) == 0:
            self.handler_ret_data(
                QDict(code=600, msg="Please choose combo item first!")
            )
        else:
            parts = []
            for item in items:
                parts.append(item.split("-"))
            self._schedule_plot(parts=parts)

    def _schedule_plot(
        self,
        parts: List = None,
        loop: Optional[Union[int, str]] = None,
        set_range: bool = True,
    ):
        if not self.exp_dict:
            return

        loop = loop or self._ui.loop_com.currentText()
        if loop is None or loop == "":
            return
        loop = int(loop)

        parts = parts or self._parts
        modules, channels = [], []
        if parts is None:
            modules.append(self._ui.module_com.currentText())
            channels.append(self._ui.channel_com.currentText())
        else:
            self._parts = parts
            for part in parts:
                m, c = part
                modules.append(m) if m else modules.append(
                    self._ui.module_com.currentText()
                )
                channels.append(c) if c else channels.append(
                    self._ui.channel_com.currentText()
                )

        waves = []
        sample_rates = []
        names = []
        for i, module in enumerate(modules):
            channel = channels[i]
            sample_rate = self.exp_dict.get("sample_rate").get(module.lower())

            if module and channel and (loop or loop == 0):
                module_data = self.exp_dict.get("measure_aio").get(module)
                if module_data is None:
                    return

                data = module_data.get(f"channel {channel}")
                if data is None:
                    return

                sweep = self.exp_dict.get("sweep_control").get(
                    f"{module}:waveform-{channel}"
                )
                self.model_data = data
                self.doc_tree_model.refresh(expand=False)

                if sweep:
                    waveform = sweep.get("waveform")
                    if "bigwavefile" in waveform:
                        wave = sweep.get("waveform").get("bigwavefile")[loop]
                    elif "wavefile" in waveform:
                        wave = sweep.get("waveform").get("wavefile")[loop]
                    else:
                        raise ValueError(
                            f"{module} - {channel} sweep documents is error!"
                        )
                else:
                    wave = data.get("waveform").get("custom_wave").get("wavefile")
                    if isinstance(wave, List):
                        wave = wave[0]

                    if self._ui.show_delay.isChecked() and module == "Read_out_control":
                        trigger_name = f"{module}:trigger_delay:0-{channel}"
                        tigger_sweep = self.exp_dict.get("sweep_control").get(
                            trigger_name
                        )
                        if tigger_sweep:
                            trigger_delay = tigger_sweep.get("points")[int(loop)]
                            wave = np.hstack(
                                (np.zeros(int(trigger_delay * sample_rate + 1)), wave)
                            )

                if wave is not None and len(wave) > 0:
                    waves.append(wave)
                    sample_rates.append(sample_rate)
                    names.append(f"{module}-{channel}-{sample_rate}")

        self.draw_raw_wave(waves, sample_rates, names, set_range)


def transform_byte_to_float(byte_):
    if b"\x18\xef\xdc\x01" in byte_:
        bo = byte_[33:-5]
        length = int(len(bo) / 2)
        data = np.array(struct.unpack(">" + "H" * length, bo))
        data_list = list(data / (2 * (2**14 - 1)) - 1)
        qaio_type = 30
    else:
        data_list = pickle.loads(byte_)[0]
        qaio_type = 72
    return [float(d) for d in data_list], qaio_type


def transform_sweep_bytes_to_array(byte_):
    if b"\x18\xef\xdc\x01" in byte_:
        pulses = byte_.split(b"\x18\xef\xdc\x01")[1:]
        ap = []
        for pulse in pulses:
            p = pulse[29:-5]
            length = int(len(p) / 2)
            data = np.array(struct.unpack(">" + "H" * length, p))
            ap.append([float(d) for d in list(((data / (2 * (2**14 - 1))) - 1))])
        return ap
    else:
        ap = pickle.loads(byte_)
        return ap


def update_dict(pre_exp):
    if isinstance(pre_exp, dict):
        for key, value in pre_exp.items():
            if key == "Z_dc_control":
                value = {f"channel {i + 1}": dc for i, dc in enumerate(value)}
            elif key == "XY_control":
                value = {f'channel {item.get("channel")}': item for item in value}
            elif key == "Z_flux_control":
                value = {f'channel {item.get("channel")}': item for item in value}
            elif key == "Read_out_control":
                value = {f'channel {item.get("channel")}': item for item in value}
            elif key == "sweep_control":
                value = {
                    f'{item.get("func")}-{item.get("channel")}': item for item in value
                }
            pre_exp[key] = update_dict(value)
    return pre_exp


def query_exp_dict(
    exp_id: str, ip: str, port: int, crosstalk: Dict, plot_trigger: bool = False
):
    global EXP_DICT

    def son_to_dict(data):
        if isinstance(data, SON):
            data = dict(data)

        if isinstance(data, dict):
            for key in data.keys():
                # bugfix 2023/03/31 (bigwavefile need read when fake pulse is False):
                if key == "bigwavefile":
                    value = son_to_dict(sweep_wave_dict[data[key]])
                else:
                    value = data[key]
                data[key] = son_to_dict(value)
        elif isinstance(data, list):
            data = [son_to_dict(v) for v in data]
        elif isinstance(data, datetime) or isinstance(data, ObjectId):
            data = str(data)
        elif isinstance(data, bytes):
            data = pickle.loads(data)

        return data

    def scan_dict_compile_pulse(data):

        if isinstance(data, dict):
            for key in data.keys():
                val = data[key]
                data[key] = scan_dict_compile_pulse(val)
        elif isinstance(data, list):
            data = [scan_dict_compile_pulse(d) for d in data]
        elif isinstance(data, PulseComponent):
            data = compile_pulse(data)

        return data

    def scan_dict_correct_pulse(data):

        if isinstance(data, dict):
            for key in data.keys():
                val = data[key]
                data[key] = scan_dict_correct_pulse(val)
        elif isinstance(data, list):
            data = [scan_dict_correct_pulse(d) for d in data]
        elif isinstance(data, PulseComponent):
            if data.type != "M":
                data.correct_pulse()

            if data.base:
                data += data.base

            data = data.pulse

        return data

    def correct_ac_crosstalk(data):
        _sweep_control = data.get("sweep_control")

        pulse_map = {}

        # scan sweep control, get pulse map
        for _control in _sweep_control:
            if _control.get("func") == "Z_flux_control:waveform":
                pulse_list = _control.get("waveform").get("wavefile")
                bit_name = pulse_list[0].bit
                pulse_map[bit_name] = pulse_list

        # if not sweep z flux wave, get form measure_aio
        if len(pulse_map) == 0:
            # cur exp not sweep waveform
            z_flux_control_list = data.get("measure_aio").get("Z_flux_control")
            for _control in z_flux_control_list:
                pulse_list = _control.get("waveform").get("custom_wave").get("wavefile")
                if pulse_list is None:
                    raise ValueError(
                        f"When there is no scan waveform, the waveform must exist here!"
                    )
                bit_name = pulse_list[0].bit
                pulse_map[bit_name] = pulse_list

        if len(pulse_map) < 2:
            return

        if crosstalk is None:
            return

        # get ac crosstalk matrix
        crosstalk_names = list(pulse_map.keys())
        infos = crosstalk.get("infos")
        ac_crosstalk = np.array(crosstalk.get("ac_crosstalk"))
        index_list = [infos.index(q) for q in crosstalk_names]
        ac_crosstalk = ac_crosstalk[index_list, :]
        ac_crosstalk = ac_crosstalk[:, index_list]
        ac_crosstalk_inv = np.linalg.inv(ac_crosstalk)

        # check pulse length
        z_pulses_length_list = []
        for _, z_pulses in pulse_map.items():
            z_pulses_length_list.append(len(z_pulses))
        z_pulses_length_list_set = set(z_pulses_length_list)
        if len(z_pulses_length_list_set) != 1:
            raise ValueError(
                f"qubit z pulse has different length: {z_pulses_length_list}"
            )
        z_pulses_length, *_ = z_pulses_length_list

        # ac crosstalk matrix correction
        for index in range(z_pulses_length):

            pulse_components = []
            for bit_name in crosstalk_names:
                pulse_components.append(pulse_map[bit_name][index])

            pulse_components = ac_crosstalk_correct(ac_crosstalk_inv, pulse_components)

            # update pulse after ac crosstalk
            for i, bit_name in enumerate(crosstalk_names):
                pulse_map[bit_name][index] = pulse_components[i]

    def add_trigger_delay(data):
        def _mian(module_name: str):
            main_module = data.get("measure_aio").get(module_name)
            sweep_crl = data.get("sweep_control")
            for item in main_module.values():
                trigger_delay = item.get("trigger_delay")[0]
                if trigger_delay > 0:
                    channel = item.get("channel")
                    sweep_name = f"{module_name}:waveform-{channel}"
                    sweep_trigger_name = f"{module_name}:trigger_delay:0-{channel}"
                    sr = data.get("sample_rate").get(module_name.lower())
                    pre_point = np.zeros(int(trigger_delay * sr + 1))
                    if sweep_trigger_name in sweep_crl:
                        continue
                    if sweep_name in sweep_crl:
                        waves = (
                            sweep_crl.get(sweep_name).get("waveform").get("wavefile")
                        )
                        for i in range(len(waves)):
                            waves[i] = np.hstack((pre_point, waves[i]))
                    else:
                        wave = item.get("waveform").get("custom_wave").get("wavefile")
                        if isinstance(wave, List):
                            wave = wave[0]
                        item["waveform"]["custom_wave"]["wavefile"] = np.hstack(
                            (pre_point, wave)
                        )

        _mian("XY_control")
        _mian("Z_flux_control")
        _mian("Read_out_control")

        return data

    sample_rate = {"xy_control": 1.6, "z_flux_control": 1.6, "read_out_control": 1.6}

    try:
        # connect server
        connect_server(ip, port)

        # query exp document
        exp_doc = ExperimentDoc.objects(id=exp_id).first()
        fake_pulse = exp_doc.fake_pulse
        measure_aio = exp_doc.measure_aio
        sweep_control = exp_doc.sweep_control

        # bugfix 2023/03/31 (bigwavefile need read when fake pulse is False):
        # load bigwavefile into cache dict
        sweep_wave_dict = {}
        for control in exp_doc.sweep_control:
            if "waveform" in control.func:
                wave_bytes = control.waveform.bigwavefile.read()
                if wave_bytes:
                    c = control.to_mongo()["waveform"]["bigwavefile"]
                    sweep_wave_dict[c] = wave_bytes

        # transform exp document to dict
        measure_aio_data = son_to_dict(measure_aio.to_mongo())
        sweep_control_data = [
            son_to_dict(control.to_mongo()) for control in sweep_control
        ]
        exp_data = son_to_dict(exp_doc.to_mongo())
        exp_data["measure_aio"] = measure_aio_data
        exp_data["sweep_control"] = sweep_control_data

        # check fake pulse to correction
        if fake_pulse is True:
            PulseComponent._fake = True
            # compile fake pulse to actual pulse
            exp_data = scan_dict_compile_pulse(exp_data)

            # correct ac crosstalk
            correct_ac_crosstalk(exp_data)

            # correct delay and distortion
            exp_data = scan_dict_correct_pulse(exp_data)

        # check sample rate
        baseband_freq = list(exp_data.get("IF").values())[0][0]
        if baseband_freq > 800:
            sample_rate.update(
                {"xy_control": 3.2, "z_flux_control": 1.2, "read_out_control": 3.2}
            )
        exp_data["sample_rate"] = sample_rate
        exp_data = update_dict(exp_data)

        if plot_trigger:
            exp_data = add_trigger_delay(exp_data)

        # schedule adapter
        EXP_DICT = exp_data
    except Exception as e:
        EXP_DICT["error"] = str(e)

    disconnect_server(alias="default")
