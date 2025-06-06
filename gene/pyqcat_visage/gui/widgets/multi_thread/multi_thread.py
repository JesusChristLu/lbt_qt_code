# -*- coding: utf-8 -*-
import time
# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/8/28
# __author:       Lang Zhu
from datetime import datetime
import pathlib
from typing import TYPE_CHECKING
from loguru import logger
from PySide6.QtCore import Slot, QStringListModel
from PySide6 import QtWidgets
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.gui.widgets.multi_thread.multi_thread_ui import Ui_mainWindow
from pyqcat_visage.gui.widgets.multi_thread.structers import ShowThreadType, ProbeStruct
import json
from pyqcat_visage.gui.widgets.multi_thread.threads_table import MultiThreadModel, LowPriorityListModel, \
    SchedulerListModel, NormalListModel

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI
from .multi_topology_widget import MultiTopologyWidget


class MultThreadWindow(TitleWindow):

    def __init__(self, gui: "VisageGUI", parent=None):
        super().__init__(parent)
        self._ui = Ui_mainWindow()
        self._ui.setupUi(self)
        self.gui = gui
        self.setWindowTitle("multi Thread View")
        self.cache_list = []
        self._current_data: ProbeStruct = None
        # flag
        self.flag_use_cache = False

        self.tr_table_model = None
        self.scheduler_queue_model = None
        self.normal_list_model = None
        self.low_priority_model = None
        self.topology.color_conf = self.gui.graphics_theme
        self.set_table_change_button_text()
        self._last_current_data = None
        self._run_task_cache = {}

    @property
    def current_data(self):
        if self._current_data:
            return self._current_data
        else:
            return {}

    @property
    def table_data(self):
        data_list = []
        if not self.current_data:
            return data_list

        for data in self.current_data.core_thread.values():
            use_bits = data.get("use_bits")
            if isinstance(use_bits, list) and len(use_bits) >= 1:
                data["use_bits"] = ", ".join(use_bits)

            data_list.append(data)

        return data_list

    @property
    def scheduler_data(self):
        if not self.current_data:
            return []
        else:
            return self.current_data.task_list

    @property
    def normal_data(self):
        if not self.current_data:
            return []
        else:
            return self.current_data.normal_task_list

    @property
    def low_priority_data(self):
        if not self.current_data:
            return []
        else:
            return self.current_data.low_task_list

    @property
    def backend(self):
        return self.gui.backend

    @property
    def topology(self) -> MultiTopologyWidget:
        return self._ui.chip_widget

    @property
    def task_text(self):
        return self._ui.detail_text

    @property
    def table_widget(self):
        return self._ui.widget_2

    def set_table_change_button_text(self):
        button_text = "show table" if self.table_widget.isHidden() else "hidden table"
        self._ui.control_table.setText(button_text)

    @Slot()
    def change_tabel_hidden_status(self):
        self.table_widget.setHidden(not self.table_widget.isHidden())
        self.set_table_change_button_text()

    @Slot(str)
    def refresh_msg_slot(self, tr_msg: str):
        tr_dict = json.loads(tr_msg)
        self._current_data = ProbeStruct(**tr_dict)
        run_task_cache = {}
        for task in self._current_data.core_thread.values():
            if task["task_id"] in self._run_task_cache:
                run_task_cache.update({task["task_id"]: self._run_task_cache[task["task_id"]]})
                task["run_time"] = time.time() - self._run_task_cache[task["task_id"]]
            else:
                run_task_cache.update({task["task_id"]: time.time()})
                task["run_time"] = 0

        self._run_task_cache = run_task_cache

        self._last_current_data = self.current_data
        if self.flag_use_cache:
            self.cache_list.append(tr_msg)

        if not self.isHidden():
            self.refresh_widget(self.current_data)

    def change_show_table(self):
        """
        change view show model.
        """
        self.refresh_widget(self.current_data)

    def screen_shot(self, *args, name="shot", type_='png'):
        name = "thread_view_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = pathlib.Path(name + "." + type_).absolute()
        # grab the main window
        screenshot = self.grab()  # type: QtGui.QPixMap
        screenshot.save(str(path), type_)

        # copy to clipboard
        QtWidgets.QApplication.clipboard().setPixmap(screenshot)
        logger.info(f"Screenshot copied to clipboard and saved to:\n {path}")

    def refresh_widget(self, show_data):
        if not show_data:
            return
        self.topology.refresh(show_data, self._ui.show_type.currentText())
        self.tr_table_model.refresh_auto(False)
        self.scheduler_queue_model.refresh_auto(False)
        self.normal_list_model.refresh_auto(False)
        self.low_priority_model.refresh_auto(False)
        thread_num = int(show_data.core_status.get("thread_num", 0)) - int(show_data.core_status.get("empty_thread", 0))
        scheduler_num = int(show_data.scheduler.get('queue_len', 0)) - thread_num
        self._ui.thread_count_lcd.display(thread_num)
        self._ui.lcd_wait.display(int(show_data.wait_task_len))
        self._ui.lcd_scheduler.display(scheduler_num)
        self._ui.lcd_len_normal.display(int(show_data.normal_task_len))
        self._ui.lcd_len_low_priority.display(int(show_data.low_task_len))

    def _set_table(self):
        self.tr_table_model = MultiThreadModel(gui=self.gui, parent=self, table_view=self._ui.tr_table_view)
        # self.tr_table_model.refresh_auto(False)
        self._ui.tr_table_view.setModel(self.tr_table_model)

        self.scheduler_queue_model = SchedulerListModel(gui=self.gui, parent=self, table_view=self._ui.task_list_view)
        self._ui.task_list_view.setModel(self.scheduler_queue_model)
        self.normal_list_model = NormalListModel(gui=self.gui, parent=self, table_view=self._ui.view_list_normal)
        self._ui.view_list_normal.setModel(self.normal_list_model)
        self.low_priority_model = LowPriorityListModel(gui=self.gui, parent=self, table_view=self._ui.view_list_low)
        self._ui.view_list_low.setModel(self.low_priority_model)
        self._ui.tr_table_view.task_detail.connect(self.show_task_detail)
        self._ui.show_type.clear()
        self._ui.show_type.addItems([ShowThreadType.ALL, ShowThreadType.HIGHER])
        self._ui.show_type.setCurrentText(ShowThreadType.ALL)
        self._ui.splitter.setStretchFactor(0, 4)
        self._ui.splitter.setStretchFactor(1, 1)
        self._ui.splitter_table.setStretchFactor(0,2)
        self._ui.splitter_table.setStretchFactor(1,3)

    def load_chip_view(self):
        if self.backend.model_channels and self.backend.model_channels.shape:
            row, col = self.backend.model_channels.shape
            qubit_names = list(self.backend.model_channels.qubit_params.keys())
            self.topology.load(row, col, qubit_names)
            self._set_table()
            self._ui.sample_name.setText(f" sample || {self.backend.config.system.sample}")

        if self._last_current_data:
            self.refresh_widget(self._last_current_data)

    @Slot(dict)
    def show_task_detail(self, task_info):
        self.task_text.clear()
        self.task_text.append(json.dumps(task_info, indent=4, ensure_ascii=False))
        self._ui.tabWidget.setCurrentWidget(self._ui.tab_2)
