# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/8/28
# __author:       Lang Zhu

from PySide6.QtCore import Slot
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from .diff_thread_ui import Ui_MainWindow
from enum import Enum
from pyQCat.invoker import DataCenter
from .near_task_table import NearTaskTableModel
from PySide6.QtWidgets import QTableView, QAbstractItemView

class DIFFStatus(Enum):
    WAITING = "wait"
    PASS = "ok"
    VALI_FAIL = "fail"
    QUERY_ERR = "query_error"


class ThreadDiffView(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent=None):
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.gui = gui
        self.setWindowTitle("multi Thread View")
        self.db = DataCenter()
        self.task_table_data = []
        self.task_table_model = None
        self._init_view()
        self._ui.splitter.setStretchFactor(3,2)


    def _init_view(self):
        self.task_table_model = NearTaskTableModel(self.gui, self, self._ui.tableView)
        self._ui.tableView.setModel(self.task_table_model)

    @Slot()
    def query_task_list(self):
        volume = self._ui.spinBoxVolume.value()
        page = self._ui.spinBoxpage.value()

        res = self.db.query_exp_policy(
            # sample=self.gui.backend.config.system.sample,
            # env_name=self.gui.backend.config.system.env_name,
            page_num=page,
            page_size=volume,
        )
        if res and res["code"] == 200:
            data = res.get("data")
            self.task_table_data = data
            self.task_table_model.refresh_auto(False)

    @Slot()
    def query_task_diff(self):
        self._set_diff_label_state(DIFFStatus.WAITING)
        task1 = self._ui.task1_input.text()
        task2 = self._ui.task2_input.text()
        if task1 and task2:
            res = self.db.compare_exp_policy([task1, task2])
            if res and res["code"] == 200:
                result = res["data"]
                if result["validate_result"]:
                    self._set_diff_label_state(state=DIFFStatus.PASS)
                    self._ui.diff_text.setText("PASS")
                else:
                    self._set_diff_label_state(state=DIFFStatus.VALI_FAIL)
                    self._ui.diff_text.setText(result["validate_reason"])
            else:
                self._set_diff_label_state(state=DIFFStatus.QUERY_ERR)
                self._ui.diff_text.setText(f"code:{res['code']}\nmsg:{res['msg']}")


    def _set_diff_label_state(self, state: DIFFStatus):
        if state == DIFFStatus.VALI_FAIL:
            self._ui.diff_status.setText("X")
            self._ui.diff_status.setStyleSheet("color:red;")
        elif state == DIFFStatus.WAITING:
            self._ui.diff_status.setText("O")
            self._ui.diff_status.setStyleSheet("color:grey;")
        elif state == DIFFStatus.PASS:
            self._ui.diff_status.setText("√")
            self._ui.diff_status.setStyleSheet("color:green;")
        elif state == DIFFStatus.QUERY_ERR:
            self._ui.diff_status.setText("0")
            self._ui.diff_status.setStyleSheet("color: blue;")

