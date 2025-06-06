# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/10/28
# __author:       xw

from PySide6.QtWidgets import QDateTimeEdit
from ..task.create_task_dialog_ui import Ui_Dialog
from ..title_window import TitleDialog
from ..task.tree_model_dag import QTreeModelDag
from typing import TYPE_CHECKING

from pyqcat_visage.gui.widgets.options.tree_delegate_options import QOptionsDelegate

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class CreateTaskDialog(TitleDialog):
    def __init__(self, gui: "VisageGUI", parent=None, sub_type="dag"):
        super().__init__(parent)
        self._ui = Ui_Dialog()
        self._ui.setupUi(self)
        self.gui = gui
        self._sub_type = sub_type

        self.exp_model = QTreeModelDag(self, self.gui, self.ui.treeView, name="exp")
        self.init_exp_model()
        self.ui.treeView.setModel(self.exp_model)
        self.exp_delegate = QOptionsDelegate(self)
        self.ui.treeView.setItemDelegate(self.exp_delegate)
        self.exp_model.load()
        self.show()

    @property
    def ui(self):
        return self._ui

    def init_exp_model(self):
        if self._sub_type == "exp":
            self.exp_model.data_dict.get("is_calibration")[2] = False

    def get_input(self):
        dag_name = self.ui.lineEdit.text()
        dag_policy = self.ui.DagPolicyText.currentText()
        conf_dict = self.exp_model.data_dict
        return dag_name, dag_policy, conf_dict

    def select_policy(self):
        selected_policy = self.ui.DagPolicyText.currentText()

        if selected_policy == "schedule":
            self.exp_model.data_dict = {
                "interval": [10, "int", True],
                "unit": ["s", ["s", "min", "h"], True],
                "priority": [1, "int", True],
                "repeat": [5, "int", True],
                "is_calibration": [False, "bool", True],
            }

        elif selected_policy == "timing":
            # edit_execution_time = QDateTimeEdit()
            # self.exp_model.data_dict = {
            #     "hour": [edit_execution_time, "obj", True]
            # }
            #
            self.exp_model.data_dict = {
                "hour": [10, "int", True],
                "minute": [10, "int", True],
                "second": [10, "int", True],
                "priority": [1, "int", True],
                "interval": [10, "int", True],
                "unit": ["s", ["s", "min", "h"], True],
                "time_nodes": [1, "int", True],
                "is_calibration": [False, "bool", True]
            }

        elif selected_policy == "repeat":
            self.exp_model.data_dict = {
                "param": ["freq", ["freq", "z_amp"], True],
                "scan_list": {
                    "start": ["", "str", True],
                    "end": ["", "str", True],
                    "step": ["", "str", True],
                    "style": ["normal", ["qarange", "normal"], True],
                    "details": [None, "list", True],
                    "describe": ["normal | None", "str", True],
                },
                "priority": [1, "int", True]
            }
        self.exp_model.load()
        self.exp_model.refresh()

    def update_task(self, task_info):
        policy_type = task_info["policy"]["type"]
        sub_type = task_info["sub_type"]
        if policy_type == "schedule":
            self.exp_model.data_dict = task_info["policy"].get("options")
            if sub_type == "exp":
                self.exp_model.data_dict.get('is_calibration')[2] = False
        elif policy_type == "timing":
            self.exp_model.data_dict = task_info["policy"].get("options")

        self.exp_model.load()
        self.exp_model.refresh()
