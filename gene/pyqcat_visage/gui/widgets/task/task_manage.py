# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/10/31
# __author:       xw

import json
from typing import TYPE_CHECKING
from loguru import logger
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from .task_manage_ui import Ui_MainWindow

from .table_model_task import QTableModelTaskManage
from pyqcat_visage.gui.widgets.task.tree_model_task_info import QTreeModelTaskInfo

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class DagManagerWindow(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent=None):
        super().__init__(parent)
        self._task_info = None
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.gui = gui
        self.tasks = []
        self._setup_table()

    @property
    def ui(self):
        return self._ui

    @property
    def backend(self):
        return self.gui.backend

    @property
    def task_info(self):
        return self._task_info

    def _setup_table(self):
        self.table_model_task = QTableModelTaskManage(
            self.gui, self, self._ui.tableTaskView
        )
        self.ui.tableTaskView.setModel(self.table_model_task)

        self.task_info_tree_model = QTreeModelTaskInfo(
            self, self.gui, self.ui.task_info_tree_view
        )

        self.ui.task_info_tree_view.setModel(self.task_info_tree_model)

        self.ui.tableTaskView.choose_task_signal.connect(self.get_task_info)

    def load_all_data(self, flag=True):
        self.ui.tableTaskView.right_click_menu.delete.setVisible(flag)
        self.ui.tableTaskView.right_click_menu.start.setVisible(flag)
        self.ui.tableTaskView.right_click_menu.stop.setVisible(flag)
        self.ui.tableTaskView.right_click_menu.update.setVisible(flag)

    def create_task(self, task_name, policy_type, policy_opt, sub_name, sub_type="dag"):
        task_obj = Task(
            self.backend,
            self.gui,
            task_name,
            policy_type,
            policy_opt,
            sub_name,
            sub_type,
        )
        ret_data = task_obj.create()
        self.handler_ret_data(ret_data)
        res_data = self.gui.backend.db.query_custom_task(task_name=task_name)
        if ret_data.get("code") != 200:
            logger.info(f"Create task failed!, message:{ret_data.get('msg')}")
        else:
            task_info = res_data.get("data")[0]
            task_id = task_info.get("id")
            task_obj.task_data.update({"id": task_id, "enable": True})
            self.backend.run_cron(task_obj.task_data)

    def query_task(self):
        ret_data = self.gui.backend.db.query_custom_task()
        if ret_data.get("code") in (200, 404):
            self.tasks = ret_data["data"]
            self.table_model_task.refresh_auto(False)
            self.load_all_data(flag=True)

    def init_task(self):
        ret_data = self.gui.backend.db.query_custom_task()
        if ret_data.get("code") == 200:
            self.tasks = ret_data["data"]
            self.backend.init_cron(self.tasks)

    def query_history(self):
        task_name = self.ui.task_name.text() or None
        task_id = self.ui.task_id.text() or None
        if task_name or task_id:
            ret_data = self.gui.backend.db.query_custom_task_his(
                doc_id=task_id, task_name=task_name
            )
            self.tasks = ret_data["data"]
            self.force_refresh()
            self.load_all_data(flag=False)

    def query(self):
        task_name = self.ui.task_name.text() or None
        task_id = self.ui.task_id.text() or None
        if task_name or task_id:
            ret_data = self.gui.backend.db.query_custom_task(
                task_name=task_name, task_id=task_id
            )
            self.tasks = ret_data["data"]
            self.force_refresh()

    def update_task(
        self,
        task_name,
        task_id,
        enable,
        policy_type,
        policy_opt,
        sub_name,
        sub_type="dag",
    ):
        task_obj = Task(
            self.backend,
            self.gui,
            task_name,
            policy_type,
            policy_opt,
            sub_name,
            sub_type,
        )
        self.task_info.update(task_obj.task_data)
        ret_data = task_obj.update()
        self.handler_ret_data(ret_data)
        if ret_data.get("code") != 200:
            logger.info(f"Update task failed!, message:{ret_data.get('msg')}")
        else:
            self.backend.remove_one_cron(cron_id=task_id)
            task_obj.task_data.update({"id": task_id, "enable": enable})
            self.backend.run_cron(task_obj.task_data)

    def delete_task(self, task: dict, index: int):
        if self.ask_ok(
            "Are you sure to delete the task? This operation will not be recoverable.",
            "Visage Message",
        ):
            ret_data = self.gui.backend.db.delete_custom_task(task.get("task_name"))
            self.backend.remove_one_cron(task.get("id"))
            self.handler_ret_data(ret_data)
            if ret_data.get("code") == 200:
                self.table_model_task.removeRows(index)
            self.force_refresh()

    def refresh(self, flag=True):
        self.load_all_data(flag)
        self.force_refresh()

    def force_refresh(self):
        self.task_info_tree_model.load()
        self.table_model_task.refresh_auto(True)

    def set_task_info(self, task_info=None):
        self._task_info = task_info

        if task_info is None:
            self.force_refresh()
            return
        self.force_refresh()
        self.ui.task_info_tree_view.autoresize_columns()

        # task_name = task_info.get("task_name")
        # task_id = task_info.get("task_id")
        # ret_data = self.gui.backend.db.query_custom_task_his(doc_id=task_id, task_name=task_name)
        # self.display_his(ret_data["data"])

    def display_his(self, task_info):
        self.ui.textEdit.setText(json.dumps(task_info, indent=4))

    def get_task_info(self, task_info):
        self.set_task_info(task_info)
        self.ui.task_info_tree_view.hide_placeholder_text()

    def update_task_status(self, status_info):
        tasks = self.tasks
        for task in tasks:
            task_id = task.get("id")
            if status_info:
                status = status_info.get(task_id)
                if status:
                    task.update({"status": status})
            else:
                task.update({"status": ""})


class Task:
    def __init__(
        self,
        backend,
        gui,
        task_name,
        policy_type,
        policy_opt,
        sub_name,
        sub_type,
    ):
        self.backend = backend
        self.gui = gui
        self.task_data = {}
        self.task_name = task_name
        self.policy_type = policy_type
        self.policy_opt = policy_opt
        self.sub_name = sub_name
        self.sub_type = sub_type
        self.dag_dict = {}
        self.exp_dict = {}

    def update(self):
        self._prepare_task_data()
        ret_data = self.gui.backend.db.update_custom_task(self.task_data)
        return ret_data

    def create(self):
        self._prepare_task_data()
        ret_data = self.gui.backend.db.add_custom_task(self.task_data)
        return ret_data

    def _prepare_task_data(self):
        global_options = self.backend.context_builder.global_options

        if self.sub_type == "dag":
            dag = self.backend.dags[self.sub_name]
            self.dag_dict = dag.to_dict(self.backend.parallel_mode)
            if self.policy_type == "schedule":
                self.task_desc = "Calibration task"
            elif self.policy_type == "repeat":
                self.task_desc = "Cycle DAG tasks"
            else:
                self.task_desc = ""
        else:
            for key, module_exps in self.backend.experiments.items():
                if self.sub_name in module_exps:
                    self.exp_dict = module_exps.get(self.sub_name).to_run_exp_dict(
                        self.backend.parallel_mode
                    )
                    self.task_desc = "Timed experimental tasks"
                    self.policy_opt.update({"is_calibration": [False, "bool", True]})
                    break

        self.task_data = {
            "task_name": self.task_name,
            "task_desc": self.task_desc,
            "policy": {"type": self.policy_type, "options": self.policy_opt},
            "sub_type": self.sub_type,
            "sub_name": self.sub_name,
            "dag": self.dag_dict,
            "exp": self.exp_dict,
            "global_options": global_options,
        }
