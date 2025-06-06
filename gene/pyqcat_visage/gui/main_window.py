# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/31
# __author:       HanQing Shi
"""GUI front-end interface for pyqcat-visage in PySide6."""
# pylint: disable=invalid-name

import copy
import json
import pickle
import time
from multiprocessing import Process
from typing import TYPE_CHECKING, Any
from typing import Union, Dict

from PySide6 import QtCore
from PySide6.QtCharts import QChart
from PySide6.QtCore import QTimer, Slot, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import QMessageBox, QLabel, QFileDialog, QCheckBox
from loguru import logger

from pyQCat.errors import BUSAllocatedError, LOAllocatedError, ExperimentParallelError, UnCollectionError
from pyQCat.structures import QDict
from pyQCat.tools.s3storage import check_s3_sample_name
from pyqcat_visage.backend import HeartThead, TaskQthread
from pyqcat_visage.backend.base_backend import BaseBackend
from pyqcat_visage.backend.reload_exp import (
    load_experiment,
    load_py_to_str,
    clear_developer,
)
from pyqcat_visage.backend.utilities import exp_class_to_model
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.exceptions import OperationError, LogicError
from pyqcat_visage.execute.start import async_process
from pyqcat_visage.gui.diagnose import DiagnoseWindow
from pyqcat_visage.gui.main_window_base import (
    QMainWindowExtensionBase,
    QMainWindowHandlerBase,
    start_qApp
)
from pyqcat_visage.gui.main_window_layout import TitleGUI
from pyqcat_visage.gui.tools.qt_handlers import qt_message_handler
from pyqcat_visage.gui.tools.utilies import slot_catch_exception
from pyqcat_visage.gui.types import QsStatus
from pyqcat_visage.gui.widgets.chimera_manage import ChimeraManagerWindow
from pyqcat_visage.gui.widgets.component_window import ComponentEditWindow
from pyqcat_visage.gui.widgets.context_window import ContextEditWindow
from pyqcat_visage.gui.widgets.copy_data_widget import CopyDataWindow
from pyqcat_visage.gui.widgets.dialog.save_library_dialog import QSaveLibraryDialog
from pyqcat_visage.gui.widgets.document_check_widget import DocumentCheckWindow
from pyqcat_visage.gui.widgets.heatmap_widget import HeatMapWindow
from pyqcat_visage.gui.widgets.library.library_tree_model import LibraryTreeModel
from pyqcat_visage.gui.widgets.multi_thread import MultThreadWindow, ThreadDiffView
from pyqcat_visage.gui.widgets.options_edit_widget import OptionsEditWidget
from pyqcat_visage.gui.widgets.report_edit import ReportWindow
from pyqcat_visage.gui.widgets.result_widget import FileSysMainWindow
from pyqcat_visage.gui.widgets.run_setting_widget import RunSettingWidget
from pyqcat_visage.gui.widgets.storm_manage import StormManagerWindow
from pyqcat_visage.gui.widgets.system_config import SystemConfigWindow
from pyqcat_visage.gui.widgets.task.task_manage import DagManagerWindow
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.gui.widgets.topolopy import TopologyView
from pyqcat_visage.gui.widgets.topolopy.context_sidebar import ContextSideBar
from pyqcat_visage.gui.widgets.user_login import UserLoginWindow
from pyqcat_visage.gui.widgets.user_operation import UserManagerWindow
from pyqcat_visage.gui.widgets.work_space import WorkSpaceWindow
from pyqcat_visage.gui.widgets.workspace_history import WorkSpaceHisWindow
from pyqcat_visage.gui.widgets.workspace_manage import WorkSpaceManageWindow
from pyqcat_visage.protocol import ExecuteOp
from pyqcat_visage.tool.utilies import FATHER_OPTIONS

if TYPE_CHECKING:
    from pyqcat_visage.backend.experiment import VisageExperiment


class VisageExtension(QMainWindowExtensionBase):

    def __init__(self):
        super().__init__()
        # children gui.
        self._simulator_gui = None
        self._report_gui = None
        self._run_setting_widget = None
        self._document_widget = None

        self.max_retry = 0
        self.heart_thread: HeartThead = None
        self.async_thread: TaskQthread = None
        self._chimera_start = QPixmap(u":/run.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
        self._chimera_stop = QPixmap(u":/close.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
        self._chimera_disconnect = QPixmap(u":/warning.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
        self.state_flag = False

    def reset_window_layout(self):
        # show or hidden chip manage
        if not self.is_super and not self.is_admin:
            # self.ui.actionChipManage.setVisible(False)
            self.ui.actionWorkSpaceManage.setVisible(False)
        else:
            # self.ui.actionChipManage.setVisible(True)
            self.ui.actionWorkSpaceManage.setVisible(True)

    # --------------------------- timer manager ---------------------------

    def setup_complete(self):
        logger.log("FLOW", "System startup completed!")
        self.gui.main_window.setEnabled(True)
        self.set_toolbar_stop_state(False)

    def setup_close(self):
        logger.log("FLOW", "System startup closed!")
        self.gui.main_window.setEnabled(False)

    def _tackle_error(self, error_obj):
        is_link = False

        if isinstance(error_obj, BUSAllocatedError):
            self.gui.error_diag.handle_error.connect(self.load_heatmap)
            is_link = True

        if isinstance(error_obj, LOAllocatedError):
            self.gui.error_diag.handle_error.connect(self.load_heatmap)
            is_link = True

        if isinstance(error_obj, UnCollectionError):
            self.gui.error_diag.handle_error.connect(self.open_question_pool)

            is_link = True

        return is_link

    @staticmethod
    def open_question_pool():
        question_pool = GUI_CONFIG.help_url.question_pool
        QDesktopServices.openUrl(QUrl(question_pool))

    @staticmethod
    def is_process_running(process):
        return process is not None and process.is_alive()

    # --------------------------- Backend attributes ---------------------------

    @property
    def gui(self) -> "VisageGUI":
        """Get the GUI."""
        return self._handler

    @property
    def backend(self):
        """Return the visage core executor."""
        return self._handler.backend

    # --------------------------- Children GUI ---------------------------
    @Slot()
    def open_context_ui(self):
        """Show context edit window"""
        self.gui.context_widget.show()

    @Slot()
    def open_system_ui(self):
        """Handles click on System config action."""
        self.gui.system_config_widget.show()

    @Slot()
    def open_component_edit_ui(self):
        self.gui.component_editor.show()

    @Slot()
    def open_files_widget(self):
        self.gui.files_widget.show()

    @Slot()
    def open_user_ui(self):
        pass

    @Slot()
    def open_schedule_tool(self):
        self.gui.document_widget.show()

    @Slot()
    def open_run_setting(self):
        self.gui.run_setting_widget.show()

    @Slot()
    def open_report_ui(self):
        if hasattr(self.gui.backend.config, "report"):
            self._report_gui = ReportWindow(self.gui, parent=self)
            self._report_gui.show()
        else:
            logger.warning("The report function is not available yet.")

    @Slot(str)
    def load_heatmap(self, table: Union[str, bool] = None):
        """Handles click on load head map action."""
        self.gui.heatmap_widget.show()
        if table:
            title, info = table.split("_")
            if title == "AMP":
                self.gui.heatmap_widget.ui.tabWidget.setCurrentIndex(2)
                self.gui.heatmap_widget.tool_query()
                self.gui.heatmap_widget.ui.bus_combo.setCurrentText(info)

            if title == "LO":
                self.gui.heatmap_widget.ui.tabWidget.setCurrentIndex(1)
                self.gui.heatmap_widget.tool_query()
                self.gui.heatmap_widget.ui.group_combo.setCurrentText(info)

    @Slot()
    def load_copy(self):
        """Handles click on load copy action."""
        self.gui.copy_widget.show()

    @Slot()
    def user_manager(self):
        self.gui.user_manage_widget.show()

    @Slot()
    def open_community(self):
        url = QUrl(GUI_CONFIG.help_url.community)
        QDesktopServices.openUrl(url)

    @Slot()
    def open_user_manual(self):
        url = QUrl(GUI_CONFIG.help_url.user_guide)
        QDesktopServices.openUrl(url)

    # --------------------------- Backend interactions ---------------------------
    @slot_catch_exception(process_reject=True)
    def run_experiment(self):
        """Run Experiment"""
        exp = self.gui.options_window.experiment

        if exp is None:
            raise OperationError(
                "Please select the experiment which you want to perform in experiment library!"
            )

        self.backend.run_experiment(exp.name)
        self._start_run()
        # self.gui.ui.tabPlot.start()

    @slot_catch_exception(process_reject=True)
    def run_dag(self):
        """Run DAG."""
        dag = self.gui.ui.mainDAGTab.node_editor.get_run_dag()

        if dag is None:
            raise OperationError(
                "Please select the dag which you want to perform in dag library!"
            )

        self.backend.run_dag(dag)
        self._start_run()

    @slot_catch_exception()
    def restart_executor_process(self):
        self.gui.backend.execute_send(ExecuteOp.stop_force)
        self.set_toolbar_state(False)
        self.set_toolbar_stop_state(False)
        self.heart_thread.execute_restart_flag = True

    @slot_catch_exception()
    def stop_execute(self):
        if self.backend.execute_is_alive:
            if self.ask_ok(
                    "Are you sure you want to end the task you are performing?",
                    "Visage Message",
            ):
                self.gui.backend.execute_send(ExecuteOp.stop_experiment, exp_id=self.gui.ui.tabPlot.data.id,
                                              use_simulator=self.backend.config.run.use_simulator)
                # self._close_run()
                self.set_toolbar_stop_state(False)
                logger.info("Exp/Dag Task stop success!")
        else:
            raise OperationError("No task is executing!")

    @slot_catch_exception()
    def save_exp(self):
        """Save exp data to database."""
        dialog = QSaveLibraryDialog()
        dialog.set_collections(self.backend.experiment_names)
        ret = dialog.exec()

        if int(ret) == 1:
            save_type, describe, items = dialog.get_input()
            self.backend.save_all_exp(save_type, describe, items)

    @slot_catch_exception()
    def save_dag(self):
        """Save data to database."""
        dag = self.ui.mainDAGTab.node_editor.dag
        if not dag:
            raise OperationError("Please choose a dag first!")
        else:
            name, ok = self.ask_items("Save dag options", ["DB", "LOCAL"])
            if ok:
                self.backend.save_dag(dag, name)

    @slot_catch_exception(process_warning=True)
    def open_experiment_file(self):
        """Loading custom experiment which developed under monster architecture."""
        cur_path = self.backend.config.system.config_path
        filenames = QFileDialog.getOpenFileNames(
            self, "Import Custom Experiment", cur_path, "(*.py)"
        )[0]

        cus_exp_list = []
        for filename in filenames:
            cus_exp_dict = load_experiment(filename)
            if cus_exp_dict.get("code") == 200:
                for exp_cls in cus_exp_dict.get("exp_list"):
                    exp_name, exp_options, ana_options = exp_class_to_model(exp_cls)
                    if exp_name in self.backend.experiments.get(
                            "CustomerExperiment", []
                    ):
                        logger.error(f"Import {exp_name} error, because it's existed!")
                        continue
                    cus_exp_list.append(
                        {
                            "exp_name": exp_name,
                            "exp_params": {
                                "experiment_options": exp_options,
                                "analysis_options": ana_options,
                                "exp_describes": load_py_to_str(filename),
                                "exp_path": cus_exp_dict.get("exp_path"),
                            },
                        }
                    )
            else:
                logger.error(
                    f'Load {filename} error!, because {cus_exp_dict.get("msg")}'
                )

        self.backend.refresh_customer_exp(cus_exp_list)
        self.gui.exp_lib_model.refresh()

    @slot_catch_exception(process_reject=True)
    def login_out(self):

        if self.backend.login_user is None:
            raise LogicError("No found login user!")

        reply = QMessageBox.question(
            self,
            "Login out",
            f"Exit Login {self.backend.login_user.get('username')}?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        elif reply == QMessageBox.StandardButton.Yes:
            self.backend.user_state = False
            self._logout_out()
            self.gui.system_config_widget.clear()
            return True
        return True

    @slot_catch_exception(process_warning=True)
    def delete_custom_exp(self):
        exp = self.gui.options_window.experiment

        if exp is None:
            raise OperationError("please choose experiment!")

        if self.backend.run_state:
            raise OperationError("Executing, please try again later!")

        self.backend.delete_custom_exp(exp)
        self.gui.exp_lib_model.load()

    @slot_catch_exception()
    def is_full_options(self):
        """options style for analysis and experiment to display."""
        self.gui.force_refresh_options_widgets(is_full=True)

    @slot_catch_exception(process_reject=True)
    def close_window(self, parent):
        # self.backend.parallel_proc.kill()
        self.backend.cache_context_manager()
        clear_developer()
        # self.thread_alive_timer.stop()
        self.backend.save_config()
        self.backend.close_save()
        self.save_window_settings()
        self.stop_communication_thread()
        self.close()
        parent.close()
        return True

    @slot_catch_exception()
    def import_exp(self):
        cur_path = self.backend.config.system.config_path

        filenames = QFileDialog.getOpenFileNames(self, "Import File", cur_path)
        if filenames[0]:
            self.backend.import_experiments(filenames[0])
        self.gui.ui.experimentLibrary.choose_exp_signal.emit(self.gui.options_window.edit_exp)
        self.gui.exp_lib_model.refresh()

    @slot_catch_exception()
    def import_dag(self):
        cur_path = self.backend.config.system.config_path

        filenames = QFileDialog.getOpenFileNames(self, "Import File", cur_path)
        if filenames[0]:
            self.backend.import_dags(filenames[0])
        self.gui.dag_lib_model.refresh()

    @slot_catch_exception()
    def user_manager(self):
        self.gui.user_manage_widget.show()

    @slot_catch_exception()
    def vali_parallel_options(self):
        # todo optimize: temp test
        ret_data = None
        exp = self.gui.options_window.experiment
        if exp is None or exp.name != "BaseParallelExperiment":
            ret_data = QDict(code=800, msg="Please Choose Parallel Experiment!")

        experiment = self.gui.options_window.select_experiment
        if not experiment:
            ret_data = QDict(code=800, msg="No Find experiment!")

        context_type = experiment.context_options.name[0]
        if self.backend.context_builder.context_data[context_type]["default"]:
            parallel_physical_unit = self.backend.context_builder.context_data[
                context_type
            ]["physical_unit"]
        else:
            parallel_physical_unit = experiment.context_options["physical_unit"][0]
        if not parallel_physical_unit:
            parallel_physical_unit = []
        else:
            parallel_physical_unit = parallel_physical_unit.split(",")

        if len(parallel_physical_unit) <= 1:
            ret_data = QDict(code=800, msg="Please Check Experiment Context!")
        else:
            exp_options = experiment.model_exp_options
            ana_options = experiment.model_ana_options

            p_exp_options = QDict(
                # is_divide_bf=exp.model_exp_options.is_divide_bf,
                # is_divide_bus=exp.model_exp_options.is_divide_bus,
                # use_union_rd=exp.model_exp_options.use_union_rd,
                exp_name=exp.model_exp_options.exp_name,
                same_options=exp.model_exp_options.same_options,
            )
            p_ana_options = QDict()

            # check exp/ana options
            def generate_parallel_options(options):
                new_options = {}
                for key, value in options.items():
                    if key in ["child_exp_options", "child_ana_options"]:
                        new_options[key] = generate_parallel_options(value)
                    elif key in FATHER_OPTIONS:
                        new_options[key] = value
                    else:
                        new_options[key] = {}
                        for physical_bit in parallel_physical_unit:
                            new_options[key][physical_bit] = copy.deepcopy(value)
                return new_options

            p_exp_options.update(generate_parallel_options(exp_options))
            p_ana_options.update(generate_parallel_options(ana_options))

            exp._model_exp_options = p_exp_options
            exp._model_ana_options = p_ana_options
            exp.physical_unit = parallel_physical_unit

            # refresh option model
            self.gui.options_window.force_refresh()

        self.handler_ret_data(ret_data)

    @slot_catch_exception()
    def show_qaio_log(self):
        self.gui.ui.dockLog.raise_()
        self.gui.refresh_qaio_log()
        if self.gui.ui.log_qstream.isVisible():
            self.gui.ui.log_qstream.hide()
        else:
            self.gui.ui.log_qstream.show()

    @slot_catch_exception()
    def load_chip_manage(self):
        self.gui.chip_manage.load_all_data()
        self.gui.chip_manage.show()

    @slot_catch_exception()
    def load_dag_manage(self):
        self.gui.dag_manage.load_all_data()
        self.gui.dag_manage.show()

    @slot_catch_exception()
    def load_storm_manage(self):
        self.gui.storm_manage.load_all_data()
        self.gui.storm_manage.show()

    @slot_catch_exception()
    def load_workspace_manage(self):
        self.gui.workspace_manage.load_all_data()
        self.gui.workspace_manage.show()

    @slot_catch_exception()
    def load_thread_view(self):
        self.gui.thread_view_widget.load_chip_view()
        self.gui.thread_view_widget.show()

    @slot_catch_exception()
    def load_workspace(self):
        self.gui.work_space.refresh_query_data()
        self.gui.work_space.show()

    @slot_catch_exception()
    def load_workspace_note(self):
        self.gui.workspace_note.load_all_data()
        self.gui.workspace_note.show()

    @Slot()
    def open_community(self):
        url = QUrl(GUI_CONFIG.help_url.community)
        QDesktopServices.openUrl(url)

    @Slot()
    def open_user_manual(self):
        url = QUrl(GUI_CONFIG.help_url.user_guide)
        QDesktopServices.openUrl(url)

    @Slot()
    def open_exp_doc(self):
        exp = self.gui.options_window.experiment
        exp_lib = GUI_CONFIG.help_url.exp_lib
        url = GUI_CONFIG.help_url.exp_indexes

        if exp is not None and exp.name in exp_lib:
            url = exp_lib.get(exp.name)

        QDesktopServices.openUrl(QUrl(url))

    # --------------------------- Solt ---------------------------
    @Slot()
    def _add_pull_remind(self):
        self.gui.ui.auto_new_pig.show()

    @Slot()
    def _update_cache(self, data: Any, qubit: bool = True):
        if isinstance(data, bytes):
            data = pickle.loads(data)
        if not isinstance(data, list):
            data = [data]
        if not data:
            return
        query_dict = {
            "name": data,
            "sample": self.backend.config.system.sample,
            "env_name": self.backend.config.system.env_name,
        }
        if qubit:
            query_dict["point_label"] = self.backend.config.system.point_label
        ret_data = self.backend.db.query_chip_all(**query_dict)
        if ret_data.get("code") == 200:
            self.backend.context_builder.refresh_chip_data(ret_data.get("data", {}))

    @Slot()
    def alert_msg(self, level, args):
        if isinstance(level, bytes):
            level = level.decode()
        if args:
            data = args[0]
            if isinstance(data, bytes):
                data = data.decode()
        else:
            data = level
            level = "Warning"
        if level.lower() == "warning":
            QMessageBox().warning(self, level, data)
        elif level.lower() == "error":
            QMessageBox().critical(self, level, data)
        elif level.lower() == "error_task":
            task_error = json.loads(data)
            msg = f"{task_error['exp_name']}| {task_error['doc_id']} error!\n {task_error['err_detail']}"
            QMessageBox().critical(self, level, msg)
        else:
            QMessageBox().information(self, level, data)

    def set_toolbar_state(self, state: bool):
        self.ui.actionRunExperiment.setEnabled(state)
        self.ui.actionRunDAG.setEnabled(state)

    def set_toolbar_stop_state(self, state: bool):
        self.ui.actionStop.setEnabled(state)

    @Slot()
    def refresh_qs_state(self, data):
        if isinstance(data, bytes):
            data = data.decode()
        if not data:
            data = QsStatus.stopped
        if data == QsStatus.disconnect:
            self.gui.ui.qs_label_pig.setPixmap(self._chimera_disconnect)
            if not self.backend.config.run.use_simulator:
                self.set_toolbar_state(False)
            self.state_flag = False
        elif data in (QsStatus.stopped, QsStatus.stop):
            self.gui.ui.qs_label_pig.setPixmap(self._chimera_stop)
            if not self.backend.config.run.use_simulator:
                self.set_toolbar_state(False)
            self.state_flag = False
        else:
            self.gui.ui.qs_label_pig.setPixmap(self._chimera_start)
            if not self.backend.config.run.use_simulator:
                self.set_toolbar_state(True)
            self.state_flag = True

    @Slot()
    def _auto_logout(self):
        if self.backend.user_state:
            self.backend.user_state = False
            QMessageBox().warning(
                self,
                f"Warning",
                "Your account has been logged in others pc.\n"
                "Your account may have security risks.\n"
                "Recommended login again after changing your password.",
            )
            self._logout_out(True)
            self.gui.login_widget._ui.stackedWidget.setCurrentIndex(1)
            self.gui.system_config_widget.clear()
            self.gui.login_widget.show()
            self.backend.login_user = None

    @Slot(bytes)
    def show_execute_error(self, error_msg):
        error_obj = pickle.loads(error_msg)
        self.gui.error_diag.error = error_obj

        if isinstance(error_obj, ExperimentParallelError):
            child_error_obj = error_obj.unit_error[0]
            is_link = self._tackle_error(child_error_obj)
        else:
            is_link = self._tackle_error(error_obj)

        self.gui.error_diag.check_link(is_link)
        self.gui.error_diag.show()

    @Slot(bytes)
    def update_dag_status(self, dag_run_map: bytes):
        dag_run_map = pickle.loads(dag_run_map)
        self.gui.ui.mainDAGTab.node_color_change(dag_run_map)

    @Slot(bytes)
    def update_task_status(self, cron_map: bytes):
        data = json.loads(cron_map.decode('utf-8'))
        self.gui.dag_manage.update_task_status(data)
        self.gui.dag_manage.table_model_task.refresh_auto(True)

    @Slot()
    def show_diff_thread_view(self):
        self.gui.thread_diff_view.query_task_list()
        self.gui.thread_diff_view.show()

    @Slot(str)
    def execute_restart(self, restart_msg):
        self.backend.run_state = False
        self.backend.execute_is_alive = False
        if self.backend.sub_proc:
            self.backend.sub_proc.terminate()
        self.backend.sub_proc = Process(target=async_process)
        self.backend.sub_proc.start()

    @Slot(str)
    def update_execute_process_status(self, execute_flag):
        if execute_flag == "free":
            self._close_run()
            self.backend.run_state = False
        elif execute_flag == "run":
            self._start_run()
            self.backend.run_state = True

    def _start_run(self):
        self.set_toolbar_state(False)
        self.set_toolbar_stop_state(True)

    def _close_run(self):
        if self.backend.config.run.use_simulator:
            self.set_toolbar_state(True)
            self.set_toolbar_stop_state(False)
        else:
            self.set_toolbar_state(self.state_flag)
            self.set_toolbar_stop_state(not self.state_flag)

    def start_communication_thread(self):
        self.backend.execute_heart = time.time()
        self.stop_communication_thread()
        self.heart_thread = HeartThead(backend=self.backend)
        self.heart_thread.qs_probe.connect(self.gui.thread_view_widget.refresh_msg_slot)
        self.heart_thread.pull_remind.connect(self._add_pull_remind)
        self.heart_thread.cache.connect(self._update_cache)
        self.heart_thread.qs_state_signal.connect(self.refresh_qs_state)
        self.heart_thread.err_signal.connect(self.alert_msg)
        self.heart_thread.logout.connect(self._auto_logout)
        self.heart_thread.report_error.connect(self.show_execute_error)
        self.heart_thread.update_dag_status.connect(self.update_dag_status)
        self.heart_thread.execute_timeout.connect(self.execute_restart)
        self.heart_thread.execute_stats.connect(self.update_execute_process_status)
        self.heart_thread.process_start.connect(self.setup_complete)
        # todoself.gui.ui.tabPlot
        self.heart_thread.dynamic_graph.connect(self.gui.ui.tabPlot.refresh)
        self.heart_thread.update_task_status.connect(self.update_task_status)
        self.heart_thread.start()

        self.async_thread = TaskQthread(backend=self.backend)
        self.async_thread.component_query_all.connect(self.gui.recall_query_all)

        self.async_thread.start()

    def stop_communication_thread(self):
        if self.heart_thread:
            self.heart_thread.close()
            self.heart_thread.wait()
            self.heart_thread = None

        if self.async_thread:
            self.async_thread.close()
            self.async_thread.wait()
            self.async_thread = None

    def _logout_out(self, courier_exit: bool = False):
        self.stop_communication_thread()
        self.gui.backend.cache_context_manager()
        clear_developer()
        self.backend.login_user = None
        self.hide()
        for sub_win in TitleGUI.sub_windows:
            sub_win.hide()
        self.gui.login_widget.login_out(courier_exit)
        self.gui.context_widget.login_out()
        self.gui.user_manage_widget.login_out()

    def ok_to_close(self):
        """Determine if it is ok to continue.

        Returns:
            bool: True to continue, False otherwise
        """
        reply = QMessageBox.question(
            self,
            "pyQCat-Visage",
            "Save unsaved changes to system?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Cancel:
            return False
        elif reply == QMessageBox.StandardButton.Yes:
            self.backend.close_save()
            self.gui.close_save()
            self.save_window_settings()

        return True


class VisageGUI(QMainWindowHandlerBase):
    _QMainWindowClass = VisageExtension
    # _stylesheet_default = "qdarkstyle-dark"
    _stylesheet_default = "visage_dark"

    def __init__(self, backend: BaseBackend = None, sub_proc=None):
        """
        Args:
            backend (): Pass in the backend that the GUI should handle.
                Defaults to None.
        """
        QtCore.qInstallMessageHandler(qt_message_handler)
        self.qApp = start_qApp()
        if not self.qApp:
            logger.error("Could not start Qt event loop using QApplication.")
        # backend for data interactions.
        self._backend = backend
        self._backend.sub_proc = sub_proc
        super().__init__()

        # Windows and Widgets
        self._system_config_widget = None
        self._library_widget = None
        self._options_window = None
        self._component_widget = None
        self._login_widget = None
        self._context_widget = None
        self._heatmap_widget = None
        self._run_setting_widget = None
        self._document_widget = None
        self._user_manage_widget = None
        self._workspace_manage_widget = None
        self._work_space_widget = None
        self._workspace_note_widget = None
        self._files_system_widget = None
        self._parallel_checkbox = None
        self._context_sidebar = None
        self._error_diag = None
        self._space_action = None
        self._pull_remind = None
        self._thread_view_widget: MultThreadWindow = None
        self._thread_diff_view: ThreadDiffView = None

        self._fig_auto_new = QPixmap(u":/new_email.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
        self._fig_auto_pull = QPixmap(u":/auto_pull.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
        self._fig_auto_push = QPixmap(u":/auto_push.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
        self._fig_auto_pull_push = QPixmap(u":/auto_pull_push.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)

        # set up all windows and widgets
        self._setup_system_widget()
        self._setup_context_sidebar()
        self._setup_library_widget()
        self._setup_options_widget()
        self._setup_component_widget()
        self._setup_login_window()
        self._setup_status_bar()
        self._setup_context_widget()
        self._setup_heatmap_widget()
        self._setup_run_setting_widget()
        self._setup_document_widget()
        self._setup_user_manage_widget()
        self._setup_files_widget()
        self._setup_copy_widget()
        self._setup_chip_manage_widget()
        self._setup_dag_manage_widget()
        self._setup_workspace_manage_widget()
        self._setup_work_space_widget()
        self._setup_workspace_note_widget()
        self._set_up_diagnose_window()
        self._set_up_thread_view_window()
        self._set_up_thread_diff_view()
        self._set_up_storm_manage_widget()

        # add check box for parallel mode.
        self._add_parallel_check_box()

        self._sub_proc = sub_proc
        QTimer.singleShot(10000, self._check_sub_process)
        QTimer.singleShot(150, self._raise)

        self.login_widget.auto_login()
        self.restore_window()

    @property
    def backend(self):
        """Get backend"""
        return self._backend

    @property
    def error_diag(self):
        return self._error_diag

    @property
    def thread_view_widget(self):
        return self._thread_view_widget

    @property
    def login_widget(self):
        """Get login widget"""
        return self._login_widget
    @property
    def thread_diff_view(self):
        return self._thread_diff_view

    @property
    def system_config_widget(self):
        """Get system config widget"""
        return self._system_config_widget

    @property
    def options_window(self):
        """Get option window"""
        return self._options_window

    @property
    def files_widget(self):
        return self._files_system_widget

    @property
    def context_widget(self):
        return self._context_widget

    @property
    def context_sidebar(self):
        return self._context_sidebar

    @property
    def component_editor(self):
        return self._component_widget

    @property
    def heatmap_widget(self):
        return self._heatmap_widget

    @property
    def chip_manage(self):
        return self._chip_manage_widget

    @property
    def dag_manage(self):
        return self._dag_manage_widget

    @property
    def storm_manage(self):
        return self._storm_manage_widget

    @property
    def workspace_manage(self):
        return self._workspace_manage_widget

    @property
    def work_space(self):
        return self._work_space_widget

    @property
    def workspace_note(self):
        return self._workspace_note_widget

    @property
    def pull_remind(self):
        return self._pull_remind

    @property
    def user_manage_widget(self):
        return self._user_manage_widget

    @property
    def run_setting_widget(self):
        return self._run_setting_widget

    @property
    def document_widget(self):
        return self._document_widget

    @property
    def copy_widget(self):
        return self._copy_widget

    def _setup_user_manage_widget(self):
        self._user_manage_widget = UserManagerWindow(self, self.main_window)

    def _setup_run_setting_widget(self):
        self._run_setting_widget = RunSettingWidget(self, self.main_window)

    def _setup_system_widget(self):
        """Sets up the system config widget."""
        self._system_config_widget = SystemConfigWindow(self, parent=self.main_window)

    def _setup_files_widget(self):
        self._files_system_widget = FileSysMainWindow(self, parent=self.main_window)

    def _setup_library_widget(self):
        """Sets up the GUI's Library tree display in Model-View-Controller framework."""
        self.exp_lib_model = LibraryTreeModel(
            self.main_window, self, self.ui.experimentLibrary, "Experiments"
        )
        self.dag_lib_model = LibraryTreeModel(
            self.main_window, self, self.ui.dagLibrary, "Dags"
        )

        self.ui.experimentLibrary.setModel(self.exp_lib_model)
        self.ui.dagLibrary.setModel(self.dag_lib_model)

        self.ui.lib_tab_widget.setCurrentIndex(0)

    def _setup_context_sidebar(self):
        """Sets up experiment options and analysis options."""
        self._context_sidebar = ContextSideBar(self, self.ui.dockContext)
        self.ui.dockContext.setWidget(self._context_sidebar)
        self.ui.dockContext.show()
        self._context_sidebar.change_physical_unit.connect(self._edit_options)
        self.ui.tabTopology.topology_view.envs_change.connect(
            self._context_sidebar.envs_refresh
        )
        self.ui.tabTopology.topology_view.physical_change.connect(
            self._context_sidebar.physical_refresh
        )
        self._context_sidebar.refresh_topology.connect(self.ui.tabTopology.refresh)
        self.ui.tabTopology.topology_view.custom_bits.connect(
            self._context_sidebar.create_custom_point
        )

    def _setup_options_widget(self):
        """Sets up experiment options and analysis options."""
        self._options_window = OptionsEditWidget(self, self.ui.dockEditOptions)

        self.ui.dockEditOptions.setWidget(self._options_window)
        self.ui.dockEditOptions.show()

        # --------------------------------------------------
        # Sigal
        self.ui.experimentLibrary.choose_exp_signal.connect(self._edit_options)
        self.ui.dagLibrary.choose_dag_signal.connect(self.ui.mainDAGTab.load_dag)

    def _setup_component_widget(self):
        """Sets up the component editor widget."""
        self._component_widget = ComponentEditWindow(self, parent=self.main_window)

    def _setup_copy_widget(self):
        self._copy_widget = CopyDataWindow(self, parent=self.main_window)

    def _setup_chip_manage_widget(self):
        self._chip_manage_widget = ChimeraManagerWindow(self, parent=self.main_window)

    def _setup_dag_manage_widget(self):
        self._dag_manage_widget = DagManagerWindow(self, parent=self.main_window)

    def _set_up_storm_manage_widget(self):
        self._storm_manage_widget = StormManagerWindow(self, parent=self.main_window)

    def _setup_workspace_manage_widget(self):
        self._workspace_manage_widget = WorkSpaceManageWindow(self, parent=self.main_window)

    def _setup_work_space_widget(self):
        self._work_space_widget = WorkSpaceWindow(self, parent=self.main_window)

    def _setup_workspace_note_widget(self):
        self._workspace_note_widget = WorkSpaceHisWindow(self, parent=self.main_window)

    def _setup_login_window(self):
        self._login_widget = UserLoginWindow(self, parent=self.main_window)

    def _setup_status_bar(self):
        self.ui.user_label = QLabel(f"Hello, ***")
        self.ui.statusbar.addPermanentWidget(self.ui.user_label)

        self.ui.sample_label = QLabel(f"| Sample(***)")
        self.ui.statusbar.addPermanentWidget(self.ui.sample_label)

        self.ui.point_label = QLabel(f"| Point(***)")
        self.ui.statusbar.addPermanentWidget(self.ui.point_label)

        self.ui.env_label = QLabel(f"| Env(***)")
        self.ui.statusbar.addPermanentWidget(self.ui.env_label)

        self.ui.qaio_label = QLabel(f"| QAIO(***)")
        self.ui.statusbar.addPermanentWidget(self.ui.qaio_label)

        self.ui.exp_label = QLabel(f"| EXP(***)")
        self.ui.statusbar.addPermanentWidget(self.ui.exp_label)

        self.ui.dag_label = QLabel(f"| DAG(***)")
        self.ui.statusbar.addPermanentWidget(self.ui.dag_label)

        self.ui.invoker_label = QLabel(f"| Server(***)")
        self.ui.statusbar.addPermanentWidget(self.ui.invoker_label)

        self.ui.qs_label = QLabel(f"|  Chimera")
        self.ui.statusbar.addPermanentWidget(self.ui.qs_label)

        self.ui.qs_label_pig = QLabel()
        self.ui.statusbar.addPermanentWidget(self.ui.qs_label_pig)

        self.ui.auto_sync_label = QLabel(f"| ")
        self.ui.statusbar.addPermanentWidget(self.ui.auto_sync_label)
        self.ui.auto_sync_pig = QLabel()
        self.ui.statusbar.addPermanentWidget(self.ui.auto_sync_pig)
        self.ui.auto_new_pig = QLabel()
        self.ui.statusbar.addPermanentWidget(self.ui.auto_new_pig)
        self.ui.auto_new_pig.setPixmap(self._fig_auto_new)
        self.ui.auto_new_pig.hide()

    def _setup_context_widget(self):
        """Sets up the context editor widget."""
        self._context_widget = ContextEditWindow(self, self.main_window)

    def _setup_heatmap_widget(self):
        """Sets up the heatmap widget."""
        self._heatmap_widget = HeatMapWindow(self, parent=self.main_window)

    def _set_up_thread_view_window(self):
        """
        set multi thread viewer window.
        """
        self._thread_view_widget = MultThreadWindow(self)

    def _set_up_thread_diff_view(self):
        self._thread_diff_view = ThreadDiffView(self)

    def _setup_document_widget(self):
        style_chart = self.settings.value("style_chart", "cs2")
        theme = int(style_chart[-1])
        self._document_widget = DocumentCheckWindow(self, parent=self.main_window)
        self._document_widget.chart.setTheme(QChart.ChartTheme(theme))

    def _set_up_diagnose_window(self):
        """Sets up the diagnosis window to catch error message and handle it."""
        self._error_diag = DiagnoseWindow(self, parent=self.main_window)

    def _edit_options(self, exp: "VisageExperiment"):
        """Show options edit widget."""
        self._options_window.set_experiment(exp)
        self._options_window.setCurrentIndex(0)
        self.options_window.ui.ana_tree_view.hide_placeholder_text()
        self.options_window.ui.exp_tree_view.hide_placeholder_text()
        self._context_sidebar.refresh_experiment_options(exp)
        # fixed bug: When collapse all dock widget and click the `NodeItem`
        # the collapsed window will show.
        if not self.collapsed:
            self.ui.dockEditOptions.show()
            self.ui.dockEditOptions.raise_()

        if exp and exp.dag:
            self.ui.dagLibrary.choose_dag_signal.emit(exp.dag)

    def _raise(self):
        """Raises the window to the top."""
        self.main_window.raise_()

        # Give keyboard focus.
        # On Windows, will change the color of the taskbar entry to indicate that the
        # window has changed in some way.
        self.main_window.activateWindow()

    def force_refresh_options_widgets(self, is_full: bool = False):
        """refresh the options widgets"""
        self._options_window.display_all = is_full
        self._options_window.force_refresh()

    def refresh_qaio_log(self):
        """
        refresh qaio addr, refresh qaio log client.
        """
        if self.backend.config.mongo.inst_log:
            addr = f"{self.backend.config.mongo.inst_host}:{self.backend.config.mongo.inst_log}"
            if addr != self.log_service.qaio_log_addr:
                logger.info("refresh qaio log.")
                self.log_service.refresh_qaio_sock(addr)

    def init_visage(self, update_config: bool = False):
        """After user login

        1. load default config
        2. set invoker env
        3. load exp and dag
        4. query chip line
        5. query heatmap
        6. init context
        7. clear view data
        """
        # load pull/push status
        workspace_data = self._work_space_widget.load_all_data()
        # start communication thread
        self.main_window.start_communication_thread()
        # set statues bar
        sample = self.backend.config.system.sample
        env_name = self.backend.config.system.env_name
        qaio_type = self.backend.config.system.qaio_type
        point_label = self.backend.config.system.point_label
        invoker_label = self.backend.config.system.invoker_addr
        save_type = self.backend.config.system.save_type
        self.ui.sample_label.setText(f"| Sample({sample}) ")
        self.ui.env_label.setText(f"| ENV({env_name}) ")
        self.ui.qaio_label.setText(f"| QAIO({qaio_type}) ")
        self.ui.point_label.setText(f"| Point({point_label}) ")
        self.ui.invoker_label.setText(f"| Server({invoker_label}) ")

        # check sample name
        if save_type == "s3":
            check_s3_sample_name(sample)

        # set env
        self.backend.set_invoker_env()
        TitleWindow.init_user(self.backend)
        # set init exp and dag
        if not update_config:
            self.backend.init_backend()
            self.exp_lib_model.load()
            self.dag_lib_model.load()

            # set user admin
            is_super_name = "Super" if self.backend.is_super else "Common"
            self.ui.user_label.setText(
                f' Hello, {self.backend.login_user.get("username")}({is_super_name}) '
            )

            # init user manage widget
            self.user_manage_widget.login()

        # clear history data
        self.backend.view_channels.clear()
        self.backend.model_channels = None
        self.backend.components.clear()

        # refresh model
        self.system_config_widget.tree_model.refresh()
        self._files_system_widget.init_file_system()
        self._run_setting_widget.refresh()
        self._update_report_theme()
        # refresh qaio log.
        self.refresh_qaio_log()

        # refresh parallel server, remove in 0.4.5
        # self.backend.refresh_parallel_server()

        # refresh context build cache data
        self.backend.context_builder.clear_cache()
        self.backend.refresh_chip_context()
        self.backend.load_user_cache_context()

        # load chip line
        self.context_widget.load_chip_line()
        self._context_sidebar.load()
        self.dag_manage.init_task()

        # load workspace bit
        self.ui.tabTopology.topology_view.workspace_flag = True
        self.refresh_workspace_cache(workspace_data)

        self.backend.refresh_meta_data()

    def refresh_workspace_cache(self, ret_data: Dict):
        bit_names = ret_data["data"].get("bit_names", [])
        bit_range = ret_data["data"].get("bit_range", [])
        conf_names = ret_data["data"].get("conf_names", [])
        point_labels = ret_data["data"].get("point_names", [])
        if self.backend.config.system.point_label not in point_labels:
            return
        TopologyView.workspace_bit_range = bit_range
        TopologyView.workspace_conf_cache = conf_names
        self.ui.tabTopology.topology_view.workspace_change.emit(bit_names)

    def close_save(self):
        self.context_widget.cache_context()

    def _check_sub_process(self):
        if self._sub_proc and not self._sub_proc.is_alive():
            logger.error(
                "The task-process did not start because of the exception, usually because the child process "
                "was not shut down properly the last time it was started (Zmq port occupancy).\nGenerally, you"
                " can restart the application to solve the problem. If this message is displayed again, close "
                "all python processes in the task Manager or restart the computer!"
            )
        # elif self._sub_proc and self._sub_proc.is_alive():
        #     logger.info("The child process start success!")

    def _update_report_theme(self, theme: str = None):
        if self.backend.config:
            r_theme = theme or GUI_CONFIG.report_theme_map.get(self._style_sheet_path)
            self.backend.config.report.theme = r_theme
            self.system_config_widget.tree_model.refresh()

    def trans_parallel_mode(self):
        self.backend.trans_parallel_mode()
        self._context_sidebar.update_options_edit()

    def _add_parallel_check_box(self):
        """Add parallel check box."""
        self._parallel_checkbox = QCheckBox("Parallel Mode", checked=False)
        self._parallel_checkbox.setObjectName("ParallelCheckBox")
        if self._options_window and self.backend.parallel_mode:
            self._parallel_checkbox.setChecked(True)
        self._parallel_checkbox.stateChanged.connect(self.trans_parallel_mode)
        self.ui.toolBarDesign.addWidget(self._parallel_checkbox)

    def add_space_action(self, auto_pull, auto_push):
        if not auto_pull and not auto_push:
            self.ui.auto_sync_pig.hide()
            return

        if auto_pull and auto_push:
            self.ui.auto_sync_pig.setPixmap(self._fig_auto_pull_push)
        elif auto_pull and not auto_push:
            self.ui.auto_sync_pig.setPixmap(self._fig_auto_pull)
        elif not auto_pull and auto_push:
            self.ui.auto_sync_pig.setPixmap(self._fig_auto_push)
        self.ui.auto_sync_pig.show()

    def query_all(self):
        self.backend.async_task_send(async_work="query_all", recall=["component_query_all"])
        self.component_editor.set_query_button_action(False)
        self.heatmap_widget.set_query_button_action(False)

    @Slot(dict)
    def recall_query_all(self, ret_data):
        if ret_data:
            self.component_editor.refresh_all()
            self.heatmap_widget.refresh_all(ret_data)
        self.component_editor.set_query_button_action(True)
        self.heatmap_widget.set_query_button_action(True)
