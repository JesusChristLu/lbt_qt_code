# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/08/10
# __author:       XuYao
import json
from typing import TYPE_CHECKING, Dict

from pyqcat_visage.gui.widgets.chip_manage_files.table_model_workspace_note import QTableModelWorkSpace
from pyqcat_visage.gui.widgets.chip_manage_files.tree_model_workspace_note import QTreeModelSpaceNote
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.gui.workspace_sync_note_ui import Ui_MainWindow

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class WorkSpaceHisWindow(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent=None):
        super().__init__(parent)
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.gui = gui
        self.data = []
        self.cache_chip_sample_data = {}
        self.user_list = {}
        self._space = {}
        self._setup_context_editor()
        self.ui.table_view_context.choose_space_signal.connect(
            self._edit_space_note
        )

    @property
    def space(self):
        return self._space

    @property
    def ui(self):
        return self._ui

    def _edit_space_note(self, space_note: Dict):
        self.set_space(space_note)
        self.ui.table_view_context.hide_placeholder_text()

    def _setup_context_editor(self):
        self.context_table_model = QTableModelWorkSpace(
            self.gui, self, self.ui.table_view_context
        )
        self.ui.table_view_context.setModel(self.context_table_model)

        self.context_tree_model = QTreeModelSpaceNote(
            self, self.gui, self.ui.tree_view_context
        )
        self.ui.tree_view_context.setModel(self.context_tree_model)

        self.ui.table_view_context.choose_space_signal.connect(self._look_space)

    def force_refresh(self):
        """Force refresh."""
        self.context_tree_model.load()

    def set_space(self, space=None):
        self._space = space

        if space is None:
            self.force_refresh()
            return

        # Labels
        # ) from {space.__class__.__module__}
        # label_text = f"{space.data_dict.name} | {space.data_dict.class_name}"
        # self.ui.labelComponentName.setText(label_text)
        # self.ui.labelComponentName.setCursorPosition(0)  # Move to left
        # self.setWindowTitle(label_text)
        # self.parent().setWindowTitle(label_text)

        self.force_refresh()

        self.ui.tree_view_context.autoresize_columns()  # resize columns

        data_dict = space
        if not isinstance(space, dict):
            data_dict = space.to_dict()

        self.ui.textEdit.setText(json.dumps(data_dict, indent=4))

    def _look_space(self, space):
        self.set_space(space)
        self.ui.tree_view_context.hide_placeholder_text()

    def load_sample_cache(self):
        ret_data = self.gui.backend.db.query_chip_sample_data()
        if ret_data.get("code") == 200:
            self.cache_chip_sample_data = ret_data["data"]

    def load_user_data(self):
        if self.is_super:
            ret_data = self.gui.backend.db.query_usernames()
            self.user_list = [""] + ret_data["data"]
        elif self.is_admin:
            ret_data = self.gui.backend.db.query_usernames(self.group_name)
            self.user_list = [""] + ret_data["data"]
        else:
            self.user_list = [self.username]
        self.load_sample_cache()

    def clear_cache_data(self):
        self.ui.UserContent.clear()
        self.ui.SampleContent.clear()
        self.ui.EnvContent.clear()
        self.ui.nameContent.clear()

    def load_all_data(self):
        self.clear_cache_data()
        self.load_user_data()
        sample_list = list(self.cache_chip_sample_data.keys())
        self.ui.UserContent.addItems(self.user_list)
        self.ui.SampleContent.addItems(sample_list)
        self.ui.EnvContent.addItems(self.cache_chip_sample_data.get("", []))

    def change_page(self, index: int):
        if index:
            self.query_workspace_his()

    def change_volume(self, index: int):
        if index:
            self.query_workspace_his()

    def query_workspace_his(self):
        if not self.is_super and not self.is_admin:
            username = self.username
        else:
            username = self.ui.UserContent.currentText()
        sample = self.ui.SampleContent.currentText()
        env_name = self.ui.EnvContent.currentText()
        name = self.ui.nameContent.text()
        page_num = self.ui.page_spinBox.value() or 1
        page_size = self.ui.volume_spinBox.value() or 10
        ret_data = self.gui.backend.db.query_workspace_his(username, sample,
                                                           env_name, name, page_num, page_size)
        if ret_data.get("code") == 200:
            self.data = ret_data["data"]
            self.context_table_model.refresh_auto(check_count=False)
        else:
            self.data = []
            self.context_table_model.refresh_auto()
