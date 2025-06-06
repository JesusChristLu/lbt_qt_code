# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/08
# __author:       YangChao Zhao

import copy
import os
from typing import TYPE_CHECKING, Union

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QTabWidget, QWidget
from loguru import logger

from pyQCat.structures import QDict
from pyqcat_visage.backend.experiment import VisageExperiment
from pyqcat_visage.gui.options_widget_ui import Ui_TabWidget
from pyqcat_visage.gui.widgets.options.tree_delegate_options import QOptionsDelegate
from pyqcat_visage.gui.widgets.options.tree_model_options import QTreeModelOptions
from pyqcat_visage.tool.utilies import FATHER_OPTIONS

if TYPE_CHECKING:
    from pyqcat_visage.gui.main_window import VisageGUI


class OptionsEditWidget(QTabWidget):
    """This is just a handler (container) for the UI; it a child object of the
    main gui.

    This class extends the `QTabWidget` class.

    PySide6 Signal / Slots Extensions:
        The UI can call up to this class to execute button clicks for instance
        Extensions in qt designer on signals/slots are linked to this class

    **Access:**
        gui.component_window
    """

    def __init__(self, gui: "VisageGUI", parent: QWidget):
        """
        Args:
            gui: (MetalGUI): The GUI
            parent (QWidget): Parent widget
        """
        # Parent is usually a dock component

        super().__init__(parent)

        # Parent GUI related
        self.gui = gui

        # UI
        self.ui = Ui_TabWidget()
        self.ui.setupUi(self)

        # Options Display states:
        self.display_all = False

        # Experiment being edited
        self.edit_exp = None

        # parallel mode default false

        # self.select_experiment = None

        # Parameter model and table view
        self.exp_model = QTreeModelOptions(self, gui, self.ui.exp_tree_view, name="exp")
        self.ana_model = QTreeModelOptions(self, gui, self.ui.ana_tree_view, name="ana")
        self.ui.exp_tree_view.setModel(self.exp_model)
        self.ui.ana_tree_view.setModel(self.ana_model)
        self.exp_delegate = QOptionsDelegate(self)
        self.exp_delegate.choose_exp_signal.connect(self.generate_parallel_options)
        self.ui.exp_tree_view.setItemDelegate(self.exp_delegate)
        self.ui.ana_tree_view.setItemDelegate(QOptionsDelegate(self))

    @property
    def backend(self):
        """Returns the design."""
        return self.gui.backend

    @property
    def experiment(self) -> "VisageExperiment":
        """Use the interface to components dict to return a QComponent.

        Returns:
            QComponent: The QComponent in design class which has name of self.component_name.
            If there are multiple usages of component_name within design._components,
            the first component using it will be returned, along with a logger.warning() message.
            None is returned if the name is not in design._components. Also warning will be posted
            through logger.warning().
        """
        return self.edit_exp

    def set_experiment(self, exp: VisageExperiment = None):
        """Main interface to set the component (by name)

        Args:
            exp (str): Set the component name, if None then clears
        """

        self.edit_exp = exp
        if exp is None:
            # TODO: handle case when name is none: just clear all
            # TODO: handle case where the component is made in jupyter notebook
            self.force_refresh()
            return

        if self.backend.parallel_mode:
            self.edit_exp.is_parallel = True
            label_text = f"P-{exp.tab.upper()} | {exp.name} | ID({exp.gid})"
            self.exp_delegate.choose_exp_signal.emit()
        else:
            self.edit_exp.is_parallel = False
            label_text = f"{exp.tab.upper()} | {exp.name} | ID({exp.gid})"
            self.edit_exp = exp
        self.setWindowTitle(label_text)
        self.parent().setWindowTitle(label_text)
        self.force_refresh()

    def force_refresh(self):
        """Force refresh."""
        self.ana_model.load()
        self.exp_model.load()
        self.ui.exp_tree_view.autoresize_columns()
        self.ui.ana_tree_view.autoresize_columns()

    @Slot()
    def save_experiment(self):
        if self.experiment:
            if self.experiment.tab == "experiments":
                ret_data = self.backend.save_one_exp_to_db(self.experiment)
                if ret_data.get("code") == 200:
                    logger.log("UPDATE", f"Save {self.experiment.name} to DB success")

    def save_exp_to_local(self):

        if self.experiment.tab == "experiments":
            save_name, ok = self.gui.main_window.ask_input("Save Experiment", "Please input file name")
            config_path = self.backend.config.system.config_path
            dirname = os.path.join(config_path, "EXP", self.backend.system_anno)
            self.experiment.to_file(dirname, self.backend.meta, self.display_all, describe=save_name)
        else:
            ret_data = QDict(code=800, msg="Only experiment support single save")
            self.gui.main_window.handler_ret_data(ret_data)

    @Slot()
    def generate_parallel_options(self):
        self.parallel_check(self.edit_exp)

    def parallel_check(self, visage_exp: VisageExperiment):
        def _get_ctx(exp):
            _ctx_options = QDict(
                {
                    "name": exp.context_options.name[0],
                    "physical_unit": exp.context_options.physical_unit[0],
                    "readout_type": exp.context_options.readout_type[0]
                }
            )

            parallel_physical_unit: Union[str, list] = _ctx_options.physical_unit

            if not parallel_physical_unit:
                parallel_physical_unit = []
            else:
                parallel_physical_unit = parallel_physical_unit.split(",")

            _ctx_options.physical_unit = parallel_physical_unit

            return _ctx_options

        def _generate_parallel_options(options, parallel_physical_unit_):
            new_options = {}
            for key, value in options.items():
                if key in ["child_exp_options", "child_ana_options"]:
                    new_options[key] = _generate_parallel_options(
                        value, parallel_physical_unit_
                    )
                elif key in FATHER_OPTIONS:
                    new_options[key] = value
                else:
                    new_options[key] = {}
                    for physical_bit in parallel_physical_unit_:
                        new_options[key][physical_bit] = copy.deepcopy(value)
            return new_options

        if self.backend.parallel_mode:
            visage_exp = visage_exp or self.edit_exp
            logger.info(f"Generate parallel options {visage_exp}")

            ctx_options = _get_ctx(visage_exp)
            if (
                visage_exp.parallel_options.model_exp_options
                and visage_exp.parallel_options.model_ana_options
                and visage_exp.parallel_options.ctx_options.to_dict() == ctx_options.to_dict()
            ):
                return

            if len(ctx_options.physical_unit) <= 1:
                logger.warning("Please Check Experiment Context!, not support parallel")
                return

            exp_options = QDict(
                same_options=visage_exp.parallel_options.model_exp_options.get(
                    "same_options", [True, "bool", True]
                )
            )
            ana_options = QDict()

            exp_options.update(
                _generate_parallel_options(
                    visage_exp.model_exp_options, ctx_options.physical_unit
                )
            )
            ana_options.update(
                _generate_parallel_options(
                    visage_exp.model_ana_options, ctx_options.physical_unit
                )
            )
            visage_exp.parallel_options.model_exp_options = exp_options
            visage_exp.parallel_options.model_ana_options = ana_options
            visage_exp.parallel_options.ctx_options = ctx_options
