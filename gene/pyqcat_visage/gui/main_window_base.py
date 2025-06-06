# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/31
# __author:       HanQing Shi
"""GUI front-end interface for pyQCat Visage in PySide6."""
import os
import pathlib
import sys
from copy import deepcopy

# pylint: disable=invalid-name
import qdarkstyle
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCharts import QChart
from PySide6.QtCore import Slot
from PySide6.QtGui import QIcon, QActionGroup
from PySide6.QtWidgets import QApplication, QFileDialog, QDockWidget
from jinja2 import Template
from loguru import logger
from pyQCat import __version__ as __monster_version__
from pyQCat.invoker import DEFAULT_PORT
from pyQCat.log import get_pubhandler, LogFormat
from pyQCat.structures import QDict
from qdarkstyle import DarkPalette, LightPalette

from pyqcat_visage import __version__ as __visage_version__
from pyqcat_visage import config
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.gui.main_window_ui import Ui_MainWindow
from pyqcat_visage.gui.tools.theme import CustomTheme
from pyqcat_visage.gui.tools.utilies import slot_catch_exception
from pyqcat_visage.gui.widgets.log.log_visage import LogHandler_for_QTextLog
from pyqcat_visage.gui.widgets.title_window import TitleWindow
from pyqcat_visage.backend.qthread_log import init_logger_service


class QMainWindowExtensionBase(TitleWindow):
    """This contains all the functions that the gui needs to call directly from
    the UI.

    Extends the `QMainWindow` class.
    """

    def __init__(self):
        """"""
        super().__init__()
        self.ui = None  # UI object.
        self.force_close = False
        self._handler = None  # type: QMainWindowHandlerBase

    @property
    def gui(self) -> "QMainWindowHandlerBase":
        """Get the GUI."""
        return self._handler

    @property
    def settings(self) -> QtCore.QSettings:
        """Get the settings."""
        return self._handler.settings

    @property
    def logger(self):
        """Get the logger."""
        return self._handler.logger

    def _remove_log_handlers(self):
        """Remove the log handlers."""
        if hasattr(self, "log_text"):
            self.log_text.remove_handlers()

    def destroy(self, destroyWindow: bool = True, destroySubWindows: bool = True):
        """When the window is cleaned up from memory.

        Args:
            destroyWindow (bool): Whether or not to destroy the window.  Defaults to True.
            destroySubWindows (bool): Whether or not to destroy sub windows  Defaults to True.
        """
        self._remove_log_handlers()
        super().destroy(
            destroyWindow=destroyWindow, destroySubWindows=destroySubWindows
        )

    def restore_window_settings(self):
        """Call a Qt built-in function to restore values from the settings
        file.

        Raises:
            Exception: Error in restoration
        """
        version_settings = self.settings.value("visage_version", defaultValue="0")
        monster_settings = self.settings.value("monster_version", defaultValue="0")
        if (
            __visage_version__ > version_settings
            or __monster_version__ > monster_settings
        ):
            logger.debug(f"Clearing window settings [{version_settings}]...")
            self.settings.clear()

        try:
            logger.debug("Restoring window settings...")

            # should probably call .encode("ascii") here
            geom = self.settings.value("geometry", "")
            if isinstance(geom, str):
                geom = geom.encode("ascii")
            self.restoreGeometry_(geom)

            # window state
            window_state = self.settings.value("windowState", "")
            if isinstance(window_state, str):
                window_state = window_state.encode("ascii")
            self.restoreState(window_state)

            # chart style
            style_chart = self.settings.value("style_chart", "cs2")
            getattr(self, GUI_CONFIG.chart_style.get(style_chart)[0])()
            eval(f"self.gui.ui.{GUI_CONFIG.chart_style.get(style_chart)[1]}.trigger()")

            # system style sheet
            style_sheet_path = self.settings.value(
                "stylesheet", self.gui._stylesheet_default
            )
            self.gui.load_stylesheet(str(style_sheet_path))
            # Dag QGraphicsView zoom factor
            dag_transform = self.settings.value("dag_transform", None)
            dag_current_scale = self.settings.value("dag_current_scale", None)
            dag_num_scaling = self.settings.value("dag_num_scaling", None)
            if dag_transform is not None:
                self.gui.ui.mainDAGTab.view.setTransform(dag_transform)
            if dag_current_scale is not None:
                self.gui.ui.mainDAGTab.view.current_scale = float(dag_current_scale)
            if dag_num_scaling is not None:
                self.gui.ui.mainDAGTab.view.num_scheduled_scaling = float(dag_num_scaling)
            # TODO: Recent files
        except Exception as e:
            logger.error(f"ERROR [restore_window_settings]: {e}")

    def get_screenshot(self, name="shot", type_="png"):
        """Grad a screenshot of the main window, save to file

        Args:
            name (string): File name without extension
            type_ (string): File format and name extension
        """
        # todo show message box and ask user to set save path.
        path = pathlib.Path(name + "." + type_).absolute()

        # grab the main window
        screenshot = self.grab()  # type: QtGui.QPixMap
        screenshot.save(str(path), type_)

        # copy to clipboard
        QtWidgets.QApplication.clipboard().setPixmap(screenshot)
        logger.info(f"Screenshot copied to clipboard and saved to:\n {path}")

    @Slot()
    def _screenshot(self):
        """Get the GUI screen shot."""
        self.get_screenshot()

    @Slot()
    def load_stylesheet_dark(self):
        """Used to call from action."""
        self._handler.load_stylesheet("qdarkstyle-dark")

    @Slot()
    def load_stylesheet_light(self):
        """Used to call from action."""
        self._handler.load_stylesheet("qdarkstyle-light")

    @Slot()
    def load_stylesheet_default(self):
        """Used to call from action."""
        self._handler.load_stylesheet("default")

    @slot_catch_exception()
    def load_stylesheet_cat_dark(self):
        """Used to call from action."""
        self._handler.load_stylesheet("visage_dark")

    @slot_catch_exception()
    def load_stylesheet_cat_light(self):
        """Used to call from action."""
        self._handler.load_stylesheet("visage_light")

    @slot_catch_exception()
    def load_stylesheet_open(self):
        """Loading style from custom files."""
        default_path = str(self.gui.stylesheets_path)
        filename = QFileDialog.getOpenFileName(
            self, "Select Qt stylesheet file `.qss`", default_path
        )[0]
        if filename:
            logger.info(f"Attempting to load stylesheet file {filename}")
            self._handler.load_stylesheet(filename)

    @slot_catch_exception()
    def save_window_settings(self):
        """Save the window settings."""
        logger.info("Saving window state")
        # get the current size and position of the window as a byte array.
        self.settings.setValue("visage_version", __visage_version__)
        self.settings.setValue("monster_version", __monster_version__)
        self.settings.setValue("geometry", self.saveGeometry_())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("stylesheet", self.gui._stylesheet)
        self.settings.setValue("dag_transform", self.gui.ui.mainDAGTab.view.transform())
        self.settings.setValue("dag_current_scale", self.gui.ui.mainDAGTab.view.current_scale)
        self.settings.setValue("dag_num_scaling", self.gui.ui.mainDAGTab.view.num_scheduled_scaling)
        self.settings.setValue(
            "style_chart", self.ui.policy_group.checkedAction().style_name
        )

    @Slot()
    def collapse_all_docks(self):
        """Show or hide all docks."""
        # Get all docks to show/hide. Ignore edit source
        docks = [
            widget for widget in self.children() if isinstance(widget, QDockWidget)
        ]
        docks = list(
            filter(
                lambda x: not x.windowTitle().lower().startswith("edit source"), docks
            )
        )
        dock_states = {dock: dock.isVisible() for dock in docks}
        do_hide = any(dock_states.values())  # if any are visible then hide all
        for dock in docks:
            if do_hide:
                dock.hide()
                self.gui.collapsed = True
            else:
                dock.show()
                self.gui.collapsed = False

    @Slot()
    def chart_light_theme(self):
        self.gui.ui.chart_view_y0.chart.setTheme(QChart.ChartTheme(0))
        self.gui.ui.chart_view_y1.chart.setTheme(QChart.ChartTheme(0))
        if hasattr(self.gui, "document_widget"):
            self.gui.document_widget.chart.setTheme(QChart.ChartTheme(0))

    @Slot()
    def chart_bc_theme(self):
        self.gui.ui.chart_view_y0.chart.setTheme(QChart.ChartTheme(1))
        self.gui.ui.chart_view_y1.chart.setTheme(QChart.ChartTheme(1))
        if hasattr(self.gui, "document_widget"):
            self.gui.document_widget.chart.setTheme(QChart.ChartTheme(1))

    @Slot()
    def chart_dark_theme(self):
        self.gui.ui.chart_view_y0.chart.setTheme(QChart.ChartTheme(2))
        self.gui.ui.chart_view_y1.chart.setTheme(QChart.ChartTheme(2))
        if hasattr(self.gui, "document_widget"):
            self.gui.document_widget.chart.setTheme(QChart.ChartTheme(2))

    @Slot()
    def chart_bs_theme(self):
        self.gui.ui.chart_view_y0.chart.setTheme(QChart.ChartTheme(3))
        self.gui.ui.chart_view_y1.chart.setTheme(QChart.ChartTheme(3))
        if hasattr(self.gui, "document_widget"):
            self.gui.document_widget.chart.setTheme(QChart.ChartTheme(3))

    @Slot()
    def chart_bn_theme(self):
        self.gui.ui.chart_view_y0.chart.setTheme(QChart.ChartTheme(4))
        self.gui.ui.chart_view_y1.chart.setTheme(QChart.ChartTheme(4))
        if hasattr(self.gui, "document_widget"):
            self.gui.document_widget.chart.setTheme(QChart.ChartTheme(4))

    @Slot()
    def chart_hc_theme(self):
        self.gui.ui.chart_view_y0.chart.setTheme(QChart.ChartTheme(5))
        self.gui.ui.chart_view_y1.chart.setTheme(QChart.ChartTheme(5))
        if hasattr(self.gui, "document_widget"):
            self.gui.document_widget.chart.setTheme(QChart.ChartTheme(5))

    @Slot()
    def chart_bi_theme(self):
        self.gui.ui.chart_view_y0.chart.setTheme(QChart.ChartTheme(6))
        self.gui.ui.chart_view_y1.chart.setTheme(QChart.ChartTheme(6))
        if hasattr(self.gui, "document_widget"):
            self.gui.document_widget.chart.setTheme(QChart.ChartTheme(6))

    @Slot()
    def chart_qt_theme(self):
        self.gui.ui.chart_view_y0.chart.setTheme(QChart.ChartTheme(7))
        self.gui.ui.chart_view_y1.chart.setTheme(QChart.ChartTheme(7))
        if hasattr(self.gui, "document_widget"):
            self.gui.document_widget.chart.setTheme(QChart.ChartTheme(7))


class QMainWindowHandlerBase:
    """Abstract Class to wrap and handle main window (QMainWindow)."""

    _appid = "pyQCat Visage"
    _QMainWindowClass = QMainWindowExtensionBase
    _stylesheet_default = "default"
    _img_folder_name = "_imgs"
    _font_folder_name = "_fonts"
    _img_logo_name = "logo.png"
    __ui_window__ = Ui_MainWindow
    log_service = None
    class_theme_special = """QCheckBox[objectName="ParallelCheckBox"]{background-color: {{checkbox_color}}}
                QWidget[objectName="TitleWidget"]{
                    border-top: 3px solid qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {{border_color1}},stop:1 {{border_color2}});
                    border-left: 3px solid qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {{border_color1}},stop:1 {{border_color2}});
                    border-right: 3px solid qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {{border_color1}},stop:1 {{border_color2}});
                    height: 30px;
                    padding: 10px;
                }
                
                QWidget[objectName="RealWidget"]{
                    border-bottom: 3px solid qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {{border_color1}},stop:1 {{border_color2}});
                    border-left: 3px solid qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {{border_color1}},stop:1 {{border_color2}});
                    border-right: 3px solid qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {{border_color1}},stop:1 {{border_color2}});
                }
                """

    def __init__(self):
        """Can pass in logger.

        Args:
            logger (logging.Logger): The logger.  Defaults to None.

        Attributes:
            _settings: Used to save the state of the window
                This information is often stored in the system
                registry on Windows, and in property list files on macOS and iOS.
        """
        # base settings and configs.
        self.config = deepcopy(config.GUI_CONFIG)
        self.settings = QtCore.QSettings(self._appid, "MainWindow")

        self._log_handler = None

        self._stylesheet = self._stylesheet_default

        # Path to gui folder and images folder
        self.gui_path = self._get_file_path()
        self.imgs_path = self._get_imgs_path()
        self.fonts_path = self._get_fonts_path()

        # Main Window and app
        self.main_window = self._QMainWindowClass()
        self.main_window._handler = self
        # todo
        self.qApp = self._setup_qApp()
        self._setup_system_tray()

        # Style and window
        self._style_sheet_path = None
        self.style_window()

        self.graphics_theme = QDict()
        # UI
        self.ui = self.__ui_window__()
        self.ui.setupUi(self.main_window)
        self._ui_extension()
        self.main_window.ui = self.ui

        # title add version
        title = self.main_window.windowTitle()
        new_title = (
            f"{title} | Visage({__visage_version__}) | Monster({__monster_version__})"
        )
        self.main_window.setWindowTitle(new_title)

        # set logger widget.
        self._setup_logger()

        # set windows configs.
        self._setup_window_size()
        self.main_window.restore_window_settings()

        # hide
        self.collapsed = False

    def restore_window(self):
        self.main_window.restore_window_settings()

    @property
    def stylesheets_path(self):
        """Returns the path to the stylesheet."""
        return pathlib.Path(self.gui_path) / "styles"

    def style_window(self):
        """Styles the window."""
        # fusion macintosh # windows
        self.main_window.setStyle(
            QtWidgets.QStyleFactory.create(config.GUI_CONFIG.main_window.style)
        )

    def _update_report_theme(self, theme: str):
        pass

    def _update_chart_theme(self, theme: QChart.ChartTheme):
        self.main_window.gui.ui.chart_view_y0.chart.setTheme(theme)
        self.main_window.gui.ui.chart_view_y1.chart.setTheme(theme)
        if hasattr(self.main_window.gui, "document_widget"):
            self.main_window.gui.document_widget.chart.setTheme(theme)

    def load_graphics_stylesheet(self, color_conf: QDict):
        self.ui.tabTopology.init_theme(
            color_conf, bool(not self.ui.tabTopology.isHidden())
        )
        self.ui.mainDAGTab.init_theme(
            color_conf, bool(not self.ui.mainDAGTab.isHidden())
        )

    def load_stylesheet(self, path=None):
        """Load and set stylesheet for the main gui.

        Args:
            path (str) : Path to stylesheet or its name.
                Can be: 'default', 'qdarkstyle' or None.

        Returns:
            bool: False if failure, otherwise nothing

        Raises:
            ImportError: Import failure
        """
        result = True
        if path == "default" or path is None:
            self._style_sheet_path = "default"
            self._update_report_theme("light")
            self.main_window.setStyleSheet("QWidget::item:hover:!selected {}")
            self._update_chart_theme(QChart.ChartTheme.ChartThemeLight)
            self.graphics_theme = GUI_CONFIG.graphics_view.theme_visage_light

        elif path == "qdarkstyle-dark":
            self._update_report_theme("dark")
            os.environ["QT_API"] = "pyside6"
            style_sheet = qdarkstyle.load_stylesheet(palette=DarkPalette)
            # patch for parallel checkbox.
            check_box_sheet = (Template(self.class_theme_special).render(checkbox_color="#455364",
                                                                         border_color1="#455364",
                                                                         border_color2="#1A72BB")
                               )
            style_sheet += check_box_sheet
            self.main_window.setStyleSheet(style_sheet)
            self._update_chart_theme(QChart.ChartTheme.ChartThemeBlueCerulean)
            self.graphics_theme = GUI_CONFIG.graphics_view.theme_classic_dark

        elif path == "qdarkstyle-light":
            self._update_report_theme("light")
            os.environ["QT_API"] = "pyside6"
            style_sheet = qdarkstyle.load_stylesheet(palette=LightPalette)
            # patch for parallel checkbox.
            check_box_sheet = (Template(self.class_theme_special).render(checkbox_color="#C9CDD0",
                                                                         border_color1="#C9CDD0",
                                                                         border_color2="#7C7C7C")
                               )
            style_sheet += check_box_sheet
            self.main_window.setStyleSheet(style_sheet)
            self._update_chart_theme(QChart.ChartTheme.ChartThemeBlueIcy)
            self.graphics_theme = GUI_CONFIG.graphics_view.theme_classic_light

        elif path == "visage_dark":
            self._update_report_theme("dark")
            # path_full = self.stylesheets_path / 'visage_dark' / 'style.qss'
            # self._load_stylesheet_from_file(path_full)
            theme = CustomTheme()
            self.main_window.setStyleSheet(theme.qss(path))
            self._update_chart_theme(QChart.ChartTheme.ChartThemeDark)
            self.graphics_theme = GUI_CONFIG.graphics_view.theme_visage_dark

        elif path == "visage_light":
            self._update_report_theme("light")
            # path_full = self.stylesheets_path / 'visage_light' / 'style.qss'
            # self._load_stylesheet_from_file(path_full)
            theme = CustomTheme()
            self.main_window.setStyleSheet(theme.qss(path))
            self._update_chart_theme(QChart.ChartTheme.ChartThemeBrownSand)
            self.graphics_theme = GUI_CONFIG.graphics_view.theme_visage_light

        if self.graphics_theme:
            self.load_graphics_stylesheet(self.graphics_theme)

        if result:
            self._stylesheet = path
            self.settings.setValue("stylesheet", self._stylesheet)
            return True
        else:
            return False

    def _ui_extension(self):
        self.ui.actionLight.style_name = "cs0"
        self.ui.actionBlue_Cerulean.style_name = "cs1"
        self.ui.actionDark.style_name = "cs2"
        self.ui.actionBrown_Sand.style_name = "cs3"
        self.ui.actionBlue_NVS.style_name = "cs4"
        self.ui.actionHigh_Contrast.style_name = "cs5"
        self.ui.actionBlue_Icy.style_name = "cs6"
        self.ui.actionQt.style_name = "cs7"

        self.ui.policy_group = QActionGroup(self.ui.tabPlot)
        self.ui.policy_group.addAction(self.ui.actionQt)
        self.ui.policy_group.addAction(self.ui.actionLight)
        self.ui.policy_group.addAction(self.ui.actionBlue_Cerulean)
        self.ui.policy_group.addAction(self.ui.actionBlue_Icy)
        self.ui.policy_group.addAction(self.ui.actionBlue_NVS)
        self.ui.policy_group.addAction(self.ui.actionDark)
        self.ui.policy_group.addAction(self.ui.actionHigh_Contrast)
        self.ui.policy_group.addAction(self.ui.actionBrown_Sand)

        self.ui.policy_group.setExclusive(True)
        self.ui.actionDark.setChecked(True)

    def _get_file_path(self):
        """Get the dir name of the current path in which this file is stored."""
        return os.path.dirname(__file__)

    def _get_imgs_path(self):
        """Get images path."""
        imgs_path = (
            pathlib.Path(self.gui_path) / self._img_folder_name
        )  # Path to GUI imgs folder
        if not imgs_path.is_dir():
            text = f"The path provided ({imgs_path}) is not a directory."
            logger.error(text)
        return imgs_path

    def _get_fonts_path(self):
        fonts_path = (
            pathlib.Path(self.gui_path) / self._font_folder_name
        )  # Path to GUI fonts folder
        if not fonts_path.is_dir():
            text = f"The path provided ({fonts_path}) is not a directory."
            logger.error(text)
        return fonts_path

    def _load_stylesheet_from_file(self, path: str):
        """Load the sylesheet from a file.

        Args:
            path (str): Path to file

        Returns:
            bool: False if failure, otherwise nothing

        Raises:
            Exception: Stylesheet load failure
        """
        logger.debug(f"path = {path}")
        try:
            path = pathlib.Path(str(path))
            if path.is_file():
                self._style_sheet_path = str(path)
                stylesheet = path.read_text()
                stylesheet = stylesheet.replace(
                    ":/visage-styles", str(self.stylesheets_path)
                )

                # if windows, double the slashes in the paths
                if os.name.startswith("nt"):
                    stylesheet = stylesheet.replace("\\", "\\\\")

                self.main_window.setStyleSheet(stylesheet)
                return True

            else:
                logger.error(
                    f"Could not find the stylesheet file where expected {path}"
                )
                return False
        except Exception as e:
            logger.error(f"Load stylesheet from file error: {e}")

    def _setup_qApp(self):
        """Only one qApp can exist at a time, so check before creating one.

        Returns:
            QApplication: a setup QApplication

        There are three classes:
            QCoreApplication - base class. Used in command line applications.
            QGuiApplication - base class + GUI capabilities. Used in QML applications.
            QApplication - base class + GUI + support for widgets. Use it in QtWidgets applications.
        """

        self.qApp = QApplication.instance()

        if self.qApp is None:
            logger.warning("QApplication.instance is None.")

            # Kickstart()

            # Did it work now
            self.qApp = QApplication.instance()

            if self.qApp is None:
                logger.error(r"""ERROR: QApplication.instance is None.""")

        # win7 compatibility.
        if os.name.startswith("nt"):
            # Arbitrary string, needed for icon in taskbar to be custom set proper
            # https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(self._appid)

        app = self.qApp
        if app:
            app.setOrganizationName(r"Origin Quantum")
            app.setOrganizationDomain(r"https://originqc.com.cn/")
            app.setApplicationName(r"pyQCat Visage")

        return self.qApp

    def _setup_logger(self):
        """Setup logging UI and show wellcome to visage message."""
        if hasattr(self.ui, "log_text"):
            # add info.
            self.ui.log_text.img_path = self.imgs_path
            self.ui.log_text.font_path = self.fonts_path
            self.ui.log_text.qwidget = self.ui.dockLog

            # create handler.
            logger.remove()
            pubhandler_visage = get_pubhandler(DEFAULT_PORT + 1)
            logger.add(pubhandler_visage, format=LogFormat.pub, level=10)

            # QTimer.singleShot(1500, self.ui.log_text.welcome_message)
            self._log_handler = LogHandler_for_QTextLog(self.ui.log_text, logger=logger)
            self._log_qaio_handler = LogHandler_for_QTextLog(self.ui.log_qstream)
            self.ui.log_qstream.hide()
            self.ui.log_text.welcome_message()
            # todo remove log sercive.
            self.log_service = init_logger_service()
            self.log_service.log_message.connect(self._log_handler.emit_msg)
            self.log_service.qaio_message.connect(self._log_qaio_handler.emit_msg)
        else:
            logger.error("UI does not have `log_text`")

    def _setup_system_tray(self):
        """Sets up the main window tray."""
        if self.imgs_path.is_dir():
            icon = QIcon(str(self.imgs_path / self._img_logo_name))
            self.main_window.setWindowIcon(icon)
            self._icon_tray = QtWidgets.QSystemTrayIcon(icon, self.main_window)
            self._icon_tray.show()
            self._icon_tray.activated.connect(self.iconActivated)
            self.qApp.setWindowIcon(icon)
            # todo remove system tray when close app.

    @Slot(QtWidgets.QSystemTrayIcon.ActivationReason)
    def iconActivated(self, reason):
        if reason == QtWidgets.QSystemTrayIcon.Trigger:
            if self.main_window.isMinimized():
                self.main_window.showNormal()
                self.main_window.activateWindow()

    def _setup_window_size(self):
        """Setup the window size."""
        if self.config.main_window.auto_size:
            screen = self.qApp.primaryScreen()
            rect = screen.availableGeometry()
            rect.setWidth(int(rect.width() * 0.9))
            rect.setHeight(int(rect.height() * 0.9))
            rect.setLeft(int(rect.left() + rect.width() * 0.1))
            rect.setTop(int(rect.top() + rect.height() * 0.1))
            self.main_window.setGeometry(rect)

    def show(self):
        """Show the main window."""
        self.main_window.show()


def start_qApp():
    """Click start the application.

    Returns:
        QtCore.QCoreApplication: the application

    Raises:
        AttributeError: Attribute only exists for Qt >= 6.1
        Exception: Magic method failure
    """
    qApp = QtCore.QCoreApplication.instance()

    if qApp is None:
        try:
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        except AttributeError:  # Attribute only exists for Qt >= 5.6
            pass

        qApp = QApplication(sys.argv)

        if qApp is None:
            logger.error("QApplication.instance is None.")
            logger.error(
                "QApplication.instance: Attempt to manually create qt5 QApplication"
            )
            qApp = QtWidgets.QApplication(["pyqcat-visage"])
            qApp.lastWindowClosed.connect(qApp.quit)

    return qApp
