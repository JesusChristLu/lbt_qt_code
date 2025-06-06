# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/31
# __author:       HanQing Shi
"""Logging widget.
"""
import collections
import html
import logging
import random
import re
from pathlib import Path
from typing import Union
from collections import deque

from pyQCat.structures import QDict
from PySide6 import QtGui
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import QDockWidget, QTextEdit, QWidget

from pyqcat_visage import __version__ as __visage_version__
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.gui.tools.utilies import clean_name, monkey_patch


class QTextEditLogger(QTextEdit):
    """A text edit logger class.

    This class extends the `QTextEdit` class
    """
    timestamp_len = 19
    _logo = 'logo.png'

    def __init__(self, qwidget: Union[QDockWidget, QWidget] = None):
        super().__init__()

        self.qwidget = qwidget
        self.img_path = "/"

        # handles the loggers
        # dict for what loggers we track and if we should show or not
        self.tracked_loggers = QDict()
        self.handler = None
        self._actions = QDict()  # menu actions. Must be an ordered Dict!
        self._auto_scroll = True  # autoscroll to end or not
        self._show_timestamps = False
        self._level_name = ''
        self._level = 0

        # Props of the Widget
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                     | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.text_format = QtGui.QTextCharFormat()
        self.text_format.setFontFamily("JetBrains Mono")

        self.logged_lines = collections.deque([],
                                              GUI_CONFIG.log_conf.num_lines)
        self.document().setMaximumBlockCount(5000)
        self.setup_menu()

    def setup_menu(self):
        """Setup the menu."""
        # Behaviour for menu: the widget displays its QWidget::actions() as context menu.
        # i.e., a local context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)

        # Click actions
        actions = self._actions

        actions.clear_log = QAction('&Clear log', self)
        actions.clear_log.triggered.connect(self.clear)

        # show helps.
        actions.print_tips = QAction('&Show tips ', self)
        actions.print_tips.triggered.connect(self.print_all_tips)

        actions.separator = QAction(self)
        actions.separator.setSeparator(True)

        # Toggle actions
        self.action_scroll_auto = QAction('&Autoscroll', self)
        self.action_scroll_auto.setCheckable(True)
        self.action_scroll_auto.setChecked(False)
        self.action_scroll_auto.toggled.connect(self.toggle_autoscroll)

        self.action_show_times = QAction('&Show timestamps', self)
        self.action_show_times.setCheckable(True)
        self.action_show_times.setChecked(False)
        self.action_show_times.toggled.connect(self.toggle_timestamps)

        # Add to actions.
        self.addActions([self.action_scroll_auto, self.action_show_times])

        # Filter level actions
        def make_trg(lvl):
            """Make a trg.

            Args:
                lvl (logging.level): The level of logging, eg.., logging.ERROR

            Returns:
                str: Value of the name attribute
            """
            name = f'set_level_{lvl}'
            setattr(self, name, lambda: self.set_level(lvl))
            func = getattr(self, name)
            return func

        actions.debug = QAction('Set filter level:  Debug', self)
        actions.debug.triggered.connect(make_trg(logging.DEBUG))

        actions.info = QAction('Set filter level:  Info', self)
        actions.info.triggered.connect(make_trg(logging.INFO))

        actions.warning = QAction('Set filter level:  Warning', self)
        actions.warning.triggered.connect(make_trg(logging.WARNING))

        actions.error = QAction('Set filter level:  Error', self)
        actions.error.triggered.connect(make_trg(logging.ERROR))

        actions.separator2 = QAction(self)
        actions.separator2.setSeparator(True)

        actions.loggers = QAction('Show/hide messages for logger:', self)
        actions.loggers.setEnabled(True)

        # Add actions to actin context menu
        self.addActions(list(actions.values()))

    def welcome_message(self):
        """Display the welcome message."""
        img_txt = ''

        # Logo
        img_path = Path(self.img_path) / self._logo
        if img_path.is_file():
            img_txt = f'<img src="{img_path}" height="80"/>'
        else:
            print(
                'WARNING: welcome_message could not locate img_path={img_path}'
            )

        # Main message
        text = f'''
        <span class="INFO">{' ' * self.timestamp_len}
            <br/>
            <table border="0" width="100%" id="tableLogo" style="margin: 0px;">
                <tr>
                    <td align="center">
                        <h2 align="center" style="text-align: center;">
                            Welcome to OriginQuantum pyqcat-visage!
                        </h2>
                        <br/>
                        <span class="version">v{__visage_version__}</span>
                    </td>
                    <td>{img_txt}</td>
                </tr>
            </table>
        </span>
        <b>Tip</b>: {random.choice(GUI_CONFIG.tips)}
        <br/>
        '''
        self.log_message(text, format_as_html=2)

    def print_all_tips(self):
        """Prints all available tips in the log window."""
        for tip in GUI_CONFIG['tips']:
            self.log_message(
                f'''<br/><span class="INFO">{' ' * self.timestamp_len} \u2022 {tip} </span>'''
            )

    def toggle_autoscroll(self, checked: bool):
        """Toggle the autoscroll.

        Args:
            checked (bool): True to toggle on, False otherwise
        """
        self._auto_scroll = bool(checked)

    def toggle_timestamps(self, checked: bool):
        """Toggle the timestamp.

        Args:
            checked (bool): True to toggle on, False otherwise
        """
        self._show_timestamps = bool(checked)
        self.show_all_messages()

    def show_all_messages(self):
        """Clear and reprint all log lines, thus refreshing toggles for
        timestamp, etc."""
        self.clear()
        for name, record in self.logged_lines:
            if name in self.get_all_checked():
                self.log_message(record, name != 'Errors')

    def get_all_checked(self):
        """Get all the checked items.

        Returns:
            list: List of checked items
        """
        res = ["INFO", "DEBUG", "ERROR", "WARNING", "CRITICAL"]
        for name, isChecked in self.tracked_loggers.items():
            if isChecked:
                res += [name]
        return res

    def set_level(self, level: int):
        """Set level on all handlers.

        Args:
            level (logging.level): The level of logging, eg.., logging.ERROR
        """
        self.set_window_title_level(level)
        self._level = level

    def set_window_title_level(self, level: int):
        """Set the window title level.

        Args:
            level (int): the level
        """
        self._level_name = logging.getLevelName(level).lower()
        if self._level_name not in ['']:
            self.qwidget.setWindowTitle(f'Log  (filter >= {self._level_name})')
        else:
            self.qwidget.setWindowTitle(f'Log (right click log for options)')
        return self._level_name

    def log_message_to(self, name, message: str, levelno: int):
        """Set where to log messages to.

        Args:
            name (str): The name
            message (str): True to send to records, False otherwise
            levelno (int): level number.
        """
        self.logged_lines.append((name, message))
        if levelno >= self._level:
            if name in self.get_all_checked():
                self.log_message(message, name != 'Errors')

    def log_message(self, message, format_as_html=True):
        """Do the actual logging.

        Args:
            message (str): The message to log.
            format_as_html (bool, int): True to format as HTML, False otherwise.  Defaults to True.
        """
        # set the write positon
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertBlock()  # add a new block, which makes a new line

        # add message
        if format_as_html:  # pylint: disable=singleton-comparison

            if not self.action_show_times.isChecked():
                # remove the timestamp -- assumes that this has been formatted
                # with pre tag by LogHandler_for_QTextLog
                res = message.split('<pre class="pre">', 1)
                if len(res) == 2:
                    message = re.sub("<time.*</time>", "", message)
                else:
                    pass  # print(f'Warning incorrect: {message}')

            cursor.insertHtml(message)

        elif format_as_html == 2:
            cursor.insertHtml(message)

        else:
            cursor.insertText(message, self.text_format)

        # make sure that the message is visible and scrolled ot
        # if self.action_scroll_auto.isChecked():
        self.moveCursor(QtGui.QTextCursor.End)
        self.moveCursor(QtGui.QTextCursor.StartOfLine)
        self.ensureCursorVisible()

    def remove_handlers(self):
        """Call on clsoe window to remove handlers from the logger."""
        self.handler.logger.remove()

    def add_logger(self, name: str, handler: logging.Handler):
        """Adds a logger to the widget.

            - adds `true bool`   to self.tracked_loggers for on/off to show
            - adds `the handler` in self.handlers
            - adds an action to the menu for the self.traceked_loggers

        For example, a logger handler is added with
            `gui.logger.addHandler(self._log_handler)`
        where
            `_log_handler is LogHandler_for_QTextLog`

        Args:
            name (string): Name of logger to be added
            handler (logging.Handler): Handler
        """
        self.handler = handler

        self.tracked_loggers[name] = True
        # Monkey patch add function
        func_name = f'toggle_{clean_name(name)}'

        def toggle_show_log(self2, val: bool):
            """Toggle the value of the.

            Args:
                self2 (QTextEdit): self
                val (bool): True or False
            """
            self2.tracked_loggers[name] = bool(val)

        monkey_patch(self, toggle_show_log, func_name=func_name)

        # Add action
        action = QAction(f' - {name}', self)
        action.setCheckable(True)
        action.setChecked(True)
        action.toggled.connect(getattr(self, func_name))

        self._actions[f'logger_{name}'] = action
        self.addAction(action)

        # style
        self.document().setDefaultStyleSheet(GUI_CONFIG.log_conf.style)

        # is this the first logger we added
        if len(self.tracked_loggers) == 1:
            self.set_window_title_level(self._level)


class LogHandler_for_QTextLog(logging.Handler):
    """Class to handle GUI logging. Handler instances dispatch logging events
    to specific destinations.

    This class extends the `logging.Handler` class.
    """

    def __init__(self, log_qtextedit: QTextEditLogger, logger=None):
        """
        Args:
            log_qtextedit (QTextEditLogger): Text edit logger.
        """
        super().__init__()
        self.log_qtextedit = log_qtextedit
        if logger is None:
            logger = self._init_log()
        self._logger = logger
        # self._logger.addHandler(self)
        # default set info.
        self.log_qtextedit.set_level(logging.getLevelName("INFO"))
        self._auto_add_filter()
        self.log_deque = deque(maxlen=10)
        self.log_qtextedit.setFont(QFont("JetBrains Mono", 13))

    def _auto_add_filter(self):
        """Add log level filter fram GUI config."""
        for level in GUI_CONFIG.log_conf.set_levels:
            name = level["name"]
            self.add_filter(name)

    def add_filter(self, name):
        """Add log level filter, default set true."""
        self.log_qtextedit.add_logger(name, self)

    @property
    def logger(self):
        """Get loggger instance."""
        return self._logger

    def _init_log(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        return logger

    @Slot(str, int)
    def emit_msg(self, message, log_type):
        """Do whatever it takes to actually log the specified logging record.
        Converts the characters '&', '<' and '>' in string s to HTML-safe
        sequences. Used to display text that might contain such characters in
        HTML.

        Args:
            record (LogRecord): The log recorder
        """
        # message = self.format(record)
        # get the log tag.

        html_record = html.escape(message)
        timestamp, level, level_no, levelname, info, = html_record.split(
            "|", maxsplit=4)
        if log_type == 2 and info in self.log_deque:
            return
        sep = '<sep class="sep"> | </sep>'
        level_no = int(level_no)
        # convert log to html format.
        html_log_message = f'<span class="{clean_name(levelname)}"><pre class="pre"><time class="time">{timestamp}</time>{sep}{level}{sep}{info}</pre></span>'

        if log_type == 1:
            self.log_deque.append(info)
            _, _, _, _, order_msg, = message.split("|", maxsplit=4)
            self.logger.log(level, order_msg)
        try:
            self.log_qtextedit.log_message_to(level, html_log_message,
                                              level_no)
        except RuntimeError as e:
            # trying to catch
            #  RuntimeError('wrapped C/C++ object of type QTextEditLogger has been deleted')
            print(f'Logger issue: {e}')
            self._logger.remove()

    def emit(self, record):
        """To fixed bug: NotImplementedError:
        emit must be implemented by Handler subclasses."""
        pass
