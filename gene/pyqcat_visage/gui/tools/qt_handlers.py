# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/07
# __author:       HanQing Shi

import logging

from PySide6 import QtCore
from loguru import logger


def qt_message_handler(mode, context, message):
    """The message handler is a function that prints out debug messages,
    warnings, critical and fatal error messages. The Qt library (debug mode)
    contains hundreds of warning messages that are printed when internal errors
    (usually invalid function arguments) occur. Qt built in release mode also
    contains such warnings unless QT_NO_WARNING_OUTPUT and/or
    QT_NO_DEBUG_OUTPUT have been set during compilation. If you implement your
    own message handler, you get total control of these messages.

    The default message handler prints the message to the standard output under X11
    or to the debugger under Windows. If it is a fatal message, the application
    aborts immediately.

    For more info, see https://doc.qt.io/qt-5/qtglobal.html#qInstallMessageHandler

    Args:
        mode (QtCore mode): the mode
        context (context): the context
        message (str): the message
    """

    if message.startswith(
            'QSocketNotifier: Multiple socket notifiers for same socket'):
        pass  # Caused by running %gui qt multiple times
    else:
        # if mode == QtCore.QtInfoMsg:
        #     mode = 'INFO'
        # elif mode == QtCore.QtWarningMsg:
        #     mode = 'WARNING'
        # elif mode == QtCore.QtCriticalMsg:
        #     mode = 'CRITICAL'
        # elif mode == QtCore.QtFatalMsg:
        #     mode = 'FATAL'
        # else:
        #     mode = 'DEBUG'
        # logger.log(
        #     getattr(logging, 'CRITICAL'), 'line: %d, func: %s(), file: %s' %
        #     (context.line, context.function, context.file) + '  %s: %s\n' %
        #     (mode, message))
        pass
