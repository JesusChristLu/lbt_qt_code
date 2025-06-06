# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/02
# __author:       HanQing Shi
"""GUI public tools."""
import re
import traceback
import types
from functools import wraps

from PySide6.QtCore import Slot
from PySide6.QtGui import QColor
from loguru import logger

from pyQCat.errors import PyQCatError
from pyqcat_visage.exceptions import PyQCatVisageError, CodeError, OperationError


def clean_name(text: str):
    """Clean a string to a proper variable name in python.

    Args:
        text (str): Original string

    Returns:
        str: Corrected string

    .. code-block:: python

        clean_name('32v2 g #Gmw845h$W b53wi ')

    *Output*
        `'_32v2_g__Gmw845h_W_b53wi_'`

    See https://stackoverflow.com/questions/3303312/how-do-i-convert-a-string-to-a-valid-variable-name-in-python
    """
    return re.sub("\W|^(?=\d)", "_", text)


def monkey_patch(self, func, func_name=None):
    """Monkey patch a method into a class at runtime.

    Use descriptor protocol when adding method as an attribute.

    For a method on a class, when you do a.some_method, python actually does:
         a.some_method.__get__(a, type(a))

    So we're just reproducing that call sequence here explicitly.

    See: https://stackoverflow.com/questions/38485123/monkey-patching-bound-methods-in-python

    Args:
        func (function): function
        func_name (str): name of the function.  Defaults to None.
    """
    func_name = func_name or func.__name__
    setattr(self, func_name, func.__get__(self, self.__class__))


def slot_catch_exception(
    *args,
    catch=Exception,
    on_exception_emit=None,
    process_warning=False,
    process_reject=False,
    deprecated=False,
):
    """This is a decorator for Slots where an exception in user code is caught,
    printed and a optional qtSignal with signature qtSignal(Exception, str) is
    emitted when that happens.

    Based on:
        https://stackoverflow.com/questions/18740884/preventing-pyqt-to-silence-exceptions-occurring-in-slots

    Args:
        args (arguments):  any valid types for the Slot.
        catch (Exception):  Type of the exception to catch.  Defaults to Exception.
        on_exception_emit (str):  name of a qtSignal to be emitted.
        process_warning (bool):  Warn the user when an experimental process is active.
        process_reject (bool):  Reject the user when an experimental process is active.
        deprecated (bool):  Interface Abandonment.
    """

    if len(args) == 0 or isinstance(args[0], types.FunctionType):
        args = []

    @Slot(*args)
    def slot_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # pylint: disable=unused-argument
            try:
                if deprecated is True:
                    logger.warning(f"{func.__name__} is deprecated.")
                    return

                if process_warning:
                    if args[0].backend.run_state:
                        ok = args[0].ask_ok(
                            "Monster is running, this action may cause data "
                            "disruption, are you sure to continue?"
                        )
                        if not ok:
                            return

                if process_reject:
                    if args[0].backend.run_state:
                        raise OperationError(f"Monster is running, prohibit operation!")

                return func(*args, **kwargs)
            except catch as e:  # pylint: disable=invalid-name,broad-except

                if issubclass(type(e), PyQCatVisageError):
                    args[0].handle_error(e)
                elif issubclass(type(e), PyQCatError):
                    args[0].gui.backend.sub_proc_error_records = e
                else:
                    message = traceback.format_exc()
                    print(message)
                    error_message = (
                        "\n\nERROR in call by Visage GUI (see traceback above)\n"
                        + f"\n{' module   :':12s} {wrapper.__module__}"
                        + f"\n{' function :':12s} {wrapper.__qualname__}"
                        + f"\n{' err msg  :':12s} {e.__repr__()}"
                        + f"\n{' args; kws:':12s} {args}; {kwargs}"
                    )
                    logger.debug(message)
                    logger.error(error_message)

                    if hasattr(args[0], "handle_error"):
                        args[0].handle_error(CodeError("Code error!"))

                    return False

                if on_exception_emit is not None:
                    # args[0] is instance of bound signal
                    qt_signal = getattr(args[0], on_exception_emit)
                    qt_signal.emit(e, wrapper.__name__)

        return wrapper

    return slot_decorator


def slot_process_warning(*args):
    @Slot(*args)
    def slot_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # pylint: disable=unused-argument

            ok = True

            if args[0].backend.run_state:
                ok = args[0].ask_ok(
                    "Monster is running, this action can not get "
                    "the latest data, are you sure to continue?"
                )

            if ok:
                return func(*args)

        return wrapper

    return slot_decorator


def slot_process_reject(*args):
    @Slot(*args)
    def slot_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # pylint: disable=unused-argument

            ok = True

            if args[0].backend.run_state:
                ok = args[0].handler_ret_data(
                    {
                        "code": 800,
                        "msg": "Monster is running, disable updating database operation!",
                    }
                )
            else:
                return func(*args)

        return wrapper

    return slot_decorator


def blend_colors(color1: QColor, color2: QColor, r: float = 0.2, alpha=255) -> QColor:
    """Blend two qt colors together.

    Args:
        color1 (QColor): first color
        color2 (QColor): second color
        r (float): ratio
        alpha (int): alpha

    Returns:
        QColor: new color
    """
    color3 = QColor(
        color1.red() * (1 - r) + color2.red() * r,
        color1.green() * (1 - r) + color2.green() * r,
        color1.blue() * (1 - r) + color2.blue() * r,
        alpha,
    )
    return color3
