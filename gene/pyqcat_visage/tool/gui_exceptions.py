# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/31
# __author:       HanQing Shi
"""Custom Exceptions."""

class VisageGUIExceptions(Exception):
    """Custom Exception super-class. Every Exception raised by pyqcat-visage
    should inherit this. Adds the pyqcat-visage prefix.

    Args:
        message (str): String describing the error raised from qiskit-metal
    """

    # pylint: disable=super-init-not-called
    def __init__(self, message: str) -> None:
        prefix = "pyQCat Visage - "
        self.args = [prefix + message]


class LibraryGUIException(VisageGUIExceptions):
    """Custom Exception for the QLibrary GUI feature
    """