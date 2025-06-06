# -*- coding: utf-8 -*-

# This code is part of pyQCat Visage.
#
# Copyright (c) 2022-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/09/16
# __author:       SS Fang

"""
Define Error Exceptions.
"""

from pyQCat.errors import PyQCatError


class PyQCatVisageError(PyQCatError):
    """Base class for errors raised by Qiskit."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)
        self._code = 600

    def __str__(self):
        """Return the message."""
        return repr(self.message)

    @property
    def code(self):
        return self._code


class ExecuteProcessError(PyQCatVisageError):
    pass


class OperationError(PyQCatVisageError):
    """Raised when the user performs an incorrect interface operation."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self._code = 601


class LogicError(PyQCatVisageError):
    """Raised when code logic errors may occur, a detailed inspection is required."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message += "\n\nMaybe code logic errors, please provide feedback to us in the community!" \
                        "\n\nTrack Addr: https://document.qpanda.cn/docs/5xkGMEYmggFdLp3X" \
                        "\n\npyQCat Community: https://document.qpanda.cn/space/9030MdOBwNfe5oqw"
        self._code = 602


class CodeError(PyQCatVisageError):
    """Raised when code unknown errors may occur, a detailed inspection is required."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message += "\n\nMaybe code unknown errors, please provide feedback to us in the community!" \
                        "\n\nTips: Log window click 'Show timestamps' or 'Filter Debug' " \
                        "can show detail error information!" \
                        "\n\nTrack Addr: https://document.qpanda.cn/docs/5xkGMEYmggFdLp3X" \
                        "\n\npyQCat Community: https://document.qpanda.cn/space/9030MdOBwNfe5oqw"
        self._code = 603


class UserPermissionError(PyQCatVisageError):

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self._code = 604


class DataServerError(PyQCatVisageError):
    """Raised when received abnormal error response from courier."""

    def __init__(self, *message, code: int = 300):
        super().__init__(*message)
        self.message += "\n\nCourier service error, please provide feedback to us in the community!" \
                        "\n\nTrack Addr: https://document.qpanda.cn/docs/5xkGMEYmggFdLp3X" \
                        "\n\npyQCat Community: https://document.qpanda.cn/space/9030MdOBwNfe5oqw"
        self._code = code

    @classmethod
    def from_ret_data(cls, ret_data):
        if ret_data:

            if ret_data.get("code") == 200:
                return

            return cls(ret_data.get("msg"), code=ret_data.get("code"))

        else:

            return cls("No ret data!")


class EnvironmentalError(PyQCatVisageError):

    def __init__(self, *message):
        super().__init__(*message)
        self._code = 700


class DagError(EnvironmentalError):
    """Exception for dag."""


class DagInvalid(DagError):
    """Dag Invalid Error."""

    def __init__(self, graph_name: str, err_msg: str = None):
        """ Initialize the exception for invalid directed acyclic graphs.

        Args:
            graph_name (str): The name of the dag that is invalid.
            err_msg (str): Error message.
        """
        super().__init__(graph_name, err_msg)
        self._graph_name = graph_name
        self._err_msg = err_msg
        self._code = 701

    def __str__(self) -> str:
        """Return description."""
        return f"<{self.__class__.__name__}> " \
               f"Valid DiGraph {self._graph_name} Error: {self._err_msg}"


class ChipError(EnvironmentalError):
    def __init__(self, *message):
        super().__init__(*message)
        self._code = 702
