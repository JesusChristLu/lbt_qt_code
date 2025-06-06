# -*- coding: utf-8 -*-

# This code is part of pyQCat Visage.
#
# Copyright (c) 2022-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/14
# __author:       Lang Zhu

"""
execute log.
"""

from pyQCat.log import pyqlog, get_pubhandler, LogFormat
from pyQCat.invoker import DEFAULT_PORT

class Log:

    def __init__(self, name: str = "FLOW") -> None:
        self._log = pyqlog
        self._name = name
        self._init_zmq_log()

    def __getattr__(self, __name: str):
        return getattr(self._log, __name)

    def __call__(self, msg: str, name: str = None, *args, **kwargs):
        if not name:
            name = self._name
        self._log.log(name, msg)

    def _init_zmq_log(self):
        pubhandler = get_pubhandler(DEFAULT_PORT)
        self._log.add(pubhandler, format=LogFormat.pub, level=10)

    @property
    def name(self):
        return self._name


logger = Log()

__ALL__ = ["logger"]
