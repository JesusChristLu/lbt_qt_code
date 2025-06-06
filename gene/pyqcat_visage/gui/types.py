# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/28
# __author:       YangChao Zhao

from enum import Enum


class LayoutTypeEnum(Enum):
    """
    Port connection types:
    :py:mod:`NodeGraphQt.constants.PortTypeEnum`
    """
    #: Connection type for input ports.
    Linear = 'Linear'
    #: Connection type for output ports.
    Grid = 'Grid'


class ExpRunMode(Enum):
    un_save = 0
    save = 1


class DagRunMode(Enum):
    un_save = 0
    save_process = 1
    save_final = 2


class QsStatus(str, Enum):
    stopped = "stopped"
    running = "running"
    connecting = "connecting"
    start = "start"  # will start
    stop = "stop"  # will stop
    restart = "restart"
    disconnect = "disconnect"
    reconnect = "reconnect"


# class QtPixmap(Enum):
#     RUN = QPixmap(u":/run.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
#     CLOSE = QPixmap(u":/close.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
#     WARNING = QPixmap(u":/warning.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
#     NEW = QPixmap(u":/new.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
#     AUTO_PULL = QPixmap(u":/auto_pull.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
#     AUTO_PUSH = QPixmap(u":/auto_push.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
#     AUTO_PULL_PUSH = QPixmap(u":/auto_pull_push.png").scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio)
