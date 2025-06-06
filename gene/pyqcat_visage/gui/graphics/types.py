# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/15
# __author:       HanQing Shi

from enum import Enum


class PortTypeEnum(Enum):
    """
    Port connection types:
    :py:mod:`NodeGraphQt.constants.PortTypeEnum`
    """

    #: Connection type for input ports.
    IN = "in"
    #: Connection type for output ports.
    OUT = "out"


class PortPosEnum(Enum):
    """
    Port position types:
    """

    LEFT = "left"
    RIGHT = "right"


class ItemsZValue(Enum):
    """Z Value enum."""
    Z_VAL_PIPE_TEXT = -2
    Z_VAL_PIPE = -1
    Z_VAL_NODE = 1
    Z_VAL_PORT = 2
    Z_VAL_NODE_WIDGET = 3


class NodeStatus(str, Enum):
    """Node status enum."""

    SUCCESS = "success"
    RUNNING = "running"
    FAILED = "failed"
    STATIC = "static"
    SELECTED = "selected"


class NodeRole(str, Enum):
    """Node role enum."""

    ROOT = "(root node)"
    TAIL = "(tail node)"
    STD = "(standard)"
