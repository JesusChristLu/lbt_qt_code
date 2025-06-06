# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/14
# __author:       HanQing Shi
"""Item class."""
from typing import TYPE_CHECKING

from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import QGraphicsPathItem

from pyQCat.structures import QDict
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.gui.graphics.types import PortTypeEnum, PortPosEnum, ItemsZValue

if TYPE_CHECKING:
    from pyqcat_visage.gui.graphics.pipe_item import PipeItem

default_conf = GUI_CONFIG.graphics_view.theme_classic_dark


class PortItem(QGraphicsPathItem):
    """The ``PortItem`` class is used for connecting one node to another."""

    def __init__(self, name: str, port_type: str, parent=None, color_conf: QDict = None):
        super().__init__(parent)
        self.setFlag(QGraphicsPathItem.ItemSendsScenePositionChanges)
        self.setAcceptHoverEvents(True)
        self.setZValue(ItemsZValue.Z_VAL_PORT.value)

        # GUI settings.
        self._text_path = QtGui.QPainterPath()
        self._hovered = False

        self._name = name.replace("_", " ").title()
        self._port_type = port_type
        if self._port_type == PortTypeEnum.IN.value:
            self._pos = PortPosEnum.LEFT.value
        else:
            self._pos = PortPosEnum.RIGHT.value
        self._node = None
        self._connections = []
        self.port_conf = color_conf if color_conf else default_conf
        self.init_theme()

    def init_theme(self, color_conf: QDict = default_conf):
        if color_conf:
            self.port_conf = color_conf
        self._create_port_path()

    def __repr__(self):
        string = f"{self._port_type}_Port({self.node})"
        return string

    @property
    def name(self):
        """Return the port name."""
        return self._name

    @property
    def pos_type(self):
        """Return the port position type."""
        return self._pos

    @property
    def node(self):
        """Get the port attached node."""
        return self._node

    @property
    def port_type(self):
        """Returns the port type."""
        return self._port_type

    @property
    def connections(self):
        """Returns the port connections."""
        return self._connections

    def attach_to_node(self, node: str):
        """Attach port to a node."""
        self._node = node

    def swap_pos(self):
        """Change port positions to draw pretty node."""
        if self._pos == PortPosEnum.LEFT.value:
            self._pos = PortPosEnum.RIGHT.value
        else:
            self._pos = PortPosEnum.LEFT.value

    def clear_connection(self, connection: "PipeItem"):
        """Clear port connections."""
        self._connections.remove(connection)
        connection.delete()

    def can_connect_to(self, port: "PortItem"):
        """Check the destination is valid."""
        if not port:
            return False
        elif port.node == self.node:
            return False
        elif self.port_type == port.port_type:
            return False
        else:
            # todo acyclic ?
            pass
        for connection in self.connections:
            if port in connection.ports:
                return False
        return True

    def set_name(self):
        """Setting port name and add text path."""
        font = QtGui.QFont()
        font_metrics = QtGui.QFontMetrics(font)
        text_height = font_metrics.height()
        text_width = font_metrics.horizontalAdvance(self._name)

        if self._pos == PortPosEnum.RIGHT.value:
            x = -self.port_conf.port_radius - self.port_conf.port_margin - text_width
        elif self._pos == PortPosEnum.LEFT.value:
            x = self.port_conf.port_radius + self.port_conf.port_margin
        else:
            raise ValueError("Port position error.")

        y = text_height / 4

        self._text_path.addText(x, y, font, self._name)

    def hoverEnterEvent(self, event):
        """Overload."""
        self._hovered = True
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Overload."""
        self._hovered = False
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        """Overload."""
        if change == QGraphicsPathItem.ItemScenePositionHasChanged:
            for connection in self._connections:
                connection.update_port_pos()
        return super().itemChange(change, value)

    def paint(self, painter, option=None, widget=None):
        """Overload."""
        rect_w = self.port_conf.port_size / 1.5
        rect_h = self.port_conf.port_size / 1.5
        rect_x = self.boundingRect().center().x() - (rect_w / 2)
        rect_y = self.boundingRect().center().y() - (rect_h / 2)
        port_rect = QtCore.QRectF(rect_x, rect_y, rect_w, rect_h)

        if self._hovered:
            color = QtGui.QColor(*self.port_conf.port_hover_color)
            border_color = QtGui.QColor(*self.port_conf.port_hover_border_color)
        elif self._connections:
            color = QtGui.QColor(*self.port_conf.port_active_color)
            border_color = QtGui.QColor(*self.port_conf.port_active_border_color)
        else:
            color = QtGui.QColor(*self.port_conf.port_color)
            border_color = QtGui.QColor(*self.port_conf.port_border_color)

        pen = QtGui.QPen(border_color, 1.8)
        painter.setPen(pen)
        painter.setBrush(color)
        painter.drawEllipse(port_rect)

        if self._connections and not self._hovered:
            painter.setBrush(border_color)
            w = port_rect.width() / 2.5
            h = port_rect.height() / 2.5
            rect = QtCore.QRectF(
                port_rect.center().x() - w / 2, port_rect.center().y() - h / 2, w, h
            )
            border_color = QtGui.QColor(*self.port_conf.port_border_color)
            pen = QtGui.QPen(border_color, 1.6)
            painter.setPen(pen)
            painter.setBrush(border_color)
            painter.drawEllipse(rect)
        elif self._hovered:
            if len(self._connections) > 1:
                pen = QtGui.QPen(border_color, 1.4)
                painter.setPen(pen)
                painter.setBrush(color)
                w = port_rect.width() / 1.8
                h = port_rect.height() / 1.8
            else:
                painter.setBrush(border_color)
                w = port_rect.width() / 3.5
                h = port_rect.height() / 3.5
            rect = QtCore.QRectF(
                port_rect.center().x() - w / 2, port_rect.center().y() - h / 2, w, h
            )
            painter.drawEllipse(rect)

        # draw text.
        text_color = QtGui.QColor(*self.port_conf.port_text_color)
        pen = QtGui.QPen(text_color)
        painter.setPen(pen)
        painter.setBrush(text_color)
        painter.drawPath(self._text_path)

    def _create_port_path(self):
        """Draw ellipse as a port."""
        path = QtGui.QPainterPath()
        path.addEllipse(
            -self.port_conf.port_radius,
            -self.port_conf.port_radius,
            2 * self.port_conf.port_radius,
            2 * self.port_conf.port_radius
        )
        self.setPath(path)
