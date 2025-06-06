# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/14
# __author:       HanQing Shi
"""Item class"""
import math
from typing import TYPE_CHECKING

from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsTextItem,
    QApplication,
    QGraphicsItem,
)

from pyQCat.structures import QDict
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.gui.graphics.types import PortTypeEnum, ItemsZValue

if TYPE_CHECKING:
    from pyqcat_visage.gui.graphics.port_item import PortItem

pipe_conf = GUI_CONFIG.graphics_view.pipe
default_conf = GUI_CONFIG.graphics_view.theme_classic_dark


class PipeItem(QGraphicsPathItem):
    """Pipe item used for drawing node connections."""

    def __init__(self, parent=None, weight: int = 1, color_conf: QDict = None):
        super().__init__(parent)

        self.setFlag(QGraphicsPathItem.ItemIsSelectable)
        self.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        self.setZValue(ItemsZValue.Z_VAL_PIPE.value)

        self._start_port = None
        self._end_port = None

        self.start_pos = QtCore.QPointF()
        self.end_pos = QtCore.QPointF()

        self._do_highlight = False

        size = 6.0
        self._arrow = QtGui.QPolygonF()
        self._arrow.append(QtCore.QPointF(-size, size))
        self._arrow.append(QtCore.QPointF(0.0, -size * 1.5))
        self._arrow.append(QtCore.QPointF(size, size))

        self._weight = weight
        self.pipe_conf = color_conf if color_conf else default_conf
        self._text = PipeWeightItem(self._weight, self, color_conf=self.pipe_conf)

        # When pipe item in sub-graph, change the color.
        self.is_sub_graph = False
        self.init_theme()

    def init_theme(self, color_conf: QDict = None):
        if color_conf:
            self.pipe_conf = color_conf
        self._color = QtGui.QColor(*self.pipe_conf.pipe_color)
        self._text.init_theme(self.pipe_conf)

    def __repr__(self):
        string = f"Connect({self.start_port.node})to({self.end_port.node})"
        return string

    @property
    def text(self):
        """Get the text item."""
        return self._text

    @property
    def ports(self):
        """Return the pipe's two ports."""
        return self._start_port, self._end_port

    @property
    def start_port(self):
        """Return the port start position."""
        return self._start_port

    @property
    def end_port(self):
        """Return the port end position."""
        return self._end_port

    @start_port.setter
    def start_port(self, port: "PortItem"):
        """Add the port start position."""
        self._start_port = port
        self._start_port.connections.append(self)

    @end_port.setter
    def end_port(self, port: "PortItem"):
        """Add the port end position."""
        self._end_port = port
        self._end_port.connections.append(self)

    @property
    def weight(self):
        """Get the pipe's weight"""
        return self._weight

    def delete(self):
        """Delete the connection."""
        for port in self.ports:
            if port:
                port.connections.remove(self)
                if len(port.connections) == 1:
                    port.connections[0]._text.setVisible(False)
        self._text.setVisible(False)
        # self.scene().removeItem(self._text)
        self.scene().removeItem(self)

    def update_path(self):
        """Draw a smooth cubic curve from the start to end ports."""
        path = QtGui.QPainterPath()
        path.moveTo(self.start_pos)

        dx = self.end_pos.x() - self.start_pos.x()
        dy = self.end_pos.y() - self.start_pos.y()

        ctr1 = QtCore.QPointF(self.start_pos.x() + dx * 0.5, self.start_pos.y())
        ctr2 = QtCore.QPointF(self.start_pos.x() + dx * 0.5, self.start_pos.y() + dy)
        path.cubicTo(ctr1, ctr2, self.end_pos)

        self.setPath(path)

    def update_port_pos(self):
        """Update the ends of the connection

        Get the start and end ports and use them to set the start and end positions.
        """
        # swap start and end port positions.
        if self.start_port and self.start_port.port_type == PortTypeEnum.IN.value:
            temp = self.end_port
            self._end_port = self.start_port
            self._start_port = temp

        if self.start_port:
            self.start_pos = self.start_port.scenePos()

        if self.end_port:
            self.end_pos = self.end_port.scenePos()

        self.update_path()

        self._text.setPos(
            self.path().pointAtPercent(0.5).x() + 15,
            self.path().pointAtPercent(0.5).y() + 15,
        )
        self._text.setPlainText(str(self._weight))

    def display_weight(self):
        """Show edges weight on view."""
        self._text.setVisible(True)
        self._text.setDefaultTextColor(self._color)
        # self._text.setPlainText(str(self._weight))
        self._weight = self._text.weight

    def paint(self, painter, option=None, widget=None):
        """Override the default paint method depending on if the object is selected."""
        pen_width = 1.5
        cen_x = self.path().pointAtPercent(0.5).x()
        cen_y = self.path().pointAtPercent(0.5).y()
        loc_pt = self.path().pointAtPercent(0.49)
        tgt_pt = self.path().pointAtPercent(0.51)

        if self.isSelected() or self._do_highlight:
            self._color = QtGui.QColor(*self.pipe_conf.pipe_highlight_color)
        elif self.is_sub_graph:
            self._color = QtGui.QColor(*self.pipe_conf.pipe_sub_graph_color)
        else:
            self._color = QtGui.QColor(*self.pipe_conf.pipe_color)

        if any(
            [
                self.start_port and len(self.start_port.connections) > 1,
                self.end_port and len(self.end_port.connections) > 1,
            ]
        ):
            self.display_weight()

        pen = QtGui.QPen(self._color, pen_width)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
        painter.save()
        painter.setPen(pen)
        painter.drawPath(self.path())

        dist = math.hypot(tgt_pt.x() - cen_x, tgt_pt.y() - cen_y)

        painter.setBrush(QtGui.QBrush(self._color.darker(200)))

        if dist < 1.0:
            pen_width *= 0.5 + dist

        transform = QtGui.QTransform()
        transform.translate(cen_x, cen_y)
        radians = math.atan2(loc_pt.y() - tgt_pt.y(), loc_pt.x() - tgt_pt.x())
        degrees = math.degrees(radians) - 90
        transform.rotate(degrees)

        if dist < 1.0:
            transform.scale(dist, dist)
        painter.drawPolygon(transform.map(self._arrow))

        painter.restore()


class PipeWeightItem(QGraphicsTextItem):
    """PipeWeightItem class used to display and edit the name of a Pipe weight."""

    def __init__(self, weight: int, parent=None, color_conf=None):
        super().__init__(parent)
        # add connection to emit signal.
        self._connection = parent
        # GUI settings.
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        self.setFlags(
            QGraphicsTextItem.ItemIsFocusable | QGraphicsTextItem.ItemIsSelectable
        )
        # self._font = QtGui.QFont(pipe_conf.fonts, pointSize=11)
        self.color_conf = color_conf if color_conf else default_conf
        self.init_theme()
        self.setFont(self._font)
        self.document().setDocumentMargin(2)
        self.setTextWidth(40)
        self.weight = weight
        self.setVisible(False)

    def init_theme(self, color_conf: QDict = None):
        if color_conf:
            self.color_conf = color_conf
        self._font = QtGui.QFont(self.color_conf.fonts, pointSize=11)

    def setPlainText(self, text: str) -> None:
        """Overwrite to display edge's weight."""
        display_text = f"w={text}"
        super().setPlainText(display_text)

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        """Overwrite to handle focus event."""
        if event.reason() != Qt.FocusReason.PopupFocusReason:
            text = self.toPlainText().strip(" ")
            if text and text != "":
                self.weight = int(text.split("=")[1])
        super().focusInEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        """Overwrite to handle focus event."""
        # todo debug.
        if (
            event.reason() == Qt.FocusReason.MouseFocusReason
            and QApplication.mouseButtons() == Qt.MouseButton.RightButton
        ):
            self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        elif event.reason() == Qt.FocusReason.PopupFocusReason:
            pass
        else:
            self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        super().focusOutEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        """Overwrite to set focus event."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
            self.setFocus()
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        """Overwrite to set weight."""
        if event.key() in [Qt.Key.Key_Return, Qt.Key.Key_Enter]:
            self.clearFocus()
        elif event.key() == 61:  # Qt.Key_Plus ctrl +
            self.weight += 1
            self.scene().update_weight_signal.emit(
                self.weight,
                self._connection.start_port.node,
                self._connection.end_port.node,
            )
            self.setPlainText(str(self.weight))
        elif event.key() == Qt.Key.Key_Minus:  # ctrl -
            if self.weight > 0:
                self.weight -= 1
                self.scene().update_weight_signal.emit(
                    self.weight,
                    self._connection.start_port.node,
                    self._connection.end_port.node,
                )
                self.setPlainText(str(self.weight))
        else:
            super().keyPressEvent(event)
