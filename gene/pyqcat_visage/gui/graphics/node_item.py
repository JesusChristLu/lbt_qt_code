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
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsPathItem
from pyQCat.structures import QDict

from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.gui.graphics.port_item import PortItem
from pyqcat_visage.gui.graphics.types import (
    ItemsZValue,
    PortPosEnum,
    NodeStatus,
    NodeRole,
)

if TYPE_CHECKING:
    from pyqcat_visage.backend.experiment import VisageExperiment
    from pyqcat_visage.gui.main_window import VisageGUI

default_conf = GUI_CONFIG.graphics_view.theme_classic_dark


class NodeItem(QGraphicsPathItem):
    """NodeItem class used to display information and provided port to edit."""

    def __init__(self, gui: "VisageGUI", name: str, id_: str = None, parent=None, color_conf: QDict = None):
        super().__init__(parent)
        self.setFlags(
            QGraphicsPathItem.ItemIsMovable | QGraphicsPathItem.ItemIsSelectable
        )

        self.setZValue(ItemsZValue.Z_VAL_NODE.value)
        self._gui = gui
        self._name = name
        self._id = id_ or hex(id(self))
        self._pos = [0.0, 0.0]

        self._node_role = NodeRole.STD.value

        # The Width of the node
        self._width = 0
        # the height of the node
        self._height = 0
        # A list of ports
        self._ports = []

        # The path for the title
        self.title_path = QtGui.QPainterPath()
        # The path for the type
        self.type_path = QtGui.QPainterPath()
        # a bunch of other stuff
        self.misc_path = QtGui.QPainterPath()
        self.node_conf = color_conf if color_conf else default_conf
        self.init_theme()

        # The instance of class VisageExperiment.
        self._node = None

        # add node status, when DAG status changed the node status change.
        self.node_status = NodeStatus.STATIC.value

    def init_theme(self, color_conf: QDict = None):
        if color_conf:
            self.node_conf = color_conf

        # horizontal margin
        self.horizontal_margin = self.node_conf.node_h_margin
        # vertical margin
        self.vertical_margin = self.node_conf.node_v_margin

        # initialize the path configuration.
        # The fonts what will be used
        self._title_font = QtGui.QFont(self.node_conf.fonts, pointSize=12)
        self._title_type_font = QtGui.QFont(self.node_conf.fonts, pointSize=10)
        self._port_font = QtGui.QFont(self.node_conf.fonts)

        # Get the dimensions of the title and type.
        self._title_dim = {
            "w": QtGui.QFontMetrics(self._title_font).horizontalAdvance(self._name),
            "h": QtGui.QFontMetrics(self._title_font).height(),
        }

        self._title_type_dim = {
            "w": QtGui.QFontMetrics(self._title_type_font).horizontalAdvance(
                self._node_role
            ),
            "h": QtGui.QFontMetrics(self._title_type_font).height(),
        }

        self._port_dim = {
            "w": 0,  # depends on in or out port.
            "h": QtGui.QFontMetrics(self._port_font).height(),
        }
        # node item color
        self.node_color = QtGui.QColor(*self.node_conf.node_color)

        # node item bound color map.
        # when node status changed, trigger the color pen change.
        self._node_status_color_pen = {
            NodeStatus.STATIC.value: self.node_color.lighter(),
            NodeStatus.RUNNING.value: QtGui.QPen(
                QtGui.QColor(*self.node_conf.node_running),
                2,
            ),
            NodeStatus.SUCCESS.value: QtGui.QPen(
                QtGui.QColor(*self.node_conf.node_success),
                3,
            ),
            NodeStatus.FAILED.value: QtGui.QPen(
                QtGui.QColor(*self.node_conf.node_failed),
                3,
            ),
            NodeStatus.SELECTED.value: QtGui.QPen(
                QtGui.QColor(*self.node_conf.node_selected),
                2,
            ),
        }
        for port in self._ports:
            port.init_theme(color_conf)

    def __repr__(self):
        return self.identifier

    @property
    def id(self):
        """Return the node memory id."""
        return self._id

    @property
    def identifier(self):
        """Return the node identifier."""
        return f"{self.name}_{self.id}"

    @property
    def name(self):
        """Return the node name."""
        return self._name

    @property
    def xy_pos(self):
        """Get the item scene position."""
        return [float(self.scenePos().x()), float(self.scenePos().y())]

    @xy_pos.setter
    def xy_pos(self, pos=None):
        """
        Set the item scene position.
        ("node.pos" conflicted with "QGraphicsItem.pos()"
        so it was refactored to "xy_pos".)

        Args:
            pos (list[float]): x, y scene position.
        """
        pos = pos or [0.0, 0.0]
        self.setPos(pos[0], pos[1])

    @property
    def input_port(self):
        """Get the in put port."""
        return self._ports[0]

    @property
    def output_port(self):
        """Get the output port."""
        return self._ports[1]

    @property
    def node(self):
        """Get the VisageExperiment object."""
        return self._node

    @property
    def node_role(self):
        """Get the current node role."""
        return self._node_role

    @node_role.setter
    def node_role(self, role: str):
        """Set the node role

        Args:
            role (str): The node role value which must be in types.NodeRole.
        """
        if role in NodeRole.__members__.values():
            self._node_role = role
            if self._node is not None:
                self._node.role = self._node_role

    def wrapper_node(self, exp: "VisageExperiment"):
        """Wrapper node as VisageExperiment object."""
        self._node = exp
        self._node.gid = self.id
        self._node.name = self._name
        self._node.tab = "dags"
        self._node.location = self.xy_pos
        self._node.role = self._node_role

    def add_port(
        self, name: str, port_type: str, pos_type: str = None, swap_pos: bool = False
    ):
        """Add port to the node.

        Args:
            name (str): Set port name. 'In' or 'Out'
            port_type (str): 'in' or 'out'.
            pos_type (str): left or right.
            swap_pos (bool): True or False, to change the port position.
        """
        port = PortItem(name=name, port_type=port_type, parent=self, color_conf=self.node_conf)
        if pos_type:
            port._pos = pos_type
        if swap_pos:
            port.swap_pos()
        port.set_name()
        port.attach_to_node(node=self.identifier)
        self._ports.append(port)

    def select_connections(self, value):
        """When select a connection, change the state."""
        for port in self._ports:
            for connection in port.connections:
                connection._do_highlight = value
                connection.update_port_pos()

    def delete(self):
        """Delete the connection.
        Remove any found connections ports by calling :any:`Port.remove_connection`.
        After connections have been removed set the stored :any:`Port` to None.
        Lastly call :any:`QGraphicsScene.removeItem` on the scene to remove this widget.
        """
        to_delete = []

        for port in self._ports:
            for connection in port.connections:
                to_delete.append(connection)

        for connection in to_delete:
            connection.delete()

        self.scene().removeItem(self)

        return to_delete

    def layout_node(self):
        """Layout the node."""
        width = 0
        height = 0
        # Get the max width
        for dim in [self._title_dim["w"], self._title_type_dim["w"]]:
            if dim > width:
                width = dim

        # Add both the title and type height together for the total height
        for dim in [self._title_dim["h"], self._title_type_dim["h"]]:
            height += dim

        # Add the height for each of the ports
        for port in self._ports:
            self._port_dim["w"] = QtGui.QFontMetrics(self._port_font).horizontalAdvance(
                port.name
            )
            if self._port_dim["w"] > width:
                width = self._port_dim["w"]
            height += self._port_dim["h"]

        # Add the margin to the total_width
        width += self.horizontal_margin
        height += self.vertical_margin

        return width, height

    def draw_type(self):
        """Draw the node type."""
        if not self.type_path.isEmpty():
            self.type_path.clear()
        self.type_path.addText(
            -self._title_type_dim["w"] / 2,
            (-self._height / 2) + self._title_dim["h"] + self._title_type_dim["h"],
            self._title_type_font,
            self._node_role,
        )

    def draw_title(self):
        """Draw the node title."""
        if not self.title_path.isEmpty():
            self.title_path.clear()
        self.title_path.addText(
            -self._title_dim["w"] / 2,
            (-self._height / 2) + self._title_dim["h"],
            self._title_font,
            self._name,
        )

    def draw_port(self):
        """Draw the port on node."""
        y = (
            (-self._height / 2)
            + self._title_dim["h"]
            + self._title_type_dim["h"]
            + self._port_dim["h"]
        )
        for port in self._ports:
            if port.pos_type == PortPosEnum.RIGHT.value:
                port.setPos(self._width / 2 - 5, y)
            else:
                port.setPos(-self._width / 2 + 5, y)
            y += self._port_dim["h"]

    def build(self):
        """Build the node"""
        path = QtGui.QPainterPath()  # The main path
        self._width, self._height = self.layout_node()

        # Draw the background rectangle, this is node bound.
        path.addRoundedRect(
            -self._width / 2, -self._height / 2, self._width, self._height, 5, 5
        )

        # Draw the title
        self.draw_title()

        # Draw the type
        self.draw_type()

        # Draw the port
        self.draw_port()

        # Set the node bound which is a rectangle.
        self.setPath(path)

    def mouseDoubleClickEvent(self, event):
        """Overload the mouse double click event to swap node port positions."""
        if not any([port.connections for port in self._ports]):
            port_dict = QDict()
            for port in self._ports:
                port_dict[port.name] = (port.port_type, port.pos_type)
                port.scene().removeItem(port)
            self._ports.clear()
            for port_name, port_info in port_dict.items():
                port_type, pos_type = port_info
                self.add_port(port_name, port_type, pos_type, swap_pos=True)

            self.build()

            # fixed bug, swap port positions forget update DAG's node positions.
            temp = self.node.port_pos[0]
            self.node.port_pos[0] = self.node.port_pos[1]
            self.node.port_pos[1] = temp

    def paint(self, painter, option=None, widget=None):
        """Overwrite the paint."""
        if self.isSelected():
            painter.setPen(self._node_status_color_pen[NodeStatus.SELECTED.value])
            painter.setBrush(self.node_color)
        else:
            # Set pen color which depends on node status to draw node bound.
            painter.setPen(self._node_status_color_pen[self.node_status])
            painter.setBrush(self.node_color)

        painter.drawPath(self.path())
        # Reset pen color to draw node title and node type.
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*self.node_conf.font_color))

        painter.drawPath(self.title_path)
        painter.drawPath(self.type_path)
        painter.drawPath(self.misc_path)

    def mousePressEvent(self, event):
        """Overwrite the mouse press event to emit signal."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._gui.ui.experimentLibrary.choose_exp_signal.emit(self._node)
            super().mousePressEvent(event)
