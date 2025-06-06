# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/14
# __author:       HanQing Shi
"""View GUI."""
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QPainter

from pyQCat.structures import QDict
from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.gui.graphics.node_item import NodeItem
from pyqcat_visage.gui.graphics.types import NodeRole

default_conf = GUI_CONFIG.graphics_view.theme_classic_dark


class View(QtWidgets.QGraphicsView):
    """Class representing NodeEditor's `Graphics View`"""

    request_node = QtCore.Signal(str)
    select_subgraph = QtCore.Signal(str, list)

    def __init__(self, parent, color_conf: QDict = None):
        super(View, self).__init__(parent)
        self.setRenderHints(
            QPainter.Antialiasing
            | QPainter.TextAntialiasing
            | QPainter.SmoothPixmapTransform
        )
        self._manipulationMode = 0
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontAdjustForAntialiasing)

        self._current_scale = 1
        self._pan = False
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._num_scheduled_scaling = 0

        self.animations = None

        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        # Record the root node and tail node if user set.
        self.root_node = None
        self.tail_nodes = []

        # status to judge signal emit successfully or not.
        self._is_valid_subgraph = True
        self.color_conf = color_conf if color_conf else default_conf
        self.init_theme()

    @property
    def scale_factor(self):
        return self.transform().m11()

    @property
    def current_scale(self):
        return self._current_scale

    @current_scale.setter
    def current_scale(self, value):
        self._current_scale = value

    @property
    def num_scheduled_scaling(self):
        return self._num_scheduled_scaling

    @num_scheduled_scaling.setter
    def num_scheduled_scaling(self, value):
        self._num_scheduled_scaling = value

    def init_theme(self, color_conf: QDict = None):
        if color_conf:
            self.color_conf = color_conf
        self._background_color = QtGui.QColor(*self.color_conf.color)

        self._grid_pen_s = QtGui.QPen(QtGui.QColor(*self.color_conf.grid_pen_s), 0.5)
        self._grid_pen_l = QtGui.QPen(QtGui.QColor(*self.color_conf.grid_pen_l), 1.0)

        self._grid_size_fine = self.color_conf.grid_size_fine
        self._grid_size_course = self.color_conf.grid_size_course

        self._mouse_wheel_zoom_rate = self.color_conf.mouse_wheel_zoom_rate

    def wheelEvent(self, event):
        """overridden Qts ``wheelEvent``. This handles zooming"""
        # sometimes you can trigger the wheel when panning, so we disable when panning
        if self._pan:
            return

        num_degrees = event.angleDelta().y() / 8.0
        num_steps = num_degrees / 5.0
        self._num_scheduled_scaling += num_steps

        # If the user moved the wheel another direction, we reset previously scheduled scalings
        if self._num_scheduled_scaling * num_steps < 0:
            self._num_scheduled_scaling = num_steps

        self.animations = QtCore.QTimeLine(350)
        self.animations.setUpdateInterval(20)

        self.animations.valueChanged.connect(self.scaling_time)
        self.animations.finished.connect(self.animations_finished)
        self.animations.start()

    def scaling_time(self, x=None):
        factor = 1.0 + self._num_scheduled_scaling / 300.0
        target_scale = self._current_scale * factor
        if 0.01 <= target_scale <= 5:
            self._current_scale = target_scale
            self.scale(factor, factor)

    def animations_finished(self):
        if self._num_scheduled_scaling > 0:
            self._num_scheduled_scaling -= 1
        else:
            self._num_scheduled_scaling += 1

    def drawBackground(self, painter, rect):
        """Draw background scene grid"""
        painter.fillRect(rect, self._background_color)

        left = int(rect.left()) - (int(rect.left()) % self._grid_size_fine)
        top = int(rect.top()) - (int(rect.top()) % self._grid_size_fine)

        # Draw horizontal fine lines
        grid_lines = []
        painter.setPen(self._grid_pen_s)
        y = float(top)
        while y < float(rect.bottom()):
            grid_lines.append(QtCore.QLineF(rect.left(), y, rect.right(), y))
            y += self._grid_size_fine
        painter.drawLines(grid_lines)

        # Draw vertical fine lines
        grid_lines = []
        painter.setPen(self._grid_pen_s)
        x = float(left)
        while x < float(rect.right()):
            grid_lines.append(QtCore.QLineF(x, rect.top(), x, rect.bottom()))
            x += self._grid_size_fine
        painter.drawLines(grid_lines)

        # Draw thick grid
        left = int(rect.left()) - (int(rect.left()) % self._grid_size_course)
        top = int(rect.top()) - (int(rect.top()) % self._grid_size_course)

        # Draw vertical thick lines
        grid_lines = []
        painter.setPen(self._grid_pen_l)
        x = left
        while x < rect.right():
            grid_lines.append(QtCore.QLineF(x, rect.top(), x, rect.bottom()))
            x += self._grid_size_course
        painter.drawLines(grid_lines)

        # Draw horizontal thick lines
        grid_lines = []
        painter.setPen(self._grid_pen_l)
        y = top
        while y < rect.bottom():
            grid_lines.append(QtCore.QLineF(rect.left(), y, rect.right(), y))
            y += self._grid_size_course
        painter.drawLines(grid_lines)

        return super().drawBackground(painter, rect)

    def judge_subgraph(self, is_valid: bool):
        """Used to determine if the generated subgraph is valid.

        Args:
            is_valid (bool): TRUE means valid, FALSE means not invalid.
        """
        self._is_valid_subgraph = is_valid

    def _node_clicked_menu(self, item: NodeItem, pos: "QtCore.QPoint"):
        """Show node role set menu when right-click the menu."""
        menu = QtWidgets.QMenu(self)

        set_root_action = QtGui.QAction("Set as root node", self)
        set_tail_action = QtGui.QAction("Set as tail node", self)
        remove_action = QtGui.QAction("Recover role", self)

        menu.addAction(set_root_action)
        menu.addAction(set_tail_action)
        menu.addAction(remove_action)

        action = menu.exec_(self.mapToGlobal(pos))

        if action == set_root_action:
            if self.root_node is not None:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Add Root node failed",
                    "Please remove existed root node before add.",
                )
            else:
                self.root_node = item.identifier
                self.select_subgraph.emit(self.root_node, self.tail_nodes)
                if self._is_valid_subgraph:
                    if item.node_role == NodeRole.TAIL.value:
                        self.tail_nodes.remove(item.identifier)
                    item.node_role = NodeRole.ROOT.value
                    item.draw_type()
                else:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Add Root node failed",
                        "The subgraph you set is invalid. Please check your subgraph structure.",
                    )
                    self.root_node = None

        if action == set_tail_action:
            if item.identifier in self.tail_nodes:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Add Tail node failed",
                    "Please remove existed tail node before add.",
                )
            else:
                self.tail_nodes.append(item.identifier)
                self.select_subgraph.emit(self.root_node, self.tail_nodes)
                if self._is_valid_subgraph:
                    if item.node_role == NodeRole.ROOT.value:
                        self.root_node = None
                    item.node_role = NodeRole.TAIL.value
                    item.draw_type()
                else:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Add Tail node failed",
                        "The subgraph you set is invalid. Please check your subgraph structure.",
                    )
                    self.tail_nodes.remove(item.identifier)

        if action == remove_action:
            if item.node_role == NodeRole.TAIL.value:
                self.tail_nodes.remove(item.identifier)
            elif item.node_role == NodeRole.ROOT.value:
                self.root_node = None
            self.select_subgraph.emit(self.root_node, self.tail_nodes)
            item.node_role = NodeRole.STD.value
            item.draw_type()

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        drop_node_name = e.mimeData().text()
        self.request_node.emit(drop_node_name)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._pan = True
            self._pan_start_x = event.x()
            self._pan_start_y = event.y()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._pan = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

        return super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan:
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - (event.x() - self._pan_start_x)
            )

            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - (event.y() - self._pan_start_y)
            )

            self._pan_start_x = event.x()
            self._pan_start_y = event.y()

        return super().mouseMoveEvent(event)

    def contextMenuEvent(self, event):
        """Overwrite context menu."""
        cursor = QtGui.QCursor()
        pos = self.mapFromGlobal(cursor.pos())
        item = self.itemAt(event.pos())

        if item and isinstance(item, NodeItem):
            self._node_clicked_menu(item, pos)
