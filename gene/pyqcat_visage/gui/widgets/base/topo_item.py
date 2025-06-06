# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/03/23
# __author:       xw

from PySide6.QtCore import Qt
from PySide6.QtGui import QPen, QPainter, QColor

from pyQCat.structures import QDict
from pyqcat_visage.config import GUI_CONFIG
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem

default_conf = GUI_CONFIG.graphics_view.theme_classic_dark


class CustomRectItem(QGraphicsRectItem):
    def __init__(self, x, y, w, h, color_conf=None):
        super().__init__(x, y, w, h)
        self.setFlags(QGraphicsRectItem.ItemIsSelectable | QGraphicsRectItem.ItemIsMovable)
        self.color_conf = color_conf if color_conf else default_conf
        self.init_theme()
        self.status = 0

    def init_theme(self, color_conf: QDict = None):
        if color_conf:
            self.color_conf = color_conf

    def paint(self, painter, option, widget=None):
        painter.setBrush(QColor(*self.color_conf.coupler_color))
        if self.isSelected():
            painter.setPen(QPen(QColor(*self.color_conf.pen_color), 2, Qt.SolidLine))
        else:
            if self.status == 1:
                painter.setPen(QPen(QColor(*self.color_conf.env_color), 2, Qt.SolidLine))
            else:
                painter.setPen(Qt.NoPen)

        painter.drawRect(self.rect())


class CustomEllipseItem(QGraphicsEllipseItem):
    def __init__(self, x, y, w, h, color_conf=None):
        super().__init__(x, y, w, h)
        self.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
        self.color_conf = color_conf if color_conf else default_conf
        self.init_theme()
        self.status = 0

    def init_theme(self, color_conf: QDict = None):
        if color_conf:
            self.color_conf = color_conf

    def paint(self, painter, option, widget=None):
        if self.isSelected():
            pen_color = QColor(*self.color_conf.pen_color)

        elif self.status == 1:
            pen_color = QColor(*self.color_conf.env_color)
        else:
            pen_color = Qt.GlobalColor.gray
        pen = QPen(pen_color, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(pen)
        painter.setBrush(QColor(*self.color_conf.qubit_color))
        painter.drawEllipse(self.rect())
