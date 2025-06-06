# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/21
# __author:       YangChao Zhao

import math

from PySide6.QtCore import Qt, QRectF, QPropertyAnimation
from PySide6.QtGui import QFont, QPen, QConicalGradient, QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QWidget, QGridLayout


def animation(parent, type=b"windowOpacity", from_value=0, to_value=1, ms=1000, connect=None):
    anim = QPropertyAnimation(parent, type)
    anim.setDuration(ms)
    anim.setStartValue(from_value)
    anim.setEndValue(to_value)
    if connect:
        anim.finished.connect(connect)
    anim.start()
    return anim


class RoundProgress(QWidget):
    m_waterOffset = 0.05
    m_offset = 50
    bg_color = QColor(255, 0, 0)
    f_size = 10

    def __init__(self, t, parent=None):
        super(RoundProgress, self).__init__(parent)
        self.resize(*t)
        self.size = t
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # 去边框
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 设置窗口背景透明
        self.percent = 0
        self.pen = QPen()
        gradient = QConicalGradient(50, 50, 91)
        gradient.setColorAt(0, QColor(75, 190, 75))
        gradient.setColorAt(1, QColor(75, 190, 75))
        # gradient.setColorAt(0.5, QColor(255, 201 ,14))
        self.pen.setBrush(gradient)  # 设置画刷渐变效果
        self.pen.setWidth(8)
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.font = QFont()
        self.font.setFamily("Share-TechMono")  # Share-TechMono
        self.font.setPointSize(self.size[0] // 4)

    def paintEvent(self, event):
        # width, height = self.size().width(), self.size().height()
        width, height = self.size
        rect = QRectF(self.f_size, self.f_size, width - self.f_size * 2, height - self.f_size * 2)
        painter = QPainter(self)
        rotateAngle = 360 * self.percent / 100
        # 绘制准备工作，启用反锯齿
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(self.pen)

        painter.drawArc(rect, (90 - 0) * 16, -rotateAngle * 16)  # 画圆环
        painter.setFont(self.font)
        painter.setPen(QColor(153 - 1.53 * self.percent,
                              217 - 0.55 * self.percent,
                              234 - 0.02 * self.percent))  # r:255, g:201 - 10/100 * percent, b: 14-4 /100*percent 当前渐变
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "%d%%" % self.percent)  # 显示进度条当前进度

        # bug: paintEvent can be invoked by repaint() or update(), cause drop-dead halt
        # self.update()

    def update_percent(self, p):
        self.percent = p
        self.update()


class WaterProgress(QWidget):
    f_size = 10

    def __init__(self, t, parent=None):
        super(WaterProgress, self).__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # 去边框
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 设置窗口背景透明
        self.resize(*t)
        self.size = t
        self.layout = QGridLayout(self)

        # 设置进度条颜色
        self.bg_color = QColor("#4040FF")
        self.m_waterOffset = 0.00001
        self.m_offset = 50
        self.m_borderwidth = 10
        self.percent = 0

    def paintEvent(self, event):
        painter = QPainter()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.begin(self)
        painter.setPen(Qt.PenStyle.NoPen)
        # 获取窗口的宽度和高度
        # size = self.size()
        # width, height = size.width(), size.height()
        width, height = self.size
        percentage = 1 - self.percent / 100
        # 水波走向：正弦函数 y = A(wx+l) + k
        # w 表示 周期，值越大密度越大
        w = 2 * math.pi / (width)
        # A 表示振幅 ，理解为水波的上下振幅
        # A = height * self.m_waterOffset
        A = self.m_waterOffset
        # k 表示 y 的偏移量，可理解为进度
        k = height * percentage

        water1 = QPainterPath()
        water2 = QPainterPath()
        # 起始点
        water1.moveTo(5, height)
        water2.moveTo(5, height)
        self.m_offset += 0.6

        if (self.m_offset > (width / 2)):
            self.m_offset = 0
        i = 5

        rect = QRectF(self.f_size, self.f_size, width - self.f_size * 2, height - self.f_size * 2)
        while (i < width - 5):
            waterY1 = A * math.sin(w * i + self.m_offset) + k
            waterY2 = A * math.sin(w * i + self.m_offset + width / 2 * w) + k

            water1.lineTo(i, waterY1)
            water2.lineTo(i, waterY2)
            i += 1

        water1.lineTo(width - 5, height)
        water2.lineTo(width - 5, height)

        totalpath = QPainterPath()
        # totalpath.addRect(rect)
        # painter.setBrush(Qt.gray)
        painter.drawRect(self.rect())
        painter.save()
        totalpath.addEllipse(rect)
        totalpath.intersected(water1)
        painter.setPen(Qt.PenStyle.NoPen)

        # 设置水波的透明度
        watercolor1 = QColor(self.bg_color)
        watercolor1.setAlpha(100)
        watercolor2 = QColor(self.bg_color)
        watercolor2.setAlpha(150)
        path = totalpath.intersected(water1)
        painter.setBrush(watercolor1)
        painter.drawPath(path)

        path = totalpath.intersected(water2)
        painter.setBrush(watercolor2)
        painter.drawPath(path)
        painter.restore()
        painter.end()
        # self.update()

    def update_percent(self, p):
        self.percent = p
        if self.m_waterOffset < 0.05:
            self.m_waterOffset += 0.001
        return p


class Progress(QWidget):
    percent = 0

    def __init__(self, text="", parent=None):
        QWidget.__init__(self, parent)

        Font = QFont()
        Font.setFamily("Consolas")
        Font.setPointSize(12)
        self.setFont(Font)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # 去边框
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 设置窗口背景透明
        width, height = 100, 100
        self.resize(width, height)

        # self.verticalLayout = QVBoxLayout()
        # self.water = WaterProgress((width, height), self)
        self.round = RoundProgress((width, height), self)
        # self.verticalLayout.addWidget(self.water)
        # self.verticalLayout.addWidget(self.round)
        # self.label = QLabel(self)
        # self.label.setText(QCoreApplication.translate("Dialog", text))
        # print(self.label.width())
        # self.label.move((width - self.label.width()) / 2, height / 3 * 2)
        # QMetaObject.connectSlotsByName(self)
        # self.anim = animation(self, )

    def percent_update(self, percent):
        self.percent = percent
        # self.water.update_percent(self.percent)
        # self.water.paintEvent(None)
        self.round.update_percent(self.percent)
