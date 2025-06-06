# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/20
# __author:       YangChao Zhao

from PySide6.QtCharts import QChart, QLineSeries, QValueAxis, QChartView
from PySide6.QtCore import Qt, QMargins
from PySide6.QtGui import QPainter, QPen
from pyqcat_visage.config import GUI_CONFIG


class QChartViewBase(QChartView):

    def __init__(self, parent: None, axis: int = 0):
        QChartView.__init__(self, parent)

        self.axis = axis

        self.show()

        self.chart = None
        self.axis_x = None
        self.axis_y = None

        self._create_chart()

    def _create_chart(self):
        self.chart = QChart()
        # self.chart.setTheme(QChart.ChartTheme(7))
        self.chart.setMargins(QMargins(0, 2, 2, 2))
        self.setChart(self.chart)
        self.setRenderHint(QPainter.Antialiasing)

        series = QLineSeries()
        series.setName(f"Y{self.axis}")
        self._cur_series = series

        # pen = QPen(GUI_CONFIG.dynamic.y0_color) if self.axis == 0 else QPen(GUI_CONFIG.dynamic.y1_color)
        # pen.setStyle(Qt.SolidLine)  # SolidLine, DashLine, DotLine, DashDotLine
        # pen.setWidth(3)
        # series.setPen(pen)

        self.chart.addSeries(series)

        axis_x = QValueAxis()
        axis_x.setLabelFormat("%.3f")  # 标签格式
        axis_x.setTickCount(10)  # 主分隔个数
        axis_x.setGridLineVisible(True)
        axis_x.setMinorGridLineVisible(False)
        self.axis_x = axis_x  # 当前坐标轴

        axis_y = QValueAxis()
        axis_y.setLabelFormat("%.2f")  # 标签格式
        axis_y.setTickCount(5)
        axis_y.setMinorGridLineVisible(False)
        self.axis_y = axis_y

        self.chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

        series.attachAxis(axis_x)
        series.attachAxis(axis_y)


class QChartViewY0(QChartViewBase):
    def __init__(self, parent=None):
        super().__init__(parent, 0)


class QChartViewY1(QChartViewBase):

    def __init__(self, parent=None):
        super().__init__(parent, 1)
