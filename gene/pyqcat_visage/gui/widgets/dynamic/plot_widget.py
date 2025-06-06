# -*- coding: utf-8 -*-
import json
# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/21
# __author:       YangChao Zhao

import os
import time

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QWidget
from pyQCat.structures import QDict
from pyqcat_visage.protocol import ExecuteMonsterOp, DYNAMIC_MIN_INTERVAL


class QPlotWidget(QWidget):

    def __init__(self, gui, parent, ui):
        super().__init__(parent)
        self.ui = ui
        self.gui = gui
        self.data = QDict(id="")
        self._init_data()
        self._create_chart()
        self._last_refresh_data = time.time()
        self._refresh_interval = DYNAMIC_MIN_INTERVAL

    def _init_data(self):
        self.data.exp = 'FakeExperiment'
        self.data.total = 1
        self.data.loop = 0
        self.data.title1 = 'y0'
        self.data.title2 = 'y1'
        self.data.x = []
        self.data.y1 = []
        self.data.y2 = []

    def _create_chart(self):
        pass

    @Slot(bytes, bytes)
    def refresh(self, run_op, data):
        try:
            if run_op == ExecuteMonsterOp.dy_start.value:
                self._init_data()
                data = json.loads(data)
                self.data.total = data.get("loop_counter")
                self.data.id = data.get("experiment_id")
                self.gui.backend.current_dirs = data.get("dirs")
            elif run_op == ExecuteMonsterOp.dy_loop.value:
                data = json.loads(data)
                if data.get("id") == self.data.id:
                    if self.add_data(data) and time.time() - self._last_refresh_data > self._refresh_interval:
                        self.draw_charts()
            elif run_op == ExecuteMonsterOp.dy_end.value:
                self.draw_charts()
                self._refresh_interval = DYNAMIC_MIN_INTERVAL
        except Exception as e:
            import traceback
            print(f"dynamic plot error, details\n", traceback.format_exc())

    def add_data(self, outer_data):
        if not isinstance(outer_data, dict):
            print("plot data is not dict")
            return False
        loop = outer_data.get('loop')
        x = outer_data.get('x')
        _id = str(outer_data.get('id'))

        if self.data.loop == loop or loop is None:
            return False

        if self.data.id != _id:
            return False
        else:
            if x is not None:
                self.data.x.extend(x)
                self.data.y1.extend(outer_data['y1'])
                self.data.y2.extend(outer_data['y2'])

        self.data.loop = loop
        return True

    def draw_charts(self):
        self._last_refresh_data = time.time()
        start_time = time.time()
        percent = int(self.data.loop / self.data.total * 100)
        self.ui.first_pregress.percent_update(percent)

        def col_axis_lim(y):
            v_min = min(y)
            v_max = max(y)
            gap = (v_max - v_min) * 0.1
            return v_min - gap, v_max + gap

        if self.data.x:
            series_y0 = self.ui.chart_view_y0.chart.series()[0]
            series_y1 = self.ui.chart_view_y1.chart.series()[0]
            axis_x_y0 = self.ui.chart_view_y0.axis_x
            axis_x_y1 = self.ui.chart_view_y1.axis_x
            axis_y_y0 = self.ui.chart_view_y0.axis_y
            axis_y_y1 = self.ui.chart_view_y1.axis_y

            series_y0.clear()
            series_y1.clear()

            axis_x_y0.setRange(self.data.x[0], self.data.x[-1])
            axis_x_y1.setRange(self.data.x[0], self.data.x[-1])

            if self.data.title1.lower() != 'p0':
                axis_y_y0.setRange(*col_axis_lim(self.data.y1))
                axis_y_y1.setRange(*col_axis_lim(self.data.y2))
            else:
                axis_y_y0.setRange(0, 1)
                axis_y_y1.setRange(0, 1)

            series_y0.setName(self.data.title1)
            series_y1.setName(self.data.title2)

            for i in range(len(self.data.x)):
                y1_data = self.data.y1[i] if len(self.data.y1) > i else None
                series_y0.append(self.data.x[i], y1_data)
                y2_data = self.data.y2[i] if len(self.data.y2) > i else None
                series_y1.append(self.data.x[i], y2_data)
        else:
            self.ui.chart_view_y0.chart.series().clear()
            self.ui.chart_view_y1.chart.series().clear()

        fresh_use_time = time.time() - start_time
        if fresh_use_time > DYNAMIC_MIN_INTERVAL:
            self._refresh_interval = fresh_use_time + 0.05
        else:
            self._refresh_interval = DYNAMIC_MIN_INTERVAL

    def resizeEvent(self, event):
        w = self.width()
        h = self.height()

        process_bar = self.ui.first_pregress
        width = process_bar.width()
        height = process_bar.height()

        process_bar.setGeometry(w * 5 / 6, h * 2 / 5, width, height)
