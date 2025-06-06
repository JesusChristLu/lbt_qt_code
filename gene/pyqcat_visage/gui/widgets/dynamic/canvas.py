# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/20
# __author:       YangChao Zhao


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as \
    FigureCanvas
import matplotlib.pyplot as plt


class MyFigureCanvas(FigureCanvas):

    def __init__(self, parent=None, width=10, height=5, dpi=100):
        # 创建一个Figure
        fig = plt.Figure(figsize=(width, height),
                         dpi=dpi,
                         tight_layout=True)
        # tight_layout: 用于去除画图时两边的空白

        FigureCanvas.__init__(self, fig)  # 初始化父类
        self.setParent(parent)

        # 调用figure下面的add_subplot方法，类似于matplotlib.pyplot
        # 下面的subplot方法
        self.axes1 = fig.add_subplot(211)
        self.axes2 = fig.add_subplot(212)
