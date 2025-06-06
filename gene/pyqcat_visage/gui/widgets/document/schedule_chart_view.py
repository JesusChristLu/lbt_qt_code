# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/20
# __author:       YangChao Zhao

from PySide6.QtCharts import QChartView
from PySide6.QtCore import Qt, Signal, QPoint, QRectF, QPointF, QRect
from PySide6.QtGui import QFont, QPainterPath, QFontMetrics, QBrush, QPen, QColor
from PySide6.QtWidgets import QGraphicsView, QGraphicsItem, QGraphicsSimpleTextItem


class Callout(QGraphicsItem):

    def __init__(self, chart):
        QGraphicsItem.__init__(self, chart)
        self._chart = chart
        self.text = ""
        self._textRect = QRectF()
        self._anchor = QPointF()
        self._font = QFont()
        self._rect = QRectF()

    def boundingRect(self):
        anchor = self.mapFromParent(self._chart.mapToPosition(self._anchor))
        rect = QRectF()
        rect.setLeft(min(self._rect.left(), anchor.x()))
        rect.setRight(max(self._rect.right(), anchor.x()))
        rect.setTop(min(self._rect.top(), anchor.y()))
        rect.setBottom(max(self._rect.bottom(), anchor.y()))

        return rect

    def paint(self, painter, option=None, widget=None):
        path = QPainterPath()
        path.addRoundedRect(self._rect, 5, 5)
        anchor = self.mapFromParent(self._chart.mapToPosition(self._anchor))
        if not self._rect.contains(anchor) and not self._anchor.isNull():
            point1 = QPointF()
            point2 = QPointF()

            # establish the position of the anchor point in relation to _rect
            above = anchor.y() <= self._rect.top()
            above_center = (self._rect.top() < anchor.y() <= self._rect.center().y())
            below_center = (self._rect.center().y() < anchor.y() <= self._rect.bottom())
            below = anchor.y() > self._rect.bottom()

            on_left = anchor.x() <= self._rect.left()
            left_of_center = (self._rect.left() < anchor.x() <= self._rect.center().x())
            right_of_center = (self._rect.center().x() < anchor.x() <= self._rect.right())
            on_right = anchor.x() > self._rect.right()

            # get the nearest _rect corner.
            x = (on_right + right_of_center) * self._rect.width()
            y = (below + below_center) * self._rect.height()
            corner_case = ((above and on_left) or (above and on_right) or
                           (below and on_left) or (below and on_right))
            vertical = abs(anchor.x() - x) > abs(anchor.y() - y)

            x1 = (x + left_of_center * 10 - right_of_center * 20 + corner_case *
                  int(not vertical) * (on_left * 10 - on_right * 20))
            y1 = (y + above_center * 10 - below_center * 20 + corner_case *
                  vertical * (above * 10 - below * 20))
            point1.setX(x1)
            point1.setY(y1)

            x2 = (x + left_of_center * 20 - right_of_center * 10 + corner_case *
                  int(not vertical) * (on_left * 20 - on_right * 10))
            y2 = (y + above_center * 20 - below_center * 10 + corner_case *
                  vertical * (above * 20 - below * 10))
            point2.setX(x2)
            point2.setY(y2)

            path.moveTo(point1)
            path.lineTo(anchor)
            path.lineTo(point2)
            path = path.simplified()

        painter.setPen(QPen(QColor(Qt.GlobalColor.yellow)))
        painter.setBrush(Qt.GlobalColor.darkMagenta)
        painter.drawPath(path)
        painter.drawText(self._textRect, self.text)

    def mousePressEvent(self, event):
        self._chart.callout_press.emit(self.text)
        event.setAccepted(True)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.setPos(self.mapToParent(
                event.pos() - event.buttonDownPos(Qt.LeftButton)))
            event.setAccepted(True)
        else:
            event.setAccepted(False)

    def set_text(self, text):
        self.text = text
        metrics = QFontMetrics(self._font)
        self._textRect = QRectF(metrics.boundingRect(
            QRect(0, 0, 150, 150), Qt.AlignLeft, self.text))
        self._textRect.translate(5, 5)
        self.prepareGeometryChange()
        self._rect = self._textRect.adjusted(-5, -5, 5, 5)

    def set_anchor(self, point):
        self._anchor = QPointF(point)

    def update_geometry(self):
        self.prepareGeometryChange()
        self.setPos(self._chart.mapToPosition(
            self._anchor) + QPointF(10, -50))


class QCharViewSchedule(QChartView):
    mouseMove = Signal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        self.__beginPoint = QPoint()
        self.__endPoint = QPoint()

        self._coordX = None
        self._coordY = None
        self.series = None
        self._callouts = []
        self._tooltip = None

        self._cur_pos = None

    def init_callout(self):
        self._coordX = QGraphicsSimpleTextItem(self.chart())

        # x_x, x_y = self.size().width() / 2 - 50, self.size().height()
        self._coordX.setPos(80, 470)
        self._coordX.setText("X: ")
        self._coordX.setBrush(QBrush(Qt.GlobalColor.yellow))

        self._coordY = QGraphicsSimpleTextItem(self.chart())
        # y_x, y_y = self.size().width() / 2 + 50, self.size().height()
        self._coordY.setPos(180, 470)
        self._coordY.setText("Y: ")
        self._coordY.setBrush(QBrush(Qt.GlobalColor.yellow))

        self._tooltip = Callout(self.chart())

    # ========== event tackle function ============
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.__beginPoint = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):

        # use to value x y
        pos = self.chart().mapToValue(event.position().toPoint())
        self._coordX.setText(f"X: {pos.x():.5f}")
        self._coordY.setText(f"Y: {pos.y():.5f}")
        self._cur_pos = pos

        point = event.pos()
        self.mouseMove.emit(point)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.__endPoint = event.pos()
            rect_f = QRectF()
            rect_f.setTopLeft(self.__beginPoint)
            rect_f.setBottomRight(self.__endPoint)
            self.chart().zoomIn(rect_f)
        elif event.button() == Qt.MouseButton.RightButton:
            self.chart().zoomReset()

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Plus:
            self.chart().zoom(1.2)
        elif key == Qt.Key.Key_Minus:
            self.chart().zoom(0.8)
        elif key == Qt.Key.Key_Left:
            self.chart().scroll(10, 0)
        elif key == Qt.Key.Key_Right:
            self.chart().scroll(-10, 0)
        elif key == Qt.Key.Key_Up:
            self.chart().scroll(0, -10)
        elif key == Qt.Key.Key_Down:
            self.chart().scroll(0, 10)
        elif key == Qt.Key.Key_PageUp:
            self.chart().scroll(0, -50)
        elif key == Qt.Key.Key_PageDown:
            self.chart().scroll(0, 50)
        elif key == Qt.Key.Key_Home:
            self.chart().zoomReset()
        elif key == Qt.Key.Key_Space:
            self.tooltip(self._cur_pos, True)
            self.keep_callout()

        super().keyPressEvent(event)

    def tooltip(self, point, state):
        if self._tooltip == 0:
            self._tooltip = Callout(self.chart())

        if state:
            x = point.x()
            y = point.y()

            self._tooltip.set_text(f"X: {x:.5f}        \nY: {y:.5f}        ")
            self._tooltip.set_anchor(point)
            self._tooltip.setZValue(11)
            self._tooltip.update_geometry()
            self._tooltip.show()
        else:
            self._tooltip.hide()

    def keep_callout(self):
        self._callouts.append(self._tooltip)
        self._tooltip = Callout(self.chart())

    def del_callout(self, text: str):
        ln = len(self._callouts)
        for i in range(ln):
            if self._callouts[i].text == text:
                self._callouts[i].setVisible(False)
                self._callouts.pop(i)
                break
