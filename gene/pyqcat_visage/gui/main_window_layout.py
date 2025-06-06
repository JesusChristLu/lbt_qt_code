from enum import Enum
from pyQCat.invoker import MULTI_THREAD
from PySide6.QtCore import Qt, QPoint, Slot, QRect, QMargins
from PySide6.QtGui import QPixmap, QEnterEvent, QMouseEvent, QPaintEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout

from pyqcat_visage.gui.title_window_ui import Ui_TitleWindow
from pyqcat_visage.protocol import ExecuteOp


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    LEFTTOP = 4
    LEFTBOTTOM = 5
    RIGHTBOTTOM = 6
    RIGHTTOP = 7
    TITLE = 8
    NONE = 9


class TitleGUI(QWidget):
    sub_windows = []
    main_window = []

    def __init__(self, sub_window=None, icon_path=None, title=None, parent=None):
        super().__init__(parent)
        self._ui = Ui_TitleWindow()
        self._ui.setupUi(self)
        if MULTI_THREAD != 0:
            self.setWindowTitle(f"PyQCat-Visage-{MULTI_THREAD}")
        self.setWindowFlags(Qt.FramelessWindowHint)  # Remove the border of window
        self.icon_path = icon_path
        self.title = title
        self.icon_and_title()
        self.setMouseTracking(True)  # Set widget mouse tracking
        self._ui.TitleWidget.installEventFilter(self)  # Initializes the event filter
        self._ui.RealWidget.installEventFilter(self)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint)
        self.m_enabled = True
        self.m_bLeftPressed = False
        self.m_bResizeable = False
        self.m_Direction = Direction.NONE
        self.m_resize_padding = 3
        if self.m_enabled:
            self.m_resize_padding = self.m_resize_padding * self.devicePixelRatioF()
            self.installEventFilter(self)
            self.m_titlebarWidget = self._ui.TitleWidget.children()
        self.m_DragPos = None
        self._margin_list = [self.m_resize_padding] * 4
        self.setResizeable(True, QMargins(0, 0, 0, 0))
        self.first_event = False

        if sub_window:
            self.sub_window = sub_window
            self.resize(self.sub_window.width(), self.sub_window.height() + self._ui.TitleWidget.height())
            self.insert_widget()
        if self.sub_window.__class__.__name__ not in ["VisageExtension", "UserLoginWindow"]:
            TitleGUI.sub_windows.append(self)
        else:
            TitleGUI.main_window.append(self)

    @property
    def is_resize(self):
        return self.m_bResizeable

    @is_resize.setter
    def is_resize(self, option: bool):
        self.m_bResizeable = option

    def insert_widget(self):
        # Initialize and create a vertical layout, QVBoxLayout
        self.main_vBoxLayout = QVBoxLayout(self._ui.RealWidget)
        self.main_vBoxLayout.setContentsMargins(*self._margin_list)
        self.main_vBoxLayout.setSpacing(0)
        self.main_vBoxLayout.setObjectName("RealWidgetBoxLayout")
        # self.main_vBoxLayout.setAlignment(QtCore.Qt.Alignment)
        self._ui.RealWidget.setLayout(self.main_vBoxLayout)
        self.main_vBoxLayout.addWidget(self.sub_window)  # add sub widget

    def icon_and_title(self):
        self._ui.IconLabel.setAlignment(Qt.AlignCenter)
        if self.icon_path:
            self._ui.IconLabel.setPixmap(QPixmap(self.icon_path))
            self._ui.TitleLabel.setScaledContents(True)
        if self.title:
            self._ui.TitleLabel.setText(self.title)

    @Slot()
    def on_WindowMin_clicked(self):
        # minimization of window
        self.showMinimized()

    @Slot()
    def on_WindowMax_clicked(self):
        # Maximization and recovery
        if self.isMaximized():
            self.showNormal()
            self._ui.WindowMax.setToolTip("<html><head/><body><p>最大化</p></body></html>")
        else:
            self.showMaximized()
            self._ui.WindowMax.setToolTip("<html><head/><body><p>恢复</p></body></html>")

    @staticmethod
    def close_sub_windows():
        for window in TitleGUI.sub_windows:
            window.close()

    @Slot()
    def on_WindowClose_clicked(self):
        # close window and process
        if self.sub_window.__class__.__name__ == "VisageExtension":
            if self.sub_window.ok_to_close():
                self.close_sub_windows()
                self.sub_window.gui.backend.execute_send(ExecuteOp.exit)
                self.sub_window.heart_thread.close()
                for win in TitleGUI.main_window:
                    if win != self:
                        win.close()
                if self.sub_window.close_window(self):
                    self.close()
        elif self.sub_window.__class__.__name__ == "UserLoginWindow":
            self.sub_window.gui.backend.execute_send(ExecuteOp.exit)
            for win in TitleGUI.main_window:
                if win != self:
                    win.close()
            self.close_sub_windows()
            self.close()
        else:
            self.close()

    def setAllWidgetMouseTracking(self, widget: QWidget):
        if not self.m_enabled:
            return
        self._ui.TitleWidget.setMouseTracking(True)
        self._ui.RealWidget.setMouseTracking(True)
        # widget.setMouseTracking(True)
        # for obj in widget.children():
        #     if obj.metaObject().className() in ["QWidget", "QDockWidget"]:
        #         obj.setMouseTracking(True)
        #         self.setAllWidgetMouseTracking(obj)

    def eventFilter(self, obj, event):
        """
        Event filter that resolves the mouse to revert to the standard mouse style after entering another control
        """
        if isinstance(event, QEnterEvent):
            self.setCursor(Qt.ArrowCursor)
        elif isinstance(event, QPaintEvent):
            if not self.first_event:
                self.first_event = True
                self.setAllWidgetMouseTracking(self)
        return super().eventFilter(obj, event)  # 注意 ,MyWindow是所在类的名称

    def mousePressEvent(self, event: QMouseEvent):
        """
        Mouse press event
            1.update mouse location and style and state
            1.update self.m_bLeftPressed
        """
        point = event.globalPos()
        tl: QPoint = self.mapToGlobal(self.rect().topLeft())
        rb: QPoint = self.mapToGlobal(self.rect().bottomRight())
        self._relocation(tl, rb, point.x(), point.y())
        # print("mouse press", self.m_Direction)
        if event.button() == Qt.LeftButton and self.m_Direction != Direction.NONE:
            self.m_bLeftPressed = True
            self.m_DragPos = event.globalPos() - self.frameGeometry().topLeft()
        return super().mousePressEvent(event)

    def is_on_border(self, pos: QPoint) -> bool:
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        return (
            0 <= x < self.m_resize_padding or
            w - self.m_resize_padding <= x < w or
            0 <= y < self.m_resize_padding or
            h - self.m_resize_padding <= y < h
        )

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Operate on the window according to the mouse position and state.
            1.resize the window
            2.move the window
            3.update mouse`s style and state
        """
        global_point = event.globalPos()
        if self.m_bLeftPressed:
            if self.m_Direction != Direction.NONE:
                tl: QPoint = self.mapToGlobal(self.rect().topLeft())
                rb: QPoint = self.mapToGlobal(self.rect().bottomRight())
                rMove = QRect(tl, rb)
                flag = True
                if self.m_Direction == Direction.LEFT:
                    if rb.x() - global_point.x() <= self.minimumWidth():
                        rMove.setX(tl.x())
                    else:
                        rMove.setX(global_point.x())
                elif self.m_Direction == Direction.RIGHT:
                    rMove.setWidth(global_point.x() - tl.x())
                elif self.m_Direction == Direction.UP:
                    if rb.y() - global_point.y() <= self.minimumHeight():
                        rMove.setY(tl.y())
                    else:
                        rMove.setY(global_point.y())
                elif self.m_Direction == Direction.DOWN:
                    rMove.setHeight(global_point.y() - tl.y())
                elif self.m_Direction == Direction.LEFTTOP:
                    rMove.setX(global_point.x())
                    rMove.setY(global_point.y())
                elif self.m_Direction == Direction.RIGHTTOP:
                    rMove.setWidth(global_point.x() - tl.x())
                    rMove.setY(global_point.y())
                elif self.m_Direction == Direction.LEFTBOTTOM:
                    rMove.setX(global_point.x())
                    rMove.setHeight(global_point.y() - tl.y())
                elif self.m_Direction == Direction.RIGHTBOTTOM:
                    rMove.setWidth(global_point.x() - tl.x())
                    rMove.setHeight(global_point.y() - tl.y())
                else:
                    flag = False
                if flag:
                    self.setGeometry(rMove)
                if self.m_Direction == Direction.TITLE:
                    if self.isMaximized():
                        self.on_WindowMax_clicked()
                    flag = True
                    self.move(event.globalPos() - self.m_DragPos)
                if flag:
                    event.accept()
                else:
                    event.ignore()
            else:
                event.ignore()
                return
        else:
            if self.is_on_border(event.pos()):
                self.region(global_point)
            else:
                self.setCursor(Qt.ArrowCursor)
        # print("m_Direction:", self.m_Direction, self.m_bLeftPressed)

    def region(self, cursor_point: QPoint):
        if not self.m_bResizeable:
            return
        rect: QRect = self.contentsRect()
        tl: QPoint = self.mapToGlobal(rect.topLeft())
        rb: QPoint = self.mapToGlobal(rect.bottomRight())
        x = cursor_point.x()
        y = cursor_point.y()
        self._relocation(tl, rb, x, y)

    def _relocation(self, tl: QPoint, rb: QPoint, x, y):
        """
        Gets the current mouse position and updates the mouse style and status!
        update:
            self.m_Direction
            self.setCursor
        """

        if tl.x() <= x <= tl.x() + self.m_resize_padding:
            # left-top  left-bottom  left
            if tl.y() <= y <= tl.y() + self.m_resize_padding:
                self.m_Direction = Direction.LEFTTOP
                self.setCursor(Qt.SizeFDiagCursor)
            elif rb.y() - self.m_resize_padding <= y <= rb.y():
                self.m_Direction = Direction.LEFTBOTTOM
                self.setCursor(Qt.SizeBDiagCursor)
            else:
                self.m_Direction = Direction.LEFT
                self.setCursor(Qt.SizeHorCursor)
        elif rb.x() - self.m_resize_padding <= x <= rb.x():
            # right-top right-bottom right
            if tl.y() <= y <= tl.y() + self.m_resize_padding:
                self.m_Direction = Direction.RIGHTTOP
                self.setCursor(Qt.SizeBDiagCursor)
            elif rb.y() - self.m_resize_padding <= y <= rb.y():
                self.m_Direction = Direction.RIGHTBOTTOM
                self.setCursor(Qt.SizeFDiagCursor)
            else:
                self.m_Direction = Direction.RIGHT
                self.setCursor(Qt.SizeHorCursor)
        else:
            # up down title none
            if tl.y() <= y <= tl.y() + self.m_resize_padding:
                self.m_Direction = Direction.UP
                self.setCursor(Qt.SizeVerCursor)
            elif rb.y() >= y >= rb.y() - self.m_resize_padding:
                self.m_Direction = Direction.DOWN
                self.setCursor(Qt.SizeVerCursor)
            elif tl.y() + self.m_resize_padding < y < tl.y() + self._ui.TitleWidget.height():
                self.m_Direction = Direction.TITLE
                self.setCursor(Qt.ArrowCursor)
            else:
                self.m_Direction = Direction.NONE
                self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.m_bResizeable:
            y = event.globalPos().y()
            tl: QPoint = self.mapToGlobal(self.rect().topLeft())
            rb: QPoint = self.mapToGlobal(self.rect().bottomRight())
            resize = False
            if tl.y() < y < tl.y() + self._ui.TitleWidget.height():
                resize = True
            elif rb.y() >= y >= rb.y() - self.m_resize_padding:
                resize = True
            if resize:
                if self.isMaximized():
                    self.showNormal()
                else:
                    self.showMaximized()
        self.m_bLeftPressed = False
        return super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.m_bLeftPressed = False
        if self.m_Direction != Direction.NONE:
            self.m_Direction = Direction.NONE
            self.setCursor(Qt.ArrowCursor)
        self.releaseMouse()
        # print("mouseReleaseEvent", self.m_Direction)
        return super().mouseReleaseEvent(event)

    def setResizeable(self, b, transparentMargins: QMargins):
        self.m_bResizeable = b
        self.m_transparentMargsins = transparentMargins
        self._ui.gridLayoutTitle.setContentsMargins(transparentMargins)
