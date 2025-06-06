from enum import Enum
from math import sqrt
from pyQCat.structures import QDict

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QTransform, QFont, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsEllipseItem,
    QGraphicsTextItem,
    QGraphicsItem
)


def diff_points_distance(p1, p2, judge_distance) -> bool:
    x1 = p1.x()
    y1 = p1.y()
    x2 = p2.x()
    y2 = p2.y()

    distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if distance > judge_distance:
        return True
    else:
        return False


def check_point_in_rect(point, range_x, range_y) -> bool:
    if range_x[0] < point.x() < range_x[1] and range_y[0] < point.y() < range_y[1]:
        return True

    return False


class BasicItem:
    font_size = 12
    COOR_NUM = 100
    R_ELLIPSE = 21

    def __init__(self, name: str, color_conf, coordinate, status=0):
        """
        status:
            0: normal
            1: env bits
            2: physical bits.
        """
        self.name = name.lower()
        self.color_conf = color_conf
        self.coordinate = coordinate
        self.status = status
        self.text = None
        self.pen_color = None
        self.brash_color = None
        self.thread = None
        self.font_status = 0

    def init_theme(self, color_conf: QDict = None, rerender: bool = False):
        if color_conf:
            self.color_conf = color_conf

    @property
    def r_center(self):
        return self.rect().center()

    def set_envs(self, flag=False):
        if flag:
            self.status = 1
        else:
            self.status = 0

    def set_physical_bits(self, flag=False):
        if flag:
            self.status = 2
        else:
            if self.status == 2:
                self.status = 1

    def set_font_bits(self, flag: bool = False):
        if flag:
            self.font_status = 1
        else:
            if self.font_status == 1:
                self.font_status = 0

    def reset_font_status(self):
        self.font_status = 0

    def reset_status(self):
        self.status = 0

    def reset(self):
        self.status = 0
        self.thread = None
        self.pen_color = None
        self.brash_color = None
        self.font_status = 0

    def get_pen_color(self):
        pen_color = None
        if self.isSelected():
            pen_color = QColor(*self.color_conf.pen_color)
        elif self.status == 1:
            pen_color = QColor(*self.color_conf.env_color)
        elif self.status == 2:
            pen_color = QColor(*self.color_conf.physical_color)
        elif self.status == 3:
            if self.pen_color:
                return QColor(*self.pen_color)
        return pen_color

    def set_color(self, thread_id, use_brash: bool = False):
        self.status = 3
        self.thread = thread_id
        self.pen_color = self.color_conf.thread_color.env_color
        if use_brash:
            self.brash_color = self.color_conf.thread_color[thread_id]["physical_color"]

    def set_high_color(self, high_color):
        self.status = 3
        if high_color == 1:
            self.pen_color = self.color_conf.thread_color.env_color
        elif high_color == 2:
            self.pen_color = self.color_conf.thread_color.secure_color
        elif high_color ==3:
            self.pen_color = self.color_conf.thread_color.alter_color
        elif high_color >= 4:
            self.pen_color = self.color_conf.thread_color.use_color
            self.brash_color = self.color_conf.thread_color.use_color
class QubitItem(BasicItem, QGraphicsEllipseItem):
    def __init__(self, name: str, color_conf, coordinate, status=0):
        BasicItem.__init__(self, name, color_conf, coordinate, status=status)
        x = self.coordinate[0] * self.COOR_NUM
        y = self.coordinate[1] * self.COOR_NUM
        w = self.R_ELLIPSE * 2
        h = self.R_ELLIPSE * 2
        QGraphicsEllipseItem.__init__(self, x, y, w, h)
        self.setFlags(QGraphicsEllipseItem.ItemIsSelectable)

        self._init_text()

    def init_font_color(self):
        if self.text:
            if self.font_status == 1:
                self.text.setDefaultTextColor(QColor(*self.color_conf.focal_font_color))
            else:
                self.text.setDefaultTextColor(QColor(*self.color_conf.font_color))

    def init_theme(self, color_conf: QDict = None, rerender: bool = False):
        super().init_theme(color_conf)
        self.init_font_color()

    def dragMoveEvent(self, event) -> None:
        event.ignore()

    def _init_text(self):
        self.text = QGraphicsTextItem(self.name.upper())
        font = QFont(self.color_conf.fonts, self.font_size)
        font.setBold(True)
        self.text.setFont(font)
        # self.text.setDefaultTextColor(QColor(*self.color_conf.font_color))
        self.init_font_color()
        self.text.setParentItem(self)
        self.text.setPos(
            self.boundingRect().center() - self.text.boundingRect().center()
        )

    def paint(self, painter, option, widget=None):
        pen_color = self.get_pen_color()
        if not pen_color:
            pen_color = Qt.GlobalColor.gray
        pen = QPen(pen_color, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        if self.status == 3 and self.brash_color:
            painter.setBrush(QColor(*self.brash_color))
        else:
            painter.setBrush(QColor(*self.color_conf.qubit_color))
        self.init_font_color()
        painter.drawEllipse(self.rect())


class CoupleDirectionEnum(Enum):
    ACROSS = "across"
    VERTICAL = "vertical"


class CoupleItem(BasicItem, QGraphicsRectItem):
    width = 20
    height = 20
    rect_width = 55
    rect_height = 10

    def __init__(self, name: str, color_conf, coordinate, direction, status=0):
        #  init basic
        BasicItem.__init__(self, name, color_conf, coordinate, status=status)
        # deal pos
        x = self.coordinate[0] * self.COOR_NUM - self.width / 2 + self.R_ELLIPSE
        y = self.coordinate[1] * self.COOR_NUM - self.height / 2 + self.R_ELLIPSE
        w = self.width
        h = self.height
        # init Rect item
        QGraphicsRectItem.__init__(self, x, y, w, h)
        self.setFlags(QGraphicsRectItem.ItemIsSelectable)

        self.direction = direction
        self.back_rect = self.create_back_rect()
        self.setZValue(1)

    def init_theme(self, color_conf: QDict = None, rerender: bool = False):
        super().init_theme(color_conf)
        if self.back_rect:
            brush = QBrush(QColor(*self.color_conf.edge_color))
            self.back_rect.setBrush(brush)

    def create_back_rect(self):
        if not isinstance(self.direction, CoupleDirectionEnum):
            return

        x = self.coordinate[0] * self.COOR_NUM
        y = self.coordinate[1] * self.COOR_NUM
        transform = None
        if self.direction == CoupleDirectionEnum.ACROSS:
            x = x - self.rect_width / 2 + self.R_ELLIPSE
            y = y - self.rect_height / 2 + self.R_ELLIPSE
        elif self.direction == CoupleDirectionEnum.VERTICAL:
            x = x - self.rect_width / 2 + self.R_ELLIPSE
            y = y - self.rect_height / 2 - self.R_ELLIPSE
            transform = (
                QTransform()
                .translate(
                    self.coordinate[0] * self.COOR_NUM,
                    self.coordinate[1] * self.COOR_NUM,
                )
                .rotate(90)
                .translate(
                    -self.coordinate[0] * self.COOR_NUM,
                    -self.coordinate[1] * self.COOR_NUM,
                )
            )

        rect = QGraphicsRectItem(x, y, self.rect_width, self.rect_height)
        brush = QBrush(QColor(*self.color_conf.edge_color))
        rect.setBrush(brush)
        if transform:
            rect.setTransform(transform)
        self.setParentItem(self)
        return rect

    def paint(self, painter, option, widget=None):
        if self.status == 3 and self.brash_color:
            painter.setBrush(QColor(*self.brash_color))
        else:
            painter.setBrush(QColor(*self.color_conf.coupler_color))
        pen_color = self.get_pen_color()
        if pen_color:
            painter.setPen(QPen(pen_color, 2, Qt.SolidLine))
        else:
            painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())


class Band:

    def __init__(self, qubit_dict, coupler_dict):
        self.start_pos = None
        self.end_pos = None
        self.select_cache = []
        self.select_mode = 0
        self.flag = False
        self.qubit_dict = qubit_dict
        self.coupler_dict = coupler_dict
        self.band_item = None
        self.max_point = None
        self.create_band()

    def create_band(self):
        self.band_item = QGraphicsRectItem(0, 0, 5, 5)
        self.init_theme()
        self.band_item.hide()
        # self.band_item.setFlag(QGraphicsItem.ItemIgnoresTransformations)

    def init_theme(self):
        self.band_item.setPen(QPen(Qt.darkBlue, 2))
        self.band_item.setBrush(QBrush(QColor(54, 192, 244, 10)))
        # self.band_item.setOpacity(0.5)

    def change_bit_select(self, bits, flag, qubit_dict, coupler_dict):
        if not isinstance(bits, list):
            bits = [bits]
        for bit in bits:
            if bit in qubit_dict:
                qubit_dict[bit].setSelected(flag)
            elif bit in coupler_dict:
                coupler_dict[bit].setSelected(flag)

    def check_select(self, qubit_dict, coupler_dict):
        temp_select = []
        range_x = (self.start_pos.x(), self.end_pos.x()) if self.start_pos.x() < self.end_pos.x() else (
            self.end_pos.x(), self.start_pos.x())
        range_y = (self.start_pos.y(), self.end_pos.y()) if self.start_pos.y() < self.end_pos.y() else (
            self.end_pos.y(), self.start_pos.y())

        if self.select_mode in [0, 1]:
            for qubit, item in qubit_dict.items():
                if check_point_in_rect(item.r_center, range_x, range_y):
                    temp_select.append(qubit)

        if self.select_mode in [0, 2]:
            for coupler, item in coupler_dict.items():
                if check_point_in_rect(item.r_center, range_x, range_y):
                    temp_select.append(coupler)

        need_pop_bit = list(set(self.select_cache) - set(temp_select))
        need_push_bit = list(set(temp_select) - set(self.select_cache))
        self.change_bit_select(need_push_bit, True, qubit_dict, coupler_dict)
        self.change_bit_select(need_pop_bit, False, qubit_dict, coupler_dict)
        self.select_cache = temp_select

    def __bool__(self):
        return self.flag

    def check_pos(self, point):
        # fix the topology will move when use band.
        if point.x() < -10:
            point.setX(-10)
        if point.y() < -10:
            point.setY(-10)

        if self.max_point:
            if point.x() > self.max_point[0]:
                point.setX(self.max_point[0])
            if point.y() > self.max_point[1]:
                point.setY(self.max_point[1])
        return point

    def get_rect(self):
        x1, x2 = (self.start_pos.x(), self.end_pos.x()) if self.start_pos.x() < self.end_pos.x() else (
            self.end_pos.x(), self.start_pos.x())

        y1, y2 = (self.start_pos.y(), self.end_pos.y()) if self.start_pos.y() < self.end_pos.y() else (
            self.end_pos.y(), self.start_pos.y())

        w = x2 - x1
        h = y2 - y1
        return x1, y1, w, h

    def start(self, point):
        self.start_pos = self.check_pos(point)
        self.end_pos = None

    def move(self, point):
        if self.start_pos:
            if not self.flag:
                if diff_points_distance(self.start_pos, point, 20):
                    self.flag = True
                    self.end_pos = self.check_pos(point)
                    self.band_item.setRect(*self.get_rect())
                    self.band_item.show()
            else:
                self.end_pos = self.check_pos(point)
                self.band_item.setRect(*self.get_rect())

    def end(self, point):
        if self.flag:
            self.end_pos = self.check_pos(point)
            self.check_select(self.qubit_dict, self.coupler_dict)
            self.band_item.hide()

    def clear(self):
        self.start_pos = None
        self.end_pos = None
        self.flag = False
        self.select_cache = []


def get_title(bit_num, color_conf, scene, text=None):
    title_item = QGraphicsTextItem(f"OriginQ Quantum {bit_num}bit")
    title_item.setFont(QFont(color_conf.fonts, 20))  # Set the font of the title
    title_item.setDefaultTextColor(
        Qt.GlobalColor.white
    )
    bounds = scene.itemsBoundingRect()
    bounds.adjust(-100, -100, 100, 100)
    scene.setSceneRect(bounds)
    title_item.setPos(
        scene.width() / 2 - 100 - title_item.boundingRect().width() / 2,
        -title_item.boundingRect().height(),
    )
    return title_item
