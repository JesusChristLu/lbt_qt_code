import matplotlib.pyplot as plt
from typing import Union, Tuple


class Location:

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __call__(self, *args, **kwargs):
        if len(args) == 2:
            self.x = args[0]
            self.y = args[1]

        self.x = kwargs.get("x", None) or self.x
        self.y = kwargs.get("y", None) or self.y

    def __repr__(self):
        return f"Location[{self.x},{self.y}]"

    @property
    def value(self):
        return self.x, self.y


class Point:

    def __init__(self, name: str, location: Union[Location, Tuple], abbreviation: str = None):
        self.name = name
        self.abbreviation = abbreviation
        if self.abbreviation is None:
            self.abbreviation = self.name[:3] if len(self.name > 3) else self.name
        if location is None:
            raise ValueError("location is None")
        elif isinstance(location, Tuple):
            self.location = Location(*location)
        elif isinstance(location, Location):
            self.location = location
        else:
            raise TypeError('location type not support')


class Graph:

    def get_ax(self) -> plt.Axes:
        return plt.subplot(1, 1, 1)

    def __init__(self):
        self.point_list: list[Point] = []
        self.path = []
        self.use_abbreviation = True
        self.ax = self.get_ax()

    def draw_point(self):
        for x in self.point_list:
            point_name = x.abbreviation if self.use_abbreviation else x.name
            # self.ax.text(x.location.x, x.location.y, point_name)
            self.ax.scatter(x.location.x, x.location.y, color='k')
            self.ax.annotate(point_name, xy=x.location.value, xytext=[x.location.x, x.location.y + 0.001])

    def draw_pre_path(self, start: Point, end: Point, rad: float = 0.0, color: str = "#818181", size: int = 2):
        arrowprops = dict(
            color=color,
            arrowstyle="simple",
            connectionstyle="arc3, rad={:.1f}".format(-rad),
            # connectionstyle = "Bar, armA=0.0, armB=0.0, fraction=0.3",
        )
        start, end = self.deal_point(start, end)
        print(arrowprops, start, end)

        self.ax.annotate("123", xy=start.value, xytext=end.value, size=size,va="center", ha="center",
                         arrowprops=arrowprops)

    def deal_point(self, start: Point, end: Point, offset=0.06) -> Tuple[Location, Location]:
        start = Location(*start.location.value)
        end = Location(*end.location.value)
        if start.x < end.x:
            start.x = start.x + offset
            end.x = end.x - offset
        else:
            start.x = start.x - offset
            end.x = end.x + offset
        return end, start


g1 = Graph()
g1.point_list = [Point("Tunable", (1, 1), "A"), Point("cavity_specturm", (2, 1), "B"),
                 Point("qubit_spectrum", (3, 1), "C"), Point("Rabi scan amp", (4, 1), "D")]

g1.draw_point()

g1.draw_pre_path(g1.point_list[0], g1.point_list[1])
g1.draw_pre_path(g1.point_list[1], g1.point_list[2])
g1.draw_pre_path(g1.point_list[2], g1.point_list[1], size=5, color="#cdb1a7", rad=0.1)
g1.draw_pre_path(g1.point_list[1], g1.point_list[0], size=5, color="#cdb1a7", rad=0.1)
g1.draw_pre_path(g1.point_list[0], g1.point_list[1], size=5, color="#cdb1a7", rad=0.1)
g1.draw_pre_path(g1.point_list[1], g1.point_list[2], size=5, color="#cdb1a7", rad=0.1)
g1.draw_pre_path(g1.point_list[2], g1.point_list[1], size=10, color="#ecf0f5", rad=0.3)
g1.draw_pre_path(g1.point_list[1], g1.point_list[2], size=10, color="#ecf0f5", rad=0.3)
g1.draw_pre_path(g1.point_list[2], g1.point_list[3], size=10, color="#ecf0f5", rad=0.3)

plt.show()
