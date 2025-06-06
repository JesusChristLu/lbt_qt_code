from pyqcat_visage.gui.widgets.topolopy import BaseTopologyView, TopologyScene
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,

)
from PySide6.QtGui import QPainter

from typing import Union
from .structers import ShowThreadType, ProbeStruct


class MultiThreadTopologyView(BaseTopologyView):

    def __init__(self, parent, scene, color_conf):
        super().__init__(parent, scene, color_conf)
        self.setRenderHint(QPainter.Antialiasing)

    def set_thread_color(self, env_bits, physic_bits, thread_id):
        for bit in env_bits:
            if bit in self.qubit_dict and self.qubit_dict[bit].thread is None:
                self.qubit_dict[bit].set_color(thread_id=thread_id)

            if bit in self.couple_dict and self.couple_dict[bit].thread is None:
                self.couple_dict[bit].set_color(thread_id=thread_id)
        for bit in physic_bits:
            if bit in self.qubit_dict:
                self.qubit_dict[bit].set_color(thread_id=thread_id, use_brash=True)

            if bit in self.couple_dict:
                self.couple_dict[bit].set_color(thread_id=thread_id, use_brash=True)

    def set_high_color(self, color_dict:dict):
        for bit, color in color_dict.items():
            if bit in self.qubit_dict:
                self.qubit_dict[bit].set_high_color(color)
            if bit in self.couple_dict:
                self.couple_dict[bit].set_high_color(color)

class MultiTopologyWidget(QWidget):

    def __init__(self, parent, color_conf=None):
        super().__init__(parent)
        self._color_conf = color_conf

        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 2, 2)
        self.setLayout(self._layout)
        self._scene = TopologyScene()
        self.multi_view: MultiThreadTopologyView = MultiThreadTopologyView(self, scene=self._scene,
                                                                           color_conf=self.color_conf)
        self.multi_view.setScene(self._scene)

        self._layout.addWidget(self.multi_view)

    @property
    def color_conf(self):
        return self._color_conf

    @color_conf.setter
    def color_conf(self, color_conf):
        if color_conf:
            self._color_conf = color_conf
            self.multi_view.color_conf = color_conf

    def init_theme(self, color_conf=None, rerender: bool = False):
        if color_conf:
            self.color_conf = color_conf
            self.multi_view.init_theme(self.color_conf, rerender=rerender)

        if rerender:
            self.hide()
            self.show()

    def load(self, row: Union[str, int], col: Union[str, int], qubit_names=None):
        self.multi_view.base_load(row=row, col=col, qubit_names=qubit_names)



    def refresh(self, thread_data: ProbeStruct, show_type: str = ShowThreadType.ALL):

        def tick_thread_info(thread_info):
            env_b = thread_info.get("env_bits", [])
            physical_b = thread_info.get("use_bits", [])
            return env_b, physical_b

        self.multi_view.reset_bits()
        if not thread_data:
            return
        if show_type == ShowThreadType.ALL:
            for tr_id, tr_info in thread_data.core_thread.items():
                env_bits, physical_bits = tick_thread_info(tr_info)
                self.multi_view.set_thread_color(env_bits, physical_bits, tr_id)
        elif show_type == ShowThreadType.HIGHER:
            self.multi_view.set_high_color(thread_data.color)


        # else:
        #     if show_type in thread_data:
        #         env_bits, physical_bits = tick_thread_info(thread_data[show_type])
        #         self.multi_view.set_thread_color(env_bits, physical_bits, show_type)

        self.multi_view.update()
