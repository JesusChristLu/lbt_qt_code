# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/14
# __author:       HanQing Shi

from PySide6 import QtWidgets, QtCore


class NodeScene(QtWidgets.QGraphicsScene):
    update_weight_signal = QtCore.Signal(int, str, str)

    def dragEnterEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        # find item at these coordinates
        item = self.itemAt(e.scenePos())
        print("NodeScene item", item)
        if item.setAcceptDrops:
            # pass on event to item at the coordinates
            try:
                item.dropEvent(e)
            except RuntimeError:
                pass  # This will supress a Runtime Error
                # generated when dropping into a widget with no ProxyWidget

    def dragMoveEvent(self, e):
        e.acceptProposedAction()
