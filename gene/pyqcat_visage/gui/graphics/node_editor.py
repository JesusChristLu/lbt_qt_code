# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/14
# __author:       HanQing Shi
"""Handle GUI events."""
from PySide6 import QtWidgets, QtCore
from loguru import logger

from pyqcat_visage.gui.graphics.node_item import NodeItem
from pyqcat_visage.gui.graphics.pipe_item import PipeItem
from pyqcat_visage.gui.graphics.port_item import PortItem
from pyqcat_visage.gui.tools.utilies import slot_catch_exception

from typing import List
from pyqcat_visage.backend.model_dag import ModelDag


class NodeEditor(QtCore.QObject):
    """Node editor object to handle all input events."""

    isvalid = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.connection = None
        self.port = None
        self.scene = None
        self.dag: ModelDag = None
        self.nodes = {}
        self.edges = {}
        self._selected_nodes = []
        self._ctrl_pressed = False
        self._sub_dag: ModelDag = None

    def install(self, scene):
        """Install scene to view."""
        self.scene = scene
        self.scene.installEventFilter(self)
        # connect signal.
        self.scene.update_weight_signal.connect(self._modify_edges_weight)

    def item_at(self, position):
        """Get the position Item."""
        items = self.scene.items(
            QtCore.QRectF(position - QtCore.QPointF(1, 1), QtCore.QSizeF(3, 3))
        )
        if items:
            return items[0]
        return None

    def _mouse_press_event(self, event):
        """Press mouse left button and select an item."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            item = self.item_at(event.scenePos())
            if isinstance(item, PortItem):
                self.connection = PipeItem()
                self.scene.addItem(self.connection)
                self.port = item
                self.connection.start_pos = item.scenePos()
                self.connection.end_pos = event.scenePos()
                self.connection.update_path()

                return True

            elif isinstance(item, PipeItem):
                self.connection = PipeItem()
                self.connection.start_pos = item.start_pos
                self.scene.addItem(self.connection)
                self.port = item.start_port
                self.connection.end_pos = event.scenePos()
                self.connection.update_port_pos()

                return True

            elif isinstance(item, NodeItem):
                if self._selected_nodes and not self._ctrl_pressed:
                    # If we clear the scene, we lose the last selection
                    try:
                        for node in self._selected_nodes:
                            node.select_connections(False)
                    except RuntimeError:
                        pass
                item.select_connections(True)
                self._selected_nodes.append(item)

            else:
                try:
                    if self._selected_nodes:
                        for node in self._selected_nodes:
                            node.select_connections(False)
                        self._ctrl_pressed = False
                except RuntimeError:
                    pass

                self._selected_nodes.clear()

    def _mouse_release_event(self, event):
        """Handle mouse release event, such as connect ports."""
        item = self.item_at(event.scenePos())
        if self.connection and event.button() == QtCore.Qt.MouseButton.LeftButton:
            # connecting a port
            if isinstance(item, PortItem):
                if self.port.can_connect_to(item):
                    self.connection.start_port = self.port
                    self.connection.end_port = item
                    self.connection.update_port_pos()

                    # update dag edges.
                    start_node, end_node = self._get_node_name(self.connection)
                    self.dag.add_edge(
                        start_node, end_node, weight=self.connection.weight
                    )
                    if len(self.dag.nodes) > 1:
                        # check DAG is acyclic
                        try:
                            self.dag.acyclic_check()
                            self.edges[f"{start_node}-{end_node}"] = self.connection
                            logger.info(f"add edges: {start_node}--->{end_node}")
                        except Exception as e:
                            logger.error(e)
                            self.dag.remove_edge(start_node, end_node)
                            self.connection.delete()
                    self.connection = None
                else:
                    # print("Deleting connection")
                    self.connection.delete()
                    self.connection = None

            if self.connection:
                self.connection.delete()
            self.connection = None
            self.port = None

            return True

        # fixed bug : 2023/03/01 HanQing Shi
        # update node position when mouse released.
        if isinstance(item, NodeItem) and event.button() == QtCore.Qt.MouseButton.LeftButton:
            item.node.location = item.xy_pos

    def _delete_key_press_event(self, event):
        """Handle delete key press event, such as delete a node."""
        if event.key() in [int(QtCore.Qt.Key.Key_Backspace), int(QtCore.Qt.Key.Key_Delete)]:
            for item in self.scene.selectedItems():
                if isinstance(item, PipeItem):
                    item.delete()
                    start_node, end_node = self._get_node_name(item)
                    self.dag.remove_edge(start_node, end_node)
                    self.edges.pop(f"{start_node}-{end_node}")
                if isinstance(item, NodeItem):
                    to_deletes = item.delete()
                    for connection in to_deletes:
                        start_node, end_node = self._get_node_name(connection)
                        self.dag.remove_edge(start_node, end_node)
                        self.edges.pop(f"{start_node}-{end_node}")
                    self.dag.remove_node(item.identifier)
                    self.nodes.pop(item.identifier)
            return True

        if event.key() == QtCore.Qt.Key.Key_Control:
            self._ctrl_pressed = True

    def _mouse_move_event(self, event):
        """Handle mouse move event, such as connect ports."""
        if self.connection:
            self.connection.end_pos = event.scenePos()
            self.connection.update_path()
            return True

    def eventFilter(self, watched, event):
        """Override the default eventFilter method to handle different kinds of events."""
        if self.dag and self.dag.official:
            return False
        if type(event) == QtWidgets.QWidgetItem:
            return False
        if event.type() == QtCore.QEvent.GraphicsSceneMousePress:
            if self._mouse_press_event(event):
                return True
        elif event.type() == QtCore.QEvent.KeyPress:
            if self._delete_key_press_event(event):
                return True
        elif event.type() == QtCore.QEvent.GraphicsSceneMouseMove:
            if self._mouse_move_event(event):
                return True
        elif event.type() == QtCore.QEvent.GraphicsSceneMouseRelease:
            if self._mouse_release_event(event):
                return True

        return super().eventFilter(watched, event)

    @staticmethod
    def _get_node_name(connection):
        """convert node to node name."""
        start_node = connection.start_port.node
        end_node = connection.end_port.node
        # start_node_name = start_node.identifier
        # end_node_name = end_node.identifier
        return start_node, end_node

    @slot_catch_exception(int, str, str)
    def _modify_edges_weight(self, weight: int, start_node: str, end_node: str):
        """Slot to update DAG edge's weight."""
        self.dag.adj[start_node][end_node]["weight"] = weight
        logger.info(f"update {start_node}--->{end_node} weight to {weight}")

    def get_subgraph(self, root_node: str, tail_nodes: List[str]):
        """Get the subgraph with root node and tail nodes.

        Args:
            root_node (str): Start node of the subgraph.
            tail_nodes (list): End node of the subgraph.
        """
        is_valid = True
        start_node = root_node or self.dag.root_node
        end_nodes = tail_nodes or self.dag.tail_nodes

        # if subgraph equal to raw graph, just need check edge's color.
        if start_node == self.dag.root_node and end_nodes == self.dag.tail_nodes:
            for pip_item in self.edges.values():
                pip_item.is_sub_graph = False
            self._sub_dag = None
            self.isvalid.emit(is_valid)
        else:
            subgraph = self.dag.subgraph(start_node, end_nodes)
            if subgraph.adj:
                for tail_node in end_nodes:
                    if tail_node not in subgraph.adj:
                        is_valid = False
                        break
                if is_valid:
                    # change path colors.
                    for pip_name, pip_item in self.edges.items():
                        if pip_name in subgraph.edges:
                            pip_item.is_sub_graph = True
                        else:
                            pip_item.is_sub_graph = False
                    self._sub_dag = subgraph
                self.isvalid.emit(is_valid)
            else:
                is_valid = False
                self.isvalid.emit(is_valid)

    def get_run_dag(self):
        """Get the real running dag."""
        if self._sub_dag is not None:
            return self._sub_dag
        else:
            return self.dag
