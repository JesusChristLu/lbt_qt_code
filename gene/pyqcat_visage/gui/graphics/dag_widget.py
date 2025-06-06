# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/15
# __author:       HanQing Shi
"""DAG widgets."""
from copy import deepcopy
from typing import TYPE_CHECKING, Dict

from PySide6 import QtWidgets, QtGui
from PySide6.QtWidgets import QGraphicsPathItem, QMessageBox
from loguru import logger

from pyQCat.structures import QDict
from pyqcat_visage.gui.graphics.node_editor import NodeEditor
from pyqcat_visage.gui.graphics.node_item import NodeItem
from pyqcat_visage.gui.graphics.pipe_item import PipeItem
from pyqcat_visage.gui.graphics.scene import NodeScene
from pyqcat_visage.gui.graphics.types import NodeRole
from pyqcat_visage.gui.graphics.view import View
from pyqcat_visage.backend.model_dag import ModelDag
from ...config import GUI_CONFIG

if TYPE_CHECKING:
    from ..main_window import VisageGUI
    from pyqcat_visage.backend.experiment import VisageExperiment
from .types import PortPosEnum

default_conf = GUI_CONFIG.graphics_view.theme_classic_dark


class DAGWidget(QtWidgets.QWidget):
    """DAG widget to control all nodes and edges."""

    def __init__(self, gui: "VisageGUI", parent=None, color_conf=None):
        super().__init__(parent)
        # Parent GUI related
        self.gui = gui
        self._current_dag = None

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        self.node_editor = NodeEditor(self)
        self.scene = NodeScene()
        self.scene.setSceneRect(-2500, -2500, 5000, 5000)
        self.color_conf = color_conf if color_conf else self.gui.graphics_theme
        self.view = View(self, self.color_conf)
        self.view.setScene(self.scene)
        self.node_editor.install(self.scene)

        main_layout.addWidget(self.view)

        # connect signals and slots.
        self.view.request_node.connect(self.create_node)
        self.view.select_subgraph.connect(self.node_editor.get_subgraph)
        self.node_editor.isvalid.connect(self.view.judge_subgraph)
        self.init_theme()

    def init_theme(self, color_conf: QDict = None, rerender: bool = False):
        if color_conf:
            self.color_conf = color_conf
        self.view.init_theme(self.color_conf)
        for _, edge in self.node_editor.edges.items():
            edge.init_theme(self.color_conf)
        for _, node in self.node_editor.nodes.items():
            node.init_theme(self.color_conf)
        if rerender:
            self.hide()
            self.show()

    def load_dag(self, dag: ModelDag):
        """Load the DAG and clear exist DAG.

        Args:
            dag(ModelDag): model dag.
        """
        if dag.name != self._current_dag:
            self.gui.ui.dag_label.setText(f'| DAG({dag.name}) ')
            self.gui.ui.tabWidget.setCurrentIndex(0)

            self.node_editor.dag = dag
            self._current_dag = dag.name
            self.scene.clear()

            # clear node role before build.
            self.view.root_node = None
            self.view.tail_nodes.clear()
            self.node_editor.nodes.clear()
            self.node_editor.edges.clear()

            node_item_dict = self.node_editor.nodes
            # add nodes and edges.
            for start_node_name, edges in self.node_editor.dag.adj.items():
                start_node_item: NodeItem = self._get_node_from_visage_exp(
                    start_node_name, node_item_dict
                )
                for end_node_name, weight in edges.items():
                    end_node_item: NodeItem = self._get_node_from_visage_exp(
                        end_node_name, node_item_dict
                    )
                    edge = PipeItem(weight=weight["weight"], color_conf=self.color_conf)
                    self.node_editor.edges[f"{start_node_name}-{end_node_name}"] = edge
                    self.scene.addItem(edge)
                    edge.start_port = start_node_item.output_port
                    edge.end_port = end_node_item.input_port
                    edge.start_pos = edge.start_port.scenePos()
                    edge.end_pos = edge.end_port.scenePos()
                    edge.update_port_pos()

            if self.node_editor.dag.adj:
                logger.info(
                    f"Load DAG {self.node_editor.dag} finished. "
                    f"Root node: {self.node_editor.dag.root_node} "
                    f"Tail nodes: {self.node_editor.dag.tail_nodes}"
                )
                self.view.select_subgraph.emit(self.view.root_node, self.view.tail_nodes)

    def create_node(self, name: str):
        """Create node from standard experiment library.

        Args:
            name (str): The experiment name.
        """
        # fixed bug. Only customize DAG can create node.
        if not self.node_editor.dag:
            QMessageBox.critical(
                self, "FAILED", "Please select or create a DAG at first."
            )
        elif self.node_editor.dag.official:
            QMessageBox.critical(
                self,
                "FAILED",
                "The standard DAG flows does not allow for changes.\n"
                "You can create a new custom DAG to do whatever you want.",
            )
        elif name.startswith("BatchExperiment"):
            QMessageBox.critical(
                self, "FAILED", "BatchExperiment no support to create a DAG."
            )
        else:
            # build node item.
            brunch_name, leaf_name = name.split("|")
            node = NodeItem(self.gui, name=leaf_name, color_conf=self.color_conf)
            node.add_port(name="In", port_type="in")
            node.add_port(name="Out", port_type="out")
            node.build()
            self.scene.addItem(node)

            # set node positions.
            global_pos = QtGui.QCursor.pos()
            logger.debug(f"{leaf_name} global pos={global_pos.x(), global_pos.y()}")
            pos = self.view.mapFromGlobal(QtGui.QCursor.pos())
            logger.debug(f"{leaf_name} view pos={pos.x(), pos.y()}")
            scene_pos = self.view.mapToScene(pos)
            node.xy_pos = [scene_pos.x(), scene_pos.y()]
            logger.debug(f"{node.id} scene pos={node.xy_pos}")

            # copy from exp dict
            node_params = deepcopy(self.gui.backend.experiments[brunch_name][leaf_name])
            node.wrapper_node(node_params)
            self.node_editor.dag.add_node(node.identifier, node_params)
            self.node_editor.nodes[node.identifier] = node
            logger.info(f"add node {node.identifier} to DAG {self.node_editor.dag}.")

    def _get_node_from_visage_exp(
        self, node_name: str, node_item_dict: Dict
    ) -> NodeItem:
        """Get node from visage experiment."""
        if node_name not in node_item_dict:
            exp: "VisageExperiment" = self.node_editor.dag.node_params[node_name]
            node_item = NodeItem(self.gui, name=exp.name, id_=str(exp.gid), color_conf=self.color_conf)

            if self.node_editor.dag.official:
                node_item.setFlags(QGraphicsPathItem.ItemIsSelectable)

            # set node position.
            node_item.xy_pos = exp.location
            in_port_pos, _ = exp.port_pos
            if in_port_pos == PortPosEnum.LEFT.value:
                swap_pos = False
            else:
                swap_pos = True
            node_item.add_port(name="In", port_type="in", swap_pos=swap_pos)
            node_item.add_port(name="Out", port_type="out", swap_pos=swap_pos)

            # set node role.
            if exp.role == NodeRole.ROOT.value:
                node_item.node_role = exp.role
                self.view.root_node = node_name
            elif exp.role == NodeRole.TAIL.value:
                node_item.node_role = exp.role
                self.view.tail_nodes.append(node_name)

            # build node.
            node_item.build()
            node_item.wrapper_node(exp)

            self.scene.addItem(node_item)

            node_item_dict[node_name] = node_item
        else:
            node_item = node_item_dict[node_name]
        return node_item

    def node_color_change(self, node_params: Dict):
        """Change node color according to the node status."""
        if node_params:
            for node_name, node_status in node_params.items():
                node_item: NodeItem = self.node_editor.nodes.get(node_name)
                if node_item:
                    node_item.node_status = node_status
                    node_item.update()
