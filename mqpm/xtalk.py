from copy import deepcopy
import networkx as nx
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag

class Xtalk():

    def __init__(self, frequency_range):
        self.frequency_range = frequency_range

    def get_circuit_subgraph(self, qc, chip, layout):
        xtalk_matrix = chip.xtalk_matrix
        if isinstance(qc, QuantumCircuit):
            qc = circuit_to_dag(qc)
        layers = list(qc.layers())
        circuit_subgraphs = []
        xtalk_graphs = []
        for layer in layers:
            gates = layer['graph'].gate_nodes()
            single_qubit_gates = []
            two_qubit_gates = []
            idle_qubits = list(chip.chip.nodes)
            circuit_subgraph = nx.Graph()
            circuit_subgraph.add_edges_from(chip.chip.edges)
            for node in circuit_subgraph.nodes:
                circuit_subgraph.nodes[node]['state'] = 0
            for gate in gates:
                virtual_qubits = gate.qargs
                physical_qubits = [layout.get_virtual_bits()[qubit] for qubit in virtual_qubits]
                for qubit in physical_qubits:
                    idle_qubits.remove(qubit)
                    circuit_subgraph.nodes[qubit]['state'] = 1
                if len(physical_qubits) == 2:
                    physical_qubits = tuple(physical_qubits)
                    two_qubit_gates.append(physical_qubits)
                    circuit_subgraph.edges[physical_qubits]['state'] = 1
                else:
                    single_qubit_gates.append(physical_qubits[0])
            circuit_subgraphs.append(circuit_subgraph)
            xtalk_graph = nx.Graph()
            for node1 in single_qubit_gates + two_qubit_gates + idle_qubits:
                if node1 in idle_qubits:
                    xtalk_graph.nodes[node1]['gate kind'] = 0
                elif node1 in single_qubit_gates:
                    xtalk_graph.nodes[node1]['gate kind'] = 1
                else:
                    xtalk_graph.nodes[node1]['gate kind'] = 2
                for node2 in single_qubit_gates + two_qubit_gates + idle_qubits:
                    if node1 == node2:
                        continue
                    if xtalk_matrix[chip.xtalk_index(node1)][chip.xtalk_index(node2)] > 0:
                        xtalk_graph.add_edge(chip.xtalk_index(node1), chip.xtalk_index(node2))
            xtalk_graphs.append(xtalk_graph)
        return tuple(zip(circuit_subgraphs, xtalk_graphs))

    def population_loss(self, frequency):
        return

    def frequency_allocation(self):
        return

    def color_distribution(self):
        return