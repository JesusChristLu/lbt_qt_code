import numpy as np
import networkx as nx

from copy import deepcopy

# We import plotting tools 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# importing Qiskit
from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute

from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

from qiskit.quantum_info import Kraus, SuperOp, Operator, Pauli

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

from qiskit.providers.aer.noise import mixed_unitary_error
from qiskit.providers.aer.noise import coherent_unitary_error
from qiskit.providers.aer.noise import reset_error
from qiskit.providers.aer.noise import phase_amplitude_damping_error
from qiskit.providers.aer.noise import amplitude_damping_error
from qiskit.providers.aer.noise import phase_damping_error
from qiskit.providers.aer.noise import kraus_error



class Get_alg():
    
    def __init__(self, fn, show=False):
        self.show = show
        self.mat, self.alg, self.edges, self.qasm_str = self.file_to_matrix(fn)
        self.graph = self.matrix_to_graph(self.mat)
        self.qc = self.to_qc(self.qasm_str)


    def file_to_matrix(self, fn): 
        # return mat, alg, edges, newCircuit, mat is the matrix of algorithm, alg contains all the gates
        # edges contains only the two bite gate block, newCircuit is the qasm whose qubit number is changed.
        with open(fn, 'r') as fp: 
            circuit = fp.read()
        fp.close()
        gates = circuit.split('\n')
        edges = []
        alg = []
        biggest = 0

        for i in gates[4:]:
            number = ''
            numbers = []
            start = False
            for j in i:
                if j == '[':
                    start = True
                elif j == ']':
                    start = False
                    if int(number) > biggest:
                        biggest = int(number)
                    numbers.append(int(number))
                    number = ''
                elif start:
                    number += j
            if len(numbers) > 0:
                alg.append(numbers)
            if len(numbers) == 2:
                edges.append(numbers)

        for i in range(len(edges) - 1):
            if len(edges[i]) == 3:
                continue
            for j in range(i + 1, len(edges)):
                if edges[i] == edges[j] or edges[i] == list(reversed(edges[j])):
                    edges[j].append(0)
                else:
                    break

        cp_edges = deepcopy(edges)
        edges = []
        for i in cp_edges:
            if len(i) == 2:
                edges.append(i)

        mat = np.zeros((biggest + 1, biggest + 1))
        for edge in edges:
            mat[edge[0]][edge[1]] += 1

        mat = mat + mat.transpose()

        newCircuit = ''
        for char in gates:
            if not 'reg' in char:
                newCircuit += char + '\n'
            elif 'q' in char:
                newCircuit += 'qreg q[' + str(biggest + 1) + '];\n'
            else:
                newCircuit += 'creg c[' + str(biggest + 1) + '];\n'
        return mat, alg, edges, newCircuit

    def to_qc(self, fn):
        # get the quantum circuit from qasm string
        qc = QuantumCircuit.from_qasm_str(fn)
        return qc


    def matrix_to_graph(self, mat): 
        # get the graph from mat 
        G = nx.from_numpy_matrix(mat)
        edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
        weights = np.array(weights)     
        max_weights = np.max(weights)
        min_weights = np.min(weights)
        if max_weights == min_weights:
            weights = 5
        else:
            weights = 10 * ((weights - min_weights) / (max_weights - min_weights) + 0.1)
        if self.show:
            plt.ion()
        
            plt.title('origin graph')
            nx.draw(G, node_color='y', edgelist=edges, width=weights, edge_cmap=plt.cm.Reds, edge_color='red', with_labels=True, node_size = 500)
            plt.draw()
            plt.pause(15)
            plt.close()
        
        return G