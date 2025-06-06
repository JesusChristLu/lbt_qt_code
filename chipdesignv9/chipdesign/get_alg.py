import numpy as np
import networkx as nx

from copy import deepcopy

# We import plotting tools 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# importing Qiskit
from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

from qiskit.quantum_info import Kraus, SuperOp, Operator, Pauli


class Get_alg():
    
    def __init__(self, fn, show=False, is_qasm=False):
        self.show = show
        if fn[-11:-1] == 'random.qas':
            b_number = int(fn[-18:-15])
            d = int(fn[-23:-19])
            while True:
                fn = self.random_circuit_generator(b_number, d)
                self.mat, self.alg, self.twoBitBlocks, self.qasm_str, self.depth = self.file_to_circuit(fn, True)
                self.graph = self.matrix_to_graph(self.mat)
                if nx.is_connected(self.graph) and len(self.mat) == b_number:
                    break
        else:
            self.mat, self.alg, self.twoBitBlocks, self.qasm_str, self.depth = self.file_to_circuit(fn, is_qasm)
            self.graph = self.matrix_to_graph(self.mat)
        self.qc = self.to_qc(self.qasm_str)

    def random_circuit_generator(self, b_number, d):
        edges = {}
        edge_p = np.random.random(int((b_number * (b_number - 1)) / 2))
        edge_p /= np.sum(edge_p)
        accumulate = 0
        n = 0
        for i in range(b_number - 1):
            for j in range(i + 1, b_number):
                edges[(i, j)] = accumulate + edge_p[n]
                accumulate = edges[(i, j)]
                n += 1
        alg = []
        bit_layer = dict(zip(range(b_number), [0 for _ in range(b_number)]))
        while True:
            choice = np.random.random()
            for p_range in list(edges.values()):
                if choice <= p_range:
                    edge = list(edges.keys())[list(edges.values()).index(p_range)]
                    max_layer = max([bit_layer[edge[0]], bit_layer[edge[1]]])
                    bit_layer[edge[0]] = max_layer + 1
                    bit_layer[edge[1]] = max_layer + 1
                    alg.append(edge) 
                    break
            if max(list(bit_layer.values())) >= d:
                break


        qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        qasm += 'qreg q[' + str(b_number) + '];\n'
        qasm += 'creg c[' + str(b_number) + '];\n'

        for gate in alg:
            qasm += 'cx q[' + str(gate[0]) + '],q['+ str(gate[1]) + '];\n'
        return qasm

    def file_to_circuit(self, fn, is_qasm): 
        # return mat, alg, edges, newCircuit, mat is the matrix of algorithm, alg contains all the gates
        # edges contains only the two bite gate block, newCircuit is the qasm whose qubit number is changed.

        if not is_qasm:
            with open(fn, 'r') as fp: 
                circuit = fp.read()
            fp.close()
        else:
            circuit = fn

        circuit = self.to_qc(circuit)
        depth = circuit.depth()
        circuit = circuit.qasm()

        gates = circuit.split('\n')
        twoBitBlocks = []
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
                twoBitBlocks.append(deepcopy(numbers))

        for i in range(len(twoBitBlocks) - 1):
            if len(twoBitBlocks[i]) == 3:
                continue
            for j in range(i + 1, len(twoBitBlocks)):
                if twoBitBlocks[i] == twoBitBlocks[j] or twoBitBlocks[i] == list(reversed(twoBitBlocks[j])):
                    twoBitBlocks[j].append(0)
                else:
                    break

        cp_twoBitBlocks = deepcopy(twoBitBlocks)
        twoBitBlocks = []
        for i in cp_twoBitBlocks:
            if len(i) == 2:
                twoBitBlocks.append(i)
        mat = np.zeros((biggest + 1, biggest + 1))
        for twoBitBlock in twoBitBlocks:
            mat[twoBitBlock[0]][twoBitBlock[1]] += 1

        mat = mat + mat.transpose()
        if not is_qasm:
            newCircuit = ''
            for char in gates:
                if not 'reg' in char:
                    newCircuit += char + '\n'
                elif 'q' in char:
                    newCircuit += 'qreg q[' + str(biggest + 1) + '];\n'
                else:
                    newCircuit += 'creg c[' + str(biggest + 1) + '];\n'
        else:
            newCircuit = circuit
        return mat, alg, twoBitBlocks, newCircuit, depth

    def to_qc(self, str):
        # get the quantum circuit from qasm string
        qc = QuantumCircuit.from_qasm_str(str)
        return qc


    def matrix_to_graph(self, mat): 
        # get the graph from mat 
        G = nx.from_numpy_matrix(mat)
        edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
        weights = np.array(weights)     
        max_weights = np.max(weights)
        min_weights = np.min(weights)
        if max_weights == min_weights:
            weights = 1
        else:
            weights = 5 * ((weights - min_weights) / (max_weights - min_weights) + 0.1)
        if self.show:
            sns.heatmap(mat, annot = True, cmap='Reds', annot_kws={'size':15})
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title('CCG matrix')
            plt.savefig("CCG matrix.pdf", dpi = 300)
            plt.show()

            labels_params = {"font_size":15}
            pos = nx.circular_layout(G)
            plt.title('CCG')
            nx.draw(G, pos, **labels_params, node_color='y', edgelist=edges, width=weights, edge_color='red', with_labels=True, node_size = 500)
            
            plt.savefig("CCG.pdf", dpi = 300)
            plt.show()
        
        return G