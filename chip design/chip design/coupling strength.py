import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import pi

# We import the tools to handle general Graphs
import networkx as nx

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit

from qiskit.compiler import transpile
from qiskit.transpiler import PassManager

from qiskit.transpiler import passes
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import Collect2qBlocks

from qiskit.converters import circuit_to_dag
from qiskit.tools.visualization import dag_drawer

from circuits import QAOA, grover, BVAlgorithm, dj_algorithm, simonAlg, QFT, phase_estimation, shor

p = 1
n = 6
# Generating the ring graph with 6 nodes
V = np.arange(0, n, 1)
E = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0)] 
beta = np.random.random((p, 2))

work_bit_num = 5
n = work_bit_num * 2 
# circ = QuantumCircuit(3, 3)
# circ = QAOA(V, p, E, beta)
# circ = grover(work_bit_num)
# circ = BVAlgorithm(work_bit_num)
# circ = dj_algorithm('balanced', work_bit_num)
# circ = simonAlg(work_bit_num)
# circ = QFT(n)
n = 6
circ = phase_estimation(n - 1)
# circ = shor()
# n = 7

circ.draw(output='mpl')
plt.show()

pass_ = Unroller(['u1', 'u2', 'u3', 'cx'])
pm = PassManager(pass_)
new_circ = pm.run(circ)
new_circ.draw(output='mpl')
plt.show()

coupling_matrix = np.zeros((n, n), dtype=int)

dag = circuit_to_dag(new_circ)

for node in dag.op_nodes():
    if len(node.qargs) == 2:
        coupling_matrix[node.qargs[0].index][node.qargs[1].index] += 1

# coupling_matrix = np.array([[0,2,0,1],
#                             [2,0,1,1],
#                             [0,1,0,1],
#                             [1,1,1,0]])

coupling_matrix = (coupling_matrix + coupling_matrix.transpose())

node_degree = np.sum(coupling_matrix, 0)

sns.heatmap(coupling_matrix, cmap='Reds', annot = True, annot_kws={'size':15})
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("profiledemo2.pdf", dpi = 300)
plt.show()


G = nx.from_numpy_matrix(coupling_matrix)
edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
weights = np.array(weights)     
max_weights = np.max(weights)
min_weights = np.min(weights)
if max_weights == min_weights:
    weights = 5
else:
    weights = 10 * (weights - min_weights + 1) / (max_weights - min_weights)
pos = nx.shell_layout(G)
# pos=nx.circular_layout(G)

labels_params = {"font_size":30}

nx.draw(G, **labels_params, node_color='y', edgelist=edges, width=weights, edge_cmap=plt.cm.Reds, with_labels=True, node_size = 2000)
edge_color=weights,
plt.savefig("profiledemo3.pdf", dpi = 300)
plt.show()
