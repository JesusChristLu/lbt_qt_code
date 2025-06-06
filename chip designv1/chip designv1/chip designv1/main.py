from get_alg import Get_alg
from clustering import Clustering
from prune import Prune
from split import Split
from lattice_chip import Lattice
from compilation import Compilation
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt

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

# compilation
from qiskit.transpiler import Layout, CouplingMap
from qiskit.transpiler import PassManager
from qiskit.transpiler import passes
from qiskit.transpiler import preset_passmanagers
from qiskit.transpiler import PropertySet
from qiskit.compiler import transpile


if __name__=="__main__":
    #path = 'F:\\vs experiment\\chip designv1\\ibm_qx_mapping-master\\examples\\'
    #path_list = os.listdir(path)
    #for file_name in path_list[:]:
    #    alg = Get_alg(os.path.join(path, file_name))
    #    g = alg.graph
    #    prog = alg.prog

    #    community = Clustering(g)
    #    for i in community.G.nodes:
    #        print(i, community.G.nodes[i]['community_rank'])

    #g = nx.random_partition_graph([10,10,10],0.8,0.05)
    #g = nx.windmill_graph(3, 8)
    #g = nx.ring_of_cliques(3, 8)

    #for edge in g.edges:
        #g.edges[edge]['weight'] = np.random.random() + 0.01


    path = 'F:\\vs experiment\\chip designv1\\ibm_qx_mapping-master\\examples\\plus63mod4096_163.qasm'
    
    #path = 'F:\\vs experiment\\chip designv1\\ibm_qx_mapping-master\\examples\\mod10_171.qasm'
    alg = Get_alg(path)
    g = alg.graph
    edges = alg.edges
    mat = nx.to_numpy_array(g)
    originQNumber = len(list(g.nodes))

    community = Clustering(g)
    for i in community.G.nodes:
        print(i, community.G.nodes[i]['community_rank'])
    graph = community.G
    prune = Prune(graph, nx.to_numpy_array(g))
    graph = prune.graph
    recover = prune.recover
    split = Split(graph, edges, recover, mat)
    algChip = split.graph
    layout = split.layout
    QNumber = len(list(graph.nodes))
    print(QNumber - originQNumber)

    lattice = Lattice(16, mat)
    triangle_chip = lattice.triangular_lattice
    check_box_chip = lattice.check_box_lattice

    liLayout = lattice.liLayout
    liChip = lattice.liChip

    compilationCheckBoxChip = Compilation(check_box_chip, alg.qc, layoutMethod='ibm', routingMethod='ibm')
    compilationTriangleChip = Compilation(triangle_chip, alg.qc, layoutMethod='ibm', routingMethod='ibm')

    compilationLiChip = Compilation(liChip, alg.qc, layoutMethod='li', routingMethod='ibm', setLayout=liLayout)

    compilation1 = Compilation(algChip, [alg.qc, alg.alg], layoutMethod='chip', routingMethod='chip', setLayout=layout)
    compilation2 = Compilation(algChip, [alg.qc, alg.alg], layoutMethod='chip', routingMethod='ibm', setLayout=layout)

