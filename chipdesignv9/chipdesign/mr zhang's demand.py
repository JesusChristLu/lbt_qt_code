from get_alg import Get_alg
from prune import Prune
from split import Split
from lattice_chip import Lattice
from compilation import Compilation
import numpy as np
import networkx as nx
from networkx.algorithms.planarity import check_planarity
import os
import matplotlib.pyplot as plt

# importing Qiskit
from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute

from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Kraus, SuperOp, Operator, Pauli

# compilation
from qiskit.transpiler import Layout, CouplingMap
from qiskit.transpiler import PassManager
from qiskit.transpiler import passes
from qiskit.transpiler import preset_passmanagers
from qiskit.transpiler import PropertySet
from qiskit.compiler import transpile


if __name__=="__main__":
    show = False
    path = 'F:\\vs experiment\\chipdesignv9\\ibm_qx_mapping-master\\examples\\'
    path_list = os.listdir(path)
    minSwapList = [0, 0, 0, 0]
    minDepthList = [0, 0, 0, 0]
    method = ['cross square', '2d lattice', 'Li', 'SPQPD']
    swapMethodBestDict = {'cross square' : [], '2d lattice' : [], 'Li' : [], 'SPQPD' : []}
    depthMethodBestDict = {'cross square' : [], '2d lattice' : [], 'Li' : [], 'SPQPD' : []}
    total = 0
    use_random = False
    kind = 'r'
    if use_random:
        name_list = []
        if kind == 'b':
            d = 30
            for b_number in range(10, 301, 10):
                str_d = str(d)
                while len(str_d) < 4:
                    str_d = '0' + str_d
                str_b_number = str(b_number)
                while len(str_b_number) < 3:
                    str_b_number = '0' + str_b_number
                for epoch in range(9):
                    str_ep = str(epoch)
                    while len(str_ep) < 2:
                        str_ep = '0' + str_ep
                    name_list.append(str_d + ' ' + str_b_number  + ' ' + str_ep + ' ' + 'random.qasm')
        elif kind == 'd':
            b_number = 20
            for d in range(50, 2001, 50):
                str_d = str(d)
                while len(str_d) < 4:
                    str_d = '0' + str_d
                str_b_number = str(b_number)
                while len(str_b_number) < 3:
                    str_b_number = '0' + str_b_number
                for epoch in range(9):
                    str_ep = str(epoch)
                    while len(str_ep) < 2:
                        str_ep = '0' + str_ep
                    name_list.append(str_d + ' ' + str_b_number  + ' ' + str_ep + ' ' + 'random.qasm')
    else:
        name_list = path_list
    name_list_len = len(name_list)
    media_vertex_per_cent = []
    larger_than_6_per_cent = []
    larger_than_4_per_cent = []
    not_planar = 0
    for file_name in name_list:
        print('reading')
        print(file_name)
        alg = Get_alg(os.path.join(path, file_name), show)
        g = alg.graph
        bit_number = len(g.nodes)
        print(max(list(dict(g.degree()).values())) / (bit_number - 1), min(list(dict(g.degree()).values())) / (bit_number - 1), 
            np.array(list(dict(g.degree()).values())).mean(), np.array(list(dict(g.degree()).values())).std())
        media_vertex = 0
        larger_than_4 = 0
        larger_than_6 = 0
        if not check_planarity(g)[0]:
            not_planar += 1
        for node in g.nodes:
            if g.degree[node] == bit_number - 1:
                media_vertex += 1
            if g.degree[node] > 6:
                larger_than_6 += 1
            if g.degree[node] > 4:
                larger_than_4 += 1
        print(media_vertex / bit_number, larger_than_6 / bit_number, larger_than_4 / bit_number)
        media_vertex_per_cent.append(media_vertex / bit_number)
        larger_than_4_per_cent.append(larger_than_4 / bit_number)
        larger_than_6_per_cent.append(larger_than_6 / bit_number)
    print(not_planar / name_list_len, np.array(media_vertex_per_cent).mean(), np.array(larger_than_6_per_cent).mean(), np.array(larger_than_4_per_cent).mean())