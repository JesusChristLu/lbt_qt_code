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
    show = False
    path = 'F:\\vs experiment\\chipdesignv2\\ibm_qx_mapping-master\\examples\\'
    path_list = os.listdir(path)
    minSwapList = [0, 0, 0, 0]
    minDepthList = [0, 0, 0, 0]
    total = 0
    for file_name in path_list[:]:
        print(file_name)
        alg = Get_alg(os.path.join(path, file_name), show)
        fsize = os.path.getsize(os.path.join(path, file_name)) / float(1024)
        print(fsize)
        if fsize > 384:
            continue
        total += 1

        g = alg.graph
        edges = alg.edges
        mat = alg.mat

        originQNumber = np.size(mat, axis=0)

        community = Clustering(g, show)
        for i in community.G.nodes:
            print(i, community.G.nodes[i]['community_rank'])
        graph = community.G
        prune = Prune(graph, nx.to_numpy_array(g), show)
        graph = prune.graph
        if prune.needPrune:
            recover = prune.recover
            split = Split(graph, edges, recover, mat, show)
            algChip = split.graph
            chipLayout = split.layout
        else:
            algChip = graph
            chipLayout = {}
            for node in range(originQNumber):
                chipLayout[node] = [node]


        lattice = Lattice(originQNumber, mat)
        triangle_chip = lattice.triangular_lattice
        check_box_chip = lattice.check_box_lattice

        liLayout = lattice.liLayout
        liChip = lattice.liChip

        print('origin circuit: ', dict(alg.qc.count_ops()), ' depth: ', alg.qc.depth(), ' bit: ', originQNumber)


        print('chip ibm qubit number', len(list(algChip.nodes)))
        compilationChip2 = Compilation(algChip, [alg.qc, alg.alg], layoutMethod='chip', routingMethod='ibm', setLayout=chipLayout)


        print('check box qubit number', len(list(check_box_chip.nodes)))
        compilationCheckBoxChip = Compilation(check_box_chip, alg.qc, layoutMethod='ibm', routingMethod='ibm')
        print('triangle chip qubit number', len(list(triangle_chip.nodes)))
        compilationTriangleChip = Compilation(triangle_chip, alg.qc, layoutMethod='ibm', routingMethod='ibm')
        print('li chip qubit number', len(list(liChip.nodes)))
        compilationLiChip = Compilation(liChip, alg.qc, layoutMethod='li', routingMethod='ibm', setLayout=liLayout)

        betterCheckBoxSwap = (-compilationChip2.additionSwap + compilationCheckBoxChip.additionSwap) * 100 / max(1, compilationCheckBoxChip.additionSwap)
        betterCheckBoxDepth = (-compilationChip2.additionDepth + compilationCheckBoxChip.additionDepth) * 100 / max(1, compilationCheckBoxChip.additionDepth)
        betterTriangleSwap = (-compilationChip2.additionSwap + compilationTriangleChip.additionSwap) * 100 / max(1, compilationTriangleChip.additionSwap)
        betterTriangleDepth = (-compilationChip2.additionDepth + compilationTriangleChip.additionDepth) * 100 / max(1, compilationTriangleChip.additionDepth)
        betterLiSwap = (-compilationChip2.additionSwap + compilationLiChip.additionSwap) * 100 / max(1, compilationLiChip.additionSwap)
        betterLiDepth = (-compilationChip2.additionDepth + compilationLiChip.additionDepth) * 100 / max(1, compilationLiChip.additionDepth)

        compareSwap = [compilationCheckBoxChip.additionSwap, compilationTriangleChip.additionSwap, compilationLiChip.additionSwap, compilationChip2.additionSwap]
        compareDepth = [compilationCheckBoxChip.additionDepth, compilationTriangleChip.additionDepth, compilationLiChip.additionDepth, compilationChip2.additionDepth]

        minSwap = np.where(np.array(compareSwap) == np.min(np.array(compareSwap)))[0]
        minDepth = np.where(np.array(compareDepth) == np.min(np.array(compareDepth)))[0]

        method = ['check box', 'triangle', 'li', 'chip']
        resultFile = 'compile\\ ' + file_name
        with open(resultFile, 'a+') as fp:
            fp.write('file size ' + str(fsize) + '\n')
            fp.write('qasm depth ' + str(alg.qc.depth()) + '\n')
            fp.write('qasm gate number:\n')
            fp.write(str(dict(alg.qc.count_ops())) + '\n')
            fp.write(str(originQNumber) + ' bit\n')
            fp.write('after check box chip\n')
            fp.write('check box qubit number ' + str(len(list(check_box_chip.nodes))) + '\n')
            fp.write('chech box add ' + str(compilationCheckBoxChip.additionSwap) + ' swap, ' + str(compilationCheckBoxChip.additionDepth) + ' depth\n')
            fp.write('after triangle chip\n')
            fp.write('triangle qubit number ' + str(len(list(triangle_chip.nodes))) + '\n')
            fp.write('triangle add ' + str(compilationTriangleChip.additionSwap) + ' swap, ' + str(compilationTriangleChip.additionDepth) + ' depth\n')
            fp.write('after li chip\n')
            fp.write('li qubit number ' + str(len(list(liChip.nodes))) + '\n')
            fp.write('li add ' + str(compilationLiChip.additionSwap) + ' swap, ' + str(compilationLiChip.additionDepth) + ' depth\n')
            fp.write('after chip\n')
            fp.write('qubit number ' + str(len(list(algChip.nodes))) + '\n')
            fp.write('add ' + str(compilationChip2.additionSwap) + ' swap, ' + str(compilationChip2.additionDepth) + ' depth\n')
            fp.write('better than check box swap ' + str(round(betterCheckBoxSwap, 1)) + '%\n')
            fp.write('better than check box depth ' + str(round(betterCheckBoxDepth, 1)) + '%\n')
            fp.write('better than triangle swap ' + str(round(betterTriangleSwap, 1)) + '%\n')
            fp.write('better than triangle depth ' + str(round(betterTriangleDepth, 1)) + '%\n')
            fp.write('better than li swap ' + str(round(betterLiSwap, 1)) + '%\n')
            fp.write('better than li depth ' + str(round(betterLiDepth, 1)) + '%\n')
            for i in minSwap:
                minSwapList[i] += 1
                fp.write('the best swap ' + method[i] + '\n')
            for i in minDepth:
                minDepthList[i] += 1
                fp.write('the best depth ' + method[i])
            

        print(minSwapList)
        print(minDepthList)






    #g = nx.random_partition_graph([10, 4, 16, 6, 5, 8, 7],0.8,0.05)
    #g = nx.windmill_graph(3, 8)
    #g = nx.ring_of_cliques(3, 8)

    #for edge in g.edges:
    #    g.edges[edge]['weight'] = np.random.random() + 0.01   