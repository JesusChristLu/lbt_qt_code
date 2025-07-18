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
    ases = np.arange(0.1, 1, 0.05)
    #ases = [0.5]
    for a in ases:
        print(a)
        show = False
        path = 'F:\\vs experiment\\chipdesignv2\\ibm_qx_mapping-master\\examples\\'
        path_list = os.listdir(path)
        minSwapList = [0, 0, 0, 0]
        minDepthList = [0, 0, 0, 0]
        depthes = []
        liAddDepth = []
        crossAddDepth = []
        triAddDepth = []
        myAddDepth = []
        cxs = []
        liAddSwap = []
        crossAddSwap = []
        triAddSwap = []
        myAddSwap = []
        method = ['check box', 'triangle', 'li', 'chip']
        swapMethodBestDict = {'check box' : [], 'triangle' : [], 'li' : [], 'chip' : []}
        depthMethodBestDict = {'check box' : [], 'triangle' : [], 'li' : [], 'chip' : []}
        total = 0
        #for file_name in path_list[:]:
        for file_name in [path_list[path_list.index('qft_16.qasm')]]:
            print(file_name)
            alg = Get_alg(os.path.join(path, file_name), show)
            fsize = os.path.getsize(os.path.join(path, file_name)) / float(1024)
            print(fsize)
            if fsize > 384:
            #if fsize > 3:
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
                split = Split(a, graph, edges, recover, mat, show)
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
            depthes.append(alg.qc.depth())
            cxs.append(dict(alg.qc.count_ops())['cx'])
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

            myAddDepth.append(compilationChip2.additionDepth)
            liAddDepth.append(compilationLiChip.additionDepth)
            crossAddDepth.append(compilationCheckBoxChip.additionDepth)
            triAddDepth.append(compilationTriangleChip.additionDepth)
            myAddSwap.append(compilationChip2.additionSwap)
            liAddSwap.append(compilationLiChip.additionSwap)
            crossAddSwap.append(compilationCheckBoxChip.additionSwap)
            triAddSwap.append(compilationTriangleChip.additionSwap)

            compareSwap = [compilationCheckBoxChip.additionSwap, compilationTriangleChip.additionSwap, compilationLiChip.additionSwap, compilationChip2.additionSwap]
            compareDepth = [compilationCheckBoxChip.additionDepth, compilationTriangleChip.additionDepth, compilationLiChip.additionDepth, compilationChip2.additionDepth]

            minSwap = np.where(np.array(compareSwap) == np.min(np.array(compareSwap)))[0]
            minDepth = np.where(np.array(compareDepth) == np.min(np.array(compareDepth)))[0]


            resultFile = 'compile' + str(a)[:3] + '\\ ' + file_name
            bestFile = 'compile' + str(a)[:3] + '\\best.txt'
            plotFile = 'compile' + str(a)[:3] + '\\plot.txt'
            with open(resultFile, 'w') as fp:
                fp.write('file size ' + str(fsize) + '\n')
                fp.write('a=' + str(a))
                fp.write('qasm depth ' + str(alg.qc.depth()) + '\n')
                fp.write('qasm gate number:\n')
                fp.write(str(dict(alg.qc.count_ops())) + str(alg.qc.__len__()) + '\n')
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
                    swapMethodBestDict[method[i]].append(file_name)
                    fp.write('the best swap ' + method[i] + '\n')
                for i in minDepth:
                    minDepthList[i] += 1
                    depthMethodBestDict[method[i]].append(file_name)
                    fp.write('the best depth ' + method[i] + '\n')
            
        
            print(minSwapList)
            print(minDepthList)

        #with open(bestFile, 'w') as fp:
        #    fp.write('best swap dict\n')
        #    for method in swapMethodBestDict:
        #        fp.write(method + ' :\n')
        #        for f in swapMethodBestDict[method]:
        #            fp.write(f + '\n')
        #    fp.write('\nbest depth dict\n')
        #    for method in depthMethodBestDict:
        #        fp.write(method + ' :\n')
        #        for f in depthMethodBestDict[method]:
        #            fp.write(f + '\n')

        #with open(plotFile, 'w') as fp:
        #    for d in range(len(depthes)):
        #        fp.write(str(depthes[d]) + ' ' + str(cxs[d]) + ' ' + 
        #                 str(myAddDepth[d]) + ' ' + str(myAddSwap[d]) + ' ' + 
        #                 str(triAddDepth[d]) + ' ' + str(triAddSwap[d]) + ' ' + 
        #                 str(liAddDepth[d]) + ' ' + str(liAddSwap[d]) + ' ' + 
        #                 str(crossAddDepth[d]) + ' ' + str(crossAddSwap[d]) + '\n')