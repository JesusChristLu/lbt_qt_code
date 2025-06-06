from get_alg import Get_alg
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
    #ases = np.arange(0.1, 1, 0.05)
    ases = [0.5]
    for a in ases:
        print(a)
        show = False
        path = 'F:\\vs experiment\\chipdesignv2\\ibm_qx_mapping-master\\examples\\'
        path_list = os.listdir(path)
        minSwapList = [0, 0, 0, 0]
        minDepthList = [0, 0, 0, 0]
        depthes = []
        ns = []
        liAddDepth = []
        crossAddDepth = []
        triAddDepth = []
        CBDDAddDepth = []
        cxs = []
        liAddSwap = []
        crossAddSwap = []
        triAddSwap = []
        CBDDAddSwap = []
        method = ['cross square', 'triangle', 'li', 'CBDD']
        swapMethodBestDict = {'cross square' : [], 'triangle' : [], 'li' : [], 'CBDD' : []}
        depthMethodBestDict = {'cross square' : [], 'triangle' : [], 'li' : [], 'CBDD' : []}
        total = 0
        #for file_name in path_list[:]:
        #for file_name in path_list[0:10]:
        for file_name in [path_list[path_list.index('rd53_251.qasm')]]:
        #for file_name in [path_list[path_list.index('4gt4-v1_74.qasm')]]:
            print('reading')
            print(file_name)
            alg = Get_alg(os.path.join(path, file_name), show)
            fsize = os.path.getsize(os.path.join(path, file_name)) / float(1024)
            print(fsize)
            if fsize > 384:
                continue
            total += 1
            g = alg.graph
            twoBitBlocks = alg.twoBitBlocks
            mat = alg.mat

            originQNumber = np.size(mat, axis=0)
            scores = {}
            algChips = {}
            chipLayouts = {}
            
            for k in range(1, min(originQNumber + 1, 5)):
                print('pruning')
                prune = Prune(g, show, k)
                graph = prune.graph
                if prune.needPrune:
                    recover = prune.recover
                    print('spliting')
                    split = Split(a, graph, alg.graph, twoBitBlocks, recover, show)
                    algChip = split.graph
                    chipLayout = split.layout
                    if algChip == {} and k > 1:
                        continue
                else:
                    algChip = graph
                    chipLayout = {}
                    for node in range(originQNumber):
                        chipLayout[node] = [node]
                score = 0
                for twoBitBlock in twoBitBlocks:
                    score += nx.shortest_path_length(algChip, twoBitBlock[0], twoBitBlock[1])
                scores[k] = score
                if len(scores) > 1:
                    for key in range(k - 1, 0, -1):
                        if scores.get(key, False):
                            if scores[k] >= scores[key]:
                                scores.pop(k) # may be no use
                                break
                algChips[k] = algChip
                chipLayouts[k] = chipLayout
            print('finish')
            algChip = algChips[list(scores.keys())[list(scores.values()).index(min(list(scores.values())))]]
            chipLayout = chipLayouts[list(scores.keys())[list(scores.values()).index(min(list(scores.values())))]]
            lattice = Lattice(originQNumber, mat, show)
            triangle_chip = lattice.triangular_lattice
            cross_square_chip = lattice.cross_square_lattice

            liLayout = lattice.liLayout
            liChip = lattice.liChip

            print('Compiling')
            print('origin circuit: ', dict(alg.qc.count_ops()), ' depth: ', alg.qc.depth(), ' bit: ', originQNumber)
            depthes.append(alg.qc.depth())
            ns.append(originQNumber)
            cxs.append(dict(alg.qc.count_ops())['cx'])
            print('chip ibm qubit number', len(list(algChip.nodes)))
            compilationCBDDChip = Compilation(algChip, [alg.qc, alg.alg], layoutMethod='chip', routingMethod='ibm', setLayout=chipLayout)

            CBDD_qasm = compilationCBDDChip.out_qc_qasm
            CBDD_analysis = Get_alg(CBDD_qasm, show=False, is_qasm=True)
            origin_mat = np.sum(alg.mat, axis=0)
            CBDD_mat = np.sum(CBDD_analysis.mat, axis=0)

            print('check box qubit number', len(list(cross_square_chip.nodes)))
            compilationCrossSquareChip = Compilation(cross_square_chip, alg.qc, layoutMethod='ibm', routingMethod='ibm')
            print('triangle chip qubit number', len(list(triangle_chip.nodes)))
            compilationTriangleChip = Compilation(triangle_chip, alg.qc, layoutMethod='ibm', routingMethod='ibm')
            print('li chip qubit number', len(list(liChip.nodes)))
            compilationLiChip = Compilation(liChip, alg.qc, layoutMethod='li', routingMethod='ibm', setLayout=liLayout)

            betterCrossSquareSwap = (-compilationCBDDChip.additionSwap + compilationCrossSquareChip.additionSwap) * 100 / max(1, compilationCrossSquareChip.additionSwap)
            betterCrossSquareDepth = (-compilationCBDDChip.additionDepth + compilationCrossSquareChip.additionDepth) * 100 / max(1, compilationCrossSquareChip.additionDepth)
            betterTriangleSwap = (-compilationCBDDChip.additionSwap + compilationTriangleChip.additionSwap) * 100 / max(1, compilationTriangleChip.additionSwap)
            betterTriangleDepth = (-compilationCBDDChip.additionDepth + compilationTriangleChip.additionDepth) * 100 / max(1, compilationTriangleChip.additionDepth)
            betterLiSwap = (-compilationCBDDChip.additionSwap + compilationLiChip.additionSwap) * 100 / max(1, compilationLiChip.additionSwap)
            betterLiDepth = (-compilationCBDDChip.additionDepth + compilationLiChip.additionDepth) * 100 / max(1, compilationLiChip.additionDepth)

            CBDDAddDepth.append(compilationCBDDChip.additionDepth)
            liAddDepth.append(compilationLiChip.additionDepth)
            crossAddDepth.append(compilationCrossSquareChip.additionDepth)
            triAddDepth.append(compilationTriangleChip.additionDepth)
            CBDDAddSwap.append(compilationCBDDChip.additionSwap)
            liAddSwap.append(compilationLiChip.additionSwap)
            crossAddSwap.append(compilationCrossSquareChip.additionSwap)
            triAddSwap.append(compilationTriangleChip.additionSwap)

            compareSwap = [compilationCrossSquareChip.additionSwap, compilationTriangleChip.additionSwap, compilationLiChip.additionSwap, compilationCBDDChip.additionSwap]
            compareDepth = [compilationCrossSquareChip.additionDepth, compilationTriangleChip.additionDepth, compilationLiChip.additionDepth, compilationCBDDChip.additionDepth]

            minSwap = np.where(np.array(compareSwap) == np.min(np.array(compareSwap)))[0]
            minDepth = np.where(np.array(compareDepth) == np.min(np.array(compareDepth)))[0]


            resultFile = 'compile' + str(a)[:3] + '\\' + file_name[:-5] + '.txt'
            occupyFile = 'compile' + str(a)[:3] + '\\' + file_name[:-5] + 'occupy.txt'
            bestFile = 'compile' + str(a)[:3] + '\\best.txt'
            plotFile = 'compile' + str(a)[:3] + '\\plot.txt'
            with open(resultFile, 'w') as fp:
                fp.write('file size ' + str(fsize) + '\n')
                fp.write('a=' + str(a))
                fp.write('qasm depth ' + str(alg.qc.depth()) + '\n')
                fp.write('qasm gate number:\n')
                fp.write(str(dict(alg.qc.count_ops())) + str(alg.qc.__len__()) + '\n')
                fp.write(str(originQNumber) + ' bit\n')
                fp.write('after cross square chip\n')
                fp.write('check cross square number ' + str(len(list(cross_square_chip.nodes))) + '\n')
                fp.write('chech box add ' + str(compilationCrossSquareChip.additionSwap) + ' swap, ' + str(compilationCrossSquareChip.additionDepth) + ' depth\n')
                fp.write('after triangle chip\n')
                fp.write('triangle qubit number ' + str(len(list(triangle_chip.nodes))) + '\n')
                fp.write('triangle add ' + str(compilationTriangleChip.additionSwap) + ' swap, ' + str(compilationTriangleChip.additionDepth) + ' depth\n')
                fp.write('after li chip\n')
                fp.write('li qubit number ' + str(len(list(liChip.nodes))) + '\n')
                fp.write('li add ' + str(compilationLiChip.additionSwap) + ' swap, ' + str(compilationLiChip.additionDepth) + ' depth\n')
                fp.write('after chip\n')
                fp.write('qubit number ' + str(len(list(algChip.nodes))) + '\n')
                fp.write('add ' + str(compilationCBDDChip.additionSwap) + ' swap, ' + str(compilationCBDDChip.additionDepth) + ' depth\n')
                fp.write('better than cross square swap ' + str(round(betterCrossSquareSwap, 1)) + '%\n')
                fp.write('better than cross square depth ' + str(round(betterCrossSquareDepth, 1)) + '%\n')
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
            
            with open(occupyFile, 'w') as fp:
                for i in range(len(CBDD_mat)):
                    if i >= len(origin_mat):
                        fp.write(str(CBDD_mat[i]) + ' ' + '0.0' + '\n')
                    else:
                        fp.write(str(CBDD_mat[i]) + ' ' + str(origin_mat[i]) + '\n')

            print(minSwapList)
            print(minDepthList)

        with open(bestFile, 'w') as fp:
           fp.write('best swap dict\n')
           for method in swapMethodBestDict:
               fp.write(method + ' :\n')
               for f in swapMethodBestDict[method]:
                   fp.write(f + '\n')
           fp.write('\nbest depth dict\n')
           for method in depthMethodBestDict:
               fp.write(method + ' :\n')
               for f in depthMethodBestDict[method]:
                   fp.write(f + '\n')

        with open(plotFile, 'w') as fp:
           for d in range(len(depthes)):
               fp.write(str(depthes[d]) + ' ' + str(ns[d]) + ' ' + str(cxs[d]) + ' ' +
                        str(CBDDAddDepth[d]) + ' ' + str(CBDDAddSwap[d]) + ' ' + 
                        str(triAddDepth[d]) + ' ' + str(triAddSwap[d]) + ' ' + 
                        str(liAddDepth[d]) + ' ' + str(liAddSwap[d]) + ' ' + 
                        str(crossAddDepth[d]) + ' ' + str(crossAddSwap[d]) + '\n')