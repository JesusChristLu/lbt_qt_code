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
    use_random = True
    kind = 'b'
    if use_random:
        name_list = []
        if kind == 'b':
            # d = 30
            d = 250
            for b_number in range(30, 301, 10):
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
            # b_number = 20
            b_number = 50
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
        # name_list = path_list
        name_list = ['radd_250.qasm']
    for file_name in name_list:
        print('reading')
        print(file_name)
        alg = Get_alg(os.path.join(path, file_name), show)
        if not file_name[-11:-1] == 'random.qas':
            fsize = os.path.getsize(os.path.join(path, file_name)) / float(1024)
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
        for k in range(1, originQNumber + 1):
            print('pruning with ' + str(k) + ' media vertices.')
            prune = Prune(g, show, k)
            graph = prune.graph
            if prune.needPrune:
                recover = prune.recover
                print('spliting')
                split = Split(graph, k, twoBitBlocks, recover, show)
                algChip = split.graph
                chipLayout = split.layout
                if algChip == {}:
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
            print('with', k, 'media vertices, score is', score)
            useless_mediaV = False
            if len(scores) > 1:
                for key in range(k - 1, 0, -1):
                    if scores.get(key, False):
                        if scores[k] >= scores[key]:
                            scores.pop(k) # may be no use
                            useless_mediaV = True
                            break
            if useless_mediaV == True:
                break
            algChips[k] = algChip
            chipLayouts[k] = chipLayout
        k = list(scores.keys())[list(scores.values()).index(min(list(scores.values())))]
        print('finish, using', k, 'media vertices.')
        algChip = algChips[k]
        chipLayout = chipLayouts[k]
        print('producing competitor')
        lattice = Lattice(originQNumber, mat, show)
        if Prune.degLimit == 6:
            twoD_lattice_chip = lattice.triangular_lattice
        elif Prune.degLimit == 4:
            twoD_lattice_chip = lattice.square_lattice
        cross_square_chip = lattice.cross_square_lattice
        liLayout = lattice.liLayout
        liChip = lattice.liChip

        print('Compiling')
        print('origin circuit: ', dict(alg.qc.count_ops()), ' depth: ', alg.depth, ' bit: ', originQNumber)
        cxs = dict(alg.qc.count_ops())['cx']
        print('SPQPD qubit number', len(list(algChip.nodes)))
        compilationSPQPDChip = Compilation(algChip, [alg.qc, alg.alg], layoutMethod='chip', routingMethod='ibm', setLayout=chipLayout)

        SPQPD_qasm = compilationSPQPDChip.out_qc_qasm
        SPQPD_analysis = Get_alg(SPQPD_qasm, show=False, is_qasm=True)
        qubit_load = np.zeros(np.shape(SPQPD_analysis.mat)[0])
        for i in SPQPD_analysis.alg:
            for j in i:
                qubit_load[j] += 1
        link_load = SPQPD_analysis.mat

        print('cross square qubit number', len(list(cross_square_chip.nodes)))
        compilationCrossSquareChip = Compilation(cross_square_chip, alg.qc, layoutMethod='ibm', routingMethod='ibm')
        print('2d lattice chip qubit number', len(list(twoD_lattice_chip.nodes)))
        compilation2dChip = Compilation(twoD_lattice_chip, alg.qc, layoutMethod='ibm', routingMethod='ibm')
        print('li chip qubit number', len(list(liChip.nodes)))
        compilationLiChip = Compilation(liChip, alg.qc, layoutMethod='li', routingMethod='ibm', setLayout=liLayout)

        betterCrossSquareSwap = (-compilationSPQPDChip.additionSwap + compilationCrossSquareChip.additionSwap) * 100 / max(1, compilationCrossSquareChip.additionSwap)
        betterCrossSquareDepth = (-compilationSPQPDChip.additionDepth + compilationCrossSquareChip.additionDepth) * 100 / max(1, compilationCrossSquareChip.additionDepth)
        better2dSwap = (-compilationSPQPDChip.additionSwap + compilation2dChip.additionSwap) * 100 / max(1, compilation2dChip.additionSwap)
        better2dDepth = (-compilationSPQPDChip.additionDepth + compilation2dChip.additionDepth) * 100 / max(1, compilation2dChip.additionDepth)
        betterLiSwap = (-compilationSPQPDChip.additionSwap + compilationLiChip.additionSwap) * 100 / max(1, compilationLiChip.additionSwap)
        betterLiDepth = (-compilationSPQPDChip.additionDepth + compilationLiChip.additionDepth) * 100 / max(1, compilationLiChip.additionDepth)

        SPQPDAddDepth = compilationSPQPDChip.additionDepth
        liAddDepth = compilationLiChip.additionDepth
        crossAddDepth = compilationCrossSquareChip.additionDepth
        twoDAddDepth = compilation2dChip.additionDepth
        SPQPDAddSwap = compilationSPQPDChip.additionSwap
        liAddSwap = compilationLiChip.additionSwap
        crossAddSwap = compilationCrossSquareChip.additionSwap
        twoDAddSwap = compilation2dChip.additionSwap

        compareSwap = [compilationCrossSquareChip.additionSwap, compilation2dChip.additionSwap, compilationLiChip.additionSwap, compilationSPQPDChip.additionSwap]
        compareDepth = [compilationCrossSquareChip.additionDepth, compilation2dChip.additionDepth, compilationLiChip.additionDepth, compilationSPQPDChip.additionDepth]

        minSwap = np.where(np.array(compareSwap) == np.min(np.array(compareSwap)))[0]
        minDepth = np.where(np.array(compareDepth) == np.min(np.array(compareDepth)))[0]


        
        if kind == 'b':
            plotFile = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\bit benchmark\\' + str(Prune.degLimit) + str(d) + ' plot.txt'
            # resultFile = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\bit benchmark\\' + file_name[:-5] + '.txt'
        elif kind == 'r':
            plotFile = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\real benchmark\\' + str(Prune.degLimit) + 'plot.txt'
            # resultFile = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\real benchmark\\' + file_name[:-5] + '.txt'
        elif kind == 'd':
            plotFile = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\depth benchmark\\' + str(Prune.degLimit) + str(b_number) + 'plot.txt'
            # resultFile = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\depth benchmark\\' + file_name[:-5] + '.txt'
        # with open(resultFile, 'w') as fp:
        #     fp.write('qasm depth ' + str(alg.depth) + '\n')
        #     fp.write('qasm gate number:\n')
        #     fp.write(str(dict(alg.qc.count_ops())) + str(alg.qc.__len__()) + '\n')
        #     fp.write(str(originQNumber) + ' bit\n')
        #     fp.write('after cross square chip\n')
        #     fp.write('cross square qubit number ' + str(len(list(cross_square_chip.nodes))) + '\n')
        #     fp.write('cross square add ' + str(compilationCrossSquareChip.additionSwap) + ' swap, ' + str(compilationCrossSquareChip.additionDepth) + ' depth\n')
        #     fp.write('after triangle chip\n')
        #     fp.write('2d lattice qubit number ' + str(len(list(twoD_lattice_chip.nodes))) + '\n')
        #     fp.write('2d lattice add ' + str(compilation2dChip.additionSwap) + ' swap, ' + str(compilation2dChip.additionDepth) + ' depth\n')
        #     fp.write('after li chip\n')
        #     fp.write('li qubit number ' + str(len(list(liChip.nodes))) + '\n')
        #     fp.write('li add ' + str(compilationLiChip.additionSwap) + ' swap, ' + str(compilationLiChip.additionDepth) + ' depth\n')
        #     fp.write('after SPQPD\n')
        #     fp.write('SPQPD qubit number ' + str(len(list(algChip.nodes))) + '\n')
        #     fp.write('SPQPD add ' + str(compilationSPQPDChip.additionSwap) + ' swap, ' + str(compilationSPQPDChip.additionDepth) + ' depth\n')
        #     fp.write('better than cross square swap ' + str(round(betterCrossSquareSwap, 1)) + '%\n')
        #     fp.write('better than cross square depth ' + str(round(betterCrossSquareDepth, 1)) + '%\n')
        #     fp.write('better than 2d lattice swap ' + str(round(better2dSwap, 1)) + '%\n')
        #     fp.write('better than 2d lattice depth ' + str(round(better2dDepth, 1)) + '%\n')
        #     fp.write('better than li swap ' + str(round(betterLiSwap, 1)) + '%\n')
        #     fp.write('better than li depth ' + str(round(betterLiDepth, 1)) + '%\n')
        for i in minSwap:
            minSwapList[i] += 1
            # swapMethodBestDict[method[i]].append(file_name)
            # fp.write('the best swap ' + method[i] + '\n')
        for i in minDepth:
            minDepthList[i] += 1
            # depthMethodBestDict[method[i]].append(file_name)
            # fp.write('the best depth ' + method[i] + '\n')
        if use_random:
            d = int(file_name.split(' ')[0])
            b_number = int(file_name.split(' ')[1])
        else:
            d = alg.depth
            b_number = originQNumber
            # dict(alg.qc.count_ops())['cx']
        with open(plotFile, 'a') as fp:
            fp.write(str(d) + ' ' + str(b_number) + ' ' + str(cxs) + ' ' +
                    str(SPQPDAddDepth) + ' ' + str(SPQPDAddSwap) + ' ' + 
                    str(twoDAddDepth) + ' ' + str(twoDAddSwap) + ' ' + 
                    str(liAddDepth) + ' ' + str(liAddSwap) + ' ' + 
                    str(crossAddDepth) + ' ' + str(crossAddSwap) + '\n')

        print(minSwapList)
        print(minDepthList)
