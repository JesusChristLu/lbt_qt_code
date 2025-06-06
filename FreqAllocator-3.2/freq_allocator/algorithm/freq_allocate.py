from pathlib import Path
import json, os
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import torch
import networkx as nx
from scipy.stats import norm
from scipy.optimize import curve_fit
from freq_allocator.model.formula import freq_var_map, draw_chip, scatter_err
from sko.PSO import PSO
import geatpy as ea
import random
from freq_allocator.dataloader.load_chip import max_Algsubgraph
from freq_allocator.model import err_model_nn
from freq_allocator.model.err_model import (
    edge_distance
)
import time

def alloc_nn(chip : nx.Graph, s: int = 1, minMaxErr: tuple = (0,)):

    single_qubit_graph = deepcopy(chip)

    two_qubit_graph = nx.Graph()
    edges_to_remove = []
    two_qubit_graph.add_nodes_from(chip.edges)
    for qcq in chip.edges():
        if chip.nodes[qcq[0]]['freq_max'] > chip.nodes[qcq[1]]['freq_max']:
            qh, ql = qcq[0], qcq[1]
        else:
            qh, ql = qcq[1], qcq[0]
        if chip.nodes[qh]['freq_min'] + chip.nodes[qh]['anharm'] > chip.nodes[ql]['freq_max'] or \
            chip.nodes[qh]['freq_max'] + chip.nodes[qh]['anharm'] < chip.nodes[ql]['freq_min']:
            edges_to_remove.append(qcq)
        else:
            two_qubit_graph.nodes[qcq]['two tq'] = 40
            two_qubit_graph.nodes[qcq]['ql'] = ql
            two_qubit_graph.nodes[qcq]['qh'] = qh
            lb = max(chip.nodes[ql]['freq_min'], chip.nodes[qh]['freq_min'] + chip.nodes[qh]['anharm'])
            ub = min(chip.nodes[ql]['freq_max'], chip.nodes[qh]['freq_max'] + chip.nodes[qh]['anharm'])
            two_qubit_graph.nodes[qcq]['allow freq'] = np.linspace(lb, ub, np.int_(ub - lb) + 1)

    two_qubit_graph.remove_nodes_from(edges_to_remove)
    maxParallelCZs = max_Algsubgraph(chip)
    for maxParallelCZ in maxParallelCZs:
        qcqHaveSeen = []
        for qcq1 in maxParallelCZ:
            if qcq1 in edges_to_remove:
                continue
            for qcq2 in maxParallelCZ:
                if qcq2 in edges_to_remove:
                    continue
                if qcq1 == qcq2:
                    continue
                qcqHaveSeen.append((qcq1, qcq2))
                if edge_distance(chip, qcq1, qcq2) == 1:
                    two_qubit_graph.add_edge(qcq1, qcq2)

    xtalk_graph = nx.union(single_qubit_graph, two_qubit_graph)

    for qcq in two_qubit_graph:
        for qubit in single_qubit_graph:
            if (nx.has_path(chip, qubit, qcq[0]) and nx.has_path(chip, qubit, qcq[1])) and \
                (nx.shortest_path_length(chip, qubit, qcq[0]) == 1 or nx.shortest_path_length(chip, qubit, qcq[1]) == 1) and \
                not(qubit in qcq):
                xtalk_graph.add_edge(qcq, qubit)

    qgnn = err_model_nn.QuantumGNN(len(xtalk_graph.nodes), xtalk_graph)
    model_path = Path.cwd() / 'results' / 'model.pth'
    qgnn.load_state_dict(torch.load(model_path))
    qgnn.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

    maxFreq = np.max([np.max(xtalk_graph.nodes[qcq]['allow freq']) for qcq in xtalk_graph.nodes])
    minFreq = np.min([np.min(xtalk_graph.nodes[qcq]['allow freq']) for qcq in xtalk_graph.nodes])

    @ea.Problem.single
    def err_model_fun(frequencys):
        frequencys = torch.tensor((frequencys - minFreq) / (maxFreq - minFreq))
        x = torch.zeros(frequencys.size()[0], frequencys.size()[0] + 1)
        x[:, :frequencys.size()[0]] = frequencys
        dataSize = x.size()[0]
        for i in range(dataSize):
            x[i, -1] = i % frequencys.size()[0]
        errList = qgnn(x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))) * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]
        return (torch.mean(errList)).cpu().detach().numpy()
        
    def err_model_fun_test(frequencys):
        frequencys = torch.tensor((frequencys - minFreq) / (maxFreq - minFreq))
        x = torch.zeros(frequencys.size()[0], frequencys.size()[0] + 1)
        x[:, :frequencys.size()[0]] = frequencys
        dataSize = x.size()[0]
        for i in range(dataSize):
            x[i, -1] = i % frequencys.size()[0]
        return qgnn(x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))).cpu().detach().numpy() * ((minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]).numpy()

    epoch = 0
    avgErrEpoch = []
    repeat_optimize_history = {
        'chip_history': [],
        'error_history': [],
        'reopt_instructions': []
    }
    
    for node in xtalk_graph.nodes:
        xtalk_graph.nodes[node]['frequency'] = xtalk_graph.nodes[node]['allow freq'][0]

    avgErr = 10
    # while avgErr > 0.001:
    while epoch < 50:
        frequencys = [xtalk_graph.nodes[node]['frequency'] for node in xtalk_graph.nodes]
        errList = err_model_fun_test(frequencys)
        for node in xtalk_graph.nodes:
            xtalk_graph.nodes[node]['all err'] = errList[list(xtalk_graph.nodes).index(node)][0]
        avgErr = np.mean(errList)
        print('check', avgErr)
        avgErrEpoch.append(avgErr)
        print('avg err estimate', avgErrEpoch)

        maxErrSum = 0
        for centerNode in xtalk_graph.nodes:
            if not(centerNode in single_qubit_graph):
                continue
            tryReoptNodes = [centerNode]
            for node in xtalk_graph.nodes:
                if centerNode == node:
                    continue
                if not(node in tryReoptNodes):
                    if node in single_qubit_graph.nodes:
                        if nx.has_path(single_qubit_graph, centerNode, node) and \
                            nx.shortest_path_length(single_qubit_graph, centerNode, node) <= s:
                            tryReoptNodes.append(node)
                    elif node in two_qubit_graph.nodes:
                        if node[0] in tryReoptNodes and node[1] in tryReoptNodes:
                            tryReoptNodes.append(node)
            errSum = np.mean([errList[list(xtalk_graph.nodes).index(node)] for node in tryReoptNodes])
            if maxErrSum < errSum: 
                if len(repeat_optimize_history['reopt_instructions']) > 0:
                    if not(tryReoptNodes in repeat_optimize_history['reopt_instructions']):
                        maxErrSum = errSum
                        reOptimizeNodes = tryReoptNodes
                else:
                    maxErrSum = errSum
                    reOptimizeNodes = tryReoptNodes

        repeat_optimize_history['chip_history'].append(deepcopy(xtalk_graph))
        repeat_optimize_history['error_history'].append(avgErr)
        repeat_optimize_history['reopt_instructions'].append(reOptimizeNodes)

        if len(repeat_optimize_history['reopt_instructions']) == len(xtalk_graph.nodes):
            repeat_optimize_history['chip_history'] = []
            repeat_optimize_history['error_history'] = []
            repeat_optimize_history['reopt_instructions'] = []

        print('optimize nodes: ', reOptimizeNodes)
        errList = [xtalk_graph.nodes[i].get('all err', 0) for i in xtalk_graph.nodes if i in xtalk_graph]
        if epoch == 0:
            draw_chip(chip, 'results\\' + str(epoch) + 'err', err=errList, centerNode=reOptimizeNodes[0], minMaxErr=minMaxErr, bar=True)
        else:
            draw_chip(chip, 'results\\' + str(epoch) + 'err', err=errList, centerNode=reOptimizeNodes[0], minMaxErr=minMaxErr)

        freqList = [xtalk_graph.nodes[i].get('frequency', 0) for i in xtalk_graph.nodes if i in xtalk_graph]
        maxFreq = np.max([np.max(xtalk_graph.nodes[qcq]['allow freq']) for qcq in xtalk_graph.nodes])
        minFreq = np.min([np.min(xtalk_graph.nodes[qcq]['allow freq']) for qcq in xtalk_graph.nodes])
        # if epoch == 49:
        if epoch == 0:
            draw_chip(chip, 'results\\' + str(epoch) + 'freq', freq=freqList, centerNode=reOptimizeNodes[0], minMaxFreq=(minFreq, maxFreq), bar=True)    
        else:
            draw_chip(chip, 'results\\' + str(epoch) + 'freq', freq=freqList, centerNode=reOptimizeNodes[0], minMaxFreq=(minFreq, maxFreq))

        lb = np.zeros(len(xtalk_graph.nodes))
        ub = np.zeros(len(xtalk_graph.nodes))
        for node in xtalk_graph.nodes:
            if node in reOptimizeNodes:
                lb[list(xtalk_graph.nodes).index(node)] = min(xtalk_graph.nodes[node]['allow freq'])
                ub[list(xtalk_graph.nodes).index(node)] = max(xtalk_graph.nodes[node]['allow freq'])
            else:
                lb[list(xtalk_graph.nodes).index(node)] = xtalk_graph.nodes[node]['frequency']
                ub[list(xtalk_graph.nodes).index(node)] = xtalk_graph.nodes[node]['frequency']

        problem = ea.Problem(
            name='soea err model',
            M=1,  # 初始化M（目标维数）
            maxormins=[1],  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
            Dim=len(xtalk_graph.nodes),  # 决策变量维数
            varTypes=[1] * len(single_qubit_graph.nodes) + [0] * len(two_qubit_graph.nodes),  # 决策变量的类型列表，0：实数；1：整数
            lb=lb,  # 决策变量下界
            ub=ub,  # 决策变量上界
            evalVars=err_model_fun
        )

        algorithm = ea.soea_DE_best_1_bin_templet(
            problem,
            # ea.Population(Encoding='RI', NIND=200),
            # MAXGEN=100,
            ea.Population(Encoding='RI', NIND=100),
            MAXGEN=50,
            # ea.Population(Encoding='RI', NIND=1),
            # MAXGEN=1,
            logTras=1,
            # trappedValue=1e-10,
            # maxTrappedCount=20
        )
        algorithm.mutOper.F = 1
        algorithm.recOper.XOVR = 1

        # algorithm.run()

        freq_bset = None
        res = ea.optimize(
            algorithm,
            prophet=freq_bset,
            # prophet=np.array(self.experiment_options.FIR0),
            verbose=True,
            drawing=0, outputMsg=True,
            drawLog=False, saveFlag=False, dirName='results\\'
        )

        freq_list_bset = res['Vars'][0]
        for node in xtalk_graph.nodes:
            if node in reOptimizeNodes:
                xtalk_graph.nodes[node]['frequency'] = freq_list_bset[list(xtalk_graph.nodes).index(node)]

        epoch += 1

    print('ave', avgErrEpoch)
    plt.plot(avgErrEpoch, label='err epoch')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('results\\' + 'err.pdf', dpi=300)
    plt.close()

    # nodeLabelList = [single_qubit_graph.nodes[qubit]['name'] for qubit in single_qubit_graph.nodes]
    # errList1 = [xtalk_graph.nodes[i].get('all err', 0) for i in xtalk_graph.nodes if i in single_qubit_graph]

    # r=(0, 0.02)

    # fig, axs = plt.subplots(1, 1, figsize=(8, 8), gridspec_kw={'hspace': 0})
    # axs.hist(errList1, bins=55, density=True, range=r)
    # axs.set_ylabel('all err')
    # axs.set_xlabel('err')
    # plt.show()

    # print('err1', np.mean(errList1), np.std(errList1))
    
    # edgeLabelList = [two_qubit_graph.nodes[qcq] for qcq in two_qubit_graph.nodes]
    # errList2 = [xtalk_graph.nodes[i].get('all err', 0) for i in xtalk_graph.nodes if i in two_qubit_graph]

    # r=(0, 0.05)

    # fig, axs = plt.subplots(1, 1, figsize=(8, 8), gridspec_kw={'hspace': 0})
    # axs.hist(errList2, bins=55, density=True, range=r)
    # axs.set_ylabel('all err')
    # axs.set_xlabel('err')
    # plt.show()

    # print('err2', np.mean(errList2), np.std(errList2))

    data = dict()
    for node in single_qubit_graph.nodes:
        data[node] = {'all err' : float(xtalk_graph.nodes[node]['all err']),
                    'frequency' : float(xtalk_graph.nodes[node]['frequency'])}
    for node in two_qubit_graph.nodes:
        data[str(node)] = {'all err' : float(xtalk_graph.nodes[node]['all err']),
                        'frequency' : float(xtalk_graph.nodes[node]['frequency'])}
    with open('results\\gates.json', 'w') as f:
        json.dump(data, f)

    return xtalk_graph