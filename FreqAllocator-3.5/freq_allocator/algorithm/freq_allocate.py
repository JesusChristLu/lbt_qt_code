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
from freq_allocator.model import err_model_nn, err_model
from freq_allocator.model.err_model import (
    edge_distance
)
import time

def xtalk_graph_gen(chip):

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
            two_qubit_graph.nodes[qcq]['allow freq'] = np.linspace(lb, ub, np.int(ub - lb) + 1)
    two_qubit_graph.remove_nodes_from(edges_to_remove)
    chip.remove_edges_from(edges_to_remove)
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
    return single_qubit_graph, two_qubit_graph, xtalk_graph


def alloc_nn(chip : nx.Graph, s: int = 1, minMaxErr: tuple = (0,)):

    single_qubit_graph, two_qubit_graph, xtalk_graph = xtalk_graph_gen(chip)

    # qgnn_s = err_model_nn.QuantumGNN(len(xtalk_graph.nodes), xtalk_graph)
    # model_path_s = Path.cwd() / 'results' / 'model_s.pth'
    # qgnn_s.load_state_dict(torch.load(model_path_s))
    # qgnn_s.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

    # qgnn_c = err_model_nn.QuantumGNN(len(xtalk_graph.nodes), xtalk_graph)
    # model_path_c = Path.cwd() / 'results' / 'model_c.pth'
    # qgnn_c.load_state_dict(torch.load(model_path_c))
    # qgnn_c.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

    qgnn = err_model_nn.QuantumGNN(len(xtalk_graph.nodes), xtalk_graph)
    model_path = Path.cwd() / 'results' / 'model.pth'
    qgnn.load_state_dict(torch.load(model_path))
    qgnn.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

    maxFreq = np.max([np.max(xtalk_graph.nodes[qcq]['allow freq']) for qcq in xtalk_graph.nodes])
    minFreq = np.min([np.min(xtalk_graph.nodes[qcq]['allow freq']) for qcq in xtalk_graph.nodes])

    @ea.Problem.single
    def err_model_fun(frequencys, qNum=8):
        # frequencys = torch.tensor((frequencys - minFreq) / (maxFreq - minFreq))
        # x_s = torch.zeros(qNum, frequencys.size()[0] + 1)
        # x_s[:, :frequencys.size()[0]] = frequencys
        # x_c = torch.zeros(frequencys.size()[0] - qNum, frequencys.size()[0] + 1)
        # x_c[:, :frequencys.size()[0]] = frequencys
        # dataSize = x_s.size()[0]
        # for i in range(dataSize):
        #     if i < qNum:
        #         x_s[i, -1] = i % frequencys.size()[0]
        #     else:
        #         x_c[i, -1] = i % frequencys.size()[0]
        # errList_s = qgnn_s(x_s.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))) * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]
        # errList_c = qgnn_c(x_c.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))) * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]
       
        # return (torch.mean(torch.cat((errList_s, errList_c), dim=0))).cpu().detach().numpy()
    
        frequencys = torch.tensor((frequencys - minFreq) / (maxFreq - minFreq))
        x = torch.zeros(frequencys.size()[0], frequencys.size()[0] + 1)
        x[:, :frequencys.size()[0]] = frequencys
        dataSize = x.size()[0]
        for i in range(dataSize):
            x[i, -1] = i % frequencys.size()[0]
        errList = qgnn(x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))) * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]
        return np.abs(torch.mean(errList).cpu().detach().numpy())
        
    def err_model_fun_test(frequencys, qNum=8):
        # frequencys = torch.tensor((frequencys - minFreq) / (maxFreq - minFreq))
        # x_s = torch.zeros(qNum, frequencys.size()[0] + 1)
        # x_s[:, :frequencys.size()[0]] = frequencys
        # x_c = torch.zeros(frequencys.size()[0] - qNum, frequencys.size()[0] + 1)
        # x_c[:, :frequencys.size()[0]] = frequencys
        # dataSize = x_s.size()[0]
        # for i in range(dataSize):
        #     if i < qNum:
        #         x_s[i, -1] = i % frequencys.size()[0]
        #     else:
        #         x_c[i, -1] = i % frequencys.size()[0]
        # errList_s = qgnn_s(x_s.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))) * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]
        # errList_c = qgnn_c(x_c.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))) * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]
        # return (torch.cat((errList_s, errList_c), dim=0)).cpu().detach().numpy()

        frequencys = torch.tensor((frequencys - minFreq) / (maxFreq - minFreq))
        x = torch.zeros(frequencys.size()[0], frequencys.size()[0] + 1)
        x[:, :frequencys.size()[0]] = frequencys
        dataSize = x.size()[0]
        for i in range(dataSize):
            x[i, -1] = i % frequencys.size()[0]
        return np.abs(qgnn(x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))).cpu().detach().numpy() * ((minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]).numpy())

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
    maxEpoch = 20
    smallestAvgErr = 1e3

    while epoch < maxEpoch:
        freqList = [xtalk_graph.nodes[node]['frequency'] for node in xtalk_graph.nodes]
        errList = err_model_fun_test(freqList)
        for node in xtalk_graph.nodes:
            xtalk_graph.nodes[node]['all err'] = errList[list(xtalk_graph.nodes).index(node)][0]
        avgErr = np.mean(errList)
        print('check', avgErr)
        avgErrEpoch.append(avgErr)
        print('avg err estimate', avgErrEpoch)
        if smallestAvgErr > avgErr:
            smallestAvgErr = avgErr
            bestErrList = errList
            bestFreqList = freqList
            bestEpoch = epoch

        maxErrSum = 0
        for centerNode in xtalk_graph.nodes:
            errSum = errList[list(xtalk_graph.nodes).index(centerNode)]
            if maxErrSum < errSum: 
                if not(centerNode in single_qubit_graph):
                    tryReoptNodes = [centerNode[1], centerNode[0]]
                    if errList[list(xtalk_graph.nodes).index(centerNode[0])] > errList[list(xtalk_graph.nodes).index(centerNode[1])]:
                        centerNode = centerNode[0]
                    else:
                        centerNode = centerNode[1]
                else:
                    centerNode = centerNode
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
                # errSum = np.mean([errList[list(xtalk_graph.nodes).index(node)] for node in tryReoptNodes])
                # errSum = np.max([errList[list(xtalk_graph.nodes).index(node)] for node in tryReoptNodes])
            
            # if maxErrSum < errSum: 
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
        if epoch == maxEpoch - 1:
            print('best', bestEpoch, smallestAvgErr)
            draw_chip(chip, 'results\\alloc\\' + str(epoch) + 'err_nn', err=bestErrList, centerNode=None, bar=True, epoch=epoch)
            draw_chip(chip, 'results\\alloc\\' + str(epoch) + 'freq_nn', freq=bestFreqList, centerNode=None, bar=True, epoch=epoch)
        else:
            draw_chip(chip, 'results\\alloc\\' + str(epoch) + 'err_nn', err=errList, centerNode=(reOptimizeNodes[0], s), bar=False, epoch=epoch)
            draw_chip(chip, 'results\\alloc\\' + str(epoch) + 'freq_nn', freq=freqList, centerNode=(reOptimizeNodes[0], s), bar=False, epoch=epoch)

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
                varTypes=[0] * len(xtalk_graph.nodes),  # 决策变量的类型列表，0：实数；1：整数
                lb=lb,  # 决策变量下界
                ub=ub,  # 决策变量上界
                evalVars=err_model_fun
            )

            algorithm = ea.soea_DE_best_1_bin_templet(
                problem,
                ea.Population(Encoding='RI', NIND=200),
                MAXGEN=30,
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
                drawLog=False, saveFlag=False, dirName='results\\alloc\\'
            )

            freq_list_bset = res['Vars'][0]
            for node in xtalk_graph.nodes:
                if node in reOptimizeNodes:
                    xtalk_graph.nodes[node]['frequency'] = freq_list_bset[list(xtalk_graph.nodes).index(node)]

        epoch += 1

    print('avg', avgErrEpoch)
    plt.plot(range(len(avgErrEpoch)), avgErrEpoch)
    plt.ylabel('avg error', fontsize=15)
    plt.xlabel('epoch', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig('results\\alloc\\' + 'loss_nn.pdf', dpi=300)
    plt.close()

    a = [0.1, 0.1, 
            0.1, 20, 0.1, 20,
            4e-4, 1e-7, 1e-2, 1e-2, 1e-5,
            0.1, 20, 0.1, 20, 
            1, 
            0.1, 20, 0.1, 20]  
    
    _, xtalk_graph, avgErr = err_model(None, xtalk_graph, a=a)
    errList1 = [xtalk_graph.nodes[i].get('all err', 0) for i in xtalk_graph.nodes if i in single_qubit_graph]
    errList2 = [xtalk_graph.nodes[i].get('all err', 0) for i in xtalk_graph.nodes if i in two_qubit_graph]
    draw_chip(chip, 'results\\alloc\\real_err_nn', err=errList1 + errList2, bar=True)

    data = dict()
    for node in single_qubit_graph.nodes:
        data[node] = {'all err' : float(xtalk_graph.nodes[node]['all err']),
                    'frequency' : float(xtalk_graph.nodes[node]['frequency'])}
    for node in two_qubit_graph.nodes:
        data[str(node)] = {'all err' : float(xtalk_graph.nodes[node]['all err']),
                        'frequency' : float(xtalk_graph.nodes[node]['frequency'])}
    with open('results\\alloc\\gates_nn.json', 'w') as f:
        json.dump(data, f)
    return np.mean(errList)


def alloc(chip : nx.Graph, a, s: int = 1):

    single_qubit_graph, two_qubit_graph, xtalk_graph = xtalk_graph_gen(chip)

    epoch = 0
    centerNode = list(chip.nodes)[0]
    avgErrEpoch = []
    newreOptimizeNodes = []
    repeat_optimize_history = {
        'chip_history': [],
        'error_history': [],
        'reopt_instructions': []
    }
    
    jumpToEmpty = True

    fixQ = []
    for node in xtalk_graph.nodes:
        if len(xtalk_graph.nodes[node]['allow freq']) <= 2:
            fixQ.append(node)
            xtalk_graph.nodes[node]['frequency'] = xtalk_graph.nodes[node]['allow freq'][0]

    while len([xtalk_graph.nodes[node]['all err'] for node in xtalk_graph.nodes if xtalk_graph.nodes[node].get('all err', False)]) < len(xtalk_graph.nodes) or \
        (not(jumpToEmpty) and len([xtalk_graph.nodes[node]['all err'] for node in xtalk_graph.nodes if xtalk_graph.nodes[node].get('all err', False)]) == len(xtalk_graph.nodes)):
        reOptimizeNodes = []
        if not(jumpToEmpty):
            for node in xtalk_graph.nodes:
                if not(node in reOptimizeNodes) and \
                    node in newreOptimizeNodes and \
                    not(node in fixQ):
                    reOptimizeNodes.append(node)
        else:
            for node in xtalk_graph.nodes:
                if not(node in reOptimizeNodes) and \
                    not(xtalk_graph.nodes[node].get('frequency', False)) and \
                    not(node in fixQ):
                    if node in single_qubit_graph.nodes:
                        if nx.has_path(single_qubit_graph, centerNode, node) and \
                            nx.shortest_path_length(single_qubit_graph, centerNode, node) <= s:
                            reOptimizeNodes.append(node)
                    elif node in two_qubit_graph.nodes:
                        if (
                                (
                                    node[0] in reOptimizeNodes or 
                                    (
                                        # xtalk_graph.nodes[node[0]].get('frequency', False) and 
                                        # nx.shortest_path_length(single_qubit_graph, centerNode, node[0]) <= s
                                        xtalk_graph.nodes[node[0]].get('frequency', False)
                                    )
                                ) 
                                and 
                                (
                                    node[1] in reOptimizeNodes or 
                                    (
                                        # xtalk_graph.nodes[node[1]].get('frequency', False) and 
                                        # nx.shortest_path_length(single_qubit_graph, centerNode, node[1]) <= s
                                        xtalk_graph.nodes[node[1]].get('frequency', False)
                                    )
                                )
                            ):
                            reOptimizeNodes.append(node)

        print('optimize nodes: ', reOptimizeNodes)

        lb = [0] * len(reOptimizeNodes)
        ub = [len(xtalk_graph.nodes[node]['allow freq']) - 1 for node in reOptimizeNodes]

        @ea.Problem.single
        def err_model_fun(frequencys):
            return err_model(frequencys, xtalk_graph, a, reOptimizeNodes)

        problem = ea.Problem(
            name='soea err model',
            M=1,  # 初始化M（目标维数）
            maxormins=[1],  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
            Dim=len(reOptimizeNodes),  # 决策变量维数
            varTypes=[1] * len(reOptimizeNodes),  # 决策变量的类型列表，0：实数；1：整数
            lb=lb,  # 决策变量下界
            ub=ub,  # 决策变量上界
            evalVars=err_model_fun
        )

        algorithm = ea.soea_DE_best_1_bin_templet(
            problem,
            ea.Population(Encoding='RI', NIND=100),
            MAXGEN=20,
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
            drawLog=False, saveFlag=False, dirName='results\\alloc\\'
        )

        freq_list_bset = res['Vars'][0]
        print(f'old node num: {len(reOptimizeNodes)}')
        for node in reOptimizeNodes:
            xtalk_graph.nodes[node]['frequency'] = xtalk_graph.nodes[node]['allow freq'][freq_list_bset[reOptimizeNodes.index(node)]]

        newreOptimizeNodes, xtalk_graph, avgErr = err_model(None, xtalk_graph, a)
        for q in fixQ:
            if q in newreOptimizeNodes:
                newreOptimizeNodes.remove(q)
                
        print(f'new node num: {len(newreOptimizeNodes)}')

        repeat_optimize_history['chip_history'].append(deepcopy(xtalk_graph))
        repeat_optimize_history['error_history'].append(avgErr)
        repeat_optimize_history['reopt_instructions'].append(newreOptimizeNodes)


        # if len(repeat_optimize_history['chip_history']) > 5 or min(repeat_optimize_history['error_history']) <= 5e-3:
        if len(repeat_optimize_history['chip_history']) > 5 or min([len(n) for n in repeat_optimize_history['reopt_instructions']]) == 0 or len(newreOptimizeNodes) > 20:
        #     xtalk_graph = repeat_optimize_history['chip_history'][repeat_optimize_history['error_history'].index(min(repeat_optimize_history['error_history']))]
        #     print('jump', repeat_optimize_history['error_history'].index(min(repeat_optimize_history['error_history'])), 'is the xtalk_graph with smallest err.')
            xtalk_graph = repeat_optimize_history['chip_history'][[len(n) for n in repeat_optimize_history['reopt_instructions']].index(min([len(n) for n in repeat_optimize_history['reopt_instructions']]))]
            print('jump', [len(n) for n in repeat_optimize_history['reopt_instructions']].index(min([len(n) for n in repeat_optimize_history['reopt_instructions']])), 'is the chip with smallest opt instructions.')
            repeat_optimize_history['error_history'] = []
            repeat_optimize_history['chip_history'] = []
            repeat_optimize_history['reopt_instructions'] = []
            jumpToEmpty = True
        else:
            print('no jump')
            print(min([len(n) for n in repeat_optimize_history['reopt_instructions']]))
            jumpToEmpty = False

        avgErrEpoch.append(avgErr)
        print('avg err estimate', avgErrEpoch)

        if jumpToEmpty:
            errList1 = [xtalk_graph.nodes[i].get('all err', 0) for i in xtalk_graph.nodes if i in single_qubit_graph]
            errList2 = [xtalk_graph.nodes[i].get('all err', 0) for i in xtalk_graph.nodes if i in two_qubit_graph]
            
            draw_chip(chip, 'results\\alloc\\' + str(epoch) + 'err_mod', err=errList1 + errList2, bar=True)

            freqList1 = [xtalk_graph.nodes[i].get('frequency', 0) for i in xtalk_graph.nodes if i in single_qubit_graph]
            freqList2 = [xtalk_graph.nodes[i].get('frequency', 0) for i in xtalk_graph.nodes if i in two_qubit_graph]
            
            draw_chip(chip, 'results\\alloc\\' + str(epoch) + 'freq_mod', freq=freqList1 + freqList2, bar=True)

        emptyNodeDict = dict()
        for qubit in single_qubit_graph.nodes():
            if not(xtalk_graph.nodes[qubit].get('frequency', False)):
                if not(nx.has_path(xtalk_graph, qubit, centerNode)):
                    emptyNodeDict[qubit] = 10000
                else:
                    emptyNodeDict[qubit] = nx.shortest_path_length(xtalk_graph, qubit, centerNode)

        if len(emptyNodeDict) > 0 and jumpToEmpty:
            print('empty qubit distance', emptyNodeDict)
            centerNode = list(sorted(emptyNodeDict.items(), key=lambda x : x[1]))[0][0]
        epoch += 1

    print('avg', avgErrEpoch)
    plt.plot(range(len(avgErrEpoch)), avgErrEpoch)
    plt.ylabel('avg error', fontsize=15)
    plt.xlabel('epoch', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig('results\\alloc\\' + 'loss_mod.pdf', dpi=300)
    plt.close()

    a = [0.1, 0.1, 
            0.1, 20, 0.1, 20,
            4e-4, 1e-7, 1e-2, 1e-2, 1e-5,
            0.1, 20, 0.1, 20, 
            1, 
            0.1, 20, 0.1, 20]  

    # a = [1, 0.5, 
    #         1, 20, 0.5, 20,
    #         4e-4, 1e-7, 1e-2, 1e-2, 1e-5,
    #         1, 20, 0.5, 20, 
    #         1, 
    #         1, 20, 0.5, 20]    
    
    _, xtalk_graph, avgErr = err_model(None, xtalk_graph, a=a)
    errList1 = [xtalk_graph.nodes[i].get('all err', 0) for i in xtalk_graph.nodes if i in single_qubit_graph]
    errList2 = [xtalk_graph.nodes[i].get('all err', 0) for i in xtalk_graph.nodes if i in two_qubit_graph]
    
    draw_chip(chip, 'results\\alloc\\real_err_mod', err=errList1 + errList2, bar=True)

    data = dict()
    for node in single_qubit_graph.nodes:
        data[node] = {'all err' : xtalk_graph.nodes[node]['all err'],
                        'frequency' : xtalk_graph.nodes[node]['frequency']}
    for node in two_qubit_graph.nodes:
        data[str(node)] = {'all err' : xtalk_graph.nodes[node]['all err'],
                            'frequency' : xtalk_graph.nodes[node]['frequency']}
    with open('results\\alloc\\gates_mod.json', 'w') as f:
        json.dump(data, f)
