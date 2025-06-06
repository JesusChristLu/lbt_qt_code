import json
from copy import deepcopy
import geatpy as ea
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from freq_allocator.model.single_qubit_model import single_err_model, singq_xtalk_err, singq_residual_err #, twoq_dist_bound
from freq_allocator.model.formula import freq_var_map, draw_chip, scatter_err
from sko.PSO import PSO
import random

def checkcoli(chip, a, varType):
    reOptimizeNodes = []
    error_chip = 0
    qubit_num = 0
    for qubit in chip.nodes():
        if chip.nodes[qubit].get('frequency', False):
            if varType == 'double':
                isolateErr = chip.nodes[qubit]['isolated_error'](chip.nodes[qubit]['frequency'])
            else:
                isolateErr = chip.nodes[qubit]['isolated_error'][chip.nodes[qubit]['allow freq'].index(chip.nodes[qubit]['frequency'])]
            xyErr = 0
            residualErr = 0
            for neighbor in chip.nodes():
                if chip.nodes[neighbor].get('frequency', False) and not(neighbor == qubit):
                    if chip.nodes[neighbor]['name'] in chip.nodes[qubit]['xy_crosstalk_coef']:
                        xyErrEachPair = singq_xtalk_err(a[2], chip.nodes[neighbor]['frequency'] - chip.nodes[qubit]['frequency'], 
                                                chip.nodes[qubit]['xy_crosstalk_coef'][chip.nodes[neighbor]['name']], 
                                                chip.nodes[qubit]['xy_crosstalk_f'])
                        xyErr += xyErrEachPair
                        if xyErrEachPair > 4e-3:
                            if chip.nodes[qubit].get('xy serious', False):
                                chip.nodes[qubit]['xy serious'].append(neighbor)
                            else:
                                chip.nodes[qubit]['xy serious'] = [neighbor]
                    if nx.has_path(chip, qubit, neighbor):
                        if nx.shortest_path_length(chip, qubit, neighbor) == 1 and \
                            not(chip.nodes[qubit].get('residual serious', False)) or \
                            not(chip.nodes[neighbor].get('residual serious', False)) or \
                            not(qubit in chip.nodes[neighbor]['residual serious']):
                            nResidualErr = singq_residual_err(a[0], a[1],                         
                                                    chip.nodes[neighbor]['frequency'],
                                                    chip.nodes[qubit]['frequency'],
                                                    chip.nodes[neighbor]['anharm'],
                                                    chip.nodes[qubit]['anharm'])
                            residualErr += nResidualErr
                            if nResidualErr > 2.5e-3:
                                if not(chip.nodes[qubit].get('residual serious', False)):
                                    chip.nodes[qubit]['residual serious'] = [neighbor]
                                else:
                                    chip.nodes[qubit]['residual serious'].append(neighbor)
                                if not(chip.nodes[neighbor].get('residual serious', False)):
                                    chip.nodes[neighbor]['residual serious'] = [neighbor]
                                else:
                                    chip.nodes[neighbor]['residual serious'].append(neighbor)
                            
                        elif nx.shortest_path_length(chip, qubit, neighbor) == 2 and \
                            not(chip.nodes[qubit].get('residual serious', False)) or \
                            not(chip.nodes[neighbor].get('residual serious', False)) or \
                            not(qubit in chip.nodes[neighbor]['residual serious']):
                            nnResidualErr = singq_residual_err(a[2], a[3], 
                                                chip.nodes[neighbor]['frequency'],
                                                chip.nodes[qubit]['frequency'],
                                                chip.nodes[neighbor]['anharm'],
                                                chip.nodes[qubit]['anharm'])
                            residualErr += nnResidualErr
                            if nResidualErr > 2.5e-3:
                                if not(chip.nodes[qubit].get('residual serious', False)):
                                    chip.nodes[qubit]['residual serious'] = [neighbor]
                                else:
                                    chip.nodes[qubit]['residual serious'].append(neighbor)
                                if not(chip.nodes[neighbor].get('residual serious', False)):
                                    chip.nodes[neighbor]['residual serious'] = [neighbor]
                                else:
                                    chip.nodes[neighbor]['residual serious'].append(neighbor)

            allErr = isolateErr + xyErr + residualErr
            error_chip += allErr
            qubit_num += 1
            if allErr > 1e-2 and not(qubit in reOptimizeNodes):
                reOptimizeNodes.append(qubit)
                print(qubit, allErr, 'single qubit err')
            chip.nodes[qubit]['xy err'] = xyErr
            chip.nodes[qubit]['residual err'] = residualErr
            chip.nodes[qubit]['isolate err'] = isolateErr
            chip.nodes[qubit]['all err'] = allErr
    print('check, large err', reOptimizeNodes)
    return reOptimizeNodes, chip, error_chip / qubit_num

def sing_alloc(chip : nx.Graph, a, s: int = 1, varType='double'):
    epoch = 0
    centerConflictNode = (0, 0)
    avgErrEpoch = []
    newreOptimizeNodes = []
    repeat_optimize_history = {
                                'chip_history': [],
                                'error_history': [],
                                'reopt_qubits' : []
                                }
    jumpToEmpty = False

    fixQ = []
    for qubit in chip.nodes:
        if len(chip.nodes[qubit]['allow freq']) == 2:
            fixQ.append(qubit)
            chip.nodes[qubit]['frequency'] = chip.nodes[qubit]['allow freq'][0]

    while len([chip.nodes[qubit]['all err'] for qubit in chip.nodes if chip.nodes[qubit].get('all err', False)]) < len(chip.nodes) or \
        (not(jumpToEmpty) and len([chip.nodes[qubit]['all err'] for qubit in chip.nodes if chip.nodes[qubit].get('all err', False)]) == len(chip.nodes)):
        
        reOptimizeNodes = [centerConflictNode]
        for qubit in chip.nodes():
            if centerConflictNode in newreOptimizeNodes and not(qubit in reOptimizeNodes) and \
                qubit in newreOptimizeNodes and \
                not(qubit in fixQ):
                reOptimizeNodes.append(qubit)
            elif not(chip.nodes[centerConflictNode].get('frequency', False)) and not(qubit in reOptimizeNodes) and \
                not(chip.nodes[qubit].get('frequency', False)) and \
                np.abs(qubit[0] - centerConflictNode[0]) + np.abs(qubit[1] - centerConflictNode[1]) <= s and \
                not(qubit in fixQ):
                reOptimizeNodes.append(qubit)
        print('optimize qubits: ', reOptimizeNodes)

        if varType == 'double':
            lb = [0] * len(reOptimizeNodes)
            ub = [1] * len(reOptimizeNodes)

            func = lambda x : single_err_model(x, chip, reOptimizeNodes, a, varType)

            pso = PSO(func=func, dim=len(reOptimizeNodes), pop=60, max_iter=200, lb=lb, ub=ub)
            pso.run()
            best_freq = freq_var_map(pso.gbest_x[reOptimizeNodes.index(qubit)], chip.nodes[qubit]['allow freq'])
            for qubit in reOptimizeNodes:
                chip.nodes[qubit]['frequency'] = best_freq[reOptimizeNodes.index(qubit)]

        else:
            lb = [0] * len(reOptimizeNodes)
            ub = [len(chip.nodes[qubit]['allow freq']) - 1 for qubit in reOptimizeNodes]

            @ea.Problem.single
            def err_model_fun(frequencys):
                return single_err_model(frequencys, chip, reOptimizeNodes, a, varType)

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
                # ea.Population(Encoding='RI', NIND=100),
                # MAXGEN=200,
                ea.Population(Encoding='RI', NIND=1),
                MAXGEN=1,
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
                verbose=True, drawing=0, outputMsg=False,
                drawLog=False, saveFlag=True, dirName='results\\'
            )

            freq_list_bset = res['Vars'][0]
            print(f'qubit num: {len(reOptimizeNodes)}')
            for qubit in reOptimizeNodes:
                chip.nodes[qubit]['frequency'] = chip.nodes[qubit]['allow freq'][freq_list_bset[reOptimizeNodes.index(qubit)]]

        newreOptimizeNodes, chip, avgErr = checkcoli(chip, a, varType)
        repeat_optimize_history['chip_history'].append(deepcopy(chip))
        repeat_optimize_history['error_history'].append(avgErr)
        repeat_optimize_history['reopt_qubits'].append(newreOptimizeNodes)

        # if len(repeat_optimize_history['chip_history']) > 5 or min(repeat_optimize_history['error_history']) <= 5e-3:
        if len(repeat_optimize_history['chip_history']) > 1 or min([len(n) for n in repeat_optimize_history['reopt_qubits']]) == 0:
            # chip = repeat_optimize_history['chip_history'][repeat_optimize_history['error_history'].index(min(repeat_optimize_history['error_history']))]
            # print('jump', repeat_optimize_history['error_history'].index(min(repeat_optimize_history['error_history'])), 'is the chip with smallest err.')
            chip = repeat_optimize_history['chip_history'][[len(n) for n in repeat_optimize_history['reopt_qubits']].index(min([len(n) for n in repeat_optimize_history['reopt_qubits']]))]
            print('jump', [len(n) for n in repeat_optimize_history['reopt_qubits']].index(min([len(n) for n in repeat_optimize_history['reopt_qubits']])), 'is the chip with smallest opt qubits.')
            repeat_optimize_history['error_history'] = []
            repeat_optimize_history['chip_history'] = []
            repeat_optimize_history['reopt_qubits'] = []

            jumpToEmpty = True
        else:
            print('no jump')
            jumpToEmpty = False

        avgErrEpoch.append(avgErr)
        print('avg err estimate', avgErrEpoch)

        errList = [np.log10(chip.nodes[i].get('all err', 1e-5)) for i in chip.nodes]
        
        draw_chip(chip, 'results\\' + str(epoch) + 'chip err', qubit_err=errList)

        reOptimizeNodeDict = dict()
        for qubit in newreOptimizeNodes:
            if not(nx.has_path(chip, qubit, centerConflictNode)):
                reOptimizeNodeDict[qubit] = 10000
            else:
                reOptimizeNodeDict[qubit] = nx.shortest_path_length(chip, qubit, centerConflictNode)

        emptyNodeDict = dict()
        for qubit in chip.nodes():
            if not(chip.nodes[qubit].get('frequency', False)):
                if not(nx.has_path(chip, qubit, centerConflictNode)):
                    emptyNodeDict[qubit] = 10000
                else:
                    emptyNodeDict[qubit] = nx.shortest_path_length(chip, qubit, centerConflictNode)

        if len(reOptimizeNodeDict) > 0 and not(jumpToEmpty):
            print('reoptimize qubit distance', reOptimizeNodeDict)
            centerConflictNode = random.choices(list(reOptimizeNodeDict.keys()), weights=[1 / max(0.5, distance) for distance in reOptimizeNodeDict.values()], k=1)[0]
        elif len(emptyNodeDict) > 0:
            jumpToEmpty = False
            print('empty qubit distance', emptyNodeDict)
            centerConflictNode = list(sorted(emptyNodeDict.items(), key=lambda x : x[1]))[0][0]
        epoch += 1

    print('ave', avgErrEpoch)
    plt.plot(avgErrEpoch, label='err epoch')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('results\\' + 'err.pdf', dpi=300)
    plt.close()

    labelList = [chip.nodes[qubit]['name'] for qubit in chip.nodes]

    errList = np.log10([chip.nodes[qubit]['isolate err'] for qubit in chip.nodes])
    draw_chip(chip, 'results\\' + 'best' + 'chip isolate err', qubit_err=errList)
    scatter_err(errList, labelList, 'results\\' + 'xy err scatter')

    errList = np.log10([chip.nodes[qubit]['xy err'] for qubit in chip.nodes])
    draw_chip(chip, 'results\\' + 'best' + 'chip xy err', qubit_err=errList)
    scatter_err(errList, labelList, 'results\\' + 'residual err scatter')

    errList = np.log10([chip.nodes[qubit]['residual err'] for qubit in chip.nodes])
    draw_chip(chip, 'results\\' + 'best' + 'chip residual err', qubit_err=errList)
    scatter_err(errList, labelList, 'results\\' + 'all err scatter')

    errList = np.log10([chip.nodes[qubit]['all err'] for qubit in chip.nodes])
    draw_chip(chip, 'results\\' + 'best' + 'chip all err', qubit_err=errList)

    freqList = [int(round(chip.nodes[qubit]['frequency'], 3)) for qubit in chip.nodes]
    draw_chip(chip, 'results\\' + 'best' + 'chip freq', qubit_err=freqList)
    scatter_err(errList, labelList, 'results\\' + 'isolate err scatter')

    data = dict()
    for qcq in chip.edges:
        data[qcq] = {'isolate err' : chip.edges[qcq]['isolate err'],
                     'xy err' : chip.edges[qcq]['xy err'],
                     'residual err' : chip.edges[qcq]['residual err'],
                     'all err' : chip.edges[qcq]['all err'],
                     'frequency' : chip.edges[qcq]['frequency']}
    with open('cz.json', 'w') as f:
        json.dump(data, f)

    return chip
