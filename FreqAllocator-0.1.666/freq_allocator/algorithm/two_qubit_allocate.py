import os, json
import geatpy as ea
import numpy as np
from copy import deepcopy
import random
from matplotlib import pyplot as plt
import networkx as nx
from freq_allocator.dataloader.load_chip import max_Algsubgraph
from freq_allocator.model.formula import gen_pos, draw_chip, scatter_err
from freq_allocator.model.two_qubit_model import (
    twoQ_err_model,
    twoq_T1_err,
    twoq_T2_err,
    twoq_xtalk_err,
    twoq_pulse_distort_err,
    is_xtalk,
    edge_distance
)
import time


def two_alloc(chip, a):
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 25}

    current_date = time.strftime("%Y-%m-%d")
    current_time = time.strftime("%H:%M:%S", time.localtime()).replace(':', '.')
    path = f'.\\results\\{current_date}\\{current_time}'

    # 做一个判断，按idle freq确定qh，ql后，如果完全不可能做门，就删掉这条边
    edges_to_remove = []
    for qcq in chip.edges():
        if chip.nodes[qcq[0]]['frequency'] > chip.nodes[qcq[1]]['frequency']:
            qh, ql = qcq[0], qcq[1]
        else:
            qh, ql = qcq[1], qcq[0]
        if chip.nodes[qh]['freq_min'] + chip.nodes[qh]['anharm'] > chip.nodes[ql]['freq_max']:
            edges_to_remove.append(qcq)
    chip.remove_edges_from(edges_to_remove)

    maxParallelCZs = max_Algsubgraph(chip)
    for level in range(len(maxParallelCZs)):
        couplerActivate = [[coupler, 'gray'] for coupler in chip.edges]
        for i in couplerActivate:
            if i[0] in maxParallelCZs[level]:
                i[1] = 'green'
        
        pos = gen_pos(chip)
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_edges(
            chip,
            pos,
            edgelist=chip.edges,
            edge_color=list(dict(couplerActivate).values()),
            edge_cmap=plt.cm.plasma,
            width=8,
        )
        path_name = os.path.join(path, f'twoq chip {level}.pdf')
        os.makedirs(os.path.dirname(path_name), exist_ok=True)
        nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, cmap=plt.cm.plasma)
        plt.axis('off')
        plt.savefig(path_name, dpi=300)
        plt.close()

    for level in range(len(maxParallelCZs)):
        print('level', level)
        if len(maxParallelCZs[level]) == 0:
            continue
        epoch = 0

        centerConflictQCQ = maxParallelCZs[level][0]
        newreOptimizeQCQs = []
        avgErrEpoch = []
        jumpToEmpty = False
        repeat_optimize_history = {'center_node': 1,
                                   'chip_history': [],
                                   'error_history': []
                                   }

        fixQcq = []


        # 先把固定节点附近的边分配好，后面不去动它了
        for qcq in maxParallelCZs[level]:
            if len(chip.nodes[qcq[0]]['allow freq']) == 2 or \
                    len(chip.nodes[qcq[1]]['allow freq']) == 2:
                if len(chip.nodes[qcq[0]]['allow freq']) == 2:
                    qfix = qcq[0]
                    qnfix = qcq[1]
                else:
                    qfix = qcq[1]
                    qnfix = qcq[0]

                fixQcq.append(qcq)

                if chip.nodes[qfix]['frequency'] > chip.nodes[qnfix]['frequency'] and \
                        chip.nodes[qfix]['frequency'] + chip.nodes[qfix]['anharm'] > chip.nodes[qnfix]['freq_min']:
                    chip.edges[qcq]['frequency'] = chip.nodes[qfix]['frequency'] + chip.nodes[qfix]['anharm']
                else:
                    chip.edges[qcq]['frequency'] = chip.nodes[qfix]['frequency']

        while len(
            [
                chip.edges[qcq]['all err']
                for qcq in chip.edges
                if chip.edges[qcq].get('all err', False) and qcq in maxParallelCZs[level]
            ]
        ) < len(maxParallelCZs[level]) or (
            len(
                [
                    chip.edges[qcq]['all err']
                    for qcq in chip.edges
                    if chip.edges[qcq].get('all err', False) and qcq in maxParallelCZs[level]
                ]
            )
            == len(maxParallelCZs[level])
            and not (jumpToEmpty)
        ):
            reOptimizeQCQs = [centerConflictQCQ]
            for qcq in maxParallelCZs[level]:
                if (
                    centerConflictQCQ in newreOptimizeQCQs
                    and not (qcq in reOptimizeQCQs)
                    and qcq in newreOptimizeQCQs
                ):
                    reOptimizeQCQs.append(qcq)
                elif (
                    not (chip.edges[centerConflictQCQ].get('frequency', False))
                    and not (qcq in reOptimizeQCQs)
                    and not (chip.edges[qcq].get('frequency', False))
                    and is_xtalk(chip, qcq, centerConflictQCQ)
                ):
                    reOptimizeQCQs.append(qcq)
            print('optimize gates: ', reOptimizeQCQs)
            
            for qcq in fixQcq:
                if qcq in reOptimizeQCQs:
                    reOptimizeQCQs.remove(qcq)
            reOptimizeQCQs = tuple(reOptimizeQCQs)

            bounds = []
            for qcq in reOptimizeQCQs:

                if chip.nodes[qcq[0]]['frequency'] > chip.nodes[qcq[1]]['frequency']:
                    qh, ql = qcq[0], qcq[1]
                else:
                    qh, ql = qcq[1], qcq[0]

                if chip.nodes[qh]['freq_max'] + chip.nodes[qh]['anharm'] < chip.nodes[ql]['freq_min']:
                    qh, ql = ql, qh

                lb = (max(chip.nodes[ql]['freq_min'], chip.nodes[qh]['freq_min'] + chip.nodes[qh]['anharm']))
                ub = (min(chip.nodes[ql]['freq_max'], chip.nodes[qh]['freq_max'] + chip.nodes[qh]['anharm']))

                bound = (lb, ub)
                assert bound[0] < bound[1]
                bounds.append(bound)

            @ea.Problem.single
            def err_model_fun(frequencys):
                return twoQ_err_model(frequencys, chip, maxParallelCZs[level], reOptimizeQCQs, a)

            problem = ea.Problem(
                name='two q err model',
                M=1,  # 初始化M（目标维数）
                maxormins=[1],  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
                Dim=len(reOptimizeQCQs),  # 决策变量维数
                varTypes=[1] * len(reOptimizeQCQs),  # 决策变量的类型列表，0：实数；1：整数
                lb=[b[0] for b in bounds],  # 决策变量下界
                ub=[b[1] for b in bounds],  # 决策变量上界
                evalVars=err_model_fun,
            )

            algorithm = ea.soea_DE_best_1_bin_templet(
                problem,
                # ea.Population(Encoding='RI', NIND=300),
                # MAXGEN=50,
                ea.Population(Encoding='RI', NIND=1),
                MAXGEN=1,
                logTras=1,
                # trappedValue=1e-10,
                # maxTrappedCount=20
            )
            algorithm.mutOper.F = 0.95
            algorithm.recOper.XOVR = 0.7

            # algorithm.run()
            path_name = os.path.join(path, f'pattern = {level+1}\\epoch={epoch + 1} soea_DE result')
            os.makedirs(os.path.dirname(path_name), exist_ok=True)

            freq_bset = None
            res = ea.optimize(
                algorithm,
                prophet=freq_bset,
                # prophet=np.array(self.experiment_options.FIR0),
                verbose=True,
                drawing=0,
                outputMsg=True,
                drawLog=False,
                saveFlag=True,
                dirName=path_name,
            )
            freq_bset = res['Vars'][0]

            for qcq in reOptimizeQCQs:
                chip.edges[qcq]['frequency'] = freq_bset[reOptimizeQCQs.index(qcq)]

            newreOptimizeQCQs, error_evarage, chip = \
                twoQ_checkcoli(chip, maxParallelCZs[level], a)
            # hisXtalkG.append(xTalkSubG)
            # hisReOptimizeQCQs.append(set(newreOptimizeQCQs))
            # ocqc = [len(h) for h in hisReOptimizeQCQs]

            repeat_optimize_history['error_history'].append(error_evarage)
            repeat_optimize_history['chip_history'].append(deepcopy(chip))

            # if len(repeat_optimize_history['error_history']) > 3 or len(newreOptimizeQCQs) == 0:
            if len(repeat_optimize_history['error_history']) > 1 or len(newreOptimizeQCQs) == 0:
                idx = repeat_optimize_history['error_history'].index(min(repeat_optimize_history['error_history']))
                chip = repeat_optimize_history['chip_history'][idx]

                repeat_optimize_history['error_history'] = []
                repeat_optimize_history['chip_history'] = []

                jumpToEmpty = True
            else:
                print('no jump')
                jumpToEmpty = False

            avgErrEpoch.append(error_evarage)

            print('avg err estimate', avgErrEpoch)

            # 保存每次迭代之后的xTalkSubG，主要是其中的error

            errList = [np.log10(chip.edges[qcq].get('all err', 1e-5)) for qcq in chip.edges]
            draw_chip(chip, 'results\\' + str(epoch) + 'cz err', qcq_err=errList)
            
            reOptimizeQCQsDict = dict([(qcq, edge_distance(chip, qcq, centerConflictQCQ)) 
                                for qcq in newreOptimizeQCQs if not(qcq == centerConflictQCQ)])

            emptyQCQDict = dict([(qcq, edge_distance(chip, qcq, centerConflictQCQ)) 
                                for qcq in maxParallelCZs[level] if not(qcq == centerConflictQCQ) and not(chip.edges[qcq].get('frequency', False))])

            if len(reOptimizeQCQsDict) > 1 and not (jumpToEmpty):
                print('reoptimize qcq distance', reOptimizeQCQsDict)
                centerConflictQCQ = random.choices(
                    list(reOptimizeQCQsDict.keys()),
                    weights=[
                        1 / max(0.5, distance)
                        for distance in reOptimizeQCQsDict.values()
                    ],
                    k=1,
                )[0]
            elif len(emptyQCQDict) > 0:
                print('empty qcq distance', emptyQCQDict)
                centerConflictQCQ = list(
                    sorted(emptyQCQDict.items(), key=lambda x: x[1])
                )[0][0]
            epoch += 1

    qcqList = [chip.nodes[qubit]['name'] for qubit in chip.nodes]

    errList = np.log10([chip.edges[qcq]['spectator err'] for qcq in chip.edges])
    draw_chip(chip, 'results\\' + 'best' + 'cz spectator err', qubit_err=errList)
    scatter_err(errList, qcqList, 'results\\' + 'spectator err scatter')

    errList = np.log10([chip.edges[qcq]['parallel err'] for qcq in chip.edges])
    draw_chip(chip, 'results\\' + 'best' + 'cz parallel err', qubit_err=errList)
    scatter_err(errList, qcqList, 'results\\' + 'parallel err scatter')

    errList = np.log10([chip.edges[qcq]['T err'] for qcq in chip.edges])
    draw_chip(chip, 'results\\' + 'best' + 'cz T err', qubit_err=errList)
    scatter_err(errList, qcqList, 'results\\' + 'T err scatter')

    errList = np.log10([chip.edges[qcq]['distort err'] for qcq in chip.edges])
    draw_chip(chip, 'results\\' + 'best' + 'cz distort err', qubit_err=errList)
    scatter_err(errList, qcqList, 'results\\' + 'distort err scatter')

    errList = np.log10([chip.edges[qcq]['all err'] for qcq in chip.edges])
    draw_chip(chip, 'results\\' + 'best' + 'cz all err', qubit_err=errList)
    scatter_err(errList, qcqList, 'results\\' + 'all err scatter')

    freqList = [int(round(chip.edges[qcq]['frequency'], 3)) for qcq in chip.edges]
    draw_chip(chip, 'results\\' + 'best' + 'cz freq', qubit_err=freqList)
    scatter_err(errList, qcqList, 'results\\' + 'cz freq')

    data = dict()
    for qcq in chip.edges:
        data[qcq] = {'spectator err' : chip.edges[qcq]['spectator err'],
                     'parallel err' : chip.edges[qcq]['parallel err'],
                     'T err' : chip.edges[qcq]['T err'],
                     'distort err' : chip.edges[qcq]['distort err'],
                     'all err' : chip.edges[qcq]['all err'],
                     'frequency' : chip.edges[qcq]['frequency']}
    with open('cz.json', 'w') as f:
        json.dump(data, f)

    return chip


def twoQ_checkcoli(chip, maxParallelCZ, a):
    reOptimizeQCQs = []
    conflictSpectator = dict()
    conflictGatePairs = []
    error_chip = 0
    qcq_num = 0
    for qcq in chip.edges:
        if chip.edges[qcq].get('frequency', False):
            if chip.nodes[qcq[0]]['frequency'] > chip.nodes[qcq[1]]['frequency']:
                qh, ql = qcq[0], qcq[1]
            else:
                qh, ql = qcq[1], qcq[0]
            if chip.nodes[qh]['freq_max'] + chip.nodes[qh]['anharm'] < chip.nodes[ql]['freq_min']:
                qh, ql = ql, qh

            fWork = chip.edges[qcq]['frequency']
            pulseql = fWork
            pulseqh = fWork - chip.nodes[qh]['anharm']

            T1Err1 = twoq_T1_err(
                pulseql,
                a[0],
                chip.edges[qcq]['two tq'],
                chip.nodes[ql]['T1 spectra']
            )
            T1Err2 = twoq_T1_err(
                pulseqh,
                a[0],
                chip.edges[qcq]['two tq'],
                chip.nodes[qh]['T1 spectra'],
            )
            T2Err1 = twoq_T2_err(
                pulseql,
                a[1],
                chip.edges[qcq]['two tq'],
                ac_spectrum_paras=chip.nodes[ql]['ac_spectrum'],
            )
            T2Err2 = twoq_T2_err(
                pulseqh,
                a[1],
                chip.edges[qcq]['two tq'],
                ac_spectrum_paras=chip.nodes[qh]['ac_spectrum'],
            )
            twoqDistErr = twoq_pulse_distort_err(
                [pulseqh, chip.nodes[qh]['frequency']],
                [pulseql, chip.nodes[ql]['frequency']],
                a[2],
                ac_spectrum_paras1=chip.nodes[qh]['ac_spectrum'],
                ac_spectrum_paras2=chip.nodes[ql]['ac_spectrum'],
            )
            
            twoqSpectatorErr = 1e-5
            for q in qcq:
                if q == ql:
                    pulse = pulseql
                else:
                    pulse = pulseqh

                for neighbor in chip[q]:
                    if neighbor in qcq:
                        continue
                    twoqSpectatorErrOnce = twoq_xtalk_err(
                            pulse,
                            chip.nodes[neighbor]['frequency'],
                            a[5:],
                            chip.nodes[q]['anharm'],
                            chip.nodes[neighbor]['anharm']
                        )
                        
                    if twoqSpectatorErrOnce > 4e-3:
                        if conflictSpectator.get(qcq, False):
                            conflictSpectator[qcq].append(neighbor)
                        else:
                            conflictSpectator[qcq] = [neighbor]
                        if qcq not in reOptimizeQCQs:
                            reOptimizeQCQs.append(qcq)
                    twoqSpectatorErr += twoqSpectatorErrOnce

            parallelErr = 1e-5
            for neighbor in chip.edges:
                if chip.edges[neighbor].get('frequency', False) and \
                    neighbor in maxParallelCZ and \
                    is_xtalk(chip, qcq, neighbor):
                    for q0 in qcq:
                        for q1 in neighbor:
                            if (q0, q1) in chip.edges:
                                if q0 == ql:
                                    pulse = pulseql
                                else:
                                    pulse = pulseqh

                                if (
                                    chip.nodes[neighbor[0]]['frequency']
                                    < chip.nodes[neighbor[1]]['frequency']
                                ):
                                    q1l = neighbor[0]
                                else:
                                    q1l = neighbor[1]
                                if q1 == q1l:
                                    nPulse = chip.edges[neighbor]['frequency']
                                else:
                                    nPulse = (
                                        chip.edges[neighbor]['frequency']
                                        - chip.nodes[q1]['anharm']
                                    )

                                parallelErrOnce = twoq_xtalk_err(
                                        pulse,
                                        nPulse,
                                        a[5:],
                                        chip.nodes[q0]['anharm'],
                                        chip.nodes[q1]['anharm']
                                    )

                                if parallelErrOnce > 4e-3 and not (
                                    (qcq, neighbor) in conflictGatePairs
                                    or (neighbor, qcq) in conflictGatePairs
                                ):
                                    conflictGatePairs.append((qcq, neighbor))

                                    if qcq not in reOptimizeQCQs:
                                        reOptimizeQCQs.append(qcq)
                                parallelErr += parallelErrOnce

            allErr = (
                twoqSpectatorErr
                + parallelErr
                + T1Err1
                + T1Err2
                + T2Err1
                + T2Err2
                + twoqDistErr
            )
            error_chip += allErr
            qcq_num += 1
            chip.edges[qcq]['spectator err'] = twoqSpectatorErr
            chip.edges[qcq]['parallel err'] = parallelErr
            chip.edges[qcq]['T err'] = T1Err1 + T1Err2 + T2Err1 + T2Err2
            chip.edges[qcq]['distort err'] = twoqDistErr
            chip.edges[qcq]['all err'] = allErr
            if allErr > 1.5e-2 and not (qcq in reOptimizeQCQs):
                reOptimizeQCQs.append(qcq)
                print(qcq, chip.edges[qcq]['all err'], 'qcq err')
    print('check, large err', reOptimizeQCQs)
    error_evarage = error_chip / qcq_num
    return reOptimizeQCQs, error_evarage, chip
