import time

import numpy as np
import json
import os
import pickle
import qutip as qp
# import numdifftools as nd
import networkx as nx
from scipy.optimize import minimize
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d, interp2d, interpn
from scipy.special import erf
# from NM_bnd import nm_minimize
from sko.PSO import PSO
# from sko.tools import set_run_mode
import geatpy as ea
import warnings
# from pyinstrument import Profiler
from math import prod
# from xtalk import xy_fun

# import seaborn as sns
from multiprocessing import Pool

# from sko.PSO import PSO
import sys

package_root = r"F:\wangpeng\code\quantum_control_rl_server"
sys.path.insert(0, package_root)

import logging
import time

# logger = logging.getLogger('RL')
# logger.propagate = False
# logger.handlers = []
# stream_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(stream_handler)
# logger.setLevel(logging.INFO)
# from quantum_control_rl_server.remote_env_tools import Client

MUTHRESHOLD = 0.005


def gen_pos(chip):
    # wStep = 1
    # hStep = 1
    pos = dict()
    for qubit in chip:
        # pos[qubit] = [qubit[0] * wStep, qubit[1] * hStep]
        pos[qubit] = [qubit[1], -qubit[0]]
    return pos


def twoQ_gen_pos(chip, xtalkG):
    bitPos = gen_pos(chip)
    for bit in bitPos:
        bitPos[bit][0] *= 2
        bitPos[bit][1] *= 2
    pos = dict()
    for coupler in xtalkG:
        pos[coupler] = [
            (bitPos[coupler[0]][0] + bitPos[coupler[1]][0]) / 2,
            (bitPos[coupler[0]][1] + bitPos[coupler[1]][1]) / 2,
        ]
    return pos


def err_model(frequencys, inducedChip, targets, a, use_rb_spectrum=False):
    # time_start = time.time()
    # profiler = Profiler()
    # profiler.start()
    chip = inducedChip
    # frequencys = np.clip(frequencys, 0, 1)
    # xy_input = []
    for target in targets:
        allow_freq = chip.nodes[target]['allow_freq']
        if frequencys.dtype == np.int32:
            chip.nodes[target]['frequency'] = allow_freq[frequencys[targets.index(target)]]
        else:
            chip.nodes[target]['frequency'] = allow_freq[
                int(round(frequencys[targets.index(target)] * (len(allow_freq) - 1)))
            ]
        # xy_input.append((chip.nodes[target]['frequency'], chip.nodes[target]['anharm']))

    cost = 0
    for target in targets:
        if chip.nodes[target]['available']:
            if (
                round(chip.nodes[target]['frequency'])
                not in chip.nodes[target]['allow_freq']
            ):
                cost += 1
            if not use_rb_spectrum:
                T1_err = singq_T1_err(
                    a[0],
                    chip.nodes[target]['t_sq'],
                    chip.nodes[target]['frequency'],
                    chip.nodes[target]['t1_spectrum'],
                )
                if T1_err < 0:
                    warnings.warn('有问题', category=None, stacklevel=1, source=None)
                cost += T1_err

                cost += singq_T2_err(
                    a[1],
                    chip.nodes[target]['t_sq'],
                    chip.nodes[target]['frequency'],
                    ac_spectrum_paras=chip.nodes[target]['ac_spectrum'],
                )
            else:
                isolated_error = chip.nodes[target]['isolated_error'][frequencys[targets.index(target)]]
                cost += isolated_error
            for neighbor in chip.nodes():
                if chip.nodes[target]['name'] not in chip.nodes[target]['xy_crosstalk_coef']:
                    continue
                if (
                    chip.nodes[neighbor].get('frequency', False)
                    and not (neighbor == target)
                    and chip.nodes[target]['xy_crosstalk_coef'][chip.nodes[neighbor]['name']] > MUTHRESHOLD
                ):  # 每次计算串扰误差的时候，计算所有已经分配的比特对target的串扰，而不是只计算分配区域内的
                    # if chip.nodes[neighbor]['xy_crosstalk_coef'][target] > MUTHRESHOLD and (
                    #     neighbor in targets
                    # ):
                    cost += singq_xtalk_err(
                        a[2],
                        chip.nodes[target]['anharm'],
                        chip.nodes[neighbor]['frequency']
                        - chip.nodes[target]['frequency'],
                        chip.nodes[target]['xy_crosstalk_coef'][chip.nodes[neighbor]['name']],
                        chip.nodes[target]['xy_crosstalk_f']
                    )
            # 遍历距离为1的比特，计算杂散耦合
            for neighbor in chip[target]:
                if chip.nodes[target]['available'] and chip.nodes[neighbor].get(
                    'frequency', False
                ):
                    cost += singq_zz_err(
                        a[3],
                        a[4],
                        chip.nodes[neighbor]['frequency'],
                        chip.nodes[target]['frequency'],
                        chip.nodes[neighbor]['anharm'],
                        chip.nodes[target]['anharm']
                    )
                # if (
                #     chip.nodes[target]['available']
                #     and chip.nodes[neighbor]['available']
                # ):
                #     cost += twoq_pulse_distort_err(
                #         chip.nodes[neighbor]['frequency'],
                #         chip.nodes[target]['frequency'],
                #         a[5],
                #     )
            for nNeighbor in chip.nodes():
                if not chip.nodes[nNeighbor]['available'] or nx.shortest_path_length(chip, nNeighbor, target) != 2:
                    continue
                elif chip.nodes[nNeighbor].get('frequency', False):
                    cost += singq_zz_err(
                        a[6],
                        a[7],
                        chip.nodes[nNeighbor]['frequency'],
                        chip.nodes[target]['frequency'],
                        chip.nodes[nNeighbor]['anharm'],
                        chip.nodes[target]['anharm']
                    )
    cost_average = cost / len(targets)
    # time_end = time.time()
    # print(
    #     f'running time: {time_start-time_end}, cost_average: {cost_average}, freq_list: {frequencys}'
    # )
    # profiler.stop()
    #
    # profiler.print()
    return cost_average


def twoQ_err_model(frequencys, chip, xtalkG, reOptimizeQCQs, a, parallelXY=[]):
    for qcq in reOptimizeQCQs:
        xtalkG.nodes[qcq]['frequency'] = frequencys[reOptimizeQCQs.index(qcq)]
    cost = 0
    for qcq in reOptimizeQCQs:
        if sum(qcq[0]) % 2:
            cost += twoq_T1_err(
                xtalkG.nodes[qcq]['frequency'],
                chip.nodes[qcq[0]]['frequency'],
                chip.nodes[qcq[0]]['sweet point'],
                a[0],
                xtalkG.nodes[qcq]['two tq'],
                chip.nodes[qcq[0]]['t1_spectrum'],
            )
            cost += twoq_T1_err(
                xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[1]]['anharm'],
                chip.nodes[qcq[1]]['frequency'],
                chip.nodes[qcq[1]]['sweet point'],
                a[0],
                xtalkG.nodes[qcq]['two tq'],
                chip.nodes[qcq[1]]['t1_spectrum'],
            )
            cost += twoq_T2_err(
                xtalkG.nodes[qcq]['frequency'],
                chip.nodes[qcq[0]]['frequency'],
                chip.nodes[qcq[0]]['sweet point'],
                a[1],
                xtalkG.nodes[qcq]['two tq'],
                chip.nodes[qcq[0]]['df/dphi'],
            )
            cost += twoq_T2_err(
                xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[1]]['anharm'],
                chip.nodes[qcq[1]]['frequency'],
                chip.nodes[qcq[1]]['sweet point'],
                a[1],
                xtalkG.nodes[qcq]['two tq'],
                chip.nodes[qcq[1]]['df/dphi'],
            )
        else:
            cost += twoq_T1_err(
                xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[0]]['anharm'],
                chip.nodes[qcq[0]]['frequency'],
                chip.nodes[qcq[0]]['sweet point'],
                a[0],
                xtalkG.nodes[qcq]['two tq'],
                chip.nodes[qcq[0]]['t1_spectrum'],
            )
            cost += twoq_T1_err(
                xtalkG.nodes[qcq]['frequency'],
                chip.nodes[qcq[1]]['frequency'],
                chip.nodes[qcq[1]]['sweet point'],
                a[0],
                xtalkG.nodes[qcq]['two tq'],
                chip.nodes[qcq[1]]['t1_spectrum'],
            )
            cost += twoq_T2_err(
                xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[0]]['anharm'],
                chip.nodes[qcq[0]]['frequency'],
                chip.nodes[qcq[0]]['sweet point'],
                a[1],
                xtalkG.nodes[qcq]['two tq'],
                chip.nodes[qcq[0]]['df/dphi'],
            )
            cost += twoq_T2_err(
                xtalkG.nodes[qcq]['frequency'],
                chip.nodes[qcq[1]]['frequency'],
                chip.nodes[qcq[1]]['sweet point'],
                a[1],
                xtalkG.nodes[qcq]['two tq'],
                chip.nodes[qcq[1]]['df/dphi'],
            )
        for q in qcq:
            if sum(q) % 2:
                fWork = xtalkG.nodes[qcq]['frequency']
            else:
                fWork = xtalkG.nodes[qcq]['frequency'] - chip.nodes[q]['anharm']
            for neighbor in chip.nodes():
                if neighbor in qcq:
                    continue
                if (
                    chip.nodes[neighbor]['xy_crosstalk_coef'][q] > MUTHRESHOLD
                    and neighbor in parallelXY
                ):
                    cost += twoq_xy_err(
                        chip.nodes[q]['anharm'],
                        [fWork, chip.nodes[q]['frequency']],
                        chip.nodes[neighbor]['frequency'],
                        chip.nodes[neighbor]['xy_crosstalk_coef'][q],
                        chip.nodes[q]['xy xtalk'],
                        a[2],
                        xtalkG.nodes[qcq]['two tq'],
                    )
                if neighbor in chip[q]:
                    cost += twoq_xtalk_err(
                        [fWork, chip.nodes[q]['frequency']],
                        [
                            chip.nodes[neighbor]['frequency'],
                            chip.nodes[neighbor]['frequency'],
                        ],
                        a[3],
                        a[4],
                        xtalkG.nodes[qcq]['two tq'],
                    )
                    cost += twoq_xtalk_err(
                        [fWork, chip.nodes[q]['frequency']],
                        [
                            chip.nodes[neighbor]['frequency']
                            + chip.nodes[neighbor]['anharm'],
                            chip.nodes[neighbor]['frequency']
                            + chip.nodes[neighbor]['anharm'],
                        ],
                        a[5],
                        a[6],
                        xtalkG.nodes[qcq]['two tq'],
                    )

        for neighbor in xtalkG[qcq]:
            if xtalkG.nodes[neighbor].get('frequency', False):
                for q0 in qcq:
                    for q1 in neighbor:
                        if (q0, q1) in chip.edges:
                            cost += twoq_xtalk_err(
                                [
                                    xtalkG.nodes[qcq]['frequency'],
                                    chip.nodes[q0]['frequency'],
                                ],
                                [
                                    xtalkG.nodes[neighbor]['frequency'],
                                    chip.nodes[q1]['frequency'],
                                ],
                                a[7],
                                a[8],
                                xtalkG.nodes[qcq]['two tq'],
                            )
                            if sum(q0) % 2:
                                cost += twoq_xtalk_err(
                                    [
                                        xtalkG.nodes[qcq]['frequency']
                                        + chip.nodes[q0]['anharm'],
                                        chip.nodes[q0]['frequency']
                                        + chip.nodes[q0]['anharm'],
                                    ],
                                    [
                                        xtalkG.nodes[neighbor]['frequency'],
                                        chip.nodes[q1]['frequency'],
                                    ],
                                    a[9],
                                    a[10],
                                    xtalkG.nodes[qcq]['two tq'],
                                )
                                cost += twoq_xtalk_err(
                                    [
                                        xtalkG.nodes[qcq]['frequency'],
                                        chip.nodes[q0]['frequency'],
                                    ],
                                    [
                                        xtalkG.nodes[neighbor]['frequency']
                                        - chip.nodes[q1]['anharm'],
                                        chip.nodes[q1]['frequency']
                                        - chip.nodes[q1]['anharm'],
                                    ],
                                    a[9],
                                    a[10],
                                    xtalkG.nodes[qcq]['two tq'],
                                )
                            else:
                                cost += twoq_xtalk_err(
                                    [
                                        xtalkG.nodes[qcq]['frequency']
                                        - chip.nodes[q0]['anharm'],
                                        chip.nodes[q0]['frequency']
                                        - chip.nodes[q0]['anharm'],
                                    ],
                                    [
                                        xtalkG.nodes[neighbor]['frequency'],
                                        chip.nodes[q1]['frequency'],
                                    ],
                                    a[9],
                                    a[10],
                                    xtalkG.nodes[qcq]['two tq'],
                                )
                                cost += twoq_xtalk_err(
                                    [
                                        xtalkG.nodes[qcq]['frequency'],
                                        chip.nodes[q0]['frequency'],
                                    ],
                                    [
                                        xtalkG.nodes[neighbor]['frequency']
                                        + chip.nodes[q1]['anharm'],
                                        chip.nodes[q1]['frequency']
                                        + chip.nodes[q1]['anharm'],
                                    ],
                                    a[9],
                                    a[10],
                                    xtalkG.nodes[qcq]['two tq'],
                                )
    return cost


def checkcoli(chip, a):
    centerNode = (H // 2, W // 2)
    reOptimizeNodes = dict()
    conflictEdge = []
    twoqForbiddenEdge = []
    for qubit in chip.nodes():
        # 在一次迭代之后，在所有已经分配的比特中，找到误差很大的比特
        if chip.nodes[qubit]['available'] and chip.nodes[qubit].get('frequency', False):
            if (
                round(chip.nodes[qubit]['frequency'], 3)
                in chip.nodes[qubit]['bad freq']
            ):
                print(qubit, 'badFreq')
                if not (qubit) in reOptimizeNodes:
                    reOptimizeNodes[qubit] = nx.shortest_path_length(
                        chip, qubit, centerNode
                    )
            relaxCost = singq_T1_err(
                a[0],
                chip.nodes[qubit]['t_sq'],
                chip.nodes[qubit]['frequency'],
                chip.nodes[qubit]['sweet point'],
                chip.nodes[qubit]['t1_spectrum'],
            )
            dephasCost = singq_T2_err(
                a[1],
                chip.nodes[qubit]['t_sq'],
                chip.nodes[qubit]['frequency'],
                chip.nodes[qubit]['sweet point'],
                chip.nodes[qubit]['df/dphi'],
            )
            if relaxCost / a[0] > 1e-2 or dephasCost / a[1] > 1e-1:
                print(qubit, dephasCost / a[1], relaxCost / a[0], 'dephas relax')
                if not (qubit) in reOptimizeNodes:
                    reOptimizeNodes[qubit] = nx.shortest_path_length(
                        chip, qubit, centerNode
                    )
            xyCost = 0
            for neighbor in chip.nodes():
                # 这里只计算了近邻比特的xy串扰
                if (
                    chip.nodes[neighbor].get('frequency', False)
                    and not (neighbor == qubit)
                    and chip.nodes[neighbor]['available']
                ):
                    if chip.nodes[neighbor]['xy_crosstalk_coef'][qubit] > MUTHRESHOLD:
                        xyCost += singq_xtalk_err(
                            a[2],
                            chip.nodes[qubit]['anharm'],
                            chip.nodes[qubit]['frequency']
                            - chip.nodes[neighbor]['frequency'],
                            chip.nodes[neighbor]['xy_crosstalk_coef'][qubit],
                            chip.nodes[qubit]['xy xtalk'],
                        )
                    if (qubit, neighbor) in chip.edges():
                        zzCost = singq_zz_err(
                            a[3],
                            a[4],
                            chip.nodes[qubit]['t_sq'],
                            chip.nodes[neighbor]['frequency'],
                            chip.nodes[qubit]['frequency'],
                        )
                        if (
                            not (
                                (qubit, neighbor) in conflictEdge
                                or (neighbor, qubit) in conflictEdge
                            )
                            and zzCost > 1e-2
                        ):
                            conflictEdge.append((qubit, neighbor))
                            print(qubit, neighbor, zzCost, 'zz')
                        if (
                            chip.nodes[qubit]['available']
                            and chip.nodes[neighbor]['available']
                        ):
                            distCost = twoq_pulse_distort_err(
                                chip.nodes[neighbor]['frequency'],
                                chip.nodes[qubit]['frequency'],
                                a[5],
                            )
                            if (
                                not (
                                    (qubit, neighbor) in twoqForbiddenEdge
                                    or (neighbor, qubit) in twoqForbiddenEdge
                                )
                                and distCost / a[5] > 0.7**2
                            ):
                                twoqForbiddenEdge.append((qubit, neighbor))
                                print(qubit, neighbor, distCost / a[5], 'dist')
                        for nNeighbor in chip[neighbor]:
                            if nNeighbor == qubit:
                                continue
                            elif chip.nodes[nNeighbor].get('frequency', False):
                                nzzCost = singq_zz_err(
                                    a[6],
                                    a[7],
                                    chip.nodes[qubit]['t_sq'],
                                    chip.nodes[nNeighbor]['frequency'],
                                    chip.nodes[qubit]['frequency'],
                                )
                                if (
                                    not (
                                        (qubit, nNeighbor) in conflictEdge
                                        or (nNeighbor, qubit) in conflictEdge
                                    )
                                    and nzzCost > 1e-2
                                ):
                                    conflictEdge.append((qubit, nNeighbor))
                                    print(qubit, nNeighbor, nzzCost, 'nzz')
            if xyCost / a[2] > 1e-1:
                print(qubit, xyCost / a[2], 'xy')
                if not (qubit in reOptimizeNodes):
                    reOptimizeNodes[qubit] = nx.shortest_path_length(
                        chip, qubit, centerNode
                    )
    print('node', reOptimizeNodes, 'edge', conflictEdge, 'dist', twoqForbiddenEdge)
    return reOptimizeNodes, conflictEdge, twoqForbiddenEdge


def check_error(chip, a, use_rb_spectrum=False):
    centerNode = (H // 2, W // 2)
    conflict_pairs = []
    error_chip = 0
    qubit_num = 0
    for qubit in chip.nodes():
        if chip.nodes[qubit]['available'] and chip.nodes[qubit].get('frequency', False):
            if not use_rb_spectrum:
                T1_err = singq_T1_err(
                    a[0],
                    chip.nodes[qubit]['t_sq'],
                    chip.nodes[qubit]['frequency'],
                    chip.nodes[qubit]['t1_spectrum'],
                )
                chip.nodes[qubit]['T1_err'] = T1_err

                T2_err = singq_T2_err(
                    a[1],
                    chip.nodes[qubit]['t_sq'],
                    chip.nodes[qubit]['frequency'],
                    ac_spectrum_paras=chip.nodes[qubit]['ac_spectrum'],
                )
                chip.nodes[qubit]['T2_err'] = T2_err
            else:
                freq = chip.nodes[qubit]['frequency']
                isolated_error = chip.nodes[qubit]['isolated_error'][chip.nodes[qubit]['allow_freq'].index(freq)]

            xy_crosstalk_error = 0
            chip.nodes[qubit]['xy_crosstalk_error'] = {}
            chip.nodes[qubit]['NN_error'] = {}
            chip.nodes[qubit]['NNN_error'] = {}
            for neighbor in chip.nodes():
                if chip.nodes[neighbor].get('frequency', False) and not (
                    neighbor == qubit
                ) and chip.nodes[neighbor]['name'] in chip.nodes[qubit]['xy_crosstalk_coef']:  # 每次计算串扰误差的时候，计算所有已经分配的比特对target的串扰，而不是只计算分配区域内的
                    xy_crosstalk_once_error = singq_xtalk_err(
                        a[2],
                        chip.nodes[qubit]['anharm'],
                        chip.nodes[neighbor]['frequency']
                        - chip.nodes[qubit]['frequency'],
                        chip.nodes[qubit]['xy_crosstalk_coef'][chip.nodes[neighbor]['name']],
                        chip.nodes[qubit]['xy_crosstalk_f']
                    )
                    chip.nodes[qubit]['xy_crosstalk_error'][
                        neighbor
                    ] = xy_crosstalk_once_error
                    if xy_crosstalk_once_error > 4e-3:
                        conflict_pair = (neighbor, qubit)
                        conflict_pairs.append(conflict_pair)

                    xy_crosstalk_error += xy_crosstalk_once_error

            NN_error = 0
            NNN_error = 0
            # 遍历距离为1的比特，计算杂散耦合
            for neighbor in chip[qubit]:
                if chip.nodes[qubit]['available'] and chip.nodes[neighbor].get(
                    'frequency', False
                ):
                    NN_error_once = singq_zz_err(
                        a[3],
                        a[4],
                        chip.nodes[neighbor]['frequency'],
                        chip.nodes[qubit]['frequency'],
                        chip.nodes[neighbor]['anharm'],
                        chip.nodes[qubit]['anharm']
                    )
                    NN_error += NN_error_once
                    chip.nodes[qubit]['NN_error'][neighbor] = NN_error_once

            for nNeighbor in chip.nodes():
                if not chip.nodes[nNeighbor]['available'] or nx.shortest_path_length(chip, nNeighbor, qubit) != 2:
                    continue
                if nNeighbor == qubit:
                    continue
                elif chip.nodes[nNeighbor].get('frequency', False):
                    NNN_error_once = singq_zz_err(
                        a[6],
                        a[7],
                        chip.nodes[nNeighbor]['frequency'],
                        chip.nodes[qubit]['frequency'],
                        chip.nodes[nNeighbor]['anharm'],
                        chip.nodes[qubit]['anharm']
                    )
                    NNN_error += NNN_error_once
                    chip.nodes[qubit]['NNN_error'][nNeighbor] = NNN_error_once
                    if NNN_error_once > 2.5e-3:
                        conflict_pair = (nNeighbor, qubit)
                        if conflict_pair not in conflict_pairs and (qubit, nNeighbor) not in conflict_pairs:
                            conflict_pairs.append(conflict_pair)

            if not use_rb_spectrum:
                chip.nodes[qubit]['error_all'] = (
                    T1_err + T2_err + xy_crosstalk_error + NN_error + NNN_error
                )
            else:
                chip.nodes[qubit]['error_all'] = (
                    isolated_error + xy_crosstalk_error + NN_error + NNN_error
                )
            error_chip += chip.nodes[qubit]['error_all']
            qubit_num += 1
    error_evarage = error_chip/qubit_num

    return conflict_pairs, chip, error_evarage


def twoQ_checkcoli(chip, xtalkG, a):
    distance = dict()
    reOptimizeQCQs = dict()
    conflictEdge = []
    for qcq in xtalkG:
        distance[qcq] = nx.shortest_path_length(
            nx.grid_2d_graph(H, W), qcq[0], (H // 2, W // 2)
        ) + nx.shortest_path_length(nx.grid_2d_graph(H, W), qcq[1], (H // 2, W // 2))
    centertwoQ = sorted(distance.items(), key=lambda x: x[1])[0][0]
    for qcq in xtalkG.nodes:
        if xtalkG.nodes[qcq].get('frequency', False):
            if sum(qcq[0]) % 2:
                T1Cost1 = twoq_T1_err(
                    xtalkG.nodes[qcq]['frequency'],
                    chip.nodes[qcq[0]]['frequency'],
                    chip.nodes[qcq[0]]['sweet point'],
                    a[0],
                    xtalkG.nodes[qcq]['two tq'],
                    chip.nodes[qcq[0]]['t1_spectrum'],
                )
                T1Cost2 = twoq_T1_err(
                    xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[1]]['anharm'],
                    chip.nodes[qcq[1]]['frequency'],
                    chip.nodes[qcq[1]]['sweet point'],
                    a[0],
                    xtalkG.nodes[qcq]['two tq'],
                    chip.nodes[qcq[1]]['t1_spectrum'],
                )
                T2Cost1 = twoq_T2_err(
                    xtalkG.nodes[qcq]['frequency'],
                    chip.nodes[qcq[0]]['frequency'],
                    chip.nodes[qcq[0]]['sweet point'],
                    a[1],
                    xtalkG.nodes[qcq]['two tq'],
                    chip.nodes[qcq[0]]['df/dphi'],
                )
                T2Cost2 = twoq_T2_err(
                    xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[1]]['anharm'],
                    chip.nodes[qcq[1]]['frequency'],
                    chip.nodes[qcq[1]]['sweet point'],
                    a[1],
                    xtalkG.nodes[qcq]['two tq'],
                    chip.nodes[qcq[1]]['df/dphi'],
                )
            else:
                T1Cost1 = twoq_T1_err(
                    xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[0]]['anharm'],
                    chip.nodes[qcq[0]]['frequency'],
                    chip.nodes[qcq[0]]['sweet point'],
                    a[0],
                    xtalkG.nodes[qcq]['two tq'],
                    chip.nodes[qcq[0]]['t1_spectrum'],
                )
                T1Cost2 = twoq_T1_err(
                    xtalkG.nodes[qcq]['frequency'],
                    chip.nodes[qcq[1]]['frequency'],
                    chip.nodes[qcq[1]]['sweet point'],
                    a[0],
                    xtalkG.nodes[qcq]['two tq'],
                    chip.nodes[qcq[1]]['t1_spectrum'],
                )
                T2Cost1 = twoq_T2_err(
                    xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[0]]['anharm'],
                    chip.nodes[qcq[0]]['frequency'],
                    chip.nodes[qcq[0]]['sweet point'],
                    a[1],
                    xtalkG.nodes[qcq]['two tq'],
                    chip.nodes[qcq[0]]['df/dphi'],
                )
                T2Cost2 = twoq_T2_err(
                    xtalkG.nodes[qcq]['frequency'],
                    chip.nodes[qcq[1]]['frequency'],
                    chip.nodes[qcq[1]]['sweet point'],
                    a[1],
                    xtalkG.nodes[qcq]['two tq'],
                    chip.nodes[qcq[1]]['df/dphi'],
                )
            if (
                T1Cost1 / a[0] > 1e-2
                or T1Cost2 / a[0] > 1e-2
                or T2Cost1 / a[1] > 3e-1
                or T2Cost2 / a[1] > 3e-1
            ):
                print(
                    qcq,
                    't1 t2',
                    T1Cost1 / a[0],
                    T1Cost2 / a[0],
                    T2Cost1 / a[1],
                    T2Cost2 / a[1],
                )
                if not (qcq in reOptimizeQCQs):
                    if nx.has_path(xtalkG, qcq, centertwoQ):
                        reOptimizeQCQs[qcq] = nx.shortest_path_length(
                            xtalkG, qcq, centertwoQ
                        )
                    else:
                        reOptimizeQCQs[qcq] = 1000

            # twoqxyCost = 0
            twoqidleCost = 0
            for q in qcq:
                if sum(q) % 2:
                    fWork = xtalkG.nodes[qcq]['frequency']
                else:
                    fWork = xtalkG.nodes[qcq]['frequency'] - chip.nodes[q]['anharm']
                for neighbor in chip.nodes():
                    if neighbor in qcq:
                        continue
                    # if chip.nodes[neighbor]['xy_crosstalk_coef'][q] > MUTHRESHOLD:
                    #     twoqxyCost += twoq_xy_err(chip.nodes[q]['anharm'], [fWork, chip.nodes[q]['frequency']],
                    #                     chip.nodes[neighbor]['frequency'], chip.nodes[neighbor]['xy_crosstalk_coef'][q],
                    #                     chip.nodes[q]['xy xtalk'], a[2], xtalkG.nodes[qcq]['two tq'])
                    if neighbor in chip[q]:
                        twoqidleCost += twoq_xtalk_err(
                            [fWork, chip.nodes[q]['frequency']],
                            [
                                chip.nodes[neighbor]['frequency'],
                                chip.nodes[neighbor]['frequency'],
                            ],
                            a[3],
                            a[4],
                            xtalkG.nodes[qcq]['two tq'],
                        )
                        twoqidleCost += twoq_xtalk_err(
                            [fWork, chip.nodes[q]['frequency']],
                            [
                                chip.nodes[neighbor]['frequency']
                                + chip.nodes[neighbor]['anharm'],
                                chip.nodes[neighbor]['frequency']
                                + chip.nodes[neighbor]['anharm'],
                            ],
                            a[5],
                            a[6],
                            xtalkG.nodes[qcq]['two tq'],
                        )
            # if twoqxyCost / a[2] > 1e-2 and not(qcq in reOptimizeQCQs):
            #     print(qcq, 'xy', twoqxyCost / a[2])
            #     reOptimizeQCQs[qcq] = nx.shortest_path_length(xtalkG, qcq, centertwoQ)
            if twoqidleCost > 3e-1 and not (qcq in reOptimizeQCQs):
                if nx.has_path(xtalkG, qcq, centertwoQ):
                    reOptimizeQCQs[qcq] = nx.shortest_path_length(
                        xtalkG, qcq, centertwoQ
                    )
                else:
                    reOptimizeQCQs[qcq] = 1000
                print(qcq, 'idle', twoqidleCost)

            for neighbor in xtalkG[qcq]:
                if xtalkG.nodes[neighbor].get('frequency', False):
                    for q0 in qcq:
                        for q1 in neighbor:
                            if (q0, q1) in chip.edges:
                                intCost = twoq_xtalk_err(
                                    [
                                        xtalkG.nodes[qcq]['frequency'],
                                        chip.nodes[q0]['frequency'],
                                    ],
                                    [
                                        xtalkG.nodes[neighbor]['frequency'],
                                        chip.nodes[q1]['frequency'],
                                    ],
                                    a[7],
                                    a[8],
                                    xtalkG.nodes[qcq]['two tq'],
                                )
                                if sum(q0) % 2:
                                    intCost += twoq_xtalk_err(
                                        [
                                            xtalkG.nodes[qcq]['frequency']
                                            + chip.nodes[q0]['anharm'],
                                            chip.nodes[q0]['frequency']
                                            + chip.nodes[q0]['anharm'],
                                        ],
                                        [
                                            xtalkG.nodes[neighbor]['frequency'],
                                            chip.nodes[q1]['frequency'],
                                        ],
                                        a[9],
                                        a[10],
                                        xtalkG.nodes[qcq]['two tq'],
                                    )
                                    intCost += twoq_xtalk_err(
                                        [
                                            xtalkG.nodes[qcq]['frequency'],
                                            chip.nodes[q0]['frequency'],
                                        ],
                                        [
                                            xtalkG.nodes[neighbor]['frequency']
                                            - chip.nodes[q1]['anharm'],
                                            chip.nodes[q1]['frequency']
                                            - chip.nodes[q1]['anharm'],
                                        ],
                                        a[9],
                                        a[10],
                                        xtalkG.nodes[qcq]['two tq'],
                                    )
                                else:
                                    intCost += twoq_xtalk_err(
                                        [
                                            xtalkG.nodes[qcq]['frequency']
                                            - chip.nodes[q0]['anharm'],
                                            chip.nodes[q0]['frequency']
                                            - chip.nodes[q0]['anharm'],
                                        ],
                                        [
                                            xtalkG.nodes[neighbor]['frequency'],
                                            chip.nodes[q1]['frequency'],
                                        ],
                                        a[9],
                                        a[10],
                                        xtalkG.nodes[qcq]['two tq'],
                                    )
                                    intCost += twoq_xtalk_err(
                                        [
                                            xtalkG.nodes[qcq]['frequency'],
                                            chip.nodes[q0]['frequency'],
                                        ],
                                        [
                                            xtalkG.nodes[neighbor]['frequency']
                                            + chip.nodes[q1]['anharm'],
                                            chip.nodes[q1]['frequency']
                                            + chip.nodes[q1]['anharm'],
                                        ],
                                        a[9],
                                        a[10],
                                        xtalkG.nodes[qcq]['two tq'],
                                    )
                                if (
                                    not (
                                        (qcq, neighbor) in conflictEdge
                                        or (neighbor, qcq) in conflictEdge
                                    )
                                    and intCost > 3e-1
                                ):
                                    print(qcq, neighbor, 'int', intCost)
                                    conflictEdge.append((qcq, neighbor))

    return reOptimizeQCQs, conflictEdge


def T1_spectra(fMax, step):
    badFreqNum = np.random.randint(5)
    fList = np.linspace(3.75, fMax, step)
    gamma = 1e-3 + (2e-2 - 1e-3) * np.random.random()
    T1 = np.random.normal(np.random.randint(20, 50), 5, step)
    for _ in range(badFreqNum):
        a = np.random.random() * 0.6
        badFreq = 3.75 + (fMax - 3.75) * np.random.random()
        T1 -= lorentzain(fList, badFreq, a, gamma)
    for T in range(len(T1)):
        T1[T] = np.max([1, T1[T]])
    return 1e-3 / T1


def f_phi_spectra(fMax, phi):
    d = 0
    return fMax * np.sqrt(
        np.abs(np.cos(np.pi * phi)) * np.sqrt(1 + d**2 * np.tan(np.pi * phi) ** 2)
    )


def phi2f(phi, fMax, step):
    phiList = np.linspace(0, 0.5, step)
    fList = f_phi_spectra(fMax, phiList)
    func_interp = interp1d(phiList, fList, kind='cubic')
    if isinstance(phi, (int, float)):
        return float(func_interp(phi))
    else:
        return func_interp(phi)


def f2phi(f, fq_max, Ec, d, w=None, g=None):
    if w:
        f = f - g ** 2 / (f - w)
    alpha = (f + Ec) / (Ec + fq_max)
    beta = (alpha**4 - d**2) / (1 - d**2)
    phi = np.arccos(np.sqrt(beta))
    return phi


def T2_spectra(fMax, step):
    fList = np.linspace(3.75, fMax, step + 1)
    phiList = f2phi(fList, fMax, step)
    df_dphi = np.abs(np.diff(fList) / np.diff(phiList))
    return df_dphi * 1e-3


def lorentzain(fi, fj, a, gamma):
    wave = (1 / np.pi) * (gamma / ((fi - fj) ** 2 + (gamma) ** 2))
    return a * wave


def singq_T1_err(a, tq, f, t1_spectrum):
    f_list = t1_spectrum['freq']
    t1_list = t1_spectrum['t1']
    # return a * (1 - np.exp(-tq * func_interp(f)))
    try:
        func_interp = interp1d(f_list, t1_list, kind='cubic')
        error = a * tq / func_interp(f)
    except:
        error = 5e-4
    if error < 0:
        error = 5e-4
    return error


def singq_T2_err(a, tq, f, t2_spectrum: dict = None, ac_spectrum_paras: list = None):
    if t2_spectrum:
        freq_list = t2_spectrum['freq']
        t2_list = t2_spectrum['t2']
        func_interp = interp1d(freq_list, t2_list, kind='cubic')
        # return a * (1 - np.exp(-tq * func_interp(f)))
        return a * tq * func_interp(f)
    else:
        ac_spectrum_paras_sub = ac_spectrum_paras[:2] + ac_spectrum_paras[4:7]
        df_dphi = 1 / (
            abs(f2phi(f, *ac_spectrum_paras_sub) - f2phi(f - 0.01, *ac_spectrum_paras_sub)) / 0.01
        )
        error = a * tq * df_dphi
        if np.isnan(error):
            return 5e-4
        else:
            return error


def singq_xtalk_err(a, anharm, detune, mu,f):
    # alpha_list = xy_crosstalk_sim['alpha_list']
    # mu_list = xy_crosstalk_sim['mu_list']
    # detune_list = xy_crosstalk_sim['detune_list']
    # error_arr = xy_crosstalk_sim['error_arr'][alpha_list.index(anharm)]

    try:
        # error = a * interpn((mu_list, detune_list), error_arr, np.array([mu, detune]))
        # x, y = np.meshgrid(mu_list, detune_list)
        # f = interp2d(detune_list, mu_list, error_arr, kind='cubic')
        error = a * f(detune, mu)
        # if error[0]<0:
        #     print(error[0])
        return error[0]
    except:
        return 0


def singq_zz_err(a, gamma, fi, fj, alpha_i, alpha_j):

    return lorentzain(fi, fj, a, gamma) + lorentzain(fi + alpha_i, fj, a, gamma) + lorentzain(fi, fj + alpha_j, a, gamma)


def twoq_T1_err(fWork, fidle, fMax, a, tq, T1Spectra):
    step = 1000
    ft = twoq_pulse(fWork, fidle, tq, step)
    fList = np.linspace(3.75, fMax, step)
    func_interp = interp1d(fList, T1Spectra, kind='cubic')
    TtList = func_interp(ft)
    return a * (1 - np.exp(-np.sum(TtList) * (tq / step)))
    # return a * np.sum(TtList) * (tq / step)


def twoq_T2_err(fWork, fidle, fMax, a, tq, df_dphiList):
    step = 1000
    ft = twoq_pulse(fWork, fidle, tq, step)
    fList = np.linspace(3.75, fMax, step)
    func_interp = interp1d(fList, df_dphiList, kind='cubic')
    TtList = func_interp(ft)
    return a * (1 - np.exp(-np.sum(TtList) * (tq / step)))
    # return a * np.sum(TtList) * (tq / step)


def twoq_xy_err(anharm, fq, fn, mu, fxy, a, twotq):
    anharms, detunes, mus = fxy[0], fxy[1], fxy[2]
    step = 100
    fqts = twoq_pulse(fq[0], fq[1], twotq, step)
    xtalkList = []
    for fqt in fqts:
        xtalkList.append(
            interpn((anharms, detunes, mus), fxy[3], np.array([anharm, fqt - fn, mu]))
        )
    return a * np.sum(xtalkList) * (twotq / step)


def twoq_xtalk_err(fi, fj, a, gamma, tq):
    step = 100
    fits = twoq_pulse(fi[0], fi[1], tq, step)
    fjts = twoq_pulse(fj[0], fj[1], tq, step)
    xtalkList = [lorentzain(fit, fjt, a, gamma) for (fit, fjt) in zip(fits, fjts)]
    return np.sum(xtalkList) * (tq / step)


def twoq_pulse_distort_err(fi, fj, a):
    return a * (fi - fj) ** 2


def twoq_pulse(freqWork, freqMax, tq, step):
    if freqWork == freqMax:
        return [freqWork] * step
    else:
        pulseLen = tq
        tList = np.linspace(0, pulseLen, step)
        sigma = [1.5]
        flattop_start = 3 * sigma[0]
        flattop_end = pulseLen - 3 * sigma[0]
        freqList = (freqWork - freqMax) * 1 / 2 * (
            erf((tList - flattop_start) / (np.sqrt(2) * sigma[0]))
            - erf((tList - flattop_end) / (np.sqrt(2) * sigma[0]))
        ) + freqMax
        return freqList


def xtalk_G(chip):
    xtalkG = nx.Graph()
    for coupler1 in chip.edges:
        if not (coupler1 in xtalkG.nodes):
            xtalkG.add_node(coupler1)
        if not (xtalkG.nodes[coupler1].get('two tq')):
            xtalkG.nodes[coupler1]['two tq'] = 60
        for coupler2 in chip.edges:
            if coupler1 == coupler2 or (coupler1, coupler2) in xtalkG.edges:
                continue
            distance = []
            for i in coupler1:
                for j in coupler2:
                    if nx.has_path(chip, i, j):
                        distance.append(nx.shortest_path_length(chip, i, j))
                    else:
                        distance.append(100000)
            if 1 in distance and not (0 in distance):
                xtalkG.add_edge(coupler1, coupler2)
    return xtalkG


def max_Algsubgraph(chip):
    dualChip = nx.Graph()
    dualChip.add_nodes_from(list(chip.edges))
    for coupler1 in dualChip.nodes:
        for coupler2 in dualChip.nodes:
            if coupler1 == coupler2 or set(coupler1).isdisjoint(set(coupler2)):
                continue
            else:
                dualChip.add_edge(coupler1, coupler2)
    maxParallelCZs = [[], [], [], []]
    for edge in chip.edges:
        if sum(edge[0]) < sum(edge[1]):
            start = edge[0]
            end = edge[1]
        else:
            start = edge[1]
            end = edge[0]
        if start[0] == end[0]:
            if sum(start) % 2:
                maxParallelCZs[0].append(edge)
            else:
                maxParallelCZs[2].append(edge)
        else:
            if sum(start) % 2:
                maxParallelCZs[1].append(edge)
            else:
                maxParallelCZs[3].append(edge)
    return maxParallelCZs


def sigq_alloc(chip, a, s: int = 1, use_rb_spectrum=False):
    current_date = time.strftime("%Y-%m-%d")
    current_time = time.strftime("%H:%M:%S", time.localtime()).replace(':', '.')
    path = f'.\\results\\{current_date}\\{current_time}'
    epoch = 0

    centerConflictNode = (H // 2, W // 2)
    conflictNodeDict = dict()
    for qubit in chip.nodes():
        conflictNodeDict[qubit] = 'gray'

    # conflictPercents = []
    reOptimizeNodes = []
    conflict_pairs_history = []
    repeat_optimize_history = {'center_node': centerConflictNode,
                             'chip_history': [],
                             'error_history': []
                             }
    for _ in range(200):
        if len(reOptimizeNodes) < 2:
            reOptimizeNodes = [centerConflictNode]

            for qubit in chip.nodes():
                # 将中心比特附近S邻域内的、可分配的、未分配的放入待优化节点中
                if (
                    # conflictNodeDict[centerConflictNode] == 'gray'
                    chip.nodes[qubit]['available']
                    # and not (qubit in reOptimizeNodes)
                    and not (chip.nodes[qubit].get('frequency', False))
                    and nx.shortest_path_length(chip, centerConflictNode, qubit) <= s
                    and centerConflictNode != qubit
                ):
                    reOptimizeNodes.append(qubit)

        bounds = []
        scale = []

        for qubit in reOptimizeNodes:
            # bounds.append((0, 1))
            scale.append(
                (
                    max(chip.nodes[qubit]['allow_freq'])
                    - min(chip.nodes[qubit]['allow_freq'])
                )
                / 2
            )

        result_list = []

        @ea.Problem.single
        def err_model_fun(frequencys):
            print(frequencys)
            return err_model(frequencys, chip, reOptimizeNodes, a, use_rb_spectrum=use_rb_spectrum)

        def err_model_fun_multi(Vars):
            # cost = []
            # for i in range(Vars.shape[0]):
            #     costi = err_model(Vars[i, :], chip, reOptimizeNodes, a, xy_crosstalk_sim)
            #     cost.append(costi)
            cost = qp.parallel_map(err_model, Vars, task_args=(chip, reOptimizeNodes, a,), num_cpus=7)
            return cost

        freq_bset = None
        for i in range(1):
            print(f'************{i}**********')
            ini_frequency = np.round(np.random.uniform(0, 1, len(reOptimizeNodes)), 3)
            # nm算法
            # res, sim = nm_minimize(
            #     err_model,
            #     ini_frequency,
            #     args=(chip, reOptimizeNodes, a, xy_crosstalk_sim),
            #     nonzdelt=0.05,
            #     maxiter=1000,
            #     maxfev=1000,
            #     xatol=1e-6,
            #     # fatol: float = 1e-4,
            #     step=[1e-3] * len(ini_frequency),
            #     bound=bounds,
            # )
            # 粒子群算法
            # set_run_mode(err_model_fun, 'multiprocessing')
            # pso = PSO(
            #     func=err_model_fun,
            #     n_dim=len(reOptimizeNodes),
            #     pop=100,
            #     max_iter=80,
            #     lb=[0] * len(reOptimizeNodes),
            #     ub=[1] * len(reOptimizeNodes),
            #     w=0.8,
            #     c1=0.5,
            #     c2=0.5,
            #     verbose=True,
            # )
            # pso.run()

            # DE算法
            problem = ea.Problem(
                name='soea err model',
                M=1,  # 初始化M（目标维数）
                maxormins=[1],  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
                Dim=len(reOptimizeNodes),  # 决策变量维数
                varTypes=[1] * len(reOptimizeNodes),  # 决策变量的类型列表，0：实数；1：整数
                lb=[0] * len(reOptimizeNodes),  # 决策变量下界
                ub=[len(chip.nodes[qubit]['allow_freq'])-1 for qubit in reOptimizeNodes],  # 决策变量上界
                evalVars=err_model_fun
            )
            if len(reOptimizeNodes) > 4:
                NIND = 80
            else:
                NIND = 40
            algorithm = ea.soea_DE_best_1_bin_templet(
                problem,
                ea.Population(Encoding='RI', NIND=NIND),
                MAXGEN=10,
                logTras=1,
                # trappedValue=1e-10,
                # maxTrappedCount=20
            )
            algorithm.mutOper.F = 0.95
            algorithm.recOper.XOVR = 0.7

            path_name = os.path.join(path, f'epoch={epoch}-{i} soea_DE result')
            os.makedirs(os.path.dirname(path_name), exist_ok=True)

            res = ea.optimize(
                algorithm,
                prophet=freq_bset,
                # prophet=np.array(self.experiment_options.FIR0),
                verbose=True, drawing=1, outputMsg=True,
                drawLog=False, saveFlag=True, dirName=path_name
            )
            freq_bset = res['Vars'][0]
            result_list.append(res)
            if res['ObjV'] < 5e-3:
                break
        # fun_list = [res.fun for res in result_list]
        # freq_list_bset = result_list[fun_list.index(min(fun_list))].x
        # fun_list = [res.gbest_y for res in result_list]
        # freq_list_bset = result_list[fun_list.index(min(fun_list))].gbest_x
        fun_list = [res['ObjV'] for res in result_list]
        freq_list_bset = result_list[fun_list.index(min(fun_list))]['Vars'][0]
        print(f'qubit num: {len(reOptimizeNodes)}')

        # print(input('start:'))
        # sleep()
        # client_socket = Client()
        # (host, port) = '127.0.0.1', 5555
        # client_socket.connect((host, port))
        # done = False
        # freq_list_process = []
        # error_process = []
        # while not done:
        #     message, done = client_socket.recv_data()
        #     logger.info('Received message from RL agent server.')
        #     logger.info('Time stamp: %f' % time.time())
        #     if not message:
        #         done = True
        #     if done:
        #         logger.info('Training finished.')
        #         break
        #     action_batch = message['action_batch']
        #     batch_size = message['batch_size']
        #     epoch_type = message['epoch_type']
        #     epoch = message['epoch']
        #     new_shape_list_freq = list(action_batch['freq'].shape)
        #
        #     new_shape_list_freq.pop(1)
        #     freq_arr = action_batch['freq'].reshape(new_shape_list_freq)
        #     logger.info('Start %s epoch %d' % (epoch_type, epoch))
        #
        #
        #     # for ii in range(batch_size):
        #     #     expectation_list[ii], prob_list[ii] = func_and_der(thetas_list[ii])
        #     error_list = err_model_fun_multi(freq_arr)
        #     for freq_list in freq_arr:
        #         freq_list_process.append(freq_list)
        #         # error_list.append(err_model(freq_list, chip, reOptimizeNodes, a, xy_crosstalk_sim))
        #     error_process.extend(error_list)
        #     print(min(error_list))
        #     reward_data = 1-np.array(error_list)
        #
        #     R = np.mean(reward_data)
        #     std_R = np.std(reward_data)
        #     logger.info('Average reward %.5f' % R)
        #     logger.info('STDev reward %.5f' % std_R)
        #     logger.info('Average prob %.5f' % np.mean(error_list))
        #
        #     # send reward data back to server (see tf_env -> reward_remote())
        #     logger.info('Sending message to RL agent server.')
        #     logger.info('Time stamp: %f' % time.time())
        #     client_socket.send_data(reward_data)
        # freq_list_bset = freq_list_process[error_process.index(min(error_process))]

        for i, qubit in enumerate(reOptimizeNodes):
            allow_freq = chip.nodes[qubit]['allow_freq']
            if isinstance(freq_list_bset[0], np.int32):
                chip.nodes[qubit]['frequency'] = allow_freq[freq_list_bset[i]]
            else:
                chip.nodes[qubit]['frequency'] = allow_freq[
                    int(round(freq_list_bset[i] * (len(allow_freq) - 1)))
                ]
            # chip.nodes[qubit]['frequency'] = res.x[i]
            # chip.nodes[qubit]['frequency'] = freq_list_bset[i]

        conflict_pairs, chip, error_evarage = check_error(chip, arb, use_rb_spectrum=use_rb_spectrum)

        # 保存每次迭代之后的chip，主要是其中的error
        path_name = os.path.join(path, f'epoch={epoch},chip_process.pickle')
        os.makedirs(os.path.dirname(path_name), exist_ok=True)
        with open(path_name, "ab") as f:
            pickle.dump(chip, f)

        drawChip = deepcopy(chip)

        pos = gen_pos(drawChip)
        labelDict = dict()

        error_dic = {}
        for qubit in chip:
            pos[qubit] = [qubit[1], -qubit[0]]
            labelDict[qubit] = chip.nodes[qubit]['name']
            error_dic[qubit] = chip.nodes[qubit].get('error_all', 1e-3)
        
        nx.draw_networkx(
            drawChip,
            pos,
            labels=labelDict,
            with_labels=True,
            nodelist=drawChip.nodes,
            node_size=600,
            node_color=np.log10(list(error_dic.values())) / np.log(10),
            # font_color=node_font_colors,
            cmap='jet',
        )
        # plt.colorbar(
        #     matplotlib.cm.ScalarMappable(
        #         norm=matplotlib.colors.LogNorm(
        #             vmin=min(list(error_dic.values())),
        #             vmax=max(list(error_dic.values())),
        #         ),
        #         cmap='jet',
        #     )
        # )
        plt.title(f"epoch = {epoch+1}")
        plt.axis('off')

        path_name = os.path.join(path, f'epoch={epoch},chip_error_log.png')
        os.makedirs(os.path.dirname(path_name), exist_ok=True)
        plt.savefig(path_name, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(list(range(len(error_dic))), list(error_dic.values()))
        ax.set_xticks(ticks=range(len(labelDict)), labels=list(labelDict.values()))
        ax.set_yscale('log')
        ax.set_title(f"epoch = {epoch + 1} scatter")
        path_name = os.path.join(path, f'epoch = {epoch + 1} scatter.png')
        os.makedirs(os.path.dirname(path_name), exist_ok=True)
        plt.savefig(path_name, bbox_inches='tight')
        plt.close()

        if len(repeat_optimize_history['error_history']) > 3:
            conflict_pairs = []
            idx = repeat_optimize_history['error_history'].index(min(repeat_optimize_history['error_history']))
            chip = repeat_optimize_history['chip_history'][idx]

        emptyNode = dict(
            [
                (qubit, nx.shortest_path_length(chip, qubit, centerConflictNode))
                for qubit in chip.nodes()
                if not (chip.nodes[qubit].get('frequency', False))
            ]
        )

        if len(conflict_pairs) > 0:
            conflict_qubits = sum(conflict_pairs, ())
            for qubit in conflict_qubits:
                if qubit not in reOptimizeNodes:
                    reOptimizeNodes.append(qubit)

            repeat_optimize_history['error_history'].append(error_evarage)
            repeat_optimize_history['chip_history'].append(drawChip)

        elif len(emptyNode) > 0:
            centerConflictNode = list(sorted(emptyNode.items(), key=lambda x: x[1]))[0][
                0
            ]
            reOptimizeNodes = []

            repeat_optimize_history['center_node'] = centerConflictNode
            repeat_optimize_history['error_history'] = []
            repeat_optimize_history['chip_history'] = []

        else:
            break

        epoch += 1

    freq_dic_result = {}
    for qubit in chip:
        pos[qubit] = [qubit[1], -qubit[0]]
        labelDict[qubit] = round(chip.nodes[qubit].get('frequency', 5000))
        freq_dic_result[qubit] = chip.nodes[qubit].get('frequency', 5000)

    nx.draw_networkx(
        drawChip,
        pos,
        labels=labelDict,
        with_labels=True,
        nodelist=drawChip.nodes,
        node_size=600,
        font_size=8,
        node_color=list(freq_dic_result.values()),
        # font_color=node_font_colors,
        cmap='coolwarm',
    )
    # plt.colorbar(
    #     matplotlib.cm.ScalarMappable(
    #         norm=matplotlib.colors.Normalize(
    #             vmin=min(list(freq_dic_result.values())),
    #             vmax=max(list(freq_dic_result.values())),
    #         ),
    #         cmap='coolwarm',
    #     )
    # )
    plt.title(f"final freq")
    plt.axis('off')
    # plt.savefig(str(epoch) + str(W) + str(H) + 'chip conflict.pdf', dpi=300)
    # plt.close()
    path_name = os.path.join(path, f'result_freq.png')
    os.makedirs(os.path.dirname(path_name), exist_ok=True)
    plt.savefig(path_name, bbox_inches='tight')
    plt.close()
    return chip, conflictNodeDict, conflictEdge, twoQForbiddenCoupler


def twoq_alloc(chip, conflictNodeDict, conflictEdge, twoQForbiddenCoupler, a):
    removeQCQs = []
    for qcq in chip.edges:
        removeQCQ = False
        for q in qcq:
            if chip.nodes[q]['frequency'] < 3.75 or conflictNodeDict[q] == 'red':
                removeQCQ = True
                break
        if removeQCQ:
            removeQCQs.append(qcq)
            continue
        if (
            (qcq in conflictEdge)
            or (qcq[::-1] in conflictEdge)
            or (qcq in twoQForbiddenCoupler)
            or (qcq[::-1] in twoQForbiddenCoupler)
        ):
            removeQCQs.append(qcq)
            continue

    chip.remove_edges_from(removeQCQs)
    maxParallelCZs = max_Algsubgraph(chip)
    xtalkG = xtalk_G(chip)

    for level in range(len(maxParallelCZs)):
        couplerActivate = [[coupler, 'gray'] for coupler in chip.edges]
        for i in couplerActivate:
            if i[0] in maxParallelCZs[level]:
                i[1] = 'green'
        pos = gen_pos(chip)
        plt.figure(figsize=(4, 8))
        nx.draw_networkx_edges(
            chip,
            pos,
            edgelist=chip.edges,
            edge_color=list(dict(couplerActivate).values()),
            edge_cmap=plt.cm.Reds_r,
            width=8,
        )
        nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, cmap=plt.cm.Reds_r)
        plt.axis('off')
        plt.savefig('twoq chip ' + str(level) + '.pdf', dpi=300)
        plt.close()

    xtalkGs = []
    for level in range(len(maxParallelCZs)):
        print('level', level)
        if len(maxParallelCZs[level]) == 0:
            continue
        epoch = 0
        conflictQCQPercents = []
        conflictQCQDict = dict()
        xTalkSubG = deepcopy(xtalkG)
        for qcq in xtalkG.nodes:
            if qcq in maxParallelCZs[level]:
                conflictQCQDict[qcq] = 'gray'
        xTalkSubG.remove_nodes_from(
            set(xtalkG.nodes).difference(set(maxParallelCZs[level]))
        )

        distance = dict()
        for qcq in xTalkSubG:
            if nx.has_path(chip, qcq[0], (H // 2, W // 2)):
                distance[qcq] = nx.shortest_path_length(
                    chip, qcq[0], (H // 2, W // 2)
                ) + nx.shortest_path_length(chip, qcq[1], (H // 2, W // 2))
            else:
                distance[qcq] = 100000
        centertwoQ = sorted(distance.items(), key=lambda x: x[1])[0]
        centerConflictQCQ = centertwoQ[0]

        for _ in range(20):
            reOptimizeQCQs = [centerConflictQCQ]
            for qcq in xTalkSubG.nodes():
                if (
                    conflictQCQDict[centerConflictQCQ] == 'gray'
                    and not (qcq in reOptimizeQCQs)
                    and not (xTalkSubG.nodes[qcq].get('frequency', False))
                    and qcq in xTalkSubG[centerConflictQCQ]
                ):
                    reOptimizeQCQs.append(qcq)
                elif (
                    conflictQCQDict[centerConflictQCQ] == 'red'
                    and not (qcq in reOptimizeQCQs)
                    and qcq in xTalkSubG[centerConflictQCQ]
                    and distance[qcq] >= distance[centerConflictQCQ]
                    and conflictQCQDict[qcq] == 'red'
                ):
                    reOptimizeQCQs.append(qcq)

            reOptimizeQCQs = tuple(reOptimizeQCQs)
            bounds = []
            for qcq in reOptimizeQCQs:
                if chip.nodes[qcq[0]]['frequency'] > chip.nodes[qcq[1]]['frequency']:
                    qh, ql = qcq[0], qcq[1]
                else:
                    qh, ql = qcq[1], qcq[0]
                bounds.append(
                    (
                        max(
                            3.75,
                            chip.nodes[qh]['sweet point']
                            - 0.7
                            + chip.nodes[qh]['anharm'],
                        ),
                        chip.nodes[ql]['sweet point'],
                    )
                )
            ini_frequency = [(max(bound) + min(bound)) / 2 for bound in bounds]
            res = minimize(
                twoQ_err_model,
                ini_frequency,
                args=(chip, xTalkSubG, reOptimizeQCQs, a),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200},
            )
            print('err estimate', res.fun)
            ini_frequency = res.x

            for qcq in reOptimizeQCQs:
                xTalkSubG.nodes[qcq]['frequency'] = round(
                    res.x[reOptimizeQCQs.index(qcq)], 3
                )

            newreOptimizeQCQs, conflictQCQEdge = twoQ_checkcoli(chip, xTalkSubG, a)

            conflictCount = dict()
            for edge in conflictQCQEdge:
                if edge[0] in conflictQCQDict:
                    if edge[0] in conflictCount:
                        conflictCount[edge[0]] += 1
                    else:
                        conflictCount[edge[0]] = 1
                if edge[1] in conflictQCQDict:
                    if edge[1] in conflictCount:
                        conflictCount[edge[1]] += 1
                    else:
                        conflictCount[edge[1]] = 1

            conflictQCQEdgeDict = dict()
            alreadyConflict = []
            for edge in xTalkSubG.edges:
                if edge in conflictQCQEdge:
                    conflictQCQEdgeDict[edge] = 'red'
                    if edge[0] in alreadyConflict:
                        if conflictQCQDict[edge[0]] == 'red':
                            conflictQCQDict[edge[1]] = 'green'
                            alreadyConflict.append(edge[1])
                        else:
                            conflictQCQDict[edge[1]] = 'red'
                            alreadyConflict.append(edge[1])
                    elif edge[1] in alreadyConflict:
                        if conflictQCQDict[edge[1]] == 'red':
                            conflictQCQDict[edge[0]] = 'green'
                            alreadyConflict.append(edge[0])
                        else:
                            conflictQCQDict[edge[0]] = 'red'
                            alreadyConflict.append(edge[0])
                    else:
                        if nx.shortest_path_length(
                            nx.grid_2d_graph(H, W), edge[0][0], (H // 2, W // 2)
                        ) + nx.shortest_path_length(
                            nx.grid_2d_graph(H, W), edge[0][1], (H // 2, W // 2)
                        ) > nx.shortest_path_length(
                            nx.grid_2d_graph(H, W), edge[1][0], (H // 2, W // 2)
                        ) + nx.shortest_path_length(
                            nx.grid_2d_graph(H, W), edge[1][1], (H // 2, W // 2)
                        ):
                            conflictQCQDict[edge[0]] = 'red'
                            conflictQCQDict[edge[1]] = 'green'
                        else:
                            conflictQCQDict[edge[1]] = 'red'
                            conflictQCQDict[edge[0]] = 'green'
                        alreadyConflict.append(edge[0])
                        alreadyConflict.append(edge[1])
                elif xTalkSubG.nodes[edge[0]].get(
                    'frequency', False
                ) and xTalkSubG.nodes[edge[1]].get('frequency', False):
                    conflictQCQEdgeDict[edge] = 'green'
                    if not (edge[0] in alreadyConflict):
                        conflictQCQDict[edge[0]] = 'green'
                    elif not (edge[1] in alreadyConflict):
                        conflictQCQDict[edge[1]] = 'green'
                else:
                    conflictQCQEdgeDict[edge] = 'gray'

            for qcq in xTalkSubG.nodes:
                if qcq in newreOptimizeQCQs:
                    conflictQCQDict[qcq] = 'red'
                if (
                    xTalkSubG.nodes[qcq].get('frequency', False)
                    and conflictQCQDict[qcq] == 'gray'
                ):
                    conflictQCQDict[qcq] = 'green'

            conflictQCQPercents.append(
                len([qcq for qcq in conflictQCQDict if conflictQCQDict[qcq] == 'red'])
                / len(
                    [
                        qcq
                        for qcq in conflictQCQDict
                        if conflictQCQDict[qcq] == 'red'
                        or conflictQCQDict[qcq] == 'green'
                    ]
                )
            )
            print('conflict percent', conflictQCQPercents[-1])

            pos = twoQ_gen_pos(chip, xTalkSubG)
            nx.draw_networkx_nodes(
                xTalkSubG,
                pos,
                nodelist=xTalkSubG.nodes,
                node_color=list(conflictQCQDict.values()),
                cmap=plt.cm.Reds_r,
            )
            nx.draw_networkx_edges(
                xTalkSubG,
                pos,
                edgelist=xTalkSubG.edges,
                edge_color=list(conflictQCQEdgeDict.values()),
                edge_cmap=plt.cm.Reds_r,
            )
            plt.axis('off')
            plt.savefig(
                str(level) + ' ' + str(epoch) + str(W) + str(H) + 'twoq conflict.pdf',
                dpi=300,
            )
            plt.close()

            reOptimizeQCQs = newreOptimizeQCQs
            emptyQCQ = dict(
                [
                    (
                        qcq,
                        nx.shortest_path_length(
                            nx.grid_2d_graph(H, W), qcq[0], (H // 2, W // 2)
                        )
                        + nx.shortest_path_length(
                            nx.grid_2d_graph(H, W), qcq[1], (H // 2, W // 2)
                        ),
                    )
                    for qcq in xTalkSubG.nodes()
                    if not (xTalkSubG.nodes[qcq].get('frequency', False))
                ]
            )

            if len(emptyQCQ) > 0:
                centerConflictQCQ = list(sorted(emptyQCQ.items(), key=lambda x: x[1]))[
                    0
                ][0]
                # centerConflictQCQ = random.choices(list(emptyQCQ.keys()), weights=[1 / (distance + 1e-5) for distance in emptyQCQ.values()], k=1)[0]
            elif len(reOptimizeQCQs) > 0:
                centerConflictQCQ = random.choices(
                    list(reOptimizeQCQs.keys()),
                    weights=[
                        1 / (distance + 1e-5) for distance in reOptimizeQCQs.values()
                    ],
                    k=1,
                )[0]
            elif conflictQCQPercents[-1] == 0:
                break
            epoch += 1

        pos = gen_pos(chip)
        intList = []
        intDict = dict()
        for qcq in chip.edges:
            if qcq in xTalkSubG.nodes:
                intList.append(xTalkSubG.nodes[qcq]['frequency'])
                intDict[qcq] = xTalkSubG.nodes[qcq]['frequency']
            else:
                intList.append(4.0)
                intDict[qcq] = 4.0
        plt.figure(figsize=(8, 16))
        nx.draw_networkx_edges(
            chip,
            pos,
            edgelist=chip.edges,
            edge_color=intList,
            edge_cmap=plt.cm.Reds_r,
            width=8,
        )
        nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, cmap=plt.cm.Reds_r)
        nx.draw_networkx_edge_labels(
            chip, pos, intDict, font_size=10, font_color='black'
        )
        plt.axis('off')
        plt.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=3.75, vmax=4.5),
                cmap=plt.cm.Reds_r,
            )
        )
        plt.savefig(
            str(level) + ' ' + str(epoch) + str(W) + str(H) + 'int freq.pdf', dpi=300
        )
        plt.close()
        xtalkGs.append(xTalkSubG)

    return xtalkGs


def data_split(x, y, percent):
    return


def loss_function(a, frequencys, RBs, chip, gates, xtalkG=None):
    outputs = []
    if xtalkG == None:
        for frequency in frequencys:
            outputs.append(err_model(frequency, chip, gates, a))
    else:
        for frequency in frequencys:
            idleFreq = frequency[0]
            intFreq = frequency[1]
            for freq in idleFreq:
                chip.nodes[qubit]['frequency'] = freq
            outputs.append(twoQ_err_model(intFreq, chip, xtalkG, gates, a))
    return np.sum(np.abs(np.array(outputs - RBs)))


def update_params(a, grads, m, v, t):
    # 定义超参数
    lr = 0.01  # 学习率
    beta1 = 0.9  # 一阶矩估计的衰减系数
    beta2 = 0.999  # 二阶矩估计的衰减系数
    epsilon = 1e-8  # 防止除零错误的小常数
    # 计算权重和偏置的一阶和二阶矩估计
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads**2)
    # 对一阶和二阶矩估计进行偏差修正
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    # 根据修正后的一阶和二阶矩估计来更新权重和偏置
    a = a - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return a


def read_data(f):
    with open(f, 'r') as fp:
        data = fp.read()
        data = data.split('\n')
        if '' in data:
            data.remove('')
    f = []
    for d in data:
        f.append(float(d))
    return f


if __name__ == '__main__':
    # sns.distplot(np.random.exponential(scale=0.01, size=1000) + 0.008, hist=False)

    # plt.show()

    # f = np.linspace(-0.1, 0.1, 100)
    # plt.plot(f, lorentzain(0, f, 0.5e-2, 0.5e-2))
    # plt.show()
    with open(
        r"./chipdata/qubit_data.json",
        "r",
        encoding="utf-8",
    ) as file:
        content = file.read()
        chip_data_dic = json.loads(content)
        file.close()

    with open(
        r"./chipdata/xy_crosstalk_sim.json",
        "r",
        encoding="utf-8",
    ) as file:
        content = file.read()
        xy_crosstalk_sim_dic = json.loads(content)
        file.close()

    H = 12
    W = 6

    # H = 2
    # W = 2

    chip = nx.grid_2d_graph(H, W)
    unused_nodes = []
    anharm_list = []
    for qubit in chip:
        qubit_name = f'q{qubit[0]*W+qubit[1]+1}'
        chip.nodes[qubit]['name'] = qubit_name
        if qubit_name in chip_data_dic:
            chip.nodes[qubit]['available'] = True
        else:
            # chip.nodes[qubit]['available'] = False
            unused_nodes.append(qubit)
            continue
        allow_freq = chip_data_dic[qubit_name]['allow_freq']
        # chip.nodes[qubit]['allow_freq'] = np.arange(min(allow_freq), max(allow_freq), 1)
        chip.nodes[qubit]['allow_freq'] = allow_freq
        chip.nodes[qubit]['isolated_error'] = chip_data_dic[qubit_name]['isolated_error']
        # chip.nodes[qubit]['bad_freq_range'] = chip_data_dic[qubit_name][
        #     'bad_freq_range'
        # ]
        chip.nodes[qubit]['ac_spectrum'] = chip_data_dic[qubit_name]['ac_spectrum']
        # chip.nodes[qubit]['sweet point'] = chip.nodes[qubit]['ac_spectrum'][0]
        chip.nodes[qubit]['t1_spectrum'] = chip_data_dic[qubit_name]['t1_spectrum']
        chip.nodes[qubit]['anharm'] = round(chip_data_dic[qubit_name]['anharm'])
        chip.nodes[qubit]['t_sq'] = 20
        chip.nodes[qubit]['xy_crosstalk_coef'] = chip_data_dic[qubit_name]['xy_crosstalk_coef']

        anharm_list.append(round(chip_data_dic[qubit_name]['anharm']))
    chip.remove_nodes_from(unused_nodes)
    anharm_list = sorted(list(set(anharm_list)), reverse=True)

    error_arr = [xy_crosstalk_sim_dic['error_arr'][xy_crosstalk_sim_dic['alpha_list'].index(anharm)] for anharm in anharm_list]
    xy_crosstalk_sim_dic['alpha_list'] = anharm_list
    xy_crosstalk_sim_dic['error_arr'] = error_arr

    for qubit in chip:
        error_arr1 = xy_crosstalk_sim_dic['error_arr'][anharm_list.index(chip.nodes[qubit]['anharm'])]
        f = interp2d(xy_crosstalk_sim_dic['detune_list'], xy_crosstalk_sim_dic['mu_list'], error_arr1, kind='linear')
        chip.nodes[qubit]['xy_crosstalk_f'] = f


    # 每个error model的系数，即训练参数
    arb = [2e-4, 1e-7, 1, 0.3, 10, 1e-2, 0.5, 10]
    # arb = [1e-2, 1e-2, 25, 1e-2, 1e-2, 2e-2]
    # axeb = [1e-2, 1e-2, 10, 1e-3, 1e-2, 1e-3, 1e-2, 1e-3, 1e-2, 1e-3, 1e-2]

    # isolate SQRB = False
    # gates = []
    # idleFreqs = np.zeros_like()
    # idleSQRBs = np.zeros_like()
    # idleFreqsTrain, idleSQRBsTrain, idleFreqsTest, idleSQRBsTest = data_split(idleFreqs, idleSQRBs, 0.6)
    # batchSize = 128
    # nBatches = int(np.ceil(idleFreqsTrain / batchSize))
    # epoches = 100
    # m = 0
    # v = 0
    # for epoch in epoches:
    #     for nbatch in range(nBatches):
    #         idleFreqsTrainBatch = idleFreqsTrain[nbatch * batchSize : (nbatch + 1) * batchSize]
    #         idleSQRBsTrainBatch = idleSQRBsTrain[nbatch * batchSize : (nbatch + 1) * batchSize]
    #         grads = nd.Gradient(loss_function, arb, step=1, order=1, args=(idleFreqsTrainBatch, idleSQRBsTrainBatch, chip, gates))
    #         t = epoch * len(nBatches) + nbatch + 1
    #         arb = update_params(arb, grads, m, v, t)

    # # isolate CZXEB = False
    # maxParallelCZs = max_Algsubgraph(chip)
    # xtalkG = xtalk_G(chip)
    # for level in range(len(maxParallelCZs)):
    #     xTalkSubG = deepcopy(xtalkG)
    #     xTalkSubG.remove_nodes_from(set(xtalkG.nodes).difference(set(maxParallelCZs[level])))

    # gates = []
    # intFreqs = np.zeros_like()
    # CZXEBs = np.zeros_like()
    # intFreqsTrain, CZXEBsTrain, intFreqTest, CZXEBsTest = data_split(intFreqs, CZXEBs, 0.6)
    # batchSize = 128
    # nBatches = int(np.ceil(intFreqsTrain / batchSize))
    # epoches = 100
    # m = 0
    # v = 0
    # for epoch in epoches:
    #     for nbatch in range(nBatches):
    #         intFreqsTrainBatch = intFreqsTrain[nbatch * batchSize : (nbatch + 1) * batchSize]
    #         intCZXEBsTrainBatch = CZXEBsTrain[nbatch * batchSize : (nbatch + 1) * batchSize]
    #         grads = nd.Gradient(loss_function, axeb, step=1, order=1, args=(intFreqsTrainBatch, intCZXEBsTrainBatch, chip, gates, xTalkSubG))
    #         t = epoch * len(nBatches) + nbatch + 1
    #         axeb = update_params(axeb, grads, m, v, t)

    chip, conflictNodeDict, conflictEdge, twoQForbiddenCoupler = sigq_alloc(
        chip, arb, s=2, use_rb_spectrum=True
    )
    for qubit in chip.nodes:
        print(
            qubit,
            chip.nodes[qubit]['frequency'],
            chip.nodes[qubit]['frequency'] - chip.nodes[qubit]['sweet point'],
        )
    # xtalkGs = twoq_alloc(chip, conflictNodeDict, conflictEdge, twoQForbiddenCoupler, axeb)
