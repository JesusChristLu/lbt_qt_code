import numpy as np
import networkx as nx
import scipy
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.special import erf
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import random
import z3
# from sko.PSO import PSO

BadFreqThreshold = 0.001
NThreshold = 0.1
NNThreshold = 0.05
TWOQThreshold = 0.7
IntThreshold = 0.05

H = 6
W = 12  

def gen_pos(chip):
    wStep = 1
    hStep = 1
    pos = dict()
    for qubit in chip:
        pos[qubit] = [qubit[0] * wStep, qubit[1] * hStep]
    return pos

def twoQ_gen_pos(chip, xtalkG):
    bitPos = gen_pos(chip)
    for bit in bitPos:
        bitPos[bit][0] *= 2
        bitPos[bit][1] *= 2
    pos = dict()
    for coupler in xtalkG:
        pos[coupler] = [(bitPos[coupler[0]][0] + bitPos[coupler[1]][0]) / 2,
                        (bitPos[coupler[0]][1] + bitPos[coupler[1]][1]) / 2]
    return pos

def checkcoli(chip, a):
    centerNode = (6 // 2, 12 // 2)
    reOptimizeNodes = dict()
    conflictEdge = []
    for qubit in chip.nodes:
        if chip.nodes[qubit].get('frequency', False) and chip.nodes[qubit]['available']:
            badFreqCost = singq_T1_err(a[0], chip.nodes[qubit]['sing tq'], chip.nodes[qubit]['frequency'], chip.nodes[qubit]['sweet point'], chip.nodes[qubit]['T1 spectra'])
            if badFreqCost / (a[0] * chip.nodes[qubit]['sing tq']) > 0.0001:
                if not(qubit in reOptimizeNodes):
                    reOptimizeNodes[qubit] = nx.shortest_path_length(chip, centerNode, qubit)
                    print(qubit, 'bad T1 freq', badFreqCost)
            dephasingCost = singq_T2_err(a[1], chip.nodes[qubit]['sing tq'], chip.nodes[qubit]['frequency'], chip.nodes[qubit]['sweet point'], chip.nodes[qubit]['df/dphi'])
            if dephasingCost / (a[1] * chip.nodes[qubit]['sing tq']) > 4.5:
                if not(qubit in reOptimizeNodes):
                    reOptimizeNodes[qubit] = nx.shortest_path_length(chip, centerNode, qubit)
                    print(qubit, 'bad T2 freq', dephasingCost)
            for neighbor in chip[qubit]:
                if not((qubit, neighbor) in conflictEdge) and not((neighbor, qubit) in conflictEdge) and chip.nodes[neighbor].get('frequency', False):
                    neighborCost1 = np.abs(chip.nodes[qubit]['frequency'] - chip.nodes[neighbor]['frequency'])
                    neighborCost2 = np.abs(chip.nodes[qubit]['frequency'] + chip.nodes[qubit]['anharm'] - chip.nodes[neighbor]['frequency'])
                    neighborCost3 = np.abs(chip.nodes[qubit]['frequency'] - (chip.nodes[neighbor]['frequency'] + chip.nodes[neighbor]['anharm']))
                    if chip.nodes[qubit]['available'] and chip.nodes[neighbor]['available']:
                        neighborCost4 = twoq_pulse_distort_err(chip.nodes[neighbor]['frequency'], chip.nodes[qubit]['frequency'], a[6])
                    else:
                        neighborCost4 = 0
                    if neighborCost1 < 0.1 or neighborCost2 < 0.1 or neighborCost3 < 0.1 or neighborCost4 / a[6] > (0.7 ** 2):
                        if neighborCost4 / (a[6]) > 0.7 ** 2:
                            print(qubit, neighbor, 'distort', neighborCost4)
                        if neighborCost1 < 0.1 or neighborCost2 < 0.1 or neighborCost3 < 0.1:
                            print(qubit, neighbor,'n', neighborCost1, 'na', neighborCost2, neighborCost3)
                        conflictEdge.append((qubit, neighbor))
                        if not(qubit in reOptimizeNodes):
                            reOptimizeNodes[qubit] = nx.shortest_path_length(chip, centerNode, qubit)
                        if not(neighbor in reOptimizeNodes):
                            reOptimizeNodes[neighbor] = nx.shortest_path_length(chip, centerNode, neighbor)

                for nextNeighbor in chip[neighbor]:
                    if not(nextNeighbor == qubit) and not((qubit, nextNeighbor) in conflictEdge) and not((nextNeighbor, qubit) in conflictEdge) and chip.nodes[nextNeighbor].get('frequency'):
                        neighborCost5 = np.abs(chip.nodes[qubit]['frequency'] - chip.nodes[nextNeighbor]['frequency'])
                        neighborCost6 = np.abs(chip.nodes[qubit]['frequency'] + chip.nodes[qubit]['anharm'] - chip.nodes[nextNeighbor]['frequency'])
                        neighborCost7 = np.abs(chip.nodes[qubit]['frequency'] - (chip.nodes[nextNeighbor]['frequency'] + chip.nodes[nextNeighbor]['anharm']))
                        if (neighborCost5 < 0.04 or neighborCost6 < 0.04 or neighborCost7 < 0.04):
                            print(qubit, nextNeighbor, 'nn', neighborCost5, 'nna', neighborCost6, neighborCost7)
                            conflictEdge.append((qubit, nextNeighbor))
                            if not(qubit in reOptimizeNodes):
                                reOptimizeNodes[qubit] = nx.shortest_path_length(chip, centerNode, qubit)
                            if not(nextNeighbor in reOptimizeNodes):
                                reOptimizeNodes[nextNeighbor] = nx.shortest_path_length(chip, centerNode, nextNeighbor)

    return reOptimizeNodes, conflictEdge
    
def twoQ_checkcoli(chip, xtalkG, a):
    distance = dict()
    reOptimizeQCQs = dict()
    conflictEdge = []
    for qcq in xtalkG:
        distance[qcq] = nx.shortest_path_length(nx.grid_2d_graph(H, W), qcq[0], (3, 6)) + \
            nx.shortest_path_length(nx.grid_2d_graph(H, W), qcq[1], (3, 6))
    centertwoQ = sorted(distance.items(), key=lambda x : x[1])[0][0]
    for qcq in xtalkG.nodes:
        if xtalkG.nodes[qcq].get('frequency', False):
            if sum(qcq[0]) % 2:
                T1Cost1 = twoq_T1_err(xtalkG.nodes[qcq]['frequency'], chip.nodes[qcq[0]]['frequency'], chip.nodes[qcq[0]]['sweet point'], 
                                    a[11], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[0]]['T1 spectra'])
                T1Cost2 = twoq_T1_err(xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[1]]['anharm'], chip.nodes[qcq[1]]['frequency'], chip.nodes[qcq[1]]['sweet point'],
                                    a[11], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[1]]['T1 spectra'])
                T2Cost1 = twoq_T2_err(xtalkG.nodes[qcq]['frequency'], chip.nodes[qcq[0]]['frequency'], chip.nodes[qcq[0]]['sweet point'], 
                                    a[12], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[0]]['df/dphi'])
                T2Cost2 = twoq_T2_err(xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[1]]['anharm'], chip.nodes[qcq[1]]['frequency'], chip.nodes[qcq[1]]['sweet point'],
                                    a[12], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[1]]['df/dphi'])
            else:
                T1Cost1 = twoq_T1_err(xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[0]]['anharm'], chip.nodes[qcq[0]]['frequency'], chip.nodes[qcq[0]]['sweet point'], 
                                    a[11], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[0]]['T1 spectra'])
                T1Cost2 = twoq_T1_err(xtalkG.nodes[qcq]['frequency'], chip.nodes[qcq[1]]['frequency'], chip.nodes[qcq[1]]['sweet point'],
                                    a[11], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[1]]['T1 spectra'])
                T2Cost1 = twoq_T2_err(xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[0]]['anharm'], chip.nodes[qcq[0]]['frequency'], chip.nodes[qcq[0]]['sweet point'], 
                                    a[12], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[0]]['df/dphi'])
                T2Cost2 = twoq_T2_err(xtalkG.nodes[qcq]['frequency'], chip.nodes[qcq[1]]['frequency'], chip.nodes[qcq[1]]['sweet point'],
                                    a[12], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[1]]['df/dphi'])
            if T1Cost1 / (a[11] * xtalkG.nodes[qcq]['two tq']) > 0.0001 or T1Cost2 / (a[11] * xtalkG.nodes[qcq]['two tq']) > 0.0001:
                if not(qcq in reOptimizeQCQs):
                    if nx.has_path(xtalkG, centertwoQ, qcq):
                        reOptimizeQCQs[qcq] = nx.shortest_path_length(xtalkG, centertwoQ, qcq)
                    else:
                        reOptimizeQCQs[qcq] = 100000
                    print(qcq, 'bad T1 freq', T1Cost1, T1Cost2)
            if T2Cost1 / (a[12] * xtalkG.nodes[qcq]['two tq']) > 4.5 or T2Cost2 / (a[12] * xtalkG.nodes[qcq]['two tq']) > 4.5:
                if not(qcq in reOptimizeQCQs):
                    if nx.has_path(xtalkG, centertwoQ, qcq):
                        reOptimizeQCQs[qcq] = nx.shortest_path_length(xtalkG, centertwoQ, qcq)
                    else:
                        reOptimizeQCQs[qcq] = 100000
                    print(qcq, 'bad T2 freq', T2Cost1, T2Cost2)
            for q in qcq:
                for neighbor in chip[q]:
                    if neighbor in qcq:
                        continue
                    neighborCost1 = twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'], chip.nodes[q]['frequency']], [chip.nodes[neighbor]['frequency'], chip.nodes[neighbor]['frequency']], 
                                                   a[13], a[14], xtalkG.nodes[qcq]['two tq'])
                    neighborCost2 = twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'], chip.nodes[q]['frequency']], 
                                                   [chip.nodes[neighbor]['frequency'] + chip.nodes[neighbor]['anharm'], chip.nodes[neighbor]['frequency'] + chip.nodes[neighbor]['anharm']], 
                                                   a[15], a[16], xtalkG.nodes[qcq]['two tq'])
                    if neighborCost1 / (a[13] * xtalkG.nodes[qcq]['two tq']) > 0.4 or neighborCost2 / (a[13] * xtalkG.nodes[qcq]['two tq']) > 0.4:
                        conflictEdge.append((qcq, neighbor))
                        if not(qcq in reOptimizeQCQs):
                            if nx.has_path(xtalkG, centertwoQ, qcq):
                                reOptimizeQCQs[qcq] = nx.shortest_path_length(xtalkG, centertwoQ, qcq)
                            else:
                                reOptimizeQCQs[qcq] = 100000
                        print(qcq, 'idle conflict', neighborCost1, neighborCost2, neighbor)
            for neighbor in xtalkG[qcq]:
                if not((qcq, neighbor) in conflictEdge) and not((neighbor, qcq) in conflictEdge) and xtalkG.nodes[neighbor].get('frequency', False):
                    for q0 in qcq:
                        for q1 in neighbor:
                            if (q0, q1) in chip.edges:
                                neighborCost3 = twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'], chip.nodes[q0]['frequency']], [xtalkG.nodes[qcq]['frequency'], chip.nodes[q1]['frequency']], 
                                                               a[17], a[18], xtalkG.nodes[qcq]['two tq'])
                                if sum(q0) % 2:
                                    neighborCost4 = twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'] + chip.nodes[q0]['anharm'], chip.nodes[q0]['frequency'] + chip.nodes[q0]['anharm']], 
                                                                   [xtalkG.nodes[qcq]['frequency'], chip.nodes[q1]['frequency']], a[19], a[20], xtalkG.nodes[qcq]['two tq'])
                                    neighborCost5 = twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'], chip.nodes[q0]['frequency']], 
                                                                   [xtalkG.nodes[qcq]['frequency'] - chip.nodes[q1]['anharm'], chip.nodes[q1]['frequency'] - chip.nodes[q1]['anharm']], 
                                                                   a[19], a[20], xtalkG.nodes[qcq]['two tq'])
                                else:
                                    neighborCost4 = twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'] - chip.nodes[q0]['anharm'], chip.nodes[q0]['frequency'] - chip.nodes[q0]['anharm']], 
                                                                   [xtalkG.nodes[qcq]['frequency'], chip.nodes[q1]['frequency']], a[19], a[20], xtalkG.nodes[qcq]['two tq'])
                                    neighborCost5 = twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'], chip.nodes[q0]['frequency']], 
                                                                   [xtalkG.nodes[qcq]['frequency'] + chip.nodes[q1]['anharm'], chip.nodes[q1]['frequency'] + chip.nodes[q1]['anharm']], 
                                                                   a[19], a[20], xtalkG.nodes[qcq]['two tq'])
                    if neighborCost3 / (a[17] * xtalkG.nodes[qcq]['two tq']) > 0.4 or \
                        neighborCost4 / (a[19] * xtalkG.nodes[qcq]['two tq']) > 0.4 or neighborCost5 / (a[19] * xtalkG.nodes[qcq]['two tq'])> 0.4:
                        conflictEdge.append((qcq, neighbor))
                        if not(qcq in reOptimizeQCQs):
                            if nx.has_path(xtalkG, qcq, centertwoQ):
                                reOptimizeQCQs[qcq] = nx.shortest_path_length(xtalkG, qcq, centertwoQ)
                            else:
                                reOptimizeQCQs[qcq] = 100000
                        if not(neighbor in reOptimizeQCQs):
                            if nx.has_path(xtalkG, neighbor, centertwoQ):
                                reOptimizeQCQs[neighbor] = nx.shortest_path_length(xtalkG, neighbor, centertwoQ)
                            else:
                                reOptimizeQCQs[neighbor] = 100000
                        print(qcq, 'int conflict', neighborCost3, neighborCost4, neighborCost5, neighbor)
    return reOptimizeQCQs, conflictEdge

def T1_spectra(fMax, step):
    badFreqNum = np.random.randint(5)
    fList = np.linspace(3.75, fMax, step)
    gamma = 0.001 + (0.1 - 0.001) * np.random.random()
    T1 = np.random.normal(np.random.randint(20, 50), 5, step)
    for _ in range(badFreqNum):
        a = np.random.randint(4, 5)
        badFreq = (3.75 + (fMax - 3.75) * np.random.random())
        T1 -= lorentzain(fList, badFreq, a, gamma)
    for T in range(len(T1)):
        T1[T] = np.max([1, T1[T]])
    return 1e-3 / T1

def f_phi_spectra(fMax, phi):
    d = 0
    return fMax * np.sqrt(np.abs(np.cos(np.pi * phi)) * np.sqrt(1 + d ** 2 * np.tan(np.pi * phi) ** 2))

def phi2f(phi, fMax, step):
    phiList = np.linspace(0, 0.5, step)
    fList = f_phi_spectra(fMax, phiList)
    func_interp = interp1d(phiList, fList, kind='cubic')
    if isinstance(phi, (int, float)):
        return float(func_interp(phi))
    else:
        return func_interp(phi)
    
def f2phi(f, fMax, step):
    phiList = np.linspace(0, 0.5, step)
    fList = f_phi_spectra(fMax, phiList)
    func_interp = interp1d(fList, phiList, kind='cubic')
    if isinstance(f, (int, float)):
        return float(func_interp(f))
    else:
        return func_interp(f)

def T2_spectra(fMax, step):
    fList = np.linspace(3.75, fMax, step + 1)
    phiList = f2phi(fList, fMax, step)
    df_dphi = np.abs(np.diff(fList) / np.diff(phiList))
    return df_dphi

def lorentzain(fi, fj, a, gamma):
    # return (a / np.pi) * (gamma / ((fi - fj) ** 2 + (gamma) ** 2))
    # return (a / (gamma * np.sqrt(2 * np.pi))) * np.exp(-(fi - fj) ** 2 / (2 * gamma ** 2))
    
    if isinstance(fi - fj, (float, int)):
        return a * max(0, gamma - np.abs(fi - fj))
    else:
        res = []
        for i in gamma - np.abs(fi - fj):
            res.append(max(0, a * i))
        return res

def singq_T1_err(a, tq, f, fMax, T1):
    step = 1000
    ft = np.linspace(3.75, fMax, step)
    func_interp = interp1d(ft, T1, kind='cubic')
    return a * tq * func_interp(f)

def singq_T2_err(a, tq, f, fMax, df_dphi):
    step = 1000
    ft = np.linspace(3.75, fMax, step)
    func_interp = interp1d(ft, df_dphi, kind='cubic')
    return a * tq * func_interp(f)

def singq_xtalk_err(a, gamma, tq, fi, fj):
    return tq * lorentzain(fi, fj, a, gamma)

def twoq_T1_err(fWork, fidle, fMax, a, tq, T1Spectra):
    step = 1000
    ft = twoq_pulse(fWork, fidle, tq, step)
    fList = np.linspace(3.75, fMax, step)
    func_interp = interp1d(fList, T1Spectra, kind='cubic')
    TtList = func_interp(ft)
    return a * np.sum(TtList) * (tq / step)

def twoq_T2_err(fWork, fidle, fMax, a, tq, df_dphiList):
    step = 1000
    ft = twoq_pulse(fWork, fidle, tq, step)
    fList = np.linspace(3.75, fMax, step)
    func_interp = interp1d(fList, df_dphiList, kind='cubic')
    TtList = func_interp(ft)
    return a * np.sum(TtList) * (tq / step)

def twoq_xtalk_err(fi, fj, a, gamma, tq):
    step = 1000
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
        sigma = [0.8]
        flattop_start = 3 * sigma[0]
        flattop_end = pulseLen - 3 * sigma[0]
        freqList = (freqWork - freqMax) * 1 / 2 * (erf((tList - flattop_start) / (np.sqrt(2) * sigma[0])) - \
                                    erf((tList - flattop_end) / (np.sqrt(2) * sigma[0]))) + freqMax
        return freqList

def phys_cost_function(frequencys, inducedChip, targets, a):
    chip = deepcopy(inducedChip)
    for target in targets:
        chip.nodes[target]['frequency'] = frequencys[targets.index(target)]
    cost = 0
    for target in targets:
        if chip.nodes[target]['available']:
            cost += singq_T1_err(a[0], chip.nodes[target]['sing tq'], chip.nodes[target]['frequency'], chip.nodes[target]['sweet point'], chip.nodes[target]['T1 spectra'])
            cost += singq_T2_err(a[1], chip.nodes[target]['sing tq'], chip.nodes[target]['frequency'], chip.nodes[target]['sweet point'], chip.nodes[target]['df/dphi'])
        for neighbor in chip[target]:
            if chip.nodes[neighbor].get('frequency', False):
                cost += singq_xtalk_err(a[2], a[3], chip.nodes[target]['sing tq'], chip.nodes[target]['frequency'], chip.nodes[neighbor]['frequency'])
                cost += singq_xtalk_err(a[4], a[5], chip.nodes[target]['sing tq'], chip.nodes[target]['frequency'] + chip.nodes[target]['anharm'], chip.nodes[neighbor]['frequency'])
                cost += singq_xtalk_err(a[4], a[5], chip.nodes[target]['sing tq'], chip.nodes[target]['frequency'], chip.nodes[neighbor]['frequency'] + chip.nodes[neighbor]['anharm'])
                if chip.nodes[target]['available'] and chip.nodes[neighbor]['available']:
                    cost += twoq_pulse_distort_err(chip.nodes[target]['frequency'], chip.nodes[neighbor]['frequency'], a[6])
            for nextNeighbor in chip[neighbor]:
                if not(nextNeighbor == target) and chip.nodes[nextNeighbor].get('frequency', False):
                    cost += singq_xtalk_err(a[7], a[8], chip.nodes[target]['sing tq'], chip.nodes[target]['frequency'], chip.nodes[nextNeighbor]['frequency'])
                    cost += singq_xtalk_err(a[9], a[10], chip.nodes[target]['sing tq'], chip.nodes[target]['frequency'] + chip.nodes[target]['anharm'], chip.nodes[nextNeighbor]['frequency'])
                    cost += singq_xtalk_err(a[9], a[10], chip.nodes[target]['sing tq'], chip.nodes[target]['frequency'], chip.nodes[nextNeighbor]['frequency'] + chip.nodes[nextNeighbor]['anharm'])
    return cost

def phys_twoq_cost_function(frequencys, chip, xTalkG, reOptimizeQCQs, a):
    xtalkG = deepcopy(xTalkG)
    for qcq in reOptimizeQCQs:
        xtalkG.nodes[qcq]['frequency'] = frequencys[reOptimizeQCQs.index(qcq)]
    cost = 0
    for qcq in reOptimizeQCQs:
        if sum(qcq[0]) % 2:
            cost += twoq_T1_err(xtalkG.nodes[qcq]['frequency'], chip.nodes[qcq[0]]['frequency'], chip.nodes[qcq[0]]['sweet point'], 
                                a[11], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[0]]['T1 spectra'])
            cost += twoq_T1_err(xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[1]]['anharm'], chip.nodes[qcq[1]]['frequency'], chip.nodes[qcq[1]]['sweet point'],
                                a[11], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[1]]['T1 spectra'])
            cost += twoq_T2_err(xtalkG.nodes[qcq]['frequency'], chip.nodes[qcq[0]]['frequency'], chip.nodes[qcq[0]]['sweet point'], 
                                a[12], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[0]]['df/dphi'])
            cost += twoq_T2_err(xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[1]]['anharm'], chip.nodes[qcq[1]]['frequency'], chip.nodes[qcq[1]]['sweet point'],
                                a[12], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[1]]['df/dphi'])
        else:
            cost += twoq_T1_err(xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[0]]['anharm'], chip.nodes[qcq[0]]['frequency'], chip.nodes[qcq[0]]['sweet point'], 
                                a[11], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[0]]['T1 spectra'])
            cost += twoq_T1_err(xtalkG.nodes[qcq]['frequency'], chip.nodes[qcq[1]]['frequency'], chip.nodes[qcq[1]]['sweet point'],
                                a[11], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[1]]['T1 spectra'])
            cost += twoq_T2_err(xtalkG.nodes[qcq]['frequency'] - chip.nodes[qcq[0]]['anharm'], chip.nodes[qcq[0]]['frequency'], chip.nodes[qcq[0]]['sweet point'], 
                                a[12], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[0]]['df/dphi'])
            cost += twoq_T2_err(xtalkG.nodes[qcq]['frequency'], chip.nodes[qcq[1]]['frequency'], chip.nodes[qcq[1]]['sweet point'],
                                a[12], xtalkG.nodes[qcq]['two tq'], chip.nodes[qcq[1]]['df/dphi'])
        for q in qcq:
            for neighbor in chip[q]:
                cost += twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'], chip.nodes[q]['frequency']], [chip.nodes[neighbor]['frequency'], chip.nodes[neighbor]['frequency']], 
                                       a[13], a[14], xtalkG.nodes[qcq]['two tq'])
                cost += twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'], chip.nodes[q]['frequency']], 
                                       [chip.nodes[neighbor]['frequency'] + chip.nodes[neighbor]['anharm'], chip.nodes[neighbor]['frequency'] + chip.nodes[neighbor]['anharm']], 
                                       a[15], a[16], xtalkG.nodes[qcq]['two tq'])
        for neighbor in xtalkG[qcq]:
            if xtalkG.nodes[neighbor].get('frequency', False):
                for q0 in qcq:
                    for q1 in neighbor:
                        if (q0, q1) in chip.edges:
                            cost += twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'], chip.nodes[q0]['frequency']], [xtalkG.nodes[qcq]['frequency'], chip.nodes[q1]['frequency']], a[17], a[18], xtalkG.nodes[qcq]['two tq'])
                            if sum(q0) % 2:
                                cost += twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'] + chip.nodes[q0]['anharm'], chip.nodes[q0]['frequency'] + chip.nodes[q0]['anharm']], 
                                                       [xtalkG.nodes[qcq]['frequency'], chip.nodes[q1]['frequency']], a[19], a[20], xtalkG.nodes[qcq]['two tq'])
                                cost += twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'], chip.nodes[q0]['frequency']], 
                                                       [xtalkG.nodes[qcq]['frequency'] - chip.nodes[q1]['anharm'], chip.nodes[q1]['frequency'] - chip.nodes[q1]['anharm']], 
                                                       a[19], a[20], xtalkG.nodes[qcq]['two tq'])
                            else:
                                cost += twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'] - chip.nodes[q0]['anharm'], chip.nodes[q0]['frequency'] - chip.nodes[q0]['anharm']], 
                                                       [xtalkG.nodes[qcq]['frequency'], chip.nodes[q1]['frequency']], a[19], a[20], xtalkG.nodes[qcq]['two tq'])
                                cost += twoq_xtalk_err([xtalkG.nodes[qcq]['frequency'], chip.nodes[q0]['frequency']], 
                                                       [xtalkG.nodes[qcq]['frequency'] + chip.nodes[q1]['anharm'], chip.nodes[q1]['frequency'] + chip.nodes[q1]['anharm']], 
                                                       a[19], a[20], xtalkG.nodes[qcq]['two tq'])
    return cost

def xtalk_G(chip):
    xtalkG = nx.Graph()
    for coupler1 in chip.edges:
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
            if 1 in distance and not(0 in distance):
                xtalkG.add_edge(coupler1, coupler2)
                xtalkG.nodes[coupler1]['two tq'] = 60
                xtalkG.nodes[coupler2]['two tq'] = 60
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
    maxParallelCZs = [[((0, 0), (0, 1))], [((0, 1), (0, 2))], [((0, 0), (1, 0))], [((1, 0), (2, 0))]]
    nb = [(1, 2), (1, 2), (2, 1), (2, 1)]
    for maxParallelCZ in maxParallelCZs:
        i, j = nb[maxParallelCZs.index(maxParallelCZ)][0], nb[maxParallelCZs.index(maxParallelCZ)][1]
        currentCZ = ((maxParallelCZ[0][0][0], maxParallelCZ[0][0][1]), (maxParallelCZ[0][1][0], maxParallelCZ[0][1][1]))
        while True:
            while currentCZ[1][1] + j <= 11:
                currentCZ = ((currentCZ[0][0], currentCZ[0][1] + j), (currentCZ[1][0], currentCZ[1][1] + j))
                if currentCZ in dualChip.nodes:
                    maxParallelCZ.append(currentCZ)
            if currentCZ[1][0] + i > 5:
                break
            currentCZ = ((currentCZ[0][0] + i, maxParallelCZ[0][0][1]), (currentCZ[1][0] + i, maxParallelCZ[0][1][1]))
            if currentCZ in dualChip.nodes:
                maxParallelCZ.append(((currentCZ[0][0], currentCZ[0][1]), (currentCZ[1][0], currentCZ[1][1])))
        if not(maxParallelCZ[0] in dualChip.nodes):
            maxParallelCZ.remove(maxParallelCZ[0])

    return maxParallelCZs

if __name__ == '__main__':
    # a = [0.00, 0.00, 10, 0.04, 10, 0.04, 6, 10, 0.013, 10, 0.013, 
            # 0.1, 0.1, 10, 0.16, 10, 0.16, 10, 0.16, 10, 0.16]
    a = [0.00, 0.00, 100, 0.1, 100, 0.1, 6, 100, 0.04, 100, 0.04, 
            0.1, 0.1, 10, 0.16, 10, 0.16, 10, 0.16, 10, 0.16]

    chip = nx.grid_2d_graph(H, W)

    sweetPointArray = np.array([[4469.05, 3989.78, 4713.768, 4186.155, 4784.413, 4159.51],
                                [3940.33, 4611.593, 4032.624, 4736.225, 4233.228, 4895.001],
                                [4712.046, 4122.505, 4718.096, 4153.639, 4860.811, 4286.59],
                                [4041.003, 4703.134, 4111.178, 4774.921, 4194.057, 4787.613],
                                [4641.077, 4057.33, 4676.5, 4054.422, 4692.44, 4265.386],
                                [4190.207, 4691.042, 4011.833, 4691.668, 4075.402, 4893.379],
                                [4888.528, 4179.015, 4792.039, 4139.709, 4729.44, 4031.202],
                                [4310.526, 4872.242, 4226.452, 4922.29, 4042.817, 4735.554],
                                [4980.339, 4354.487, 4889.19, 4270.34, 4819.38, 4165.983],
                                [4430.961, 4958.593, 4303.285, 4844.546, 4250.114, 4903.312],
                                [4923.431, 4223.7597, 4842.507, 4092.872, 4895.467, 4279.065],
                                [4399.842, 5033.471, 4242.685, 4962.309, 4359.198, 4886.872]])

    sweetPointArray = sweetPointArray.T
    for qubit in chip.nodes():
        # if sum(qubit) % 2:
        #     chip.nodes[qubit]['sweet point'] = round(4.3 + 0.15 * (np.random.random() - 0.5), 3)】
        # else:
        #     chip.nodes[qubit]['sweet point'] = round(4.7 + 0.15 * (np.random.random() - 0.5), 3)

        chip.nodes[qubit]['sweet point'] = round(sweetPointArray[qubit[0], qubit[1]] * 1e-3, 3)
        chip.nodes[qubit]['T1 spectra'] = T1_spectra(chip.nodes[qubit]['sweet point'], 1000)
        chip.nodes[qubit]['df/dphi'] = T2_spectra(chip.nodes[qubit]['sweet point'], 1000)
        chip.nodes[qubit]['anharm'] = -round((200 + 50 * np.random.random()) * 1e-3, 3)
        chip.nodes[qubit]['sing tq'] = 30
        chip.nodes[qubit]['available'] = True

    qubitList = list(chip.nodes)

    epoch = 0

    centerConflictNode = (3, 6)
    conflictNodeDict = dict()
    for qubit in chip.nodes():
        conflictNodeDict[qubit] = 'gray'

    conflictPercents = []

    for _ in range(30):
        reOptimizeNodes = [centerConflictNode]
        for qubit in chip.nodes():
            if conflictNodeDict[centerConflictNode] == 'gray' and not(qubit in reOptimizeNodes) and \
                not(chip.nodes[qubit].get('frequency', False)) and \
                np.abs(qubit[0] - centerConflictNode[0]) <= 1 and np.abs(qubit[1] - centerConflictNode[1]) <= 1:
                reOptimizeNodes.append(qubit)
            elif conflictNodeDict[centerConflictNode] == 'red' and not(qubit in reOptimizeNodes) and \
                np.abs(qubit[0] - centerConflictNode[0]) <= 1 and np.abs(qubit[1] - centerConflictNode[1]) <= 1 and \
                (nx.shortest_path_length(chip, qubit, (3, 6)) >= (nx.shortest_path_length(chip, centerConflictNode, (3, 6)))) and \
                conflictNodeDict[qubit] == 'red':
                reOptimizeNodes.append(qubit)

        reOptimizeNodes = tuple(reOptimizeNodes)
        ini_frequency = [chip.nodes[qubit].get('frequency', chip.nodes[qubit]['sweet point']) for qubit in reOptimizeNodes]

        bounds = []
        for qubit in reOptimizeNodes:
            if chip.nodes[qubit]['available']:
                bounds.append((max(3.75, chip.nodes[qubit]['sweet point'] - 0.4), chip.nodes[qubit]['sweet point']))
            else:
                bounds.append((2, 3))

        res = minimize(phys_cost_function, ini_frequency, args=(chip, reOptimizeNodes, a),
                    method='nelder-mead', bounds=bounds, options={'maxiter' : 200})
        print(res.fun)

        for qubit in reOptimizeNodes:
            chip.nodes[qubit]['frequency'] = round(res.x[reOptimizeNodes.index(qubit)], 3)

        newreOptimizeNodes, conflictEdge = checkcoli(chip, a)

        conflictCount = dict()
        for edge in conflictEdge:
            if edge[0] in conflictCount:   
                conflictCount[edge[0]] += 1
            else:
                conflictCount[edge[0]] = 1
            if edge[1] in conflictCount:   
                conflictCount[edge[1]] += 1
            else:
                conflictCount[edge[1]] = 1

        unAllocPercent = len([qubit for qubit in conflictNodeDict if conflictNodeDict[qubit] == 'gray']) / len(conflictNodeDict)

        for qubit in conflictCount:
            if conflictCount[qubit] == max(list(conflictCount.values())) and unAllocPercent == 0.0 and conflictCount[qubit] >= 4:
                print('all have allocate, choose', qubit, 'which have', max(list(conflictCount.values())), 'conflict edge')
                chip.nodes[qubit]['available'] = False

        print('multi')

        twoQForbiddenCoupler = []
        for coupler in chip.edges:
            if chip.nodes[coupler[0]].get('frequency', False) and chip.nodes[coupler[1]].get('frequency', False) and \
                np.abs(chip.nodes[coupler[0]]['frequency'] - chip.nodes[coupler[1]]['frequency']) > TWOQThreshold:
                twoQForbiddenCoupler.append(coupler)
        for coupler in twoQForbiddenCoupler:
            if coupler in conflictEdge:
                conflictEdge.remove(coupler)

        drawChip = deepcopy(chip)
        drawChip.add_edges_from(conflictEdge)
        drawChip.remove_edges_from(twoQForbiddenCoupler)

        conflictEdgeDict = dict()
        alreadyConflict = []
        for coupler in drawChip.edges:
            if coupler in conflictEdge:
                conflictEdgeDict[coupler] = 'red'
                if coupler[0] in alreadyConflict:
                    if conflictNodeDict[coupler[0]] == 'red':
                        conflictNodeDict[coupler[1]] = 'green'
                        alreadyConflict.append(coupler[1])
                    else:
                        conflictNodeDict[coupler[1]] = 'red'
                        alreadyConflict.append(coupler[1])
                elif coupler[1] in alreadyConflict:
                    if conflictNodeDict[coupler[1]] == 'red':
                        conflictNodeDict[coupler[0]] = 'green'
                        alreadyConflict.append(coupler[0])
                    else:
                        conflictNodeDict[coupler[0]] = 'red'
                        alreadyConflict.append(coupler[0])
                else:
                    if nx.shortest_path_length(chip, coupler[0], (3, 6)) > nx.shortest_path_length(chip, coupler[1], (3, 6)):
                        conflictNodeDict[coupler[0]] = 'red'
                        conflictNodeDict[coupler[1]] = 'green'
                    else:
                        conflictNodeDict[coupler[1]] = 'red'
                        conflictNodeDict[coupler[0]] = 'green'
                    alreadyConflict.append(coupler[0])
                    alreadyConflict.append(coupler[1])
            elif drawChip.nodes[coupler[0]].get('frequency', False) and drawChip.nodes[coupler[1]].get('frequency', False):
                conflictEdgeDict[coupler] = 'green'
                if not(coupler[0] in alreadyConflict):
                    conflictNodeDict[coupler[0]] = 'green'
                if not(coupler[1] in alreadyConflict):
                    conflictNodeDict[coupler[1]] = 'green'
            else:
                conflictEdgeDict[coupler] = 'gray'

        for qubit in drawChip.nodes:
            if not(drawChip.nodes[qubit]['available']):
                conflictNodeDict[qubit] = 'black'
            if qubit in newreOptimizeNodes and (conflictNodeDict[qubit] == 'green' or conflictNodeDict[qubit] == 'gray'):
                isBadFreq = True
                for coupler in conflictEdgeDict:
                    if conflictEdgeDict[coupler] == 'red' and qubit in coupler:
                        isBadFreq = False
                        break
                for coupler in twoQForbiddenCoupler:
                    if qubit in coupler:
                        isBadFreq = False
                        break
                if isBadFreq:
                    conflictNodeDict[qubit] = 'red'
            if drawChip.nodes[qubit].get('frequency', False) and conflictNodeDict[qubit] == 'gray':
                conflictNodeDict[qubit] = 'green'

        conflictPercents.append(len([qubit for qubit in conflictNodeDict if conflictNodeDict[qubit] == 'red' or conflictNodeDict[qubit] == 'black']) / 
                                len([qubit for qubit in conflictNodeDict 
                                    if conflictNodeDict[qubit] == 'red' or  conflictNodeDict[qubit] == 'green' or conflictNodeDict[qubit] == 'black']))
        print(conflictPercents[-1])

        pos = gen_pos(drawChip)
        labelDict = dict([(i, i) for i in drawChip.nodes])
        plt.figure(figsize=(4, 8))
        nx.draw_networkx_labels(drawChip, pos, labelDict, font_size=10, font_color="black")
        nx.draw_networkx_nodes(drawChip, pos, nodelist=drawChip.nodes, node_color=list(conflictNodeDict.values()), cmap=plt.cm.Reds_r)
        nx.draw_networkx_edges(drawChip, pos, edgelist=drawChip.edges, edge_color=list(conflictEdgeDict.values()), edge_cmap=plt.cm.Reds_r)
        plt.axis('off')
        plt.savefig(str(epoch) + str(W) + str(H) + 'chip conflict.pdf', dpi=300)
        plt.close()

        reOptimizeNodes = dict([(qubit, nx.shortest_path_length(chip, qubit, (3, 6))) for qubit in conflictNodeDict if conflictNodeDict[qubit] == 'red'])
        emptyNode = dict([(qubit, nx.shortest_path_length(chip, qubit, (3, 6))) for qubit in chip.nodes() if not(chip.nodes[qubit].get('frequency', False))])
        noavailableNode = dict([(qubit, nx.shortest_path_length(chip, qubit, (3, 6))) for qubit in chip.nodes() if not chip.nodes[qubit]['available']])

        noConflictnoAvailableNode = []
        for qubit in noavailableNode:
            noConflict = True
            for coupler in conflictEdge:
                if qubit in coupler:
                    noConflict = False
                    break
            if noConflict:
                noConflictnoAvailableNode.append(qubit)
        
        for qubit in noConflictnoAvailableNode:
            del noavailableNode[qubit]

        if len(noavailableNode) > 0:
            centerConflictNode = random.choices(list(noavailableNode.keys()), weights=[1 / (distance + 1e-5) for distance in noavailableNode.values()], k=1)[0]
        elif len(emptyNode) > 0:
            centerConflictNode = list(sorted(emptyNode.items(), key=lambda x : x[1]))[0][0]
        elif len(reOptimizeNodes) > 0:
            centerConflictNode = random.choices(list(reOptimizeNodes.keys()), weights=[1 / (distance + 1e-5) for distance in reOptimizeNodes.values()], k=1)[0]
        elif conflictPercents[-1] == 0:
            break
        epoch += 1

    print(conflictPercents)
    plt.plot(range(epoch), conflictPercents)
    plt.xlabel('epoch')
    plt.ylabel('percent')
    plt.savefig('conflict percent.pdf', dpi=300)
    plt.close()

    pos = gen_pos(chip)
    freqList = [chip.nodes[qubit]['frequency'] for qubit in chip.nodes]
    qlow = min(freqList)
    qhigh = max(freqList)
    freqDict = dict([(i, round(chip.nodes[i]['frequency'], 3)) for i in chip.nodes])
    plt.figure(figsize=(4, 8))
    nx.draw_networkx_labels(chip, pos, freqDict, font_size=10, font_color="black")
    nx.draw_networkx_edges(chip, pos, edgelist=chip.edges, edge_cmap=plt.cm.Reds_r)
    nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, node_color=freqList, cmap=plt.cm.Reds_r)
    plt.axis('off')
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=qlow, vmax=qhigh), cmap=plt.cm.Reds_r))
    plt.savefig(str(W) + str(H) + 'chip freq.pdf', dpi=300)
    plt.close()


    # twoq gate 
    xTalkG = xtalk_G(chip)
    pos = twoQ_gen_pos(chip, xTalkG)

    removeQCQs = []
    for qcq in xTalkG.nodes:
        removeQCQ = False
        for q in qcq:
            if chip.nodes[q]['frequency'] < 3.75 or conflictNodeDict[q] == 'red':
                removeQCQ = True
                break
        if removeQCQ:
            removeQCQs.append(qcq)
            continue
        if (qcq in conflictEdge) or (qcq[::-1] in conflictEdge) or \
            (qcq in twoQForbiddenCoupler) or (qcq[::-1] in twoQForbiddenCoupler):
            removeQCQs.append(qcq)
            continue
        if chip.nodes[qcq[0]]['frequency'] > chip.nodes[qcq[1]]['frequency']:
            qh, ql = qcq[0], qcq[1]
        else:
            qh, ql = qcq[1], qcq[0]
        if (max(3.75, chip.nodes[qh]['sweet point'] - TWOQThreshold) + chip.nodes[qh]['anharm'] > chip.nodes[ql]['sweet point']):
            removeQCQs.append(qcq)
            continue

    xTalkG.remove_nodes_from(removeQCQs)
    chip.remove_edges_from(removeQCQs)
    maxParallelCZs = max_Algsubgraph(chip)
    xTalkG = xtalk_G(chip)

    for level in range(len(maxParallelCZs)):
        couplerActivate = [[coupler, 'gray'] for coupler in chip.edges]
        for i in couplerActivate:
            if i[0] in maxParallelCZs[level]:
                i[1] = 'green'
        pos = gen_pos(chip)
        plt.figure(figsize=(4, 8))
        nx.draw_networkx_edges(chip, pos, edgelist=chip.edges, edge_color=list(dict(couplerActivate).values()), edge_cmap=plt.cm.Reds_r, width=8)
        nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, cmap=plt.cm.Reds_r)
        plt.axis('off')
        plt.savefig('twoq chip ' + str(level) + '.pdf', dpi=300)
        plt.close()

    for level in range(len(maxParallelCZs)):
        epoch = 0
        conflictQCQPercents = []
        conflictQCQDict = dict()
        for qcq in xTalkG.nodes:
            xTalkSubG = deepcopy(xTalkG)
            if qcq in maxParallelCZs[level]:
                conflictQCQDict[qcq] = 'gray'
        xTalkSubG.remove_nodes_from(set(xTalkG.nodes).difference(set(maxParallelCZs[level])))

        distance = dict()
        for qcq in xTalkSubG:
            if nx.has_path(chip, qcq[0], (3, 6)):
                distance[qcq] = nx.shortest_path_length(chip, qcq[0], (3, 6)) + \
                    nx.shortest_path_length(chip, qcq[1], (3, 6))
            else:
                distance[qcq] = 100000
        centertwoQ = sorted(distance.items(), key=lambda x : x[1])[0]
        centerConflictQCQ = centertwoQ[0]

        for _ in range(40):
            reOptimizeQCQs = [centerConflictQCQ]
            for qcq in xTalkSubG.nodes():
                if conflictQCQDict[centerConflictQCQ] == 'gray' and not(qcq in reOptimizeQCQs) and \
                    not(xTalkSubG.nodes[qcq].get('frequency', False)) and qcq in xTalkSubG[centerConflictQCQ]:
                    reOptimizeQCQs.append(qcq)
                elif conflictQCQDict[centerConflictQCQ] == 'red' and not(qcq in reOptimizeQCQs) and \
                    qcq in xTalkSubG[centerConflictQCQ] and distance[qcq] >= distance[centerConflictQCQ] and \
                    conflictQCQDict[qcq] == 'red':
                    reOptimizeQCQs.append(qcq)

            reOptimizeQCQs = tuple(reOptimizeQCQs)

            bounds = []
            for qcq in reOptimizeQCQs:
                if chip.nodes[qcq[0]]['frequency'] > chip.nodes[qcq[1]]['frequency']:
                    qh, ql = qcq[0], qcq[1]
                else:
                    qh, ql = qcq[1], qcq[0]
                bounds.append((max(3.75, chip.nodes[qh]['sweet point'] - TWOQThreshold) + chip.nodes[qh]['anharm'], chip.nodes[ql]['sweet point']))

            ini_frequency = [(max(bound) + min(bound)) / 2 for bound in bounds]

            res = minimize(phys_twoq_cost_function, ini_frequency, args=(chip, xTalkSubG, reOptimizeQCQs, a),
                        method='Nelder-Mead', bounds=bounds, options={'maxiter' : 200})
            print(res.fun)
            
            for qcq in reOptimizeQCQs:
                xTalkSubG.nodes[qcq]['frequency'] = round(res.x[reOptimizeQCQs.index(qcq)], 3)

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

            unAllocPercent = len([qcq for qcq in conflictQCQDict if conflictQCQDict[qcq] == 'gray']) / len(conflictQCQDict)

            print('multi')

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
                        if nx.shortest_path_length(nx.grid_2d_graph(H, W), edge[0][0], (3, 6)) + nx.shortest_path_length(nx.grid_2d_graph(H, W), edge[0][1], (3, 6)) > \
                            nx.shortest_path_length(nx.grid_2d_graph(H, W), edge[1][0], (3, 6)) + nx.shortest_path_length(nx.grid_2d_graph(H, W), edge[1][1], (3, 6)):
                            conflictQCQDict[edge[0]] = 'red'
                            conflictQCQDict[edge[1]] = 'green'
                        else:
                            conflictQCQDict[edge[1]] = 'red'
                            conflictQCQDict[edge[0]] = 'green'
                        alreadyConflict.append(edge[0])
                        alreadyConflict.append(edge[1])
                elif xTalkSubG.nodes[edge[0]].get('frequency', False) and xTalkSubG.nodes[edge[1]].get('frequency', False):
                    conflictQCQEdgeDict[edge] = 'green'
                    if not(edge[0] in alreadyConflict):
                        conflictQCQDict[edge[0]] = 'green'
                    elif not(edge[1] in alreadyConflict):
                        conflictQCQDict[edge[1]] = 'green'
                else:
                    conflictQCQEdgeDict[edge] = 'gray'

            for qcq in xTalkSubG.nodes:
                if qcq in newreOptimizeQCQs and (conflictQCQDict[qcq] == 'green' or conflictQCQDict[qcq] == 'gray'):
                    isBadFreq = True
                    for edge in conflictQCQEdgeDict:
                        if conflictQCQEdgeDict[edge] == 'red' and qcq in edge:
                            isBadFreq = False
                            break
                    if isBadFreq:
                        conflictQCQDict[qcq] = 'red'
                if xTalkSubG.nodes[qcq].get('frequency', False) and conflictQCQDict[qcq] == 'gray':
                    conflictQCQDict[qcq] = 'green'

            conflictQCQPercents.append(len([qcq for qcq in conflictQCQDict if conflictQCQDict[qcq] == 'red']) / 
                                    len([qcq for qcq in conflictQCQDict 
                                        if conflictQCQDict[qcq] == 'red' or  conflictQCQDict[qcq] == 'green']))
            print(conflictQCQPercents[-1])

            pos = twoQ_gen_pos(chip, xTalkSubG)
            nx.draw_networkx_nodes(xTalkSubG, pos, nodelist=xTalkSubG.nodes, node_color=list(conflictQCQDict.values()), cmap=plt.cm.Reds_r)
            nx.draw_networkx_edges(xTalkSubG, pos, edgelist=xTalkSubG.edges, edge_color=list(conflictQCQEdgeDict.values()), edge_cmap=plt.cm.Reds_r)
            plt.axis('off')
            plt.savefig(str(level) + ' ' + str(epoch) + str(W) + str(H) + 'twoq conflict.pdf', dpi=300)
            plt.close()
            
            
            reOptimizeQCQs = newreOptimizeQCQs
            emptyQCQ = dict([(qcq, nx.shortest_path_length(nx.grid_2d_graph(H, W), qcq[0], (3, 6)) + nx.shortest_path_length(nx.grid_2d_graph(H, W), qcq[1], (3, 6))) 
                            for qcq in xTalkSubG.nodes() if not(xTalkSubG.nodes[qcq].get('frequency', False))])

            if len(emptyQCQ) > 0:
                centerConflictQCQ = list(sorted(emptyQCQ.items(), key=lambda x : x[1]))[0][0]
                # centerConflictQCQ = random.choices(list(emptyQCQ.keys()), weights=[1 / (distance + 1e-5) for distance in emptyQCQ.values()], k=1)[0]
            elif len(reOptimizeQCQs) > 0:
                centerConflictQCQ = random.choices(list(reOptimizeQCQs.keys()), weights=[1 / (distance + 1e-5) for distance in reOptimizeQCQs.values()], k=1)[0]
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
        nx.draw_networkx_edges(chip, pos, edgelist=chip.edges, edge_color=intList, edge_cmap=plt.cm.Reds_r, width=8)
        nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, cmap=plt.cm.Reds_r)
        nx.draw_networkx_edge_labels(chip, pos, intDict, font_size=10, font_color='black')
        plt.axis('off')
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=3.75, vmax=4.5), cmap=plt.cm.Reds_r))
        plt.savefig(str(level) + ' ' + str(epoch) + str(W) + str(H) + 'int freq.pdf', dpi=300)
        plt.close()

    # fMax = 5
    # fWork = 4.4
    # fidle = 4.8

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # f = np.linspace(3.75, fMax, 100)
    # ax1.plot(f, T1_spectra(fMax, len(f)), 'b-')
    # ax1.set_ylabel('T1^(-1)ns^(-1)', color='b')
    # ax2 = ax1.twinx()
    # ax2.plot(f, T2_spectra(fMax, len(f)), 'r-')
    # ax2.set_ylabel('df/dphi(GHz/phi0)', color='r')
    # plt.show()

    # fList = np.linspace(-1, 1, 100000)
    # plt.plot(fList, lorentzain(fList, 0, 1, 0.3))
    # plt.show()