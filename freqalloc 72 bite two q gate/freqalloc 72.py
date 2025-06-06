import numpy as np
import networkx as nx
from scipy.optimize import minimize
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import random
import z3
# from sko.PSO import PSO

BadFreqThreshold = 0.001
NThreshold = 0.1
NNThreshold = 0.04
TWOQThreshold = 0.7
IntThreshold = 0.05
Anharm = -0.2

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

def cost_function(frequencys, inducedChip, targets):
    chip = deepcopy(inducedChip)
    for target in targets:
        chip.nodes[target]['frequency'] = frequencys[targets.index(target)]
    cost = 0
    for target in targets:
        if chip.nodes[target]['available']:
            cost += 0 * sweet_point_cost(chip.nodes[target]['sweet point'], chip.nodes[target]['frequency'])
            # cost += 0.05 * sweet_point_cost(chip.nodes[target]['sweet point'], chip.nodes[target]['frequency'])
            cost += 50000 * bad_freq_constraint(chip.nodes[target]['frequency'], chip.nodes[target]['bad freq'])
        for neighbor in chip[target]:
            if chip.nodes[neighbor].get('frequency', False):
                cost += 500 * n_constraint(chip.nodes[neighbor]['frequency'], chip.nodes[target]['frequency'])
                cost += 500 * na_constraint(chip.nodes[neighbor]['frequency'], chip.nodes[target]['frequency'])
                if chip.nodes[target]['available'] and chip.nodes[neighbor]['available']:
                    cost += 200 * twoQGate_constraint(chip.nodes[neighbor]['frequency'], chip.nodes[target]['frequency'])
            for nextNeighbor in chip[neighbor]:
                if not(nextNeighbor == target) and chip.nodes[nextNeighbor].get('frequency', False):
                    cost += 2000 * nn_constraint(chip.nodes[nextNeighbor]['frequency'], chip.nodes[target]['frequency'])
                    cost += 2000 * nna_constraint(chip.nodes[nextNeighbor]['frequency'], chip.nodes[target]['frequency'])
    return cost

def twoQ_cost_function(frequencys, chip, xTalkG, reOptimizeQCQs):
    xtalkG = deepcopy(xTalkG)
    for qcq in reOptimizeQCQs:
        xtalkG.nodes[qcq]['frequency'] = frequencys[reOptimizeQCQs.index(qcq)]
    cost = 0
    for qcq in reOptimizeQCQs:
        cost += 50000 * twoQ_bad_freq_constraint(xtalkG.nodes[qcq]['frequency'], chip, qcq)
        for q in qcq:
            for neighbor in chip[q]:
                if neighbor in qcq:
                    continue   
                cost += 2000 * twoQ_idle_constraint(xtalkG.nodes[qcq]['frequency'], chip.nodes[neighbor]['frequency'])
        for neighbor in xtalkG[qcq]:
            if xtalkG.nodes[neighbor].get('frequency', False):
                cost += 200 * twoQ_int_constraint(chip, xtalkG, qcq, neighbor)
    return cost

def sweet_point_cost(sweetPoint, f):
    if isinstance(f, float) or isinstance(f, int):
        return round(sweetPoint - f, 3)
    else:
        cost = np.zeros(len(f))
        for ff in f:
            cost[list(f).index(ff)] = round(sweetPoint - ff, 3)
        return cost

def bad_freq_constraint(f, badFreqRange):
    if isinstance(f, float) or isinstance(f, int):
        cost = 0
        for badFreq in badFreqRange:
            cost += max(0, BadFreqThreshold - np.abs(f - badFreq))
        return cost
    else:
        cost = np.zeros(len(f))
        for ff in f:
            for badFreq in badFreqRange:
                cost[list(f).index(ff)] += max(0, BadFreqThreshold - np.abs(ff - badFreq))
        return cost

def n_constraint(neighborf, f):
    if isinstance(f, float) or isinstance(f, int):
        return max(0, NThreshold - np.abs(neighborf - f))
    else:
        cost = np.zeros(len(f))
        for ff in f:
            cost[list(f).index(ff)] = max(0, NThreshold - np.abs(neighborf - ff))
        return cost

def na_constraint(neighborf, f):
    if isinstance(f, float) or isinstance(f, int):
        return max(0, NThreshold - np.abs(np.abs(neighborf - f) + Anharm))
    else:
        cost = np.zeros(len(f))
        for ff in f:
            cost[list(f).index(ff)] = max(0, NThreshold - np.abs(np.abs(neighborf - ff) + Anharm))
        return cost

def nn_constraint(neighborf, f):
    if isinstance(f, float) or isinstance(f, int):
        return max(0, NNThreshold - np.abs(neighborf - f))
    else:
        cost = np.zeros(len(f))
        for ff in f:
            cost[list(f).index(ff)] = max(0, NNThreshold - np.abs(neighborf - ff))
        return cost

def nna_constraint(neighborf, f):
    if isinstance(f, float) or isinstance(f, int):
        return max(0, NNThreshold - np.abs(np.abs(neighborf - f) - 0.2))
    else:
        cost = np.zeros(len(f))
        for ff in f:
            cost[list(f).index(ff)] = max(0, NNThreshold - np.abs(np.abs(neighborf - ff) - 0.2))
        return cost

def twoQGate_constraint(neighborf, f):
    if isinstance(f, float) or isinstance(f, int):
        return max(0, np.abs(neighborf - f) - TWOQThreshold)
    else:
        cost = np.zeros(len(f))
        for ff in f:
            cost[list(f).index(ff)] = max(0, np.abs(neighborf - ff) - TWOQThreshold)
        return cost

def checkcoli(chip):
    centerNode = (6 // 2, 12 // 2)
    reOptimizeNodes = dict()
    conflictEdge = []
    for qubit in chip.nodes:
        if chip.nodes[qubit].get('frequency', False) and chip.nodes[qubit]['available']:
            badFreqCost = round(bad_freq_constraint(chip.nodes[qubit]['frequency'], chip.nodes[qubit]['bad freq']), 4)
            if badFreqCost > 0:
                if not(qubit in reOptimizeNodes):
                    reOptimizeNodes[qubit] = nx.shortest_path_length(chip, centerNode, qubit)
                    print(qubit, 'bad freq', badFreqCost)

            for neighbor in chip[qubit]:
                if not((qubit, neighbor) in conflictEdge) and not((neighbor, qubit) in conflictEdge) and chip.nodes[neighbor].get('frequency', False):
                    neighborCost1 = round(n_constraint(chip.nodes[neighbor]['frequency'], chip.nodes[qubit]['frequency']), 4)
                    neighborCost2 = round(na_constraint(chip.nodes[neighbor]['frequency'], chip.nodes[qubit]['frequency']), 4)
                    if chip.nodes[qubit]['available'] and chip.nodes[neighbor]['available']:
                        neighborCost3 = round(twoQGate_constraint(chip.nodes[neighbor]['frequency'], chip.nodes[qubit]['frequency']), 4)
                    else:
                        neighborCost3 = 0
                    if neighborCost1 > 0 or neighborCost2 > 0 or neighborCost3 > 0:
                        print(qubit, neighbor,'n', neighborCost1, 'na', neighborCost2, 'twoq', neighborCost3)
                        conflictEdge.append((qubit, neighbor))
                        if not(qubit in reOptimizeNodes):
                            reOptimizeNodes[qubit] = nx.shortest_path_length(chip, centerNode, qubit)
                        if not(neighbor in reOptimizeNodes):
                            reOptimizeNodes[neighbor] = nx.shortest_path_length(chip, centerNode, neighbor)

                for nextNeighbor in chip[neighbor]:
                    if not(nextNeighbor == qubit) and not((qubit, nextNeighbor) in conflictEdge) and not((nextNeighbor, qubit) in conflictEdge) and chip.nodes[nextNeighbor].get('frequency'):
                        neighborCost4 = round(nn_constraint(chip.nodes[nextNeighbor]['frequency'], chip.nodes[qubit]['frequency']), 4)
                        neighborCost5 = round(nna_constraint(chip.nodes[nextNeighbor]['frequency'], chip.nodes[qubit]['frequency']), 4)
                        if (neighborCost4 > 0 or neighborCost5 > 0):
                            print(qubit, nextNeighbor, 'nn', neighborCost4, 'nna', neighborCost5)
                            conflictEdge.append((qubit, nextNeighbor))
                            if not(qubit in reOptimizeNodes):
                                reOptimizeNodes[qubit] = nx.shortest_path_length(chip, centerNode, qubit)
                            if not(nextNeighbor in reOptimizeNodes):
                                reOptimizeNodes[nextNeighbor] = nx.shortest_path_length(chip, centerNode, nextNeighbor)

    return reOptimizeNodes, conflictEdge
    
def twoQ_checkcoli(chip, xTalkG):
    distance = dict()
    reOptimizeQCQs = dict()
    conflictEdge = []
    for qcq in xTalkG:
        distance[qcq] = nx.shortest_path_length(nx.grid_2d_graph(H, W), qcq[0], (3, 6)) + \
            nx.shortest_path_length(nx.grid_2d_graph(H, W), qcq[1], (3, 6))
    centertwoQ = sorted(distance.items(), key=lambda x : x[1])[0][0]
    for qcq in xTalkG.nodes:
        if xTalkG.nodes[qcq].get('frequency', False):
            badFreqCost = round(twoQ_bad_freq_constraint(xTalkG.nodes[qcq]['frequency'], chip, qcq), 3)
            if badFreqCost > 0:
                if not(qcq in reOptimizeQCQs):
                    if nx.has_path(xTalkG, centertwoQ, qcq):
                        reOptimizeQCQs[qcq] = nx.shortest_path_length(xTalkG, centertwoQ, qcq)
                    else:
                        reOptimizeQCQs[qcq] = 100000
                    print(qcq, 'bad freq', badFreqCost)
            for q in qcq:
                for neighbor in chip[q]:
                    if neighbor in qcq:
                        continue
                    neighborCost1 = round(twoQ_idle_constraint(xTalkG.nodes[qcq]['frequency'], chip.nodes[neighbor]['frequency']), 3)
                    if neighborCost1 > 0:
                        conflictEdge.append((qcq, neighbor))
                        if not(qcq in reOptimizeQCQs):
                            if nx.has_path(xTalkG, centertwoQ, qcq):
                                reOptimizeQCQs[qcq] = nx.shortest_path_length(xTalkG, centertwoQ, qcq)
                            else:
                                reOptimizeQCQs[qcq] = 100000
                        print(qcq, 'idle conflict', neighborCost1, neighbor)
            for neighbor in xTalkG[qcq]:
                if not((qcq, neighbor) in conflictEdge) and not((neighbor, qcq) in conflictEdge) and xTalkG.nodes[neighbor].get('frequency', False):
                    neighborCost2 = round(twoQ_int_constraint(chip, xTalkG, qcq, neighbor), 3)
                    if neighborCost2 > 0:
                        conflictEdge.append((qcq, neighbor))
                        if not(qcq in reOptimizeQCQs):
                            if nx.has_path(xTalkG, qcq, centertwoQ):
                                reOptimizeQCQs[qcq] = nx.shortest_path_length(xTalkG, qcq, centertwoQ)
                            else:
                                reOptimizeQCQs[qcq] = 100000
                        if not(neighbor in reOptimizeQCQs):
                            if nx.has_path(xTalkG, neighbor, centertwoQ):
                                reOptimizeQCQs[neighbor] = nx.shortest_path_length(xTalkG, neighbor, centertwoQ)
                            else:
                                reOptimizeQCQs[neighbor] = 100000
                        print(qcq, 'int conflict', neighborCost2, neighbor)
    return reOptimizeQCQs, conflictEdge

def create_z3_vars(reOptmizNodes):
    freqVarDict = dict()
    for qubit in reOptmizNodes:
        freqVarDict[qubit] = z3.Real(str(qubit[0]) + ',' + str(qubit[1]))
    return freqVarDict

def twoQ_create_z3_vars(reOptimizeQCQs):
    freqVarDict = dict()
    for qcq in reOptimizeQCQs:
        freqVarDict[qcq] = z3.Real(str(qcq[0][0]) + ',' + str(qcq[0][1]) + ';' + 
                                    str(qcq[1][0]) + ',' + str(qcq[1][1]))
    return freqVarDict

def basic_bounds(chip, freqVarDict, opt):
    for qubit in freqVarDict:
        opt.add(freqVarDict[qubit] > max(3.75, chip.nodes[qubit]['sweet point'] - 0.4))
        opt.add(freqVarDict[qubit] < chip.nodes[qubit]['sweet point'])
        for badFreq in chip.nodes[qubit]['bad freq']:
            opt.add(z3.Or(freqVarDict[qubit] - badFreq > BadFreqThreshold, freqVarDict[qubit] - badFreq < -BadFreqThreshold))
    return opt

def twoQ_basic_bounds(chip, freqVarDict, opt):
    for qcq in freqVarDict:
        if chip.nodes[qcq[0]]['frequency'] > chip.nodes[qcq[1]]['frequency']:
            qh, ql = qcq[0], qcq[1]
        else:
            qh, ql = qcq[1], qcq[0]
        opt.add(freqVarDict[qcq] > max(3.75, chip.nodes[qh]['sweet point'] - TWOQThreshold) + Anharm)
        opt.add(freqVarDict[qcq] < chip.nodes[ql]['sweet point'])
        for badFreq in chip.nodes[qh]['bad freq']:
            opt.add(z3.Or(freqVarDict[qcq] - Anharm - badFreq > BadFreqThreshold, freqVarDict[qcq] - Anharm - badFreq < -BadFreqThreshold))
        for badFreq in chip.nodes[ql]['bad freq']:
            opt.add(z3.Or(freqVarDict[qcq] - badFreq > BadFreqThreshold, freqVarDict[qcq] - badFreq < -BadFreqThreshold))
    return opt

def constraints(chip, freqVarDict, opt):
    for qubit in freqVarDict:
        for neighbor in chip[qubit]:
            if chip.nodes[neighbor].get('frequency'):
                opt.add(z3.Or(freqVarDict[qubit] - chip.nodes[neighbor]['frequency'] > NThreshold, freqVarDict[qubit] - chip.nodes[neighbor]['frequency'] < -NThreshold))
                opt.add(z3.Or(freqVarDict[qubit] - chip.nodes[neighbor]['frequency'] - np.abs(Anharm) > NThreshold, freqVarDict[qubit] - chip.nodes[neighbor]['frequency'] - np.abs(Anharm) < -NThreshold))
                opt.add(z3.Or(freqVarDict[qubit] - chip.nodes[neighbor]['frequency'] + np.abs(Anharm) > NThreshold, freqVarDict[qubit] - chip.nodes[neighbor]['frequency'] + np.abs(Anharm) < -NThreshold))
                opt.add(z3.And(freqVarDict[qubit] - chip.nodes[neighbor]['frequency'] < TWOQThreshold, freqVarDict[qubit] - chip.nodes[neighbor]['frequency'] > -TWOQThreshold))
            elif neighbor in freqVarDict:
                opt.add(z3.Or(freqVarDict[qubit] - freqVarDict[neighbor] > NThreshold, freqVarDict[qubit] - freqVarDict[neighbor] < -NThreshold))
                opt.add(z3.Or(freqVarDict[qubit] - freqVarDict[neighbor] - np.abs(Anharm) > NThreshold, freqVarDict[qubit] - freqVarDict[neighbor] - np.abs(Anharm) < -NThreshold))
                opt.add(z3.Or(freqVarDict[qubit] - freqVarDict[neighbor] + np.abs(Anharm) > NThreshold, freqVarDict[qubit] - freqVarDict[neighbor] + np.abs(Anharm) < -NThreshold))
                opt.add(z3.And(freqVarDict[qubit] - freqVarDict[neighbor] < TWOQThreshold, freqVarDict[qubit] - freqVarDict[neighbor] > -TWOQThreshold))

            for nextNeighbor in chip[neighbor]:
                if not(nextNeighbor == qubit) and chip.nodes[nextNeighbor].get('frequency'):
                    opt.add(z3.Or(freqVarDict[qubit] - chip.nodes[nextNeighbor]['frequency'] > NNThreshold, freqVarDict[qubit] - chip.nodes[nextNeighbor]['frequency'] < -NNThreshold))
                    opt.add(z3.Or(freqVarDict[qubit] - chip.nodes[nextNeighbor]['frequency'] - np.abs(Anharm) > NNThreshold, freqVarDict[qubit] - chip.nodes[nextNeighbor]['frequency'] - np.abs(Anharm) < -NNThreshold))
                    opt.add(z3.Or(freqVarDict[qubit] - chip.nodes[nextNeighbor]['frequency'] + np.abs(Anharm) > NNThreshold, freqVarDict[qubit] - chip.nodes[nextNeighbor]['frequency'] + np.abs(Anharm) < -NNThreshold))
                elif not(nextNeighbor == qubit) and nextNeighbor in freqVarDict:
                    opt.add(z3.Or(freqVarDict[qubit] - freqVarDict[nextNeighbor] > NNThreshold, freqVarDict[qubit] - freqVarDict[nextNeighbor] < -NNThreshold))
                    opt.add(z3.Or(freqVarDict[qubit] - freqVarDict[nextNeighbor] - np.abs(Anharm) > NNThreshold, freqVarDict[qubit] - freqVarDict[nextNeighbor] - np.abs(Anharm) < -NNThreshold))
                    opt.add(z3.Or(freqVarDict[qubit] - freqVarDict[nextNeighbor] + np.abs(Anharm) > NNThreshold, freqVarDict[qubit] - freqVarDict[nextNeighbor] + np.abs(Anharm) < -NNThreshold))
    return opt

def twoQ_constraints(chip, xtalkG, freqVarDict, opt):
    for qcq in freqVarDict:
        for q in qcq:
            for neighbor in chip[q]:
                if neighbor in qcq:
                    continue
                opt.add(z3.Or(freqVarDict[qcq] - chip.nodes[neighbor]['frequency'] > IntThreshold, freqVarDict[qcq] - chip.nodes[neighbor]['frequency'] < -IntThreshold))
                opt.add(z3.Or(freqVarDict[qcq] - chip.nodes[neighbor]['frequency'] - Anharm > IntThreshold, freqVarDict[qcq] - chip.nodes[neighbor]['frequency'] - Anharm < -IntThreshold))
        for neighbor in xtalkG[qcq]:
            if neighbor in freqVarDict:
                opt.add(z3.Or(freqVarDict[qcq] - freqVarDict[neighbor] > IntThreshold, freqVarDict[qcq] - freqVarDict[neighbor] < -IntThreshold))
                for q1 in qcq:
                    for q2 in neighbor:
                        if (q1, q2) in chip.edges:
                            if sum(q1) % 2:
                                opt.add(z3.Or(freqVarDict[qcq] + Anharm - freqVarDict[neighbor] > IntThreshold, freqVarDict[qcq] + Anharm - freqVarDict[neighbor] < -IntThreshold))
                                opt.add(z3.Or(freqVarDict[qcq] - freqVarDict[neighbor] + Anharm > IntThreshold, freqVarDict[qcq] - freqVarDict[neighbor] + Anharm < -IntThreshold))
                            else:
                                opt.add(z3.Or(freqVarDict[qcq] - Anharm - freqVarDict[neighbor] > IntThreshold, freqVarDict[qcq] - Anharm - freqVarDict[neighbor] < -IntThreshold))
                                opt.add(z3.Or(freqVarDict[qcq] - freqVarDict[neighbor] - Anharm > IntThreshold, freqVarDict[qcq] - freqVarDict[neighbor] - Anharm < -IntThreshold))
            elif xtalkG.nodes[neighbor].get('frequency', False):
                opt.add(z3.Or(freqVarDict[qcq] - xtalkG.nodes[neighbor]['frequency'] > IntThreshold, freqVarDict[qcq] - xtalkG.nodes[neighbor]['frequency'] < -IntThreshold))
                for q1 in qcq:
                    for q2 in neighbor:
                        if (q1, q2) in chip.edges:
                            if sum(q1) % 2:
                                opt.add(z3.Or(freqVarDict[qcq] + Anharm - xtalkG.nodes[neighbor]['frequency'] > IntThreshold, freqVarDict[qcq] + Anharm - xtalkG.nodes[neighbor]['frequency'] < -IntThreshold))
                                opt.add(z3.Or(freqVarDict[qcq] - xtalkG.nodes[neighbor]['frequency'] + Anharm > IntThreshold, freqVarDict[qcq] - xtalkG.nodes[neighbor]['frequency'] + Anharm < -IntThreshold))
                            else:
                                opt.add(z3.Or(freqVarDict[qcq] - Anharm - xtalkG.nodes[neighbor]['frequency'] > IntThreshold, freqVarDict[qcq] - Anharm - xtalkG.nodes[neighbor]['frequency'] < -IntThreshold))
                                opt.add(z3.Or(freqVarDict[qcq] - xtalkG.nodes[neighbor]['frequency'] - Anharm > IntThreshold, freqVarDict[qcq] - xtalkG.nodes[neighbor]['frequency'] - Anharm < -IntThreshold))
    return opt

def twoQ_bad_freq_constraint(f, chip, qcq):
    cost = 0
    if chip.nodes[qcq[0]]['frequency'] > chip.nodes[qcq[1]]['frequency']:
        qh, ql = qcq[0], qcq[1]
    else:
        qh, ql = qcq[1], qcq[0]
    for badFreq in chip.nodes[qh]['bad freq']:
        cost += max(BadFreqThreshold - np.abs(f - Anharm - badFreq), 0)
    for badFreq in chip.nodes[ql]['bad freq']:
        cost += max(BadFreqThreshold - np.abs(f - badFreq), 0)
    return cost

def twoQ_idle_constraint(f, neighborf):
    return max(IntThreshold - np.abs(f - neighborf), 0) + max(IntThreshold - np.abs(f - neighborf - Anharm), 0)

def twoQ_int_constraint(chip, xtalkG, qcq, neighbor):
    cost = max(IntThreshold - np.abs(xtalkG.nodes[qcq]['frequency'] - xtalkG.nodes[neighbor]['frequency']), 0)
    for q1 in qcq:
        for q2 in neighbor:
            if (q1, q2) in chip.edges:
                if sum(q1) % 2:
                    cost += max(IntThreshold - np.abs(xtalkG.nodes[qcq]['frequency'] + Anharm - xtalkG.nodes[neighbor]['frequency']), 0)
                else:
                    cost += max(IntThreshold - np.abs(xtalkG.nodes[qcq]['frequency'] - Anharm - xtalkG.nodes[neighbor]['frequency']), 0)
    return cost

def objective_function(chip, freqVarDict, opt):
    cost = []
    for qubit in freqVarDict:
        cost.append(chip.nodes[qubit]['sweet point'] - freqVarDict[qubit])
    opt.minimize(z3.Sum(cost))
    return opt

def twoQ_Objective_function(freqVarDict, opt):
    cost = []
    for qcq in freqVarDict:
        cost.append(freqVarDict[qcq])
    opt.maximize(z3.Sum(cost))
    return opt

def r2f(val):
    """
    Convert Z3 Real to Python float
    """
    return float(val.as_decimal(16).rstrip("?"))

def extract_solution(freqVarDict, opt):
    if opt.check() == z3.sat:
        model = opt.model()
        result = dict()
        for qubit in freqVarDict:
            result[qubit] = r2f(model[freqVarDict[qubit]])
        return result
    else:
        return False

def solve_optimization(chip, reOptimizeQCQs, xtalkG=None, isTwoQ=False):
    opt = z3.Optimize()
    if isTwoQ:
        freqVarDict = twoQ_create_z3_vars(reOptimizeQCQs)
        opt = twoQ_basic_bounds(chip, freqVarDict, opt)
        opt = twoQ_constraints(chip, xtalkG, freqVarDict, opt)
        opt = twoQ_Objective_function(freqVarDict, opt)
    else:
        freqVarDict = create_z3_vars(reOptimizeQCQs)
        opt = basic_bounds(chip, freqVarDict, opt)
        opt = constraints(chip, freqVarDict, opt)
        opt = objective_function(chip, freqVarDict, opt)
    result = extract_solution(freqVarDict, opt)
    return result

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
    # while True:
    #     if len(dualChip.nodes) == 0:
    #         break
    #     else:
    #         nodeSeed = random.choice(list(dualChip.nodes))
    #         maxParallelCZs.append(nx.maximal_independent_set(dualChip, [nodeSeed]))
    #         dualChip.remove_nodes_from(maxParallelCZs[-1])

    return maxParallelCZs

f = np.arange(3.75, 5, 0.001)
plt.plot(f, 50000 * bad_freq_constraint(f, badFreqRange=[4, 4.2, 4.4, 4.6, 4.8]))
plt.plot(f, 0.05 * sweet_point_cost(5, f))
plt.plot(f, 500 * (n_constraint(4.7, f) + na_constraint(4.7, f)))
plt.plot(f, 2000 * (nn_constraint(4.7, f) + nna_constraint(4.7, f)))
plt.plot(f, 200 * twoQGate_constraint(5, f))
plt.show()

H = 6
W = 12  

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
                            [4923.431, 4223.597, 4842.507, 4092.872, 4895.467, 4279.065],
                            [4399.842, 5033.471, 4242.685, 4962.309, 4359.198, 4886.872]])

sweetPointArray = sweetPointArray.T
for qubit in chip.nodes():
    # if sum(qubit) % 2:
    #     chip.nodes[qubit]['sweet point'] = round(4.3 + 0.15 * (np.random.random() - 0.5), 3)】
    # else:
    #     chip.nodes[qubit]['sweet point'] = round(4.7 + 0.15 * (np.random.random() - 0.5), 3)

    chip.nodes[qubit]['sweet point'] = round(sweetPointArray[qubit[0], qubit[1]] * 1e-3, 3)
    badFreqNum = random.randint(0, 10)
    # badFreqNum = 0
    chip.nodes[qubit]['bad freq'] = random.choices(np.arange(3.75, chip.nodes[qubit]['sweet point'], 0.001), k=badFreqNum)
    chip.nodes[qubit]['available'] = True

qubitList = list(chip.nodes)

epoch = 0

centerConflictNode = (3, 6)
conflictNodeDict = dict()
for qubit in chip.nodes():
    conflictNodeDict[qubit] = 'gray'

conflictPercents = []

for _ in range(30):
    # reOptimizeNodes = [qubit for qubit in chip.nodes() 
                    #    if np.abs(qubit[0] - centerConflictNode[0]) <= 3 and np.abs(qubit[1] - centerConflictNode[1]) <= 3 and not(chip.nodes[qubit].get())]
    
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
        # elif conflictNodeDict[centerConflictNode] == 'red' and not(qubit in reOptimizeNodes) and \
            # np.abs(qubit[0] - centerConflictNode[0]) <= 2 and np.abs(qubit[1] - centerConflictNode[1]) <= 2 and \
            # (nx.shortest_path_length(chip, qubit, (3, 6)) >= (nx.shortest_path_length(chip, centerConflictNode, (3, 6)))):

            reOptimizeNodes.append(qubit)

    reOptimizeNodes = tuple(reOptimizeNodes)
    ini_frequency = [chip.nodes[qubit].get('frequency', chip.nodes[qubit]['sweet point']) for qubit in reOptimizeNodes]

    bounds = []
    for qubit in reOptimizeNodes:
        if chip.nodes[qubit]['available']:
            bounds.append((max(3.75, chip.nodes[qubit]['sweet point'] - 0.4), chip.nodes[qubit]['sweet point']))
        else:
            bounds.append((2, 3))

    result = solve_optimization(chip, reOptimizeNodes)
    if not result:
        print('z3 无解')
        cost = 100000
        for _ in range(10):
            res = minimize(cost_function, ini_frequency, args=(chip, reOptimizeNodes),
                        method='Powell', bounds=bounds, options={'maxiter' : 200})
            print(res.fun)
            if round(float(res.fun), 3) == 0.0:
                result = res.x
                break
            elif round(float(res.fun), 3) < cost:
                ini_frequency = [i + np.random.random() * 1e-2 for i in res.x]
                result = res.x

        for qubit in reOptimizeNodes:
            chip.nodes[qubit]['frequency'] = round(result[reOptimizeNodes.index(qubit)], 3)
    else:
        print('z3 有解')
        for qubit in reOptimizeNodes:
            chip.nodes[qubit]['frequency'] = round(result[qubit], 3)

    newreOptimizeNodes, conflictEdge = checkcoli(chip)

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
    if (max(3.75, chip.nodes[qh]['sweet point'] - TWOQThreshold) + Anharm > chip.nodes[ql]['sweet point']):
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
        # ini_frequency = [(max(chip.nodes[qcq[0]]['frequency'], chip.nodes[qcq[1]]['frequency']) + Anharm +
                        # min(chip.nodes[qcq[0]]['frequency'], chip.nodes[qcq[1]]['frequency'])) / 2
                        # for qcq in xTalkSubG.nodes]
        
        bounds = []
        for qcq in xTalkSubG.nodes:
            if chip.nodes[qcq[0]]['frequency'] > chip.nodes[qcq[1]]['frequency']:
                qh, ql = qcq[0], qcq[1]
            else:
                qh, ql = qcq[1], qcq[0]
            bounds.append((max(3.75, chip.nodes[qh]['sweet point'] - TWOQThreshold) + Anharm, chip.nodes[ql]['sweet point']))

        ini_frequency = [(max(bound) + min(bound)) / 2 for bound in bounds]

        result = solve_optimization(chip, reOptimizeQCQs, xtalkG=xTalkSubG, isTwoQ=True)
        if not(result):
            print('z3 无解')
            cost = np.inf
            for _ in range(1):
                res = minimize(twoQ_cost_function, ini_frequency, args=(chip, xTalkSubG, reOptimizeQCQs),
                            method='powell', bounds=bounds, options={'maxiter' : 200})
                print(res.fun)
                if round(float(res.fun), 3) == 0.0:
                    result = res.x
                    break
                elif round(float(res.fun), 3) < cost:
                    ini_frequency = res.x
                    result = res.x
            for qcq in reOptimizeQCQs:
                xTalkSubG.nodes[qcq]['frequency'] = round(result[reOptimizeQCQs.index(qcq)], 3)

        else:
            print('z3 有解')
            for qcq in reOptimizeQCQs:
                xTalkSubG.nodes[qcq]['frequency'] = round(result[qcq], 3)

        newreOptimizeQCQs, conflictQCQEdge = twoQ_checkcoli(chip, xTalkSubG)

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