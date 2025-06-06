import numpy as np
import networkx as nx
from scipy.optimize import minimize
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import random
# from sko.PSO import PSO


def gen_pos(chip):
    wStep = 1
    hStep = 1
    pos = dict()
    for qubit in chip:
        pos[qubit] = [qubit[0] * wStep, qubit[1] * hStep]
    return pos

def cost_function(frequencys, inducedChip, targets):
    haveSeen = []
    lowestQ = 3.75
    largestFSD = 0.4

    chip = deepcopy(inducedChip)
    for target in targets:
        chip.nodes[target]['frequency'] = frequencys[targets.index(target)]

    cost = 0
    for target in targets:

        cost += sweet_point_constraint(chip.nodes[target]['sweet point'], chip.nodes[target]['frequency'], chip.nodes[target]['bad freq'])
        for neighbor in inducedChip[target]:
            if not(((neighbor, target) in haveSeen) or ((target, neighbor) in haveSeen)):
                cost += n_constraint(chip.nodes[target]['frequency'], chip.nodes[neighbor]['frequency'])
                cost += twoQGate_constraint(chip.nodes[target]['frequency'], chip.nodes[neighbor]['frequency'])
                haveSeen.append((target, neighbor))
            for nextNeighbor in chip[neighbor]:
                if not(nextNeighbor == target) and \
                    not((target, neighbor, nextNeighbor) in haveSeen or (nextNeighbor, neighbor, target) in haveSeen):
                    cost += nn_constraint(chip.nodes[target]['frequency'], chip.nodes[nextNeighbor]['frequency'])
                    haveSeen.append((target, neighbor, nextNeighbor))
    return cost

def sweet_point_constraint(sweetPoint, f, badFreqRange=None):
    cost = 0
    a = 1
    b = 1e-5
    if not(badFreqRange == None):
        for badFreq in badFreqRange:
            cost += a / np.sqrt(2 * np.pi * b) * np.exp(-(badFreq - f) ** 2 / (2 * b))
    return 1e-30 * (f - sweetPoint) ** 2 + cost

def n_constraint(neighborf, f):
    a = 50
    b = 0.001
    anharm = -0.2
    return a / np.sqrt(2 * np.pi * b) * np.exp(-(neighborf - f) ** 2 / (2 * b)) + \
        a / np.sqrt(2 * np.pi * b) * np.exp(-(np.abs(neighborf - f) + anharm) ** 2 / (2 * b))

def nn_constraint(neighborf, f):
    a = 50
    b = 0.00025
    anharm = -0.2
    return a / np.sqrt(2 * np.pi * b) * np.exp(-(neighborf - f) ** 2 / (2 * b)) + \
        a / np.sqrt(2 * np.pi * b) * np.exp(-(np.abs(neighborf - f) + anharm) ** 2 / (2 * b))

def twoQGate_constraint(neighborf, f):
    a = 50
    b = 0.0004
    lowestDetune = 1
    # a = 10
    # b = 0.01
    # lowestDetune = 1
    if isinstance(f, float) or isinstance(f, int):
        if np.abs(neighborf - f) < lowestDetune:
            return a / np.sqrt(2 * np.pi * b) * np.exp(-(np.abs(neighborf - f) - lowestDetune) ** 2 / (2 * b))
        else:
            return a / np.sqrt(2 * np.pi * b) + f
    else:
        cost = []
        for ff in f:
            if np.abs(neighborf - ff) < lowestDetune:
                cost.append(a / np.sqrt(2 * np.pi * b) * np.exp(-(np.abs(neighborf - ff) - lowestDetune) ** 2 / (2 * b)))
            else:
                cost.append(a / np.sqrt(2 * np.pi * b))
        return cost

def checkcoli(chip):
    reOptimizeGraph = nx.Graph()
    conflictEdge = []
    for qubit in chip.nodes:
        badFreqCost = 0
        for badFreq in chip.nodes[qubit]['bad freq']:
            badFreqCost += np.abs(chip.nodes[qubit]['frequency'] - badFreq) 
        if badFreqCost < 0.001:
            print(qubit, 'bad freq', badFreqCost)
            if not(qubit in reOptimizeGraph.nodes):
                reOptimizeGraph.add_node(qubit)
                reOptimizeGraph.nodes[qubit]['sweet point'] = chip.nodes[qubit]['sweet point']
                reOptimizeGraph.nodes[qubit]['frequency'] = chip.nodes[qubit]['frequency']
                reOptimizeGraph.nodes[qubit]['bad freq'] = chip.nodes[qubit]['bad freq']
        for neighbor in chip[qubit]:
            neighborCost1 = round(np.abs(chip.nodes[neighbor]['frequency'] - chip.nodes[qubit]['frequency']), 6)
            neighborCost2 = round(np.abs(np.abs(chip.nodes[neighbor]['frequency'] - chip.nodes[qubit]['frequency']) - 0.2), 6)
            neighborCost3 = round(np.abs(chip.nodes[neighbor]['frequency'] - chip.nodes[qubit]['frequency']), 6)
            if neighborCost1 < 0.1 or neighborCost2 < 0.1 or neighborCost3 > 1:
                if not(qubit in reOptimizeGraph.nodes):
                    reOptimizeGraph.add_node(qubit)
                    reOptimizeGraph.nodes[qubit]['sweet point'] = chip.nodes[qubit]['sweet point']
                    reOptimizeGraph.nodes[qubit]['frequency'] = chip.nodes[qubit]['frequency']
                    reOptimizeGraph.nodes[qubit]['bad freq'] = chip.nodes[qubit]['bad freq']

                if reOptimizeGraph.nodes[qubit].get('cost', False):
                    reOptimizeGraph.nodes[qubit]['cost'] = max([1 / (neighborCost1 + 1e-5), 1 / (neighborCost2 + 1e-5), reOptimizeGraph.nodes[qubit]['cost']])
                else:
                    reOptimizeGraph.nodes[qubit]['cost'] = max([1 / (neighborCost1 + 1e-5), 1 / (neighborCost2 + 1e-5)])

                if not(neighbor in reOptimizeGraph.nodes):
                    reOptimizeGraph.add_node(neighbor)
                    reOptimizeGraph.nodes[neighbor]['sweet point'] = chip.nodes[neighbor]['sweet point']
                    reOptimizeGraph.nodes[neighbor]['frequency'] = chip.nodes[neighbor]['frequency']
                    reOptimizeGraph.nodes[neighbor]['bad freq'] = chip.nodes[neighbor]['bad freq']
                if not((qubit, neighbor) in reOptimizeGraph.edges):
                    reOptimizeGraph.add_edge(qubit, neighbor)
                    print(qubit, neighbor,'n', neighborCost1, 'na', neighborCost2, 'twoq', neighborCost3)
                    conflictEdge.append((qubit, neighbor))

            # if sum(qubit) % 2:
            for nextNeighbor in chip[neighbor]:
                if not(nextNeighbor == qubit):
                    neighborCost4 = round(np.abs(chip.nodes[nextNeighbor]['frequency'] - chip.nodes[qubit]['frequency']), 6)
                    neighborCost5 = round(np.abs(np.abs(chip.nodes[nextNeighbor]['frequency'] - chip.nodes[qubit]['frequency']) - 0.2), 6)
                    if (neighborCost4 < 0.05 or neighborCost5 < 0.05):
                        if not(qubit in reOptimizeGraph.nodes) or not(nextNeighbor in reOptimizeGraph.nodes):
                            print(qubit, nextNeighbor, 'nn', neighborCost4, 'nna', neighborCost5)
                            conflictEdge.append((qubit, nextNeighbor))
                        if not(qubit in reOptimizeGraph.nodes):
                            reOptimizeGraph.add_node(qubit)
                            reOptimizeGraph.nodes[qubit]['sweet point'] = chip.nodes[qubit]['sweet point']
                            reOptimizeGraph.nodes[qubit]['frequency'] = chip.nodes[qubit]['frequency']
                            reOptimizeGraph.nodes[qubit]['bad freq'] = chip.nodes[qubit]['bad freq']

                        if reOptimizeGraph.nodes[qubit].get('cost', False):
                            reOptimizeGraph.nodes[qubit]['cost'] = max([1 / (neighborCost4 + 1e-5), reOptimizeGraph.nodes[qubit]['cost']])
                        else:
                            reOptimizeGraph.nodes[qubit]['cost'] = 1 / (neighborCost4 + 1e-5)

                        if not(nextNeighbor in reOptimizeGraph.nodes):
                            reOptimizeGraph.add_node(nextNeighbor)
                            reOptimizeGraph.nodes[nextNeighbor]['sweet point'] = chip.nodes[nextNeighbor]['sweet point']
                            reOptimizeGraph.nodes[nextNeighbor]['frequency'] = chip.nodes[nextNeighbor]['frequency']
                            reOptimizeGraph.nodes[nextNeighbor]['bad freq'] = chip.nodes[nextNeighbor]['bad freq']
    return reOptimizeGraph, conflictEdge
    
# f = np.arange(3.75, 5, 0.001)
# plt.plot(f, sweet_point_constraint(5, f, badFreqRange=[4, 4.2, 4.4, 4.6, 4.8]))
# plt.plot(f, n_constraint(4.7, f))
# print((n_constraint(4.7, 4.7 - 0.1) - n_constraint(4.7, 4.7 - 0.1 - 1e-5)) / 1e-5)
# plt.plot(f, nn_constraint(4.7, f))
# print((nn_constraint(4.7, 4.7 - 0.05) - nn_constraint(4.7, 4.7 - 0.05 - 1e-5)) / 1e-5)
# plt.plot(f, twoQGate_constraint(5, f))
# print((twoQGate_constraint(5, 4) - twoQGate_constraint(5, 4 - 1e-5)) / 1e-5)
# plt.show()

H = 6
W = 12  

rawchip = nx.grid_2d_graph(H, W)

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
for qubit in rawchip.nodes():
    # if sum(qubit) % 2:
    #     rawchip.nodes[qubit]['sweet point'] = round(4.3 + 0.15 * (np.random.random() - 0.5), 3)
    # else:
    #     rawchip.nodes[qubit]['sweet point'] = round(4.7 + 0.15 * (np.random.random() - 0.5), 3)

    rawchip.nodes[qubit]['sweet point'] = round(sweetPointArray[qubit[0], qubit[1]] * 1e-3, 3)
    badFreqNum = random.randint(0, 10)
    rawchip.nodes[qubit]['bad freq'] = random.choices(np.arange(3.75, rawchip.nodes[qubit]['sweet point'], 0.001), k=badFreqNum)

qubitList = list(rawchip.nodes)

chip = deepcopy(rawchip)


for qubit in qubitList:
    chip.nodes[qubit]['frequency'] = chip.nodes[qubit]['sweet point']#############################################################################################
    # if qubitList.index(qubit) == 0:
    #     chip.nodes[qubit]['frequency'] = chip.nodes[qubit]['sweet point']
    # else:
    #     inducedChip = nx.Graph()
    #     inducedChip.add_node(qubit)
    #     inducedChip.nodes[qubit]['sweet point'] = chip.nodes[qubit]['sweet point']
    #     inducedChip.nodes[qubit]['bad freq'] = chip.nodes[qubit]['bad freq']
    #     neighbors = [neighbor for neighbor in chip[qubit] if chip.nodes[neighbor].get('frequency', False)]
    #     for neighbor in neighbors:
    #         inducedChip.add_edge(qubit, neighbor)
    #         inducedChip.nodes[neighbor]['sweet point'] = chip.nodes[neighbor]['sweet point']
    #         inducedChip.nodes[neighbor]['frequency'] = chip.nodes[neighbor]['frequency']
    #         inducedChip.nodes[neighbor]['bad freq'] = chip.nodes[neighbor]['bad freq']
    #         for nextNeighbor in chip[neighbor]:
    #             if not(nextNeighbor == qubit) and chip.nodes[nextNeighbor].get('frequency', False):
    #                 inducedChip.add_edge(neighbor,nextNeighbor)
    #                 inducedChip.nodes[nextNeighbor]['sweet point'] = chip.nodes[nextNeighbor]['sweet point']
    #                 inducedChip.nodes[nextNeighbor]['frequency'] = chip.nodes[nextNeighbor]['frequency']
    #                 inducedChip.nodes[nextNeighbor]['bad freq'] = chip.nodes[nextNeighbor]['bad freq']

    #     bounds = [(chip.nodes[qubit]['sweet point'] - 0.4, chip.nodes[qubit]['sweet point'])]

    #     minRes = cost_function([inducedChip.nodes[qubit]['sweet point']], inducedChip, [qubit])
    #     minFreq = inducedChip.nodes[qubit]['sweet point']
    #     for frequency in np.arange(bounds[0][1], bounds[0][0], -0.001):
    #         tempRes = cost_function([frequency], inducedChip, [qubit])
    #         if tempRes < minRes:
    #             minFreq = frequency
    #             minRes = tempRes
    #     chip.nodes[qubit]['frequency'] = minFreq

    #     print(round(chip.nodes[qubit]['sweet point'], 3), round(chip.nodes[qubit]['frequency'], 3), round(minRes, 3))

firstChip = deepcopy(chip)

for qubit in chip:
    firstChip.nodes[qubit]['frequency'] = round(firstChip.nodes[qubit]['frequency'], 3)

reOptimizeGraph, conflictEdge = checkcoli(firstChip)

epoch = 0
conFlictPercent = []
for _ in range(40):
    
    reOpSubGraphs = dict()
    for reOpSubGraph in nx.connected_components(reOptimizeGraph):
        reOpSubGraph = tuple(reOpSubGraph)
        reOpSubGraphs[reOpSubGraph] = sum([reOptimizeGraph.nodes[qubit]['cost'] for qubit in reOpSubGraph if 
                                           reOptimizeGraph.nodes[qubit].get('cost', False)])
    reOpSubGraphs = dict(sorted(reOpSubGraphs.items(), key=lambda x : x[1], reverse=True))

    # print(list(reOpSubGraphs.values()))
    if len(reOpSubGraphs) == 0 or sum(list(reOpSubGraphs.values())) == 0:
        conFlictPercent.append(0)
        drawChip = deepcopy(chip)
        break
    
    for reOpSubGraph in random.choices(list(reOpSubGraphs.keys()), weights=list(reOpSubGraphs.values()), k=len(reOpSubGraphs)):
        reOpSubGraph = tuple(reOpSubGraph)
        ini_frequency = [chip.nodes[qubit]['frequency'] for qubit in reOpSubGraph]

        bounds = []
        for qubit in reOpSubGraph:
            bounds.append((max(3.75, chip.nodes[qubit]['sweet point'] - 0.4), chip.nodes[qubit]['sweet point']))

        cost = np.inf
        for _ in range(1):
            res = minimize(cost_function, ini_frequency, args=(chip, reOpSubGraph),
                        method='slsqp', bounds=bounds, options={'maxiter' : 200})
            if round(res.fun, 3) == 0.0:
                result = res.x
                break
            else:
                ini_frequency = [i + np.random.random() * 1e-3 for i in res.x]
                result = res.x

        for qubit in reOpSubGraph:
            chip.nodes[qubit]['frequency'] = round(result[reOpSubGraph.index(qubit)], 3)

    reOptimizeGraph, conflictEdge = checkcoli(chip)

    twoQForbiddenCoupler = []
    for coupler in conflictEdge:
        if np.abs(chip.nodes[coupler[0]]['frequency'] - chip.nodes[coupler[1]]['frequency']) > 1:
            twoQForbiddenCoupler.append(coupler)
    for coupler in twoQForbiddenCoupler:
        conflictEdge.remove(coupler)

    drawChip = deepcopy(chip)
    drawChip.add_edges_from(conflictEdge)
    drawChip.remove_edges_from(twoQForbiddenCoupler)

    conflictNodeDict = dict()
    for qubit in drawChip.nodes:
        if qubit in reOptimizeGraph.nodes:
            conflictNodeDict[qubit] = 'red'
        else:
            conflictNodeDict[qubit] = 'green'

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
        else:
            conflictEdgeDict[coupler] = 'green'

    pos = gen_pos(drawChip)
    labelDict = dict([(i, i) for i in drawChip.nodes])
    nx.draw_networkx_labels(drawChip, pos, labelDict, font_size=10, font_color="black")
    nx.draw_networkx_nodes(drawChip, pos, nodelist=drawChip.nodes, node_color=list(conflictNodeDict.values()), cmap=plt.cm.Reds_r)
    nx.draw_networkx_edges(drawChip, pos, edgelist=drawChip.edges, edge_color=list(conflictEdgeDict.values()), edge_cmap=plt.cm.Reds_r)
    plt.axis('off')
    plt.savefig(str(epoch) + str(W) + str(H) + 'chip conflict.pdf', dpi=300)
    plt.close()
    epoch += 1
    conFlictPercent.append(len([qubit for qubit in conflictNodeDict if conflictNodeDict[qubit] == 'red']) / len(chip.nodes()))
    print(conFlictPercent[-1])

print(conFlictPercent)

plt.plot(range(len(conFlictPercent)), conFlictPercent)
plt.xlabel('epoch')
plt.ylabel('conflict qubit percentage')
plt.savefig(str(W) + str(H) + 'conflict qubit percentage.pdf', dpi=300)
plt.close()

for qubit in chip.nodes:
    print(qubit, chip.nodes[qubit]['sweet point'], round(chip.nodes[qubit]['frequency'], 3), round(np.abs(chip.nodes[qubit]['frequency'] - chip.nodes[qubit]['sweet point']), 3))

pos = gen_pos(chip)
freqList = [chip.nodes[qubit]['frequency'] for qubit in chip.nodes]
qlow = min(freqList)
qhigh = max(freqList)
freqDict = dict([(i, round(chip.nodes[i]['frequency'], 3)) for i in chip.nodes])
nx.draw_networkx_labels(chip, pos, freqDict, font_size=10, font_color="black")
nx.draw_networkx_edges(chip, pos, edgelist=chip.edges, edge_cmap=plt.cm.Reds_r)
nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, node_color=freqList, cmap=plt.cm.Reds_r)
plt.axis('off')
plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=qlow, vmax=qhigh), cmap=plt.cm.Reds_r))
plt.savefig(str(W) + str(H) + 'chip freq.pdf', dpi=300)
plt.close()