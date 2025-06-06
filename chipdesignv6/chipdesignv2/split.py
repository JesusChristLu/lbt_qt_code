import math
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.quality import performance
from networkx.algorithms.centrality import betweenness_centrality
from networkx.algorithms.centrality import edge_betweenness_centrality
from networkx.algorithms.components import number_connected_components
from networkx.algorithms.components import node_connected_component
from networkx.algorithms.planarity import check_planarity
from networkx.classes.function import subgraph
from networkx.algorithms.isomorphism import is_isomorphic
from networkx.algorithms.shortest_paths.generic import shortest_path_length
from networkx.algorithms.operators.binary import compose
from networkx.relabel import relabel_nodes
from clustering import Clustering
from prune import Prune
from get_alg import Get_alg
from collections import Iterable
from copy import deepcopy


class Split():
    def __init__(self, a, g, prog, recover, mat, show=False):
        self.show = show
        self.graph = deepcopy(g)
        self.prog = prog
        self.recover = recover
        self.mat = mat
        self.splitVertex = self.get_split_vertex()
        self.graph, self.layout = self.split_vertex(a, self.splitVertex, recover)

    def get_split_vertex(self):
        splitVertex = list(np.where(np.array(list(dict(self.graph.degree(self.graph.nodes)).values())) > Prune.degLimit)[0])
        return splitVertex

    def split_vertex(self, a, splitVertexes, recover):
        graph = deepcopy(self.graph)
        mediaStructureSet = set()
        mediaStructures = []
        allocations = []
        I1s = []
        I2s = []
        node2indexes = []
        index2nodes = []
        layout = {}
        for node in graph.nodes:
            if node not in splitVertexes:
                layout[node] = [node]
        for v in splitVertexes:
            splitMethod = self.choose_media_structure(a, v, graph, recover)
            I1 = splitMethod[2]
            I2 = splitMethod[3]
            node2index = splitMethod[4]
            index2node = splitMethod[5]
            neighbors = dict(graph.adj[v])
            mapping = {}
            allocation = {}
            for node in splitMethod[0]:
                if node == 0:
                    mapping[node] = v
                else:
                    mapping[node] = node + max(list(dict(graph.nodes).keys()))
                allocation[mapping[node]] = splitMethod[1][node + min(list(splitMethod[1].keys()))]
            mediaStructure = relabel_nodes(splitMethod[0], mapping)
            mediaStructures.append(mediaStructure)
            graph.remove_node(v)
            graph = compose(graph, mediaStructure)
            layout[v] = list(mediaStructure.nodes)
            for body in list(allocation.keys()):
                for branch in allocation[body]:
                    graph.add_edge(body, branch)
                    #graph.edges[body, branch]['weight'] = neighbors[branch]['weight']##########################################
            mediaStructureSet = mediaStructureSet.union(set(mediaStructure.nodes))
            allocations.append(allocation)
            I1s.append(I1)
            I2s.append(I2)
            node2indexes.append(node2index)
            index2nodes.append(index2nodes)

        if self.show:
            node_color = []
            for node in graph.nodes:
                if node in mediaStructureSet:
                    node_color.append('red')
                else:
                    node_color.append('green')

            plt.ion() 
            plt.title('graph after split')
            nx.draw(graph, with_labels=False, node_color=node_color, font_color='white', node_size=50)
            plt.pause(15)
            plt.close()

        while len(self.recover) > 0:
            add_edge = list(self.recover.keys())[list(self.recover.values()).index(max(list(self.recover.values())))]
            graph.add_edge(add_edge[0], add_edge[1])
            
            is_planar, c = check_planarity(graph)

            is_break = False
            if max(list(dict(graph.degree).values())) > Prune.degLimit:
                is_break = True

            if not is_planar or is_break:
                graph.remove_edge(add_edge[0], add_edge[1])
            else:
                graph.edges[add_edge]['weight'] = self.recover[add_edge]###########################################
            self.recover.pop(add_edge)

        #for mediaStructure in mediaStructures:
        #    allocation = allocations[mediaStructures.index(mediaStructure)]
        #    I1 = I1s[mediaStructures.index(mediaStructure)]
        #    node2index = node2indexes[mediaStructures.index(mediaStructure)]
        #    index2nodes = index2nodes[mediaStructures.index(mediaStructure)]
        #    for edge in mediaStructure.edges:
        #        graph.edges[edge]['weight'] = 0
        #    nodels = []
        #    for node1 in mediaStructure.nodes:
        #        for node2 in mediaStructure.nodes:
        #            if (node1, node2) in nodels:
        #                continue
        #            branches = []
        #            for branch1 in allocation[node1]:
        #                for branch2 in allocation[node2]:
        #                    if (branch1, branch2) in branches:
        #                        continue
        #                    if branch1 == branch2:
        #                        continue
        #                    if not node1 == node2:
        #                        shortestPath = nx.shortest_path(mediaStructure, source=node1, target=node2)
        #                    else:
        #                        shortestPath = []
        #                    if not ((branch1, branch2) in graph.edges):
        #                        graph.edges[node2, branch2]['weight'] += self.mat[branch1, branch2] * 2
        #                        if len(shortestPath) > 0:
        #                            for path in range(len(shortestPath) - 1):
        #                                graph.edges[shortestPath[path], shortestPath[path + 1]]['weight'] += (self.mat[branch1, branch2] + I1[node2index[branch1], node2index[branch2]]) * 2
        #                    elif len(shortestPath) > 0:
        #                        for path in range(len(shortestPath) - 1):
        #                            graph.edges[shortestPath[path], shortestPath[path + 1]]['weight'] += I1[node2index[branch1], node2index[branch2]] * 2
        #                    branches.append((branch2, branch1))
        #                    branches.append((branch2, branch2))
        #            nodels.append((node2, node1))
        #            nodels.append((node1, node2))
        if self.show:
            plt.ion()
            plt.title('graph after recover')
            pos = nx.planar_layout(graph)
            #nx.draw(graph, with_labels=True, node_color=node_color, font_color='white')
            nx.draw(graph, pos=pos, with_labels=True, node_color=node_color, font_color='white')
            plt.pause(15)
            plt.close()
        return graph, layout

    def get_media_structure(self, n, k):
        mediaStructures = {}
        simpliestStructure = nx.Graph()
        for i in range(n - 1):
            simpliestStructure.add_edge(i, i + 1, weight=1)
        mediaStructures[n] = [simpliestStructure]
        if (Prune.degLimit - 1) * n - k < 0:
            return False
        for i in range(1, min((Prune.degLimit - 1) * n - k + 2, int(n * (n - 1) / 2) - n + 2)):
            structures = []
            for struct in mediaStructures[n + i - 1]:
                for node1 in struct.nodes:
                    for node2 in struct.nodes:
                        if not node1 == node2 and not (node1, node2) in struct.edges:
                            structure = deepcopy(struct)
                            structure.add_edge(node1, node2)
                            inValid = False
                            for s in structures:
                                is_planar, certificate = check_planarity(s)
                                if is_isomorphic(s, structure) or not is_planar:
                                    inValid = True
                                    break
                            if not inValid:
                                structures.append(structure)
            mediaStructures[n + i] = structures
        ms = set()
        for i in mediaStructures:
            ms = ms.union(mediaStructures[i])
        return ms

    def choose_media_structure(self, a, v, graph, recover):
        I, node2index, index2node, I1, I2 = self.get_I(a, self.prog, v, graph, recover)
        k = graph.degree[v]
        n = 2
        scoresN = []
        mediaStructures = {}
        while True:
            mediaStructures[n] = self.get_media_structure(n, k)
            if not mediaStructures[n]:
                n += 1
                continue
            scoreS = {}
            scoreM = {}
            for mediaStructure in mediaStructures[n]:
                neighbors = list(dict(graph.adj[v]).keys())
                neighborNum = max(neighbors)
                D = self.get_D(mediaStructure)
                allocation = self.weight_allocation(graph, v, mediaStructure, I, node2index, index2node, D, neighbors, neighborNum)
                if allocation == {}:
                    continue
                score = 0
                for body1 in allocation:
                    for branch1 in allocation[body1]:
                        for body2 in allocation:
                            for branch2 in allocation[body2]:
                                if branch1 == branch2:
                                    continue
                                score += D[body1 - neighborNum - 1, body2 - neighborNum - 1] * I[node2index[branch1], node2index[branch2]]
                scoreS[mediaStructure] = score
                scoreM[mediaStructure] = allocation
            if scoreS == {}:
                n += 1
                continue
            bestS = list(scoreS.keys())[list(scoreS.values()).index(min(list(scoreS.values())))]
            scoresN.append({scoreS[bestS] : scoreM[bestS]})
            if len(scoresN) >= 2 and list(scoresN[-1].keys())[0] >= list(scoresN[-2].keys())[0]:
                break
            else:
                n += 1
                oldBestS = bestS
        oldBestA = list(scoresN[-2].values())[0]

        for edge in oldBestS.edges:
            oldBestS.edges[edge[0], edge[1]]['weight'] = 1000

        return oldBestS, oldBestA, I1, I2, node2index,index2node

    def weight_allocation(self, graph, v, mediaStructure, I, node2index, index2node, D, neighbors, neighborNum):
        mapping = {}
        for node in mediaStructure.nodes:
            mediaStructure.nodes[node]['branch'] = False
            mapping[node] = node + neighborNum + 1
        mediaStructure = relabel_nodes(mediaStructure, mapping)
        for node in neighbors:
            mediaStructure.add_node(node)
            mediaStructure.nodes[node]['branch'] = True
        

        allocation = {}
        for node in mediaStructure:
            if not mediaStructure.nodes[node]['branch']:
                allocation[node] = []

        full_allocation = {}

        already = []
        while True:
            if allocation == {}:
                return {}
            best_body = list(dict(mediaStructure.degree(list(allocation.keys()))).keys())[list(dict(mediaStructure.degree(list(allocation.keys()))).values()).\
                           index(max(list(dict(mediaStructure.degree(list(allocation.keys()))).values())))]
            best_branch = np.where(np.sum(I, axis=0) == np.max(np.sum(I, axis=0)))[0][0]

            mediaStructure.add_edge(best_body, index2node[best_branch])
            is_planar, c = check_planarity(mediaStructure)

            if max(list(dict(mediaStructure.degree(list(allocation.keys()))).values())) > Prune.degLimit or not is_planar:
                mediaStructure.remove_edge(best_body, index2node[best_branch])
                full_allocation[best_body] = allocation.pop(best_body)
            else:
                allocation[best_body].append(index2node[best_branch])
                already.append(index2node[best_branch])
                neighbors.remove(index2node[best_branch])
                break


        while len(neighbors) > 0:
            I_l = np.zeros(len(I))
            for i in already:
                I_l = I_l + I[:, node2index[i]]
            I_l = list(I_l)
            while True:
                if index2node[I_l.index(max(I_l))] not in already: 
                    next = I_l.index(max(I_l))
                    break
                elif sum(I_l) == 0:
                    next = node2index[neighbors[0]]
                    break
                else:
                    I_l[I_l.index(max(I_l))] = 0
            scores = {}
            full = False
            for n in allocation:
                if mediaStructure.nodes[n]['branch']:
                    continue
                if mediaStructure.degree[n] == Prune.degLimit:
                    full_allocation[n] = allocation.pop(n)
                    full = True
                    break
                score = 0
                iter = deepcopy(allocation)
                if not full_allocation == {}:
                    iter.update(full_allocation)
                for body in iter:
                    for branch in iter[body]:
                        score += D[n - neighborNum - 1, body - neighborNum - 1] * I[node2index[branch], next]
                mediaStructure.add_edge(n, index2node[next])
                is_planar, c = check_planarity(mediaStructure)
                mediaStructure.remove_edge(n, index2node[next])
                if not is_planar:
                    continue
                scores[n] = score
            if len(neighbors) > 0 and len(allocation) == 0:
                return {}
            if full:
                continue
            else:
                best_body = list(scores.keys())[list(scores.values()).index(min(list(scores.values())))]
                mediaStructure.add_edge(best_body, index2node[next])
                allocation[best_body].append(index2node[next])
                already.append(index2node[next])
                neighbors.remove(index2node[next])
        if not full_allocation == {}:
            allocation.update(full_allocation)
        return allocation

    def get_I(self, a, prog, v, graph, recover):
        neighbors = list(dict(graph.adj[v]).keys())
        neighborLen = len(neighbors)
        node2index = dict(zip(neighbors, range(neighborLen)))
        index2node = dict(zip(range(neighborLen), neighbors))
        I1 = np.zeros((neighborLen, neighborLen))
        I2 = np.zeros((neighborLen, neighborLen))
        for node1 in list(dict(graph.adj[v]).keys()):
            for node2 in list(dict(graph.adj[v]).keys()):
                if node1 == node2:
                    continue
                if (node1, node2) in list(recover.keys()):
                    I2[node2index[node1], node2index[node2]] = recover[(node1, node2)]
                state = 0
                for gate in prog:
                    if not set(gate) == set([node1, v]) and not set(gate) == set([node2, v]):
                        continue
                    elif set(gate) == set([node1, v]):
                        if state == 2:
                            I1[node2index[node1], node2index[node2]] += 1
                        state = 1
                    elif set(gate) == set([node2, v]):
                        if state == 1:
                            I1[node2index[node1], node2index[node2]] += 1
                        state = 2
        I1 = I1 / 2
        I2 = I2 + I2.transpose()
        I = a * I1 + (1 - a) * I2
        return I, node2index, index2node, I1, I2

    def get_D(self, mediaStructure):
        mediaLen = len(mediaStructure.nodes)
        D = np.zeros((mediaLen, mediaLen))
        for node1 in range(mediaLen):
            for node2 in range(mediaLen):
                if node1 == node2 or D[node1, node2] > 0:
                    continue
                D[node1, node2] = shortest_path_length(mediaStructure, node1, node2)
        return D