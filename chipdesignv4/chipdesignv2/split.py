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
from prune import Prune
from get_alg import Get_alg
from collections import Iterable
from copy import deepcopy


class Split():
    def __init__(self, g, prog, mat, vertexRanking, show=False):
        self.show = show
        self.graph = deepcopy(g)
        self.prog = prog
        self.mat = mat
        self.vertexRanking = vertexRanking
        self.graph, self.layout = self.get_graph()
        if self.show:
            plt.ion()
            nx.draw(self.graph, with_labels=True)
            plt.draw()
            plt.pause(5)
            plt.close()
        

    def get_graph(self):
        for n in range(min(8, len(self.vertexRanking)), len(self.vertexRanking) + 1):
            mediaStructures = self.get_media_structure(n, len(self.vertexRanking) - n)
            if mediaStructures:
                allocationMethod = self.choose_media_structure(n, mediaStructures)
                if not allocationMethod:
                    continue
                else:
                    break
        allocationMethod = self.recover(allocationMethod[0], allocationMethod[2])
        layout = {}
        for node in allocationMethod.nodes:
            layout[node] = [node]
        return allocationMethod, layout

    def recover(self, allocation, recoverEdge):
        edges  = {}
        for edge in recoverEdge:
            edges[edge] = self.graph.edges[edge]['weight']
        recoverEdge = sorted(edges.items(),key = lambda x : x[1], reverse = True)
        for edge in recoverEdge:
            allocation.add_edge(edge[0][0], edge[0][1])
            if allocation.degree[edge[0][0]] > Prune.degLimit or allocation.degree[edge[0][1]] > Prune.degLimit or not check_planarity(allocation)[0]:
                allocation.remove_edge(edge[0][0], edge[0][1])
            recoverEdge.remove(recoverEdge[0])
        return allocation


    def get_media_structure(self, n, k):
        mediaStructures = {}
        simpliestStructure = nx.Graph()
        for i in range(n - 1):
            simpliestStructure.add_edge(i, i + 1, weight=1)
        mediaStructures[n - 1] = [simpliestStructure]
        if (Prune.degLimit - 1) * n - k < 0:
            return False
        for i in range(1, min((Prune.degLimit - 1) * n - k + 2, int(n * (n - 1) / 2) - n + 2)):
            structures = []
            for struct in mediaStructures[n + i - 2]:
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
            mediaStructures[n + i - 1] = structures
        ms = set()
        for i in mediaStructures:
            ms = ms.union(mediaStructures[i])
        return ms

    def choose_media_structure(self, n, mediaStructures):
        I, node2index, index2node, I1, I2 = self.get_I(n)
        scoreS = {}
        for mediaStructure in mediaStructures:
            neighbors = list(self.vertexRanking.values())[n:]
            neighborNum = len(self.vertexRanking) - n
            D = self.get_D(mediaStructure)
            allocation = self.weight_allocation(mediaStructure, I, node2index, index2node, D, n)
            if not allocation:
                continue
            mapping = allocation[1]
            invMap = allocation[2]
            mediaStructure = allocation[0]
            allocation = allocation[3]
            score = 0
            for node1 in mediaStructure:
                for node2 in mediaStructure:
                    if node1 == node2 or not mediaStructure.nodes[node1]['branch'] or not mediaStructure.nodes[node2]['branch']:
                        continue
                    score += D[invMap[allocation[node1]], invMap[allocation[node2]]] * I[node2index[node1], node2index[node2]]
            scoreS[mediaStructure] = score
        if len(scoreS) > 0:
            bestS = list(scoreS.keys())[list(scoreS.values()).index(min(list(scoreS.values())))]
            recover = set(self.graph.edges).difference(set(bestS.edges))
        else:
            return False
        return bestS, scoreS[bestS], recover

    def weight_allocation(self, mediaStructure, I, node2index, index2node, D, n):
        neighbors = list(self.vertexRanking.values())[n:]
        mapping = {}
        degreeSort = sorted(dict(mediaStructure.degree()).items(),key = lambda x : x[1], reverse = True)
        i = 1
        for node in degreeSort:
            mediaStructure.nodes[node[0]]['branch'] = False
            mapping[node[0]] = self.vertexRanking[i]
            i += 1
        invMap = dict(zip(mapping.values(), mapping.keys()))
        mediaStructure = relabel_nodes(mediaStructure, mapping)
        for node in neighbors:
            mediaStructure.add_node(node)
            mediaStructure.nodes[node]['branch'] = True

        already = {}

        empty = True
        for neighbor in neighbors:
            connectBody = set(dict(self.graph.adj[neighbor]).keys()).intersection(set(list(self.vertexRanking.values())[:n]))
            if empty:
                bodySort = sorted(dict(mediaStructure.degree(connectBody)).items(),key = lambda x : x[1], reverse = True)

            else:
                scores = {}
                for body in connectBody:
                    score = 0
                    for alreadyNeighbor in already:
                        score += D[invMap[body], invMap[already[alreadyNeighbor]]] * I[node2index[neighbor], node2index[alreadyNeighbor]]
                    scores[body] = score
                bodySort = sorted(dict(scores).items(),key = lambda x : x[1])

            while len(bodySort) > 0:
                mediaStructure.add_edge(bodySort[0][0], neighbor)
                if mediaStructure.degree(bodySort[0][0]) > Prune.degLimit or not check_planarity(mediaStructure)[0]:
                    mediaStructure.remove_edge(bodySort[0][0], neighbor)
                    bodySort.remove(bodySort[0])
                else:
                    if empty == True:
                        empty = False
                    already[neighbor] = bodySort[0][0]
                    break
            else:
                return False
                   
        return mediaStructure, mapping ,invMap, already

    def get_I(self, n):
        neighbors = list(self.vertexRanking.values())[n:]
        neighborLen = len(self.vertexRanking) - n
        node2index = dict(zip(neighbors, range(neighborLen)))
        index2node = dict(zip(range(neighborLen), neighbors))
        I1 = np.zeros((neighborLen, neighborLen))
        I2 = np.zeros((neighborLen, neighborLen))
        for node1 in neighbors:
            for node2 in neighbors:
                if node1 == node2:
                    continue
                if dict(self.graph.edges).get((node1, node2), False) and self.graph.edges[(node1, node2)]['weight'] > 0:
                    I2[node2index[node1], node2index[node2]] = self.graph.edges[(node1, node2)]['weight']
                for v in list(self.vertexRanking.values())[:n]:
                    state = 0
                    for gate in self.prog:
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
        a = 0.5#$######################################################################################
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