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
from collections import Iterable
from copy import deepcopy

class Prune():
    degLimit = 6
    def __init__(self, graph):
        self.vertexRanking, self.needPrune = self.vertex_ranking(graph)

    def vertex_ranking(self, g):
        graph = deepcopy(g)
        bigDeg = np.max(np.array(list(graph.degree()) + [(0, 0)]), axis = 0)[1]
        if bigDeg <= self.degLimit and check_planarity(graph)[0]:
            return [], False
        vertexRanking = {}
        recordVertex = []
        degreeDict = dict(graph.degree())
        while not len(vertexRanking) == len(graph.nodes):
            importance = []
            for vertex in graph.nodes:
                if (graph.degree[vertex]) == max(list(degreeDict.values())):
                    importance.append(vertex)
            for vertex in importance:
                degreeDict.pop(vertex)
            weights = {}
            if len(importance) > 1:
                weightC = {}
                weightDict = {}
                for mV in importance:
                    weight = 0
                    weights = []
                    adj_W = list(dict(graph.adj[mV]).values())
                    for w in adj_W:
                        weight += w['weight']
                        weights.append(w['weight'])
                    weightDict[mV] = weight
                    weightC[mV] = np.std(np.array(weights))

                sort = np.lexsort((tuple(weightC.values()), tuple(weightDict.values())))[::-1]
                rule = dict(zip(sort, range(len(sort))))
                sortedImportance = []
                for i in sort:
                    vertexRanking[len(vertexRanking) + 1] = importance[i]

            else:
                vertexRanking[len(vertexRanking) + 1] = importance[0]
            recordVertex += importance
        return vertexRanking, True