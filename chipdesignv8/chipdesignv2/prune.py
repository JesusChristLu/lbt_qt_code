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
#from collections import Iterable
from copy import deepcopy

class Prune():
    degLimit = 4 #############################################################
    def __init__(self, graph, show=False, k=3):
        self.show = show
        self.graph = graph
        self.mediaVertex = self.media_vertex(self.graph, k)
        self.needPrune = True
        if not self.mediaVertex == []:
            self.graph, self.recover = self.prune(self.graph)
        else:
            self.needPrune = False

    def media_vertex(self, g, k = 3):
        graph = deepcopy(g)
        bigDeg = np.max(np.array(list(graph.degree()) + [(0, 0)]), axis = 0)[1]
        if bigDeg <= self.degLimit and check_planarity(graph)[0]:
            return []
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
                #rule = dict(zip(sort, range(len(sort))))
                #sortedImportance = []
                for i in sort:
                    vertexRanking[len(vertexRanking) + 1] = importance[i]

            else:
                vertexRanking[len(vertexRanking) + 1] = importance[0]
            recordVertex += importance
        

        return list(vertexRanking.values())[:k]

    def prune(self, g):
        graph = deepcopy(g)
        recover = {}
        while True:
            for edge in graph.edges:
                if len(set(edge).intersection(set(self.mediaVertex))) == 0:
                    recover[edge] = graph.edges[edge]['weight']
                    graph.remove_edge(edge[0], edge[1])
                # else: ##############################################
                #     graph.edges[edge]['weight'] = 1 ############################################
            if check_planarity(graph, False)[0]:
                break
            else:
                for edge in list(graph.edges.keys()):
                    if len(set(edge).intersection(set(self.mediaVertex))) == 1 and \
                       self.graph.degree[edge[0]] > 1 and \
                       self.graph.degree[edge[1]] > 1:
                        tempEdge = edge
                        break
                smallestWeight = dict(graph.edges)[tempEdge]['weight']
                for edge in graph.edges:
                    if len(set(edge).intersection(set(self.mediaVertex))) == 1 and \
                       dict(graph.edges)[edge]['weight'] < smallestWeight and \
                       self.graph.degree[edge[0]] > 1 and \
                       self.graph.degree[edge[1]] > 1:
                        tempEdge = edge
                        smallestWeight = dict(graph.edges)[edge]['weight']
                recover[tempEdge] = graph.edges[tempEdge]['weight']
                graph.remove_edge(tempEdge[0], tempEdge[1])
        if self.show:

            node_color = []
            for node in graph.nodes:
                if node in self.mediaVertex:
                    node_color.append('red')
                else:
                    node_color.append('blue') 
            plt.ion()
            g = deepcopy(graph)
            for edge in g.edges:
                g.edges[edge]['weight'] = 1
            labels_params = {"font_size":20}
            nx.draw(g, **labels_params, with_labels=True, node_color=node_color, font_color='white', node_size = 1000, width=5)
            plt.savefig("afterpruning.pdf", dpi = 300)
            plt.pause(5)
            plt.close()

        return graph, recover
