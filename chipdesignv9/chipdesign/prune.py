from os import remove
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path_length
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
    degLimit = 6 #############################################################
    def __init__(self, graph, show=False, k=3):
        self.show = show
        self.graph = graph
        self.mediaVertex = self.media_vertex(self.graph, k)
        self.needPrune = True
        if not self.mediaVertex == []:
            self.graph, self.recover = self.prune(self.graph, k)
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

    def prune(self, g, k):
        graph = deepcopy(g)
        recover = {}
        edge_list = list(graph.edges)
        for edge in edge_list:
            if len(set(edge).intersection(set(self.mediaVertex))) == 0:
                weight = graph.edges[edge]['weight']
                graph.remove_edge(edge[0], edge[1])
                if number_connected_components(graph) == 1:
                    recover[edge] = weight
                else:
                    graph.add_weighted_edges_from([(edge[0], edge[1], weight)])
        for node in graph.nodes:
            cut_candidate = {}
            for mv in self.mediaVertex:
                if shortest_path_length(graph, mv, node) == 1:
                    cut_candidate[(mv, node)] = graph.edges[(mv, node)]['weight']
            while len(cut_candidate) > 1:
                remove_edge = list(cut_candidate.keys())[list(cut_candidate.values()).index(min(list(cut_candidate.values())))]
                weight = graph.edges[remove_edge]['weight']
                graph.remove_edge(remove_edge[0], remove_edge[1])
                if number_connected_components(graph) == 1:
                    recover[remove_edge] = weight
                else:
                    graph.add_weighted_edges_from([(remove_edge[0], remove_edge[1], weight)])
                del cut_candidate[list(cut_candidate.keys())[list(cut_candidate.values()).index(min(list(cut_candidate.values())))]]
        while not check_planarity(graph)[0]:
            edge_list = list(graph.edges)
            smallestWeight = 1e50
            for edge in edge_list:
                if len(set(edge).intersection(set(self.mediaVertex))) >= 1:
                    weight = graph.edges[edge]['weight']
                    if weight < smallestWeight:
                        graph.remove_edge(edge[0], edge[1])
                        if number_connected_components(graph) == 1:
                            tempEdge = edge
                            smallestWeight = weight
                        graph.add_weighted_edges_from([(edge[0], edge[1], weight)])
            graph.remove_edge(tempEdge[0], tempEdge[1])

        if self.show:

            node_color = []
            for node in graph.nodes:
                if node in self.mediaVertex:
                    node_color.append('red')
                else:
                    node_color.append('blue') 
            g = deepcopy(graph)
            for edge in g.edges:
                g.edges[edge]['weight'] = 1
            labels_params = {"font_size":15}
            plt.title('after pruning')
            nx.draw(g, **labels_params, with_labels=True, node_color=node_color, font_color='white', node_size = 500, width=2)
            plt.savefig(str(Prune.degLimit) + ' ' + str(k) + ' ' + "after pruning.pdf", dpi = 300)
            plt.show()

        return graph, recover
