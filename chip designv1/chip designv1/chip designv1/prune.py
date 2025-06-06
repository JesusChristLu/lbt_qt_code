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
from clustering import Clustering
from collections import Iterable
from copy import deepcopy

class Prune():
    biggestDeg = 6
    def __init__(self, graph, mat):
        self.graph = graph
        self.get_weight(mat)
        self.mediaVertex = self.media_vertex(self.graph)
        self.graph, self.recover = self.prune(self.graph)

    def get_weight(self, mat):
        for edge in self.graph.edges:
            self.graph.edges[edge]['weight'] = mat[edge[0], edge[1]]

    def media_vertex(self, g):
        graph = deepcopy(g)
        cluster = {}
        mediaVertex = []
        for node in graph.nodes:
            if len(graph.nodes[node]['community_rank']) > 1:# and graph.degree[node] > self.biggestDeg:
                mediaVertex.append(node)
            for i in set(graph.nodes[node]['community_rank']).intersection(set(cluster.keys())):
                cluster[i].append(node)
            for i in set(graph.nodes[node]['community_rank']).difference(set(cluster.keys())):
                cluster[i] = [node]

        for i in cluster:
            mediaVertexInCluster = []
            bigDeg = np.max(np.array(list(graph.degree(cluster[i])) + [(0, 0)]), axis = 0)[1]
            #if bigDeg > self.biggestDeg:
            for vertex in cluster[i]:
                if not (vertex in mediaVertex):# and (graph.degree[vertex]) > self.biggestDeg:
                    mediaVertexInCluster.append(vertex)
            weights = {}
            if len(mediaVertexInCluster) > 1:
                for mV in mediaVertexInCluster:
                    weight = 0
                    adj_W = list(dict(graph.adj[mV]).values())
                    for w in adj_W:
                        weight += w['weight']
                    weights[mV] = weight
                for mV in weights:
                    if weights[mV] < max(list(weights.values())):
                        mediaVertexInCluster.remove(mV)
            weightsC = {}
            if len(mediaVertexInCluster) > 1:
                for mV in mediaVertexInCluster:
                    weights = []
                    adj_W = list(dict(graph.adj[mV]).values())
                    for w in adj_W:
                        weights.append(w['weight'])
                    weightsC[mV] = np.std(np.array(weights))
                for mV in weightsC:
                    if weightsC[mV] > min(list(weightsC.values())):
                        mediaVertexInCluster.remove(mV)

            if len(mediaVertexInCluster) > 1:
                mediaVertexInCluster = [mediaVertexInCluster[0]]
            mediaVertex += mediaVertexInCluster
        return mediaVertex 

    def prune(self, g):
        graph = deepcopy(g)
        recover = {}
        
        for edge in graph.edges:
            if len(set(edge).intersection(set(self.mediaVertex))) == 0:
                recover[edge] = graph.edges[edge]['weight']
                graph.remove_edge(edge[0], edge[1])


        node_color = []
        for node in graph.nodes:
            if node in self.mediaVertex:
                node_color.append('red')
            else:
                node_color.append('green')
        is_planar, certificate = check_planarity(graph, True)

        while not is_planar:
            notPlanarSubgraph = nx.Graph()
            notPlanarSubgraph.add_edges_from(certificate.edges()) 

            for edge in notPlanarSubgraph.edges:
                notPlanarSubgraph.edges[edge]['weight'] = graph.edges[edge]['weight']
            edgesCentrality = edge_betweenness_centrality(notPlanarSubgraph)
            
            while True:
                edge = list(edgesCentrality.keys())[list(list(edgesCentrality.values())).index(min(list(edgesCentrality.values())))]
                if not (edge[0] in self.mediaVertex and edge[1] in self.mediaVertex):
                    break
                edgesCentrality.pop(edge)
            graph.remove_edge(edge[0], edge[1])
            recover.append(edge)

            node_color = []
            for node in graph.nodes:
                if node in self.mediaVertex:
                    node_color.append(10)
                else:
                    node_color.append(0)

            is_planar, certificate = check_planarity(graph, True)
        
        plt.ion()
        plt.title('graph after first pruning')
        nx.draw(graph, with_labels=True, node_color=node_color, font_color='white')###################
        plt.pause(5)
        plt.close()

        return graph, recover