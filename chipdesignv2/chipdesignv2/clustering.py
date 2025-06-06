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
from collections import Iterable
from copy import deepcopy

class Clustering():
    def __init__(self, graph, show=False):
        self.show = show
        self.G = self.binary_G(graph)
        self.GMat = np.array(nx.to_numpy_matrix(self.G))
        self.community_detection()
        if self.show:
            self.draw_community()

    def binary_G(self, graph):
        mat = nx.to_numpy_matrix(graph)
        mat = mat / (mat + 1e-50)
        return nx.from_numpy_matrix(mat)

    def community_detection(self):
        if self.show:
            plt.ion()
            plt.title('binary weight')
            nx.draw(self.G, with_labels=True)
            plt.draw()
            plt.pause(5)
            plt.close()
        
        for node in self.G.nodes:
            self.G.nodes[node]['community_rank'] = [0]
        G = deepcopy(self.G)

        bestG = [deepcopy(G)]

        modularity = [0]
        atoms = False
        while not atoms:


            componentsNumber = number_connected_components(G)

            nodesCentrality = betweenness_centrality(G)
            edgesCentrality = edge_betweenness_centrality(G)
            maxEdge = list(edgesCentrality.keys())[list(edgesCentrality.values()).index(max(edgesCentrality.values()))]
            maxEdgeCentrality = max(edgesCentrality.values())
            maxNode = list(nodesCentrality.keys())[list(nodesCentrality.values()).index(max(nodesCentrality.values()))]
            maxNodeCentrality = max(nodesCentrality.values())
            trySplit = False
            split = False
            if maxNodeCentrality > maxEdgeCentrality:
                if maxNode in maxEdge:
                    if abs(nodesCentrality[maxEdge[0]] - nodesCentrality[maxEdge[1]]) > nodesCentrality[maxNode] * 0.5:############
                        paths = dict(nx.all_pairs_shortest_path(G))
                        neighbors = list(dict(G.adj[maxNode]).keys())
                        for i in range(len(neighbors)):
                            neighbors[i] = [neighbors[i]]
                        pairBetweennessMat = np.zeros((len(neighbors), len(neighbors)))
                        for node in paths:
                            for path in paths[node]:
                                for i in range(len(neighbors) - 1):
                                    for j in range(i + 1, len(neighbors)):
                                        if set([neighbors[i][0], maxNode]).intersection(set(paths[node][path])) == set([neighbors[i][0], maxNode]) and \
                                           set([neighbors[j][0], maxNode]).intersection(set(paths[node][path])) == set([neighbors[j][0], maxNode]):
                                            pairBetweennessMat[i, j] += 1
                                            pairBetweennessMat[j, i] += 1
                        pairBetweennessMat /= len(G.nodes) * (len(G.nodes) - 1) / 2
                
                        for k in range(len(neighbors) - 2):
                            pairBetweennessMat += np.eye(len(neighbors)) * 1e50
                            contract = [np.where(pairBetweennessMat == np.min(pairBetweennessMat))[0][0], 
                                        np.where(pairBetweennessMat == np.min(pairBetweennessMat))[1][0]]
                            pairBetweennessMat -= np.eye(len(neighbors)) * 1e50
                            for i in range(len(neighbors) - 1):
                                contract_i = set(neighbors[i])
                                if len(contract_i.intersection(set(neighbors[contract[0]]).union(set(neighbors[contract[1]])))):
                                    for j in range(i + 1, len(neighbors)):
                                        contract_j = set(neighbors[j])
                                        if len(contract_j.intersection(set(neighbors[contract[0]]).union(set(neighbors[contract[1]])))):
                                            neighbors[i] = list(contract_i.union(contract_j))
                                            del neighbors[j] 
                                            pairBetweennessMat[i] = pairBetweennessMat[i] + pairBetweennessMat[j]
                                            pairBetweennessMat[:,i] = pairBetweennessMat[:,i] + pairBetweennessMat[:,j]
                                            pairBetweennessMat = np.delete(pairBetweennessMat, j, axis = 0)
                                            pairBetweennessMat = np.delete(pairBetweennessMat, j, axis = 1)
                                            break
                                    break
                        split_betweenness = np.max(pairBetweennessMat)
                        if split_betweenness > maxEdgeCentrality:
                            split = True
                            if isinstance(maxNode, str):
                                infList = maxNode.split(' ')
                                copyNode = str(infList[0]) + ' ' + str(int(infList[1]) + 1)
                            else:
                                copyNode = str(maxNode) + ' 1'
                            G.add_node(copyNode)
                            for i in neighbors[1]:
                                G.remove_edge(maxNode, i)
                                G.add_edge(i, copyNode)
            if not split:
                G.remove_edge(maxEdge[0], maxEdge[1])

            if number_connected_components(G) > componentsNumber:
                componentsNumber = number_connected_components(G)
                cluster = {}
                community_rank = 0
                for node in G.nodes:
                    node_set = node_connected_component(G, node)
                    if not (node_set in cluster.values()):
                        cluster[community_rank] = node_set
                        G.nodes[node]['community_rank'] = [community_rank]
                        community_rank += 1
                    else:
                        G.nodes[node]['community_rank'] = [list(cluster.keys())[list(cluster.values()).index(node_set)]]
                performance = 0
                for community in cluster:
                    dc = 0
                    for nodeSg in cluster[community]:
                        dc += G.degree[nodeSg]
                    sg = G.subgraph(cluster[community])
                    if len(G.edges) == 0:
                        atoms = True
                        break
                    performance += len(sg.edges) / (len(G.edges)) - (dc / (2 * len(G.edges))) ** 2
                bestG.append(deepcopy(G))
                modularity.append(performance)

        if self.show:
            plt.ion()
            plt.title('modularity')
            plt.plot(range(1, len(modularity) + 1), modularity, marker='o',color='g')
        
            plt.grid()

            plt.draw()
            plt.pause(5)
            plt.close()
        
        for i in modularity:

            if i > 0.6 and modularity.index(i) < 0.5 * len(modularity):

                nodes = deepcopy(list(bestG[modularity.index(i)].nodes))

                for node in nodes:
                    if isinstance(node, str):
                        self.G.nodes[int(node.split()[0])]['community_rank'].append(bestG[modularity.index(i)].nodes[node]['community_rank'][0])
                        continue
                    else:
                        self.G.nodes[node]['community_rank'] = bestG[modularity.index(i)].nodes[node]['community_rank']
                print(i)
                break


    def draw_community(self):
        node_color = []
        for node in self.G.nodes:
            if len(self.G.nodes[node]['community_rank']) == 1:
                node_color.append(10 * (self.G.nodes[node]['community_rank'][0] + 1))
            else:
                node_color.append(0)
        plt.ion()
        plt.title('graph after clustering')
        nx.draw(self.G, with_labels=True, node_color=node_color, font_color='white')###################
        plt.draw()
        plt.pause(5)
        plt.close()
        