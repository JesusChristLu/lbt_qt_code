import os
import networkx as nx
import numpy as np
from numpy.lib.shape_base import split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
#from collections import Iterable
from copy import deepcopy


class Split():
    def __init__(self, g, out_degree, prog, recover, show=False, a = 0.5):
        self.show = show
        self.graph = deepcopy(g)
        self.prog = prog
        self.recover = recover
        self.splitVertex = self.get_split_vertex()
        self.graph, self.layout = self.split_vertex(a, self.splitVertex, recover, out_degree)

    def get_split_vertex(self):
        splitVertex = list(np.where(np.array(list(dict(self.graph.degree(self.graph.nodes)).values())) > Prune.degLimit)[0])
        return splitVertex

    def split_vertex(self, a, splitVertexes, recover, out_degree):
        graph = deepcopy(self.graph)
        mediaStructures = []
        layout = {}
        for node in graph.nodes:
            if node not in splitVertexes:
                layout[node] = [node]
        for v in splitVertexes:
            print('spliting the vertex ', v, ' in ', splitVertexes)
            mediaStructure, allocation, invMap = self.choose_media_structure(a, v, graph, recover, out_degree)
            if mediaStructure == {}:
                return {}, {}
            mediaStructure = relabel_nodes(mediaStructure, invMap)
            mediaStructures += list(mediaStructure.nodes)
            graph.remove_node(v)
            graph = compose(graph, mediaStructure)
            layout[v] = list(mediaStructure.nodes)
            for body in list(allocation.keys()):
                for branch in allocation[body]:
                    graph.add_edge(body, branch)

        if self.show:
            node_color = []
            for node in graph.nodes:
                if node in mediaStructures:
                    node_color.append('red')
                else:
                    node_color.append('blue')

            labels_params = {"font_size":15}
            plt.title('after split')
            for edge in graph.edges():
                graph.edges[edge]['weight'] = 1
            nx.draw(graph, **labels_params, with_labels=True, node_color=node_color, font_color='white', node_size = 500, width=2)
            plt.savefig(str(Prune.degLimit) + ' ' + str(out_degree) + ' ' + "after split.pdf", dpi = 300)
            plt.show()

        while len(self.recover) > 0:
            add_edge = list(self.recover.keys())[list(self.recover.values()).index(max(list(self.recover.values())))]
            graph.add_edge(add_edge[0], add_edge[1])
            
            is_planar = check_planarity(graph)[0]

            is_break = False
            if max(list(dict(graph.degree).values())) > Prune.degLimit:
                is_break = True

            if not is_planar or is_break:
                graph.remove_edge(add_edge[0], add_edge[1])
            else:
                graph.edges[add_edge]['weight'] = self.recover[add_edge]
            self.recover.pop(add_edge)

        if self.show:
            pos = nx.planar_layout(graph)
            labels_params = {"font_size":15}
            # nx.draw(graph, **labels_params, with_labels=True, node_color=node_color, font_color='white', node_size = 1000, width=5)
            plt.title('after recover')
            nx.draw(graph, **labels_params, pos=pos, with_labels=True, node_color=node_color, font_color='white', node_size = 500, width=2)
            plt.savefig(str(Prune.degLimit) + ' ' + str(out_degree) + ' ' + "after recover.pdf", dpi = 300)
            plt.show()
        return graph, layout

    def get_media_structure(self, k):
        path = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\media structures ' + str(Prune.degLimit) + '\\'
        path_list = os.listdir(path)
        ms = []
        for file in path_list:
            if int(file.split(' ')[1]) >= k and int(file.split(' ')[1]) < k + 2:
                with open(os.path.join(path, file), 'r') as fp:
                    data = fp.read()
                    data = data.split('\n')[:-1]
                    mediaStructure = np.zeros((len(data), len(data)))
                    for raw in range(len(data)):
                        data_raw = data[raw].split(' ')[:-1]
                        for column in range(len(data)):
                            mediaStructure[raw, column] = int(data_raw[column])
                    ms.append(nx.from_numpy_array(mediaStructure))
        return ms

    def choose_media_structure(self, a, v, graph, recover, out_degree):
        I, node2index, index2node = self.get_I(a, self.prog, v, graph, recover, out_degree)
        k = graph.degree[v]
        mediaStructures = self.get_media_structure(k)
        print('there are', len(mediaStructures), 'candidate structures')
        if len(mediaStructures) == 0:
            return {}, {}, {}
        scoreDict = {}
        alloDict = {}
        for mediaStructure in mediaStructures:
            neighbors = list(dict(graph.adj[v]).keys())
            neighborNum = max(neighbors)
            D = self.get_D(mediaStructure)
            allocation, invMap = self.weight_allocation(graph, v, mediaStructure, I, node2index, index2node, D, neighbors, neighborNum)
            if allocation == {}:
                continue
            score = 0
            for body1 in allocation:
                for branch1 in allocation[body1]:
                    for body2 in allocation:
                        for branch2 in allocation[body2]:
                            if branch1 == branch2:
                                continue
                            score += D[invMap[body1], invMap[body2]] * I[node2index[branch1], node2index[branch2]]
            scoreDict[mediaStructure] = score
            alloDict[mediaStructure] = [allocation, dict(zip(invMap.values(), invMap.keys()))]
            print('after allocation: ', score)
        bestS = list(scoreDict.keys())[list(scoreDict.values()).index(min(list(scoreDict.values())))]
        return bestS, alloDict[bestS][0], alloDict[bestS][1]

    def weight_allocation(self, graph, v, mediaStructure, I, node2index, index2node, D, neighbors, neighborNum):
        mapping = {}

        for node in mediaStructure.nodes:
            if node == 0:
                mapping[node] = v
            else:
                mapping[node] = node + max(list(dict(graph.nodes).keys()))
            mediaStructure.nodes[node]['branch'] = 'trunk'
        invMap = dict(zip(mapping.values(), mapping.keys()))

        mediaStructure = relabel_nodes(mediaStructure, mapping)

        tryGraph = deepcopy(graph)
        tryGraph.remove_node(v)

        mediaStructure = nx.compose(mediaStructure, tryGraph)

        for node in mediaStructure.nodes:
            if node in neighbors:
                mediaStructure.nodes[node]['branch'] = 'branch'
            elif not node in mapping.values():
                mediaStructure.nodes[node]['branch'] = 'other'

        allocation = {}
        for node in mediaStructure:
            if mediaStructure.nodes[node]['branch'] == 'trunk':
                allocation[node] = []

        full_allocation = {}

        already = []
        while True:
            if allocation == {}:
                return {}, {}
            best_body = list(dict(mediaStructure.degree(list(allocation.keys()))).keys())[list(dict(mediaStructure.degree(list(allocation.keys()))).values()).\
                           index(max(list(dict(mediaStructure.degree(list(allocation.keys()))).values())))]
            best_branch = np.where(np.sum(I, axis=0) == np.max(np.sum(I, axis=0)))[0][0]

            mediaStructure.add_edge(best_body, index2node[best_branch])
            is_planar = check_planarity(mediaStructure)[0]

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
                if not mediaStructure.nodes[n]['branch'] == 'trunk':
                    continue
                if mediaStructure.degree[n] >= Prune.degLimit:
                    full_allocation[n] = allocation.pop(n)
                    full = True
                    break

                mediaStructure.add_edge(n, index2node[next])
                is_planar = check_planarity(mediaStructure)[0]
                mediaStructure.remove_edge(n, index2node[next])
                if not is_planar:
                    continue

                score = 0
                iter = deepcopy(allocation)
                if not full_allocation == {}:
                    iter.update(full_allocation)
                for body in iter:
                    for branch in iter[body]:
                        score += D[invMap[n], invMap[body]] * I[node2index[branch], next]

                scores[n] = score
            if (len(neighbors) > 0 and len(allocation) == 0) or not is_planar:
                return {}, {}
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
        return allocation, invMap

    def get_I(self, a, prog, v, graph, recover, out_degree):
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

        if self.show:
            sns.heatmap(I, annot = True, cmap='Reds', annot_kws={'size':10})
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title('matrix I')
            plt.savefig(str(Prune.degLimit) + ' ' + str(out_degree) + ' ' + "matrix I.pdf", dpi = 300)
            plt.show()
        return I, node2index, index2node

    def get_D(self, mediaStructure):
        mediaLen = len(mediaStructure.nodes)
        D = np.zeros((mediaLen, mediaLen))
        for node1 in range(mediaLen):
            for node2 in range(mediaLen):
                if node1 == node2 or D[node1, node2] > 0:
                    continue
                D[node1, node2] = shortest_path_length(mediaStructure, node1, node2)
        return D