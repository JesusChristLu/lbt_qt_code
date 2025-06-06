import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

class Lattice():
    def __init__(self, n, mat):
        if n < 4:
            n = 4
        self.height = int(np.ceil(n / np.sqrt(n)))
        self.width = int(np.ceil(n / self.height))
        self.mat = mat
        self.coordinate2node, self.node2coordinate = self.map(self.height, self.width)
        self.triangular_lattice = self.triangular_chip(self.height, self.width)
        self.check_box_lattice = self.check_box_chip(self.height, self.width)
        self.liChip, self.liLayout = self.liGuShu()

    def triangular_chip(self, h, w):
        graph = nx.triangular_lattice_graph(h, w * 2)
        graph = nx.relabel.relabel_nodes(graph, self.coordinate2node)
        graph = nx.subgraph(graph, self.coordinate2node.values())
        return graph

    def check_box_chip(self, h, w):
        graph = nx.grid_2d_graph(h, w)
        for i in range(h - 1):
            for j in range(w // 2):
                if i % 2:
                    if 2 * (j + 1) > w - 1:
                        break
                    edge1 = [(i, 2 * j + 1), (i + 1, 2 * (j + 1))]
                    edge2 = [(i, 2 * (j + 1)), (i + 1, 2 * j + 1)]
                else:
                    if 2 * j + 1 > w - 1:
                        break
                    edge1 = [(i, 2 * j), (i + 1, 2 * j + 1)]
                    edge2 = [(i, 2 * j + 1), (i + 1, 2 * j)]
                graph.add_edges_from([edge1, edge2])
        graph = nx.relabel.relabel_nodes(graph, self.coordinate2node)
        return graph
    
    def map(self, h, w):
        c2n = dict([((i, j), h * i + j) for i in range(h) for j in range(w)])
        n2c = dict(zip(list(c2n.values()), list(c2n.keys())))
        return c2n, n2c

    def liGuShu(self):
        rawLattice = nx.grid_2d_graph(self.height, self.width)
        layout = self.li_layout(rawLattice)
        busAddLattice = self.li_4_bus(rawLattice, layout)
        #busAddLattice = nx.subgraph(busAddLattice, layout.keys())
        busAddLattice = nx.relabel.relabel_nodes(busAddLattice, self.coordinate2node)

        mapLayout = {}
        while len(layout) > 0:
            key = list(layout.keys())[0]
            value = layout.pop(key)
            mapLayout[self.coordinate2node[key]] = value
        return busAddLattice, mapLayout

    def li_layout(self, rawLattice):
        halfRawLattice = deepcopy(rawLattice)
        qRank = dict(zip(range(len(self.mat)), list(np.sum(self.mat, axis=0))))
        layout = {}
        while len(qRank) > 0:
            qBiggest = list(qRank.keys())[list(qRank.values()).index(max(list(qRank.values())))]
            qRank.pop(qBiggest)
            if len(layout) == 0:
                layout[int(self.height / 2), int(self.width / 2)] = qBiggest
            else:
                scores = {}
                for Q in set(rawLattice.nodes).difference(set(layout.keys())):
                    score = 0
                    for obsessQ in layout:
                        score += nx.shortest_path_length(rawLattice, source=obsessQ, target=Q) * self.mat[qBiggest, layout[obsessQ]]
                    scores[Q] = score
                bestPosition = list(scores.keys())[list(scores.values()).index(min(scores.values()))]
                layout[bestPosition] = qBiggest

        return layout

    def li_4_bus(self, lattice, layout):
        rawLattice = deepcopy(lattice)
        avaliable = []
        for node in rawLattice.nodes:
            if not node[0] == self.height - 1 and not node[1] == self.width - 1:
                avaliable.append(node)
        bestBox = []
        while len(avaliable) > 0:
            filterWeight = {}
            for square in avaliable:
                filterWeight[square] = 0
                neighbors = [(square[0] - 1, square[1]), (square[0] + 1, square[1]),
                             (square[0], square[1] - 1), (square[0], square[1] + 1)]
                if layout.get(square, False) and layout.get((square[0] + 1, square[1] + 1), False):
                    filterWeight[square] += self.mat[layout[square], layout[(square[0] + 1, square[1] + 1)]]
                if layout.get((square[0], square[1] + 1), False) and layout.get((square[0] + 1, square[1]), False):
                    filterWeight[square] += self.mat[layout[(square[0], square[1] + 1)], layout[(square[0] + 1, square[1])]]
                for neighbor in neighbors:
                    if layout.get(neighbor, False) and layout.get((neighbor[0] + 1, neighbor[1] + 1), False):
                        filterWeight[square] -= self.mat[layout[neighbor], layout[(neighbor[0] + 1, neighbor[1] + 1)]]
                    if layout.get((neighbor[0], neighbor[1] + 1), False) and layout.get((neighbor[0] + 1, neighbor[1]), False):
                        filterWeight[square] -= self.mat[layout[(neighbor[0], neighbor[1] + 1)], layout[(neighbor[0] + 1, neighbor[1])]]
            bestSquare = list(filterWeight.keys())[list(filterWeight.values()).index(max(filterWeight.values()))]
            bestBox.append(bestSquare)
            rawLattice.add_edge(bestSquare, (bestSquare[0] + 1, bestSquare[1] + 1))
            rawLattice.add_edge((bestSquare[0] + 1, bestSquare[1]), (bestSquare[0], bestSquare[1] + 1))
            avaliable.remove(bestSquare)
            neighborBest = [(bestSquare[0] - 1, bestSquare[1]), (bestSquare[0] + 1, bestSquare[1]),
                            (bestSquare[0], bestSquare[1] - 1), (bestSquare[0], bestSquare[1] + 1)]
            for block in neighborBest:
                if block in avaliable:

                    avaliable.remove(block)
        return rawLattice
