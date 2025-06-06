import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class TriangleChip():
    def __init__(self, n):
        self.chip = self.get_chip(n)

    def get_chip(self, n):
        if n < 4:
            n = 4
        height = int(np.ceil(n / np.sqrt(n)))
        width = int(np.ceil(n / height))
        graph = nx.Graph()
        graph.add_nodes_from(range(width * height))
        for h in range(height):
            w_edges = [(h * height + w1, h * height + (w1 + 1)) for w1 in range(width - 1)]
            graph.add_edges_from(w_edges)
            if h > 0:
                if h % 2:
                    for w in range(width):
                        graph.add_edge(h * height + w, (h - 1) * height + w)
                        if w > 0:
                            graph.add_edge(h * height + w, (h - 1) * height + w - 1)
                else:
                    for w in range(width):
                        graph.add_edge(h * height + w, (h - 1) * height + w)
                        if not w == width - 1:
                            graph.add_edge(h * height + w, (h - 1) * height + w + 1)

        plt.ion()
        nx.draw(graph)
        plt.draw()
        plt.pause(10)
        plt.close()

        return graph
        
chip = TriangleChip(16)
