import networkx as nx
import  matplotlib.pyplot as plt

nodes = ["rabi","RB","t1","t2","ramesy"]
edges = [("rabi","RB"), ("RB", "t1"), ("t1","t2"), ("t2", "ramesy")]
G = nx.DiGraph()
G.add_nodes_from(nodes)
# G.add_edges_from(edges)

pos = nx.kamada_kawai_layout(G)
# pos = nx.spring_layout(G)
_, ax = plt.subplots(1)
# nx.draw(G, ax=ax, with_labels=True)
nx.draw_networkx_nodes(G,  pos=pos, node_size=300,node_color="#210070", alpha=0.9, ax=ax)
# label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
nx.draw_networkx_labels(G, pos, font_size=10, font_color="w", bbox=None)
nx.draw_networkx_edges(G,pos=pos,  ax=ax, edgelist= edges, edge_color="m", arrowsize=20,connectionstyle="arc3, rad=0.1")
nx.draw_networkx_edges(G,pos=pos,  ax=ax, edgelist= [("t2","t2")], edge_color="m",alpha=0.3, arrowsize=10,connectionstyle="arc3, rad=0.1")
plt.show()

