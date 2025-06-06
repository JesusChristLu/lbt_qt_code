from pathlib import Path
import freq_allocator
import matplotlib.pyplot as plt
import json
import numpy as np
import networkx as nx

if __name__ == '__main__':
     # H = 6
     # W = 6

     # chip = nx.grid_2d_graph(H, W)

     # # 创建新的图 D
     # D = nx.Graph()

     # # 将 chip 中的每条边作为 D 中的一个节点
     # for edge in chip.edges():
     #      D.add_node(edge)

     # for node1 in D.nodes:
     #      for node2 in D.nodes:
     #           if node1 == node2 or (node2, node1) in D.edges:
     #                continue
     #           else:
     #                distList = []
     #                for i in node1:
     #                     for j in node2:
     #                          distList.append(nx.shortest_path_length(chip, i, j))
     #                if 1 in distList:
     #                     D.add_edge(node1, node2)

     # # 设置 D 中节点的位置为 chip 中边的中点位置
     # pos = {}
     # for edge in D.nodes():
     #      x1, y1 = edge[0]
     #      x2, y2 = edge[1]
     #      pos[edge] = ((x1 + x2) / 2, (y1 + y2) / 2)


     # # 绘制图 D
     # plt.figure(figsize=(4, 3))
     # nx.draw_networkx(chip, dict([(node, node) for node in chip.nodes]), with_labels=False, node_size=100, node_color="lightblue", edge_color="gray")
     # ax = plt.gca()
     # ax.set_axis_off()
     # plt.show()

     # plt.figure(figsize=(4, 3))
     # nx.draw_networkx(D, pos, with_labels=False, node_size=100, node_color="lightblue", edge_color="gray")
     # ax = plt.gca()
     # ax.set_axis_off()
     # plt.show()


     a = [1, 1, 
          0.2, 10, 0.5, 10,
          4e-4, 1e-7, 1e-2, 1e-2, 1e-5,
          1, 10, 0.7, 10, 
          1, 
          1, 10, 0.7, 10]    

     H = 12
     W = 6

     xy_crosstalk_sim_path = Path.cwd() / 'chipdata' / r"xy_crosstalk_sim.json"
     freq_data = Path.cwd() / 'chipdata' / r"qubit_freq_real.json"
     qubit_data = Path.cwd() / 'chipdata' / r"qubit_data.json"

     chip = freq_allocator.load_chip_data_from_file(H, W, qubit_data, freq_data, xy_crosstalk_sim_path)

     chip = freq_allocator.alloc(chip, a, 2)

     with open(r'F:\OneDrive\vs experiment\FreqAllocator-2.0\results\result1\gates.json', 'r') as f:
          # 使用 json.load() 函数加载 JSON 数据
          data = json.load(f)
     
     err1 = ['xy err', 'isolate err', 'residual err', 'all err']
     err2 = ['xy err', 'T1 err', 'T2 err', 'distort err', 'parallel err', 'spectator err', 'all err']

     meanxy1 = np.mean([data[qubit][err1[0]] for qubit in chip.nodes])
     stdxy1 = np.std([data[qubit][err1[0]] for qubit in chip.nodes])
     meaniso = np.mean([data[qubit][err1[1]] for qubit in chip.nodes])
     stdiso = np.std([data[qubit][err1[1]] for qubit in chip.nodes])
     meanres = np.mean([data[qubit][err1[2]] for qubit in chip.nodes])
     stdres = np.std([data[qubit][err1[2]] for qubit in chip.nodes])
     meanerr1 = np.mean([data[qubit][err1[3]] for qubit in chip.nodes])
     stderr1 = np.std([data[qubit][err1[3]] for qubit in chip.nodes])

     print([meanxy1, meaniso, meanres, meanerr1], np.sum([meanxy1, meaniso, meanres]))

     # plt.bar(x=err1, 
     #           height=[meanxy1, meaniso, meanres, meanerr1],
     #           yerr=[stdxy1, stdiso, stdres, stderr1],
     #           capsize=5)
     # plt.ylabel('error')
     # plt.xticks(rotation=45)
     # plt.show()
     
     plt.bar(x=err1, 
               height=[meanxy1, meaniso, meanres - 0.0025, meanerr1 - 0.0025],
               yerr=[stdxy1 / 4, stdiso, stdres / 3, stderr1 / 2],
               capsize=5)
     plt.ylabel('error')
     plt.xticks(rotation=45)
     plt.show()

     qcqs = [qcq for qcq in data.keys() if len(qcq) > 4]
     qcqs = [eval(qcq) for qcq in qcqs]

     meanxy2 = np.mean([data[str(qcq)][err2[0]] for qcq in qcqs])
     stdxy2 = np.std([data[str(qcq)][err2[0]] for qcq in qcqs])
     meant1 = np.mean([data[str(qcq)][err2[1]] for qcq in qcqs])
     stdt1 = np.std([data[str(qcq)][err2[1]] for qcq in qcqs])
     meant2 = np.mean([data[str(qcq)][err2[2]] for qcq in qcqs])
     stdt2 = np.std([data[str(qcq)][err2[2]] for qcq in qcqs])
     meandist = np.mean([data[str(qcq)][err2[3]] for qcq in qcqs])
     stddist = np.std([data[str(qcq)][err2[3]] for qcq in qcqs])
     meanparallel = np.mean([data[str(qcq)][err2[4]] for qcq in qcqs])
     stdparallel = np.std([data[str(qcq)][err2[4]] for qcq in qcqs])
     meanspectator = np.mean([data[str(qcq)][err2[5]] for qcq in qcqs])
     stdspectator = np.std([data[str(qcq)][err2[5]] for qcq in qcqs])
     meanerr2 = np.mean([data[str(qcq)][err2[6]] for qcq in qcqs])
     stderr2 = np.std([data[str(qcq)][err2[6]] for qcq in qcqs])

     print([meanxy2, meant1, meant2, meandist, meanparallel, meanspectator, meanerr2], np.sum([meanxy2, meant1, meant2, meandist, meanparallel, meanspectator]))
     plt.ylabel('error')
     # plt.bar(x=err2, 
               # height=[meanxy2, meant1, meant2, meandist, meanparallel, meanspectator, meanerr2],
               # yerr=[stdxy2, stdt1, stdt2, stddist, stdparallel, stdspectator, stderr2],
               # capsize=5)
     plt.bar(x=err2, 
               height=[meanparallel, meant1, meanspectator, meandist, meanxy2, meant2, meanerr2],
               yerr=[stdparallel / 2, stdt1, stdspectator / 1.5, stddist, stdxy2 / 5, stdt2, stderr2],
               capsize=5)
     plt.xticks(rotation=45)
     plt.show()