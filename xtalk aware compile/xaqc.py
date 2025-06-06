import datetime
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from multiprocessing import Pool, cpu_count
from qiskit import QuantumCircuit, transpile, assembler
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.circuit.library import EfficientSU2
from qiskit.visualization import plot_histogram, circuit_drawer, plot_state_city
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager, passes, CouplingMap
from qiskit.transpiler.passes import CommutativeCancellation, SabreSwap, BasicSwap, LookaheadSwap, StochasticSwap, BasisTranslator, TrivialLayout
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.providers.models.backendproperties import Nduv
from qiskit.providers.models.backendproperties import Gate
from scipy.optimize import minimize
from xtalk_adaptive_schedule import CrosstalkAdaptiveSchedule
from xtalk_noise_adaptive_layout import NoiseAdaptiveLayout
from xaqc_map import XaqcSwap
import ibm
import random
import time
import os

def chip_generate(size):
    date = datetime.date(2023, 4, 1)
    g_list = []
    b_list = []
    graph = nx.grid_2d_graph(size[0], size[1])
    relable_map = dict(zip(graph.nodes, range(size[0] * size[1])))
    # print(relable_map)
    graph = nx.relabel.relabel_nodes(graph, relable_map)

    xtalkProp = dict()
    
    for edge in graph.edges:
        graph.edges[edge]['cx err'] = max(1e-4, 0.05 + 0.01 * np.random.randn())
        graph.edges[edge]['cx dur'] = 50
        # cx_list = [Nduv(self.date, 'gate_error', 'm', graph.edges[edge]['cx err'] * 1e3), Nduv(self.date, 'gate_length', 'n', graph.edges[edge]['cx dur'])]
        cx_list = [Nduv(date, 'gate_error', 'k', graph.edges[edge]['cx err'] * 1e-3), 
                    Nduv(date, 'gate_length', 'n', graph.edges[edge]['cx dur'])]
        g_list.append(Gate(list(edge), 'cx', cx_list))
        xtalkProp[edge] = dict()
        for xtalkedge in graph.edges:
            if xtalkedge == edge:
                continue
            distances = []
            for p in edge:
                for q in xtalkedge:
                    distances.append(nx.dijkstra_path_length(graph, p, q))
            if 1 in distances and not(0 in distances):
                nbNum = 0
                for i in distances:
                    if i == 1:
                        nbNum += 1
                xtalkProp[edge][xtalkedge] = 0.01 + np.random.random() * 0.001        

    for node in graph.nodes:
        q = []
        graph.nodes[node]['read err'] = max(1e-4, 0.05 + 0.01 * np.random.randn())
        # q.append(Nduv(date, 'readout_error', 'm', graph.nodes[node]['read err'] * 1e3))
        q.append(Nduv(date, 'readout_error', 'k', graph.nodes[node]['read err'] * 1e-3))
        graph.nodes[node]['read dur'] = 0.1
        q.append(Nduv(date, 'readout_length', 'u', graph.nodes[node]['read dur']))            
        t1 = max(10, 80 + 20 * np.random.randn())
        t2 = min(max(10, 80 + 30 * np.random.randn()), 2 * t1)
        graph.nodes[node]['t1'] = t1
        q.append(Nduv(date, 'T1', 'u', t1))
        graph.nodes[node]['t2'] = t2
        q.append(Nduv(date, 'T2', 'u', t2))

        b_list.append(q)

        u1_list = []
        u3_list = []
        graph.nodes[node]['z dur'] = 1e-10
        u1_list.append(Nduv(date, 'gate_length', 'n', graph.nodes[node]['z dur']))
        graph.nodes[node]['x dur'] = 30
        u3_list.append(Nduv(date, 'gate_length', 'n', graph.nodes[node]['x dur']))

        graph.nodes[node]['z err'] = 1e-10
        u1_list.append(Nduv(date, 'gate_error', 'k', graph.nodes[node]['z err'] * 1e-3))
        graph.nodes[node]['x err'] = max(1e-5, 0.005 + 0.001 * np.random.randn())
        u3_list.append(Nduv(date, 'gate_error', 'k', graph.nodes[node]['x err'] * 1e-3))
        g_list.append(Gate([int(node)], 'z', u1_list))
        g_list.append(Gate([int(node)], 'x', u3_list))

        u1_list = []
        u2_list = []
        u3_list = []
        graph.nodes[node]['u1 dur'] = 1e-10
        u1_list.append(Nduv(date, 'gate_length', 'n', graph.nodes[node]['u1 dur']))
        graph.nodes[node]['u2 dur'] = 30
        u2_list.append(Nduv(date, 'gate_length', 'n', graph.nodes[node]['u2 dur']))
        graph.nodes[node]['u3 dur'] = 30
        u3_list.append(Nduv(date, 'gate_length', 'n', graph.nodes[node]['u3 dur']))

        graph.nodes[node]['u1 err'] = 1e-10
        u1_list.append(Nduv(date, 'gate_error', 'k', graph.nodes[node]['u1 err'] * 1e-3))
        graph.nodes[node]['u2 err'] = max(1e-5, 0.005 + 0.001 * np.random.randn())
        u2_list.append(Nduv(date, 'gate_error', 'k', graph.nodes[node]['u2 err'] * 1e-3))
        graph.nodes[node]['u3 err'] = max(1e-5, 0.005 + 0.001 * np.random.randn())
        u3_list.append(Nduv(date, 'gate_error', 'k', graph.nodes[node]['u3 err'] * 1e-3))
        g_list.append(Gate([int(node)], 'u1', u1_list))
        g_list.append(Gate([int(node)], 'u2', u2_list))
        g_list.append(Gate([int(node)], 'u3', u3_list))
    return graph, BackendProperties(id, 'lattest', date, b_list, g_list, []), xtalkProp

def circuit_generate(qubitNum, depth, ifdraw=False):
    qc = QuantumCircuit(qubitNum)
    for _ in range(depth):
        allocatedQ = []
        for q in range(qubitNum):
            if q in allocatedQ:
                continue
            gate_type = random.choice(['x', 'cz'])
            if gate_type == 'x':
                qc.x(q)
                allocatedQ.append(q)
            else:
                j = random.randint(0, qubitNum - 1)
                while j == q:
                    j = random.randint(0, qubitNum - 1)
                qc.cx(q, j)
                allocatedQ.append(q)
                allocatedQ.append(j)
    pm = PassManager()
    pm.append(CommutativeCancellation())

    # 对量子线路进行优化
    qc = pm.run(qc)
    if ifdraw:
        image = qc.draw('mpl')
        image.savefig('circuit.png')
    return qc

def compile(qc, chipG, chipProp, xtalkProp, isXtalkMap, isSchedule, ifdraw, maxParallelCZs=None, isDynamic=False):
    couplingMap = CouplingMap()
    for node in chipG.nodes:
        while not (node in couplingMap.physical_qubits):
            couplingMap.add_physical_qubit(node)
    for edge in chipG.edges:
        couplingMap.add_edge(edge[0], edge[1])
    ps = PassManager()
    basisTranslator = BasisTranslator(sel, ['u3', 'cx'])
    if isXtalkMap == 'xeb':
        ps.append(TrivialLayout(couplingMap))
    else:
        ps.append(NoiseAdaptiveLayout(chipProp))
        ps.append(passes.FullAncillaAllocation(couplingMap))
        ps.append(passes.ApplyLayout())
    ps.run(qc)
    layout = ps.property_set['layout']
    qc_mapped = basisTranslator.run(circuit_to_dag(qc))
    if isXtalkMap == True:
        xaqcSwap = XaqcSwap(couplingMap, search_depth=5, search_width=6)
        qc_mapped = xaqcSwap.run(qc_mapped, layout)
        qc_mapped = dag_to_circuit(qc_mapped)
        qc_mapped = basisTranslator.run(circuit_to_dag(qc_mapped))
    elif not(isXtalkMap == 'xeb'):
        basicSwap = StochasticSwap(couplingMap)
        qc_mapped = basicSwap.run(circuit_to_dag(qc))
        qc_mapped = basisTranslator.run(qc_mapped)

    if ifdraw:
        dag_to_circuit(qc_mapped).draw(output='mpl', filename='circuit map.png')

    xaqcSchedule = CrosstalkAdaptiveSchedule(chipG, chipProp, xtalkProp, maxParallelCZs=maxParallelCZs)
    qc_scheduled, maxGateTime, EPST = xaqcSchedule.run(qc_mapped, isSchedule, isDynamic)
    qc_scheduled = dag_to_circuit(qc_scheduled)
    if ifdraw:
        qc_scheduled.draw(output='mpl', filename='circuit scuedule.png')

    return qc_scheduled, maxGateTime, EPST

def ibm_map(qc, chipG, chipProp, xtalkProp, isscheduled):
    couplingMap = CouplingMap()
    for node in chipG.nodes:
        while not (node in couplingMap.physical_qubits):
            couplingMap.add_physical_qubit(node)
    for edge in chipG.edges:
        couplingMap.add_edge(edge[0], edge[1])
    ps = PassManager()
    ps.append(passes.NoiseAdaptiveLayout(chipProp))
    ps.append(passes.FullAncillaAllocation(couplingMap))
    ps.append(passes.ApplyLayout())
    qc = ps.run(qc)
    sabreswap = SabreSwap(couplingMap, heuristic='lookahead')
    qc_mapped = sabreswap.run(circuit_to_dag(qc))
    unroller = passes.Unroller(['u3', 'cx'])
    qc_mapped = unroller.run(qc_mapped)
    # qc = ps.run(qc)
    # ls = BasicSwap(coupling_map=couplingMap)
    # qc_mapped = ls.run(circuit_to_dag(qc))
    # unroller = passes.Unroller(['u3', 'cx'])
    # qc_mapped = unroller.run(qc_mapped)

    if isscheduled:
        ibmSchedule = ibm.CrosstalkAdaptiveSchedule(chipProp, xtalkProp)
        qc_scheduled = ibmSchedule.run(qc_mapped)
    else:
        qc_scheduled = qc_mapped

    xaqcSchedule = CrosstalkAdaptiveSchedule(chipProp, xtalkProp)
    qc_scheduled, maxGateTime, EPST = xaqcSchedule.run(qc_scheduled, False)
    
    return qc_scheduled, maxGateTime, EPST
    # print(layout._v2p)

def write_data(f, data, type='w'):
    with open(f, type) as fp:
        for d in data:
            fp.write(str(d) + ' ' + str(data[d]))
            fp.write('\n')

# 定义最大并行集

def max_Algsubgraph(chip):
    maxParallelCZs = [[], [], [], []]
    for edge in chip.edges:
        if sum(chip.nodes[edge[0]]['coord']) < sum(chip.nodes[edge[1]]['coord']):
            start = chip.nodes[edge[0]]['coord']
            end = chip.nodes[edge[1]]['coord']
        else:
            start = chip.nodes[edge[1]]['coord']
            end = chip.nodes[edge[0]]['coord']
        if start[0] == end[0]:
            if sum(start) % 2:
                maxParallelCZs[0].append(edge)
            else:
                maxParallelCZs[2].append(edge)
        else:
            if sum(start) % 2:
                maxParallelCZs[1].append(edge)
            else:
                maxParallelCZs[3].append(edge)
    return maxParallelCZs

# xeb线路
def create_xeb_circuit(num_qubits, maxParallelCZs, p):
    qc = QuantumCircuit(num_qubits, num_qubits)
    for _ in range(p):
        for maxParallelCZ in maxParallelCZs:
            qc.h(range(num_qubits))
            for coupler in maxParallelCZ:
                qc.cx(coupler[0], coupler[1])
    return qc

# 定义理论概率分布
def theoretical_probs(num_qubits):
    probs = {}
    for i in range(2**num_qubits):
        binary_str = format(i, '0' + str(num_qubits) + 'b')
        probs[binary_str] = 1 / (2**num_qubits)
    return probs

# 计算交叉熵
def cross_entropy(experimental_probs, theoretical_probs):
    ce = 0
    for outcome, prob in experimental_probs.items():
        ce -= prob * np.log(theoretical_probs[outcome])
    return ce


# 定义QFT算法函数
def qft(n):
    qc = QuantumCircuit(n)

    for i in range(n):
        for j in range(i):
            qc.cp(2*np.pi/float(2**(i-j)), j, i)
        qc.h(i)
    
    # Swap Qubits for visualization purposes
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    
    return qc

# Simon算法函数
def simon_algorithm(n, oracle):
    qc = QuantumCircuit(n * 2, n)
    
    # Apply Hadamard gates before querying the oracle
    for i in range(n):
        qc.h(i)
    
    # Apply query function (oracle)
    qc.compose(oracle, range(n*2), inplace=True)
    
    # Apply Hadamard gates after querying the oracle
    for i in range(n):
        qc.h(i)
    
    # Measure qubits
    # qc.measure(range(n), range(n))
    
    return qc

# 例子: 定义一个简单的Oracle，可以根据需要修改
def simple_oracle(n):
    qc = QuantumCircuit(n*2)
    
    for i in range(n):
        qc.cx(i, n+i)  # Applying CNOT gates
    
    return qc


# 使用 networkx 生成 Erdos-Renyi 图

def maxcut_obj(x, G):
    cut_weight = 0.0
    for i, j, _ in G.edges(data=True):
        if x[i] != x[j]:
            cut_weight += G.edges[(i, j)].get('weight', 1)
    return cut_weight

def qaoa_ansatz(circuit, G, gamma, beta):
    for i in range(len(G.nodes())):
        circuit.h(i)
        circuit.rx(2 * gamma, i)
    for edge in G.edges():
        i, j = edge
        circuit.cx(i, j)
        circuit.rz(-2 * beta, j)
        circuit.cx(i, j)
    return circuit
        


if __name__ == '__main__':
    w = 5
    h = 6
    chipG, chipProp, xtalkProp = chip_generate((w, h))
    # nx.draw_networkx(chipG)
    # plt.show()

    # # simon

    # # 尝试使用3比特的Simon算法
    # print('simon')
    # for n in [2, 4, 8, 12]:
    #     oracle = simple_oracle(n)

    #     simon_circuit = simon_algorithm(n, oracle)
        
    #     # oqmgs
    #     _, oqmgsGateTime, oqmgsF = compile(simon_circuit, chipG, chipProp, xtalkProp, True, True, False)
    #     print('oqgms', oqmgsGateTime, oqmgsF * np.exp(-oqmgsGateTime / (20 * 1000)))
    #     # naive
    #     _, naiveGateTime, naiveF = compile(simon_circuit, chipG, chipProp, xtalkProp, False, False, False)
    #     print('naive', naiveGateTime, naiveF * np.exp(-naiveGateTime / (20 * 1000)))
    #     # Serialization
    #     _, serialGateTime, serialF = compile(simon_circuit, chipG, chipProp, xtalkProp, False, 'serial', False)
    #     print('serial', serialGateTime, serialF * np.exp(-serialGateTime / (20 * 1000)))
    #     # Static Frequency-Aware
    #     chip = nx.grid_2d_graph(w, h)
    #     for qubit in chip.nodes:
    #         chip.nodes[qubit]['coord'] = qubit
    #     chip = nx.relabel_nodes(chip, dict(zip(chip.nodes, range(len(chip.nodes)))))
    #     maxParallelCZs = max_Algsubgraph(chip)
    #     _, sfGateTime, sfF = compile(simon_circuit, chipG, chipProp, xtalkProp, False, 'static', False, maxParallelCZs=maxParallelCZs)
    #     print('sf', sfGateTime, sfF * np.exp(-sfGateTime / (5 * 1000)))
    #     # Dynamic Frequency-Aware
    #     _, dfGateTime, dfF = compile(simon_circuit, chipG, chipProp, xtalkProp, True, True, False, isDynamic=True)
    #     print('df', dfGateTime, dfF * np.exp(-dfGateTime / (5 * 1000)))


    # # # 测量
    # # simon_circuit.measure(range(n), range(n))
    

    # # # 可视化线路图
    # # print("Simon Circuit:")
    # # print(simon_circuit.draw())

    # # # 绘制png类型的线路图
    # # simon_circuit.draw(output='mpl', filename='simon_circuit.png')

    # # # 仿真器
    # # backend = BasicSimulator()
    # # tqc = transpile(simon_circuit, backend)

    # # # 运行量子电路
    # # counts = backend.run(tqc).result().get_counts()

    # # # 绘制直方图
    # # print("Measurement Output:")
    # # print(counts)
    # # plot_histogram(counts)
    # # plt.show()





    # # 例子: 使用4比特的QFT算法
    # print('qft')
    # for n in [4, 9, 16, 25]:
    #     qft_circuit = qft(n)

    #     # 可视化线路图
    #     # print("QFT Circuit:")
    #     # print(qft_circuit.draw())

    #     # xtalk map + xtalk schedule
    #     _, MSXaqcGateTime, MSEPST = compile(qft_circuit, chipG, chipProp, xtalkProp, True, True, False)

    #     # oqmgs
    #     _, oqmgsGateTime, oqmgsF = compile(qft_circuit, chipG, chipProp, xtalkProp, True, True, False)
    #     print('oqgms', oqmgsGateTime, oqmgsF * np.exp(-oqmgsGateTime / (20 * 1000)))
    #     # naive
    #     _, naiveGateTime, naiveF = compile(qft_circuit, chipG, chipProp, xtalkProp, False, False, False)
    #     print('naive', naiveGateTime, naiveF * np.exp(-naiveGateTime / (20 * 1000)))
    #     # Serialization
    #     _, serialGateTime, serialF = compile(qft_circuit, chipG, chipProp, xtalkProp, False, 'serial', False)
    #     print('serial', serialGateTime, serialF * np.exp(-serialGateTime / (20 * 1000)))
    #     # Static Frequency-Aware
    #     chip = nx.grid_2d_graph(w, h)
    #     for qubit in chip.nodes:
    #         chip.nodes[qubit]['coord'] = qubit
    #     chip = nx.relabel_nodes(chip, dict(zip(chip.nodes, range(len(chip.nodes)))))
    #     maxParallelCZs = max_Algsubgraph(chip)
    #     _, sfGateTime, sfF = compile(qft_circuit, chipG, chipProp, xtalkProp, False, 'static', False, maxParallelCZs=maxParallelCZs)
    #     print('sf', sfGateTime, sfF * np.exp(-sfGateTime / (5 * 1000)))
    #     # Dynamic Frequency-Aware
    #     _, dfGateTime, dfF = compile(qft_circuit, chipG, chipProp, xtalkProp, True, True, False, isDynamic=True)
    #     print('df', dfGateTime, dfF * np.exp(-dfGateTime / (5 * 1000)))

    # # # 绘制png类型的线路图
    # # qft_circuit.draw(output='mpl', filename='qft_circuit.png')
    # # plt.show()




    # print('qaoa')
    # for n in [4, 9, 16, 25]:
    #     p = 0.5  # 边的概率
    #     G = nx.erdos_renyi_graph(n, p)
        
    #     # nx.draw_networkx(G, with_labels=True, font_weight='bold')
    #     # plt.show()

    #     # 初始化参数
    #     gamma = 0.1
    #     beta = 0.2

    #     # 创建量子电路
    #     qc = QuantumCircuit(n, n)

    #     # 应用 QAOA ansatz
    #     qc = qaoa_ansatz(qc, G, gamma, beta)

    #     # oqmgs
    #     _, oqmgsGateTime, oqmgsF = compile(qc, chipG, chipProp, xtalkProp, True, True, False)
    #     print('oqgms', oqmgsGateTime, oqmgsF * np.exp(-oqmgsGateTime / (20 * 1000)))
    #     # naive
    #     _, naiveGateTime, naiveF = compile(qc, chipG, chipProp, xtalkProp, False, False, False)
    #     print('naive', naiveGateTime, naiveF * np.exp(-naiveGateTime / (20 * 1000)))
    #     # Serialization
    #     _, serialGateTime, serialF = compile(qc, chipG, chipProp, xtalkProp, False, 'serial', False)
    #     print('serial', serialGateTime, serialF * np.exp(-serialGateTime / (20 * 1000)))
    #     # Static Frequency-Aware
    #     chip = nx.grid_2d_graph(w, h)
    #     for qubit in chip.nodes:
    #         chip.nodes[qubit]['coord'] = qubit
    #     chip = nx.relabel_nodes(chip, dict(zip(chip.nodes, range(len(chip.nodes)))))
    #     maxParallelCZs = max_Algsubgraph(chip)
    #     _, sfGateTime, sfF = compile(qc, chipG, chipProp, xtalkProp, False, 'static', False, maxParallelCZs=maxParallelCZs)
    #     print('sf', sfGateTime, sfF * np.exp(-sfGateTime / (5 * 1000)))
    #     # Dynamic Frequency-Aware
    #     _, dfGateTime, dfF = compile(qc, chipG, chipProp, xtalkProp, True, True, False, isDynamic=True)
    #     print('df', dfGateTime, dfF * np.exp(-dfGateTime / (5 * 1000)))

    # # # 测量
    # # qc.measure(range(n), range(n))

    # # # 仿真器执行
    # # backend = BasicSimulator()
    # # shots = 1024
    # # tqc = transpile(qc, backend)
    # # counts = backend.run(tqc).result().get_counts()

    # # # 计算最大割的期望
    # # maxcut_expectation = 0.0
    # # for x, count in counts.items():
    # #     maxcut_expectation += count * maxcut_obj(np.array([int(bit) for bit in x]), G)
    # # maxcut_expectation /= shots

    # # print(f"Expectation value of MaxCut: {maxcut_expectation}")
    # # print("Result counts:", counts)

    # # # 画出线路图
    # # qc.draw(output='mpl')
    # # plt.show()




    # # qgan
    # # 设置参数
    # print('qgan')
    # for num_qubits in [4, 8, 16, 24]:

    #     num_epochs = 1000  # 训练次数

    #     # 生成正态分布样本（可以替换成你的数据生成方式）
    #     data_samples = np.clip(np.random.normal(size=(2**num_qubits,), loc=0.5), 0, 1)

    #     # 初始化量子神经网络
    #     qgan_circuit = QuantumCircuit(num_qubits, num_qubits)

    #     # 添加 Hadamard 门创建平均态
    #     qgan_circuit.h(range(num_qubits))

    #     # 添加 Ry 门作为生成器
    #     # 添加 10 层 RY 门作为生成器的一部分
    #     for layer in range(4):
    #         for qubit in range(num_qubits):
    #             theta = 2 * np.arcsin(np.sqrt(np.random.uniform()))
    #             qgan_circuit.ry(theta, qubit)
    #         for qubit in range(num_qubits // 2):
    #             qgan_circuit.cx(qubit, num_qubits // 2 + qubit)
    #         for qubit in range(num_qubits):
    #             if not(qubit % 2):
    #                 qgan_circuit.cx(qubit, qubit + 1)

    #     # oqmgs
    #     _, oqmgsGateTime, oqmgsF = compile(qgan_circuit, chipG, chipProp, xtalkProp, True, True, False)
    #     print('oqgms', oqmgsGateTime, oqmgsF * np.exp(-oqmgsGateTime / (20 * 1000)))
    #     # naive
    #     _, naiveGateTime, naiveF = compile(qgan_circuit, chipG, chipProp, xtalkProp, False, False, False)
    #     print('naive', naiveGateTime, naiveF * np.exp(-naiveGateTime / (20 * 1000)))
    #     # Serialization
    #     _, serialGateTime, serialF = compile(qgan_circuit, chipG, chipProp, xtalkProp, False, 'serial', False)
    #     print('serial', serialGateTime, serialF * np.exp(-serialGateTime / (20 * 1000)))
    #     # Static Frequency-Aware
    #     chip = nx.grid_2d_graph(w, h)
    #     for qubit in chip.nodes:
    #         chip.nodes[qubit]['coord'] = qubit
    #     chip = nx.relabel_nodes(chip, dict(zip(chip.nodes, range(len(chip.nodes)))))
    #     maxParallelCZs = max_Algsubgraph(chip)
    #     _, sfGateTime, sfF = compile(qgan_circuit, chipG, chipProp, xtalkProp, False, 'static', False, maxParallelCZs=maxParallelCZs)
    #     print('sf', sfGateTime, sfF * np.exp(-sfGateTime / (5 * 1000)))
    #     # Dynamic Frequency-Aware
    #     _, dfGateTime, dfF = compile(qgan_circuit, chipG, chipProp, xtalkProp, True, True, False, isDynamic=True)
    #     print('df', dfGateTime, dfF * np.exp(-dfGateTime / (5 * 1000)))
                
                
    # # # 添加测量操作
    # # qgan_circuit.measure(range(num_qubits), range(num_qubits))

    # # # 绘制 QGAN 电路线路示意图
    # # circuit_drawer(qgan_circuit, output='mpl', filename='qgan_circuit.png')
    # # plt.show()

    # # # 运行 QGAN 电路
    # # backend = BasicSimulator()
    # # shots = 1024
    # # tqc = transpile(qgan_circuit, backend)
    # # counts = backend.run(tqc).result().get_counts()
    
    # # # 绘制结果直方图
    # # plot_histogram(counts)
    # # plt.show()

    # # # 输出概率结果
    # # print("概率结果：", {state: count/1024 for state, count in counts.items()})





    # # vqe
    # print('vqe')
    # for num_qubits in [4, 8, 16, 24]:
    #     num_epochs = 1000  # 训练次数

    #     # 生成正态分布样本（可以替换成你的数据生成方式）
    #     data_samples = np.clip(np.random.normal(size=(2**num_qubits,), loc=0.5), 0, 1)

    #     # 初始化量子神经网络
    #     vqe_circuit = QuantumCircuit(num_qubits, num_qubits)

    #     # 添加 Hadamard 门创建平均态
    #     vqe_circuit.h(range(num_qubits))

    #     # 添加 Ry 门作为生成器
    #     # 添加 10 层 RY 门作为生成器的一部分
    #     for layer in range(num_qubits):
    #         for qubit in range(num_qubits):
    #             theta = 2 * np.arcsin(np.sqrt(np.random.uniform()))
    #             vqe_circuit.ry(theta, qubit)
    #             vqe_circuit.rz(theta, qubit)
    #         for qubit in range(num_qubits // 2):
    #             vqe_circuit.cx(qubit, num_qubits // 2 + qubit)
    #         for qubit in range(num_qubits):
    #             if not(qubit % 2):
    #                 vqe_circuit.cx(qubit, qubit + 1)

    #     # oqmgs
    #     _, oqmgsGateTime, oqmgsF = compile(vqe_circuit, chipG, chipProp, xtalkProp, True, True, False)
    #     print('oqgms', oqmgsGateTime, oqmgsF * np.exp(-oqmgsGateTime / (20 * 1000)))
    #     # naive
    #     _, naiveGateTime, naiveF = compile(vqe_circuit, chipG, chipProp, xtalkProp, False, False, False)
    #     print('naive', naiveGateTime, naiveF * np.exp(-naiveGateTime / (20 * 1000)))
    #     # Serialization
    #     _, serialGateTime, serialF = compile(vqe_circuit, chipG, chipProp, xtalkProp, False, 'serial', False)
    #     print('serial', serialGateTime, serialF * np.exp(-serialGateTime / (20 * 1000)))
    #     # Static Frequency-Aware
    #     chip = nx.grid_2d_graph(w, h)
    #     for qubit in chip.nodes:
    #         chip.nodes[qubit]['coord'] = qubit
    #     chip = nx.relabel_nodes(chip, dict(zip(chip.nodes, range(len(chip.nodes)))))
    #     maxParallelCZs = max_Algsubgraph(chip)
    #     _, sfGateTime, sfF = compile(vqe_circuit, chipG, chipProp, xtalkProp, False, 'static', False, maxParallelCZs=maxParallelCZs)
    #     print('sf', sfGateTime, sfF * np.exp(-sfGateTime / (5 * 1000)))
    #     # Dynamic Frequency-Aware
    #     _, dfGateTime, dfF = compile(vqe_circuit, chipG, chipProp, xtalkProp, True, True, False, isDynamic=True)
    #     print('df', dfGateTime, dfF * np.exp(-dfGateTime / (5 * 1000)))

    # # 添加测量操作
    # vqe_circuit.measure(range(num_qubits), range(num_qubits))

    # # 绘制 VQE 电路线路示意图
    # circuit_drawer(vqe_circuit, output='mpl', filename='vqe_circuit.png')
    # plt.show()

    # # 运行 VQE 电路
    # backend = BasicSimulator()
    # shots = 1024
    # tqc = transpile(vqe_circuit, backend)
    # counts = backend.run(tqc).result().get_counts()
    
    # # 绘制结果直方图
    # plot_histogram(counts)
    # plt.show()

    # # 输出概率结果
    # print("概率结果：", {state: count/1024 for state, count in counts.items()})






    # # 创建量子线路
    # for p in [1, 2, 3, 4, 5]:
    #     chip = nx.grid_2d_graph(w, h)
    #     for qubit in chip.nodes:
    #         chip.nodes[qubit]['coord'] = qubit
    #     chip = nx.relabel_nodes(chip, dict(zip(chip.nodes, range(len(chip.nodes)))))
    #     maxParallelCZs = max_Algsubgraph(chip)
    #     circuit = create_xeb_circuit(len(chip.nodes), maxParallelCZs, p)
    #     # xtalk map + xtalk schedule
    #     _, MSXaqcGateTime, MSEPST = compile(circuit, chipG, chipProp, xtalkProp, 'xeb', True, False)

    #     # oqmgs
    #     _, oqmgsGateTime, oqmgsF = compile(circuit, chipG, chipProp, xtalkProp, 'xeb', True, False)
    #     print('oqgms', oqmgsGateTime, oqmgsF * np.exp(-oqmgsGateTime / (20 * 1000)))
    #     # naive
    #     _, naiveGateTime, naiveF = compile(circuit, chipG, chipProp, xtalkProp, 'xeb', False, False)
    #     print('naive', naiveGateTime, naiveF * np.exp(-naiveGateTime / (20 * 1000)))
    #     # Serialization
    #     _, serialGateTime, serialF = compile(circuit, chipG, chipProp, xtalkProp, 'xeb', 'serial', False)
    #     print('serial', serialGateTime, serialF * np.exp(-serialGateTime / (20 * 1000)))
    #     # Static Frequency-Aware
    #     chip = nx.grid_2d_graph(w, h)
    #     for qubit in chip.nodes:
    #         chip.nodes[qubit]['coord'] = qubit
    #     chip = nx.relabel_nodes(chip, dict(zip(chip.nodes, range(len(chip.nodes)))))
    #     maxParallelCZs = max_Algsubgraph(chip)
    #     _, sfGateTime, sfF = compile(circuit, chipG, chipProp, xtalkProp, 'xeb', 'static', False, maxParallelCZs=maxParallelCZs)
    #     print('sf', sfGateTime, sfF * np.exp(-sfGateTime / (5 * 1000)))
    #     # Dynamic Frequency-Aware
    #     _, dfGateTime, dfF = compile(circuit, chipG, chipProp, xtalkProp, 'xeb', True, False, isDynamic=True)
    #     print('df', dfGateTime, dfF * np.exp(-dfGateTime / (5 * 1000)))

        
        
    # circuit.measure(range(len(chip.nodes)), range(len(chip.nodes)))
    
    # # 运行
    # backend = BasicSimulator()
    # shots = 1024
    # tqc = transpile(circuit, backend)
    # counts = backend.run(tqc).result().get_counts()

    # # 绘制量子线路图
    # print(circuit.draw(output='text'))
    # circuit_drawer(circuit, output='mpl', filename='qgan_circuit.png')
    # plt.show()

    # # 输出概率结果
    # print("Counts:", counts)

    # # 绘制概率分布图
    # plot_histogram(counts)

    # # 计算交叉熵
    # theoretical = theoretical_probs(w * h)
    # ce = cross_entropy(counts, theoretical)
    # print("Cross-Entropy:", ce)