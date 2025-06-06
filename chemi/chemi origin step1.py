from copy import deepcopy
import qpandalite
import qpandalite.task.origin_qcloud as originq
import numpy as np
import networkx as nx
from pychemiq import Molecules
from pychemiq.Transform.Mapping import jordan_wigner
import re

def variational_circuit(params, measurements, layer, qubits, maxParallelCZs):
    c = qpandalite.Circuit()
    for l in range(layer):
        for i, qid in enumerate(qubits):
            # 在每层的旋转门前应用相位阻尼和振幅阻尼通道
            c.rx(qn=int(qid[1:]), theta=params[l * len(qubits) + i])
            c.rz(qn=int(qid[1:]), theta=params[l * len(qubits) + i])
            c.rx(qn=int(qid[1:]), theta=params[l * len(qubits) + i])

        for maxParallelCZ in maxParallelCZs:
            for qs in maxParallelCZ:
                if not(qs[0] in qubits) or not(qs[1] in qubits):
                    continue
                c.cnot(controller=int(qs[0][1:]), target=int(qs[1][1:]))
            # c.barrier(*[int(qi[1:]) for qi in qubits])

    measureList = []
    for measurement in measurements:
        qid, op = measurement
        if op == 'X':
            c.h(qn=int(qubits[qid][1:]))
        elif op == 'Y':
            c.rx(qn=int(qubits[qid][1:]), theta=-np.pi / 2)
        measureList.append(int(qubits[qid][1:]))
    c.measure(*measureList)
    # print(c.circuit)
    return c.circuit

def remapping(c, mapping):
    c = deepcopy(c)
    for old_qubit, new_qubit in mapping.items():
        c = c.replace(f'q[{old_qubit}]', f'q[_{old_qubit}]')

    for old_qubit, new_qubit in mapping.items():
        c = c.replace(f'q[_{old_qubit}]', f'q[{new_qubit}]')

    return c

def max_Algsubgraph(chip, patternid):
    maxParallelCZs = [[], [], [], []]
    for edge in chip.edges:
        if sum(chip.nodes[edge[0]]['coord']) < sum(chip.nodes[edge[1]]['coord']):
            start = chip.nodes[edge[0]]['coord']
            end = chip.nodes[edge[1]]['coord']
        else:
            start = chip.nodes[edge[1]]['coord']
            end = chip.nodes[edge[0]]
        if patternid == 0:
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
        else:
            if start[0] == end[0]:
                if (sum(start) % 2 and start[0] % 2) or (not(sum(start) % 2) and not(start[0] % 2)):
                    maxParallelCZs[0].append(edge)
                else:
                    maxParallelCZs[2].append(edge)
            else:
                if (sum(start) % 2 and start[1] % 2) or (not(sum(start) % 2) and not(start[1] % 2)):
                    maxParallelCZs[1].append(edge)
                else:
                    maxParallelCZs[3].append(edge)
    return maxParallelCZs

def generate_geometry(dist):
    # return f"H 0 0 0, H 0 0 {dist}"
    # return f"H 0 0 0, H 0 0 {dist}, H 0 0 {2 * dist}, H 0 0 {3 * dist}"
    # return f"H 0 0 0, Li 0 0 {dist}"
    # return f"N 0 0 0, N 0 0 {dist}"
    return f"H {dist} {dist} {dist}, H {-dist} {-dist} {dist}, H {-dist} {dist} {-dist}, H {dist} {-dist} {-dist}"

if __name__ == '__main__':
    
    chip = nx.grid_2d_graph(12, 6)

    for node in chip.nodes:
        chip.nodes[node]['coord'] = node

    relabel_map = dict(zip(chip.nodes, ['q' + str(i) for i in range(len(chip.nodes))]))
    chip = nx.relabel_nodes(chip, relabel_map)
    patternid = 1
    maxParallelCZs = max_Algsubgraph(chip, patternid)
    layer = 5
    n_qubits = 8
    qubits = ['q0', 'q1', 'q2', 'q3', 'q6', 'q7', 'q8', 'q9']
    physicalQ = ['q30', 'q31', 'q32', 'q33', 'q36', 'q37', 'q38', 'q39']

    paramses = [
        [
            np.array([4.81613038, 6.9893218 , 7.37427245, 3.10745704, 3.20767496,
            3.34813399, 0.10506995, 3.25683781, 6.33657139, 1.62573856,
            0.97536005, 0.38932706, 2.3089098 , 2.73751354, 2.26379622,
            2.37616298, 0.28742916, 4.42747598, 4.08458625, 2.7796029 ,
            5.84729859, 2.67548226, 3.17885646, 0.06675477, 4.69439022,
            3.24078736, 1.70466343, 1.40327669, 0.2348827 , 6.09807399,
            6.2235199 , 3.60093336, 3.58679647, 3.12372896, 4.5906869 ,
            3.03109613, 2.62149302, 3.28512794, 3.64665848, 6.32407973]), 
            np.array([0.03989208, 3.39198402, 0.03820583, 5.99910863, 2.63789348,
            3.57038903, 0.69897681, 5.39231368, 0.66937249, 0.42254586,
            1.3962062 , 4.44621776, 0.92172306, 3.601732  , 4.38397682,
            5.3710506 , 3.37222264, 3.80593264, 1.7623775 , 1.38284237,
            2.55308785, 5.16611857, 3.27052979, 0.93116844, 4.49045269,
            0.39236812, 0.12402535, 0.28811617, 1.53204636, 1.08037845,
            0.48826901, 1.73523807, 2.57438749, 4.26668629, 5.09733395,
            4.78773681, 4.45644124, 1.701422  , 6.35953432, 3.30732722]), 
            np.array([2.03721753, 5.81068393, 0.76216793, 1.91796618, 0.8746449 ,
            1.11029487, 1.29661109, 4.79483348, 0.95346569, 4.8703985 ,
            5.92465231, 0.26343105, 0.06683985, 4.37353349, 2.09663516,
            1.06750239, 2.05857852, 3.106683  , 4.36436844, 0.51770448,
            5.90596565, 3.47663143, 4.00458506, 4.90491814, 4.71076207,
            7.32876078, 6.12378736, 2.05792313, 4.45097519, 5.20631371,
            1.06880999, 4.26442476, 2.92121606, 3.79353317, 3.97190136,
            0.59360093, 0.40317103, 4.5076215 , 4.55386999, 3.72959586]), 
            np.array([4.74736518, 4.72725784, 4.19484046, 3.63572199, 4.69622095,
            4.75550812, 3.35283816, 4.67671948, 3.09855509, 0.57632185,
            4.58657554, 0.26158854, 4.53178788, 1.12554417, 6.34118923,
            3.43666722, 4.7358958 , 3.57404982, 5.15519844, 3.05991537,
            3.38027961, 0.88144594, 5.05856696, 4.73866318, 1.23792679,
            2.90645574, 1.67624505, 6.43121367, 1.76402478, 0.30618396,
            3.61047599, 3.43275094, 1.70245852, 0.77785856, 6.27752049,
            6.19801878, 4.87598697, 3.44475044, 0.03854849, 3.45847272]), 
            np.array([4.71810758, 1.50654314, 0.16554125, 1.8989477 , 3.83402742,
            3.66951873, 6.11724334, 3.29829601, 3.15896756, 0.11191249,
            0.15607047, 1.83104402, 2.73269302, 4.07298026, 4.69926759,
            4.86285422, 6.26940159, 3.15764884, 5.69443117, 2.88362054,
            3.80060932, 2.73833005, 1.56831741, 2.72495469, 6.26366246,
            3.09965947, 6.28863089, 1.28854074, 4.63127148, 5.94834158,
            3.13717309, 5.40694489, 1.55181607, 3.78564273, 5.78235889,
            0.97985369, 6.46829695, 0.63820807, 4.64097871, 2.85439531])
        ],
        [
            np.array([2.04417912, 4.55819711, 0.07646744, 0.07636452, 2.5261531 ,
            4.38695115, 3.15003222, 3.12612247, 0.32190993, 3.69600716,
            0.83180404, 3.39078115, 3.31483119, 3.99521731, 2.84086653,
            0.18914461, 0.93207667, 4.50055453, 4.52058869, 0.01875196,
            1.85317951, 1.23636317, 5.10328092, 3.3220146 , 4.86499377,
            0.80597202, 3.91991627, 2.9276964 , 3.02850683, 6.94291715,
            4.28768126, 3.07043312, 0.59673045, 3.41845257, 4.40812166,
            0.06243402, 3.27441092, 0.8484476 , 6.38311209, 0.04846492]), 
            np.array([ 4.6633373 ,  4.72741712,  4.68979342,  6.12015932,  1.54361247,
            4.69163727,  2.79336117,  4.666391  ,  3.15513144,  6.30271878,
            3.24574762,  5.32098975,  6.30053815, -0.06318297,  2.59331906,
            4.74222709,  1.58598386,  2.99104307,  3.12861806,  6.86423619,
            1.58581594,  4.47721396,  1.88729907,  1.53481454,  4.70750768,
            4.70287224,  4.65440728,  2.86520241,  2.83995516,  2.91311138,
            3.37621849,  4.61519855,  0.06353155,  0.18734655,  1.62150897,
            3.22466622,  6.28373109,  3.34148355,  2.9833168 ,  2.87422369]), 
            np.array([4.65894938, 5.79817467, 1.29159165, 3.16564790,
            6.30640212, 5.52515876, 6.79622715e-01, 7.67778710e-01,
            3.15684813, 5.90734634, 3.00702782, 2.68411500,
            1.30956396e-03, 1.66453597, 4.68243608, 6.23530153,
            6.40457618, 1.32237503, 3.10766723, 2.71694342,
            4.53246716, 4.64028898, 3.08158744, 2.66571960,
            3.07469724, 6.06911154, 3.18897488, 3.54682692,
            4.64914561, 9.09722775e-01, 4.64170542, 3.02094912,
            1.39193123, 1.29009201e-01, 4.55512369, 2.58375815,
            3.24529858, 4.35755706, 2.27000741e-02, 1.11187564]), 
            np.array([4.70153018, 1.48518483, 1.97930412, 5.5944567 , 1.70660921,
            3.0757076 , 6.30244812, 3.37010584, 2.37487087, 4.71551509,
            1.89271697, 5.57750362, 1.4682907 , 0.05070253, 1.67632667,
            6.30409879, 3.8850755 , 3.09860683, 6.20763567, 4.41791841,
            4.62163243, 2.87269234, 2.90557341, 0.14284882, 1.5451868 ,
            6.7074828 , 2.0621688 , 3.47843411, 1.52261611, 3.33993012,
            1.11889325, 3.65864538, 3.41391276, 5.05161193, 3.29170773,
            0.4017261 , 2.86350034, 0.70035864, 2.97864717, 3.06834533]), 
            np.array([0.20774389, 1.56768132, 1.56769606, 4.67999326, 4.95899568,
            3.30434925, 4.72097035, 0.2976059 , 4.57581092, 1.49233432,
            1.55415959, 0.90797886, 4.6642418 , 0.03766193, 3.23969508,
            3.27551088, 0.32433832, 4.71222617, 1.54422372, 5.37671564,
            3.09286999, 3.04461539, 1.59925572, 0.03531607, 1.63393515,
            1.72561121, 4.72646146, 0.96425827, 0.80510728, 0.05160013,
            3.31300222, 3.18286912, 1.54035518, 0.80721967, 5.32589538,
            4.55739622, 4.88932118, 3.43566378, 3.05409177, 0.27163591])
        ]
    ]

    rs = [np.linspace(0.18, 3.5, 5)[4]]
    
    # for _ in range(5):
    for _ in [0]:
        circuits = []
        for rid, r in enumerate(rs):
            geom = generate_geometry(r)

            # 创建分子对象
            # molecule = Molecules(geom, charge=0, multiplicity=1, basis="sto-3g")
            molecule = Molecules(geom, charge=1, multiplicity=2, basis="sto-3g")
            n_qubits = molecule.n_qubits
            n_elec = molecule.n_electrons
            # print(n_qubits, n_elec)

            # 获取分子哈密顿量
            hamiltonian = molecule.get_molecular_hamiltonian()

            # 使用 Jordan-Wigner 变换将费米子哈密顿量转换为量子比特哈密顿量
            qubit_hamiltonian = jordan_wigner(hamiltonian)

            # 遍历 PauliOperator 对象的项
            for term, coeff in qubit_hamiltonian.data():
                ops = []
                for index, op in term[0].items():
                    ops.append((index, op))
                if ops == []:
                    continue

                params = paramses[patternid][rid]

                circuit = variational_circuit(params=params, measurements=ops, layer=layer, qubits=qubits, maxParallelCZs=maxParallelCZs)
                mapping = dict(zip([int(qid[1:]) for qid in qubits], [int(qid[1:]) for qid in physicalQ]))
                # print(circuit)
                circuit = remapping(circuit, mapping)
                # 修改前面的部分
                circuit = re.sub(r'QINIT \d+\nCREG \d+', 'QINIT 50\nCREG 20', circuit, 1)
                # print(circuit)
                circuits.append(circuit)

        print(len([circuits[0]]))
        taskid = originq.submit_task([circuits[0]], shots=1000, auto_mapping=False, measurement_amend=True, circuit_optimize=False)
        # print(len(circuits))
        # taskid = originq.submit_task(circuits, shots=1000, auto_mapping=False, measurement_amend=True, circuit_optimize=False)
        print('pattern', patternid, taskid)

