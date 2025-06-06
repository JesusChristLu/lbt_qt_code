from copy import deepcopy
import qpandalite
import qpandalite.task.origin_qcloud as originq
import numpy as np
import networkx as nx
from pychemiq import Molecules
from pychemiq.Transform.Mapping import jordan_wigner
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def variational_circuit(params, measurements, layer, qubits, maxParallelCZs):
    c = qpandalite.Circuit()
    for l in range(layer):
        for i, _ in enumerate(qubits):
            # 在每层的旋转门前应用相位阻尼和振幅阻尼通道
            c.rx(qn=i, theta=params[l * len(qubits) + i])
            c.rz(qn=i, theta=params[l * len(qubits) + i])
            c.rx(qn=i, theta=params[l * len(qubits) + i])

        for maxParallelCZ in maxParallelCZs:
            for qs in maxParallelCZ:
                if not(qs[0] in qubits) or not(qs[1] in qubits):
                    continue
                c.cnot(controller=int(qubits.index(qs[0])), target=int(qubits.index(qs[1])))

    measureList = []
    for measurement in measurements:
        qid, op = measurement
        if op == 'X':
            c.h(qn=qid)
        elif op == 'Y':
            c.rx(qn=qid, theta=-np.pi / 2)
        measureList.append(qid)
    c.measure(*measureList)
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
    layer = 5
    n_qubits = 8
    patternid = 0
    maxParallelCZs = max_Algsubgraph(chip, patternid)
    qubits = ['q0', 'q1', 'q2', 'q3', 'q6', 'q7', 'q8', 'q9']
    rs = np.linspace(0.18, 3.5, 5)

    circuits = []
    energyExps = []
    params = []
    
    for i, r in enumerate(rs):
        print('dist', r)
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

        # 打印qubit_hamiltonian对象的内容及其方法或属性
        # print(qubit_hamiltonian)

        def energy(params):
        
            for term, coeff in qubit_hamiltonian.data():
                ops = []
                for index, op in term[0].items():
                    ops.append((index, op))
                if ops == []:
                    continue

                circuit = variational_circuit(params=params, measurements=ops, layer=layer, qubits=qubits, maxParallelCZs=maxParallelCZs)
                mapping = {0 : 31, 1 : 32, 2 : 33, 3 : 34, 4 : 37, 5 : 38, 6 : 39, 7 : 40}
                # print(circuit)
                circuit = remapping(circuit, mapping)
                # 修改前面的部分
                circuit = circuit.replace('QINIT 8', 'QINIT 50').replace('CREG ' + str(len(ops)), 'CREG 8')
                # print(circuit)
                circuits.append(circuit)

            taskid = originq.submit_task(circuits, shots=1000, auto_mapping=False, measurement_amend=True)
            # print('pattern', patternid, taskid)

            while 1:
                finished, task_count = originq.query_all_task()
                if finished == task_count:
                    print('oh good!')
                    break

            taskid = originq.get_last_taskid()
            results = originq.query_by_taskid_sync(taskid)

            results = qpandalite.convert_originq_result(
                results, 
                style='list', 
                prob_or_shots='shots',
                key_style='dec'
            )

            # 遍历 PauliOperator 对象的项
            hid = 0
            for term, coeff in qubit_hamiltonian.data():
                ops = []
                for index, op in term[0].items():
                    ops.append((index, op))
                if ops == []:
                    energyExp = np.real(coeff)
                else:
                    result = results[hid]
                    for j, prob in enumerate(result):
                        binI = bin(j)[2:]
                        pm = binI.count('1')
                        if pm % 2:
                            energyExp -= np.real(coeff * prob)
                        else:
                            energyExp += np.real(coeff * prob)
                    hid += 1
            print(params)
            print(energyExp)
            return energyExp
        
        params = np.random.random(size=layer * n_qubits) * (2 * np.pi)
        res = minimize(energy, params, method='nelder-mead')
        print(res.x, res.fun)
        energyExps.append(res.fun)
        params.append(res.x)
        
    print(energyExps)
    print(params)
