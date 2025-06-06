import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import qutip as qt
from pychemiq import Molecules
from pychemiq.Transform.Mapping import jordan_wigner
import scipy.optimize as optimize
from qutip_qip.circuit import QubitCircuit
from qutip_qip.circuit import CircuitSimulator
from qutip_qip.operations import Gate
from qutip import tensor, basis, sigmax, sigmay, sigmaz, identity, Qobj, expand_operator
from scipy.optimize import minimize
import qpandalite
import qpandalite.task.origin_qcloud as originq
from copy import deepcopy
import re
import json

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

def circuit(layer, qubits, maxParallelCZ, qs, parallel=True):
    if not(qs[0] in qubits) or not(qs[1] in qubits):
        return False
    c_sim = QubitCircuit(N=2)
    c_exp = qpandalite.Circuit()
    for _ in range(layer):
        for i in range(len(qubits)):
            arg = np.random.random() * np.pi
            # 在每层的旋转门前应用相位阻尼和振幅阻尼通道
            if qubits[i] in qs:
                c_sim.add_gate('RY', arg_value=arg, targets=[list(qs).index(qubits[i])])
                if not parallel:
                    c_exp.ry(qn=int(qubits[i][1:]), theta=arg)
            if parallel:
                c_exp.ry(qn=int(qubits[i][1:]), theta=arg)

        c_sim.add_gate('CNOT', controls=[list(qs).index(qs[0])], targets=[list(qs).index(qs[1])])
        if not parallel:
            c_exp.cnot(controller=int(qs[0][1:]), target=int(qs[1][1:]))
        else:
            for cz in maxParallelCZ:
                if not(cz[0] in qubits) or not(cz[1] in qubits):
                    continue
                c_exp.cnot(controller=int(cz[0][1:]), target=int(cz[1][1:]))
            c_exp.barrier(*[int(qi[1:]) for qi in qubits])

    measureList = [int(qid[1:]) for qid in qs]
    c_exp.measure(*measureList)
    # print(c_exp.circuit)
    # print(c_sim.gates)
    return c_sim, c_exp.circuit

def remapping(c, mapping):
    c = deepcopy(c)
    for old_qubit, new_qubit in mapping.items():
        c = c.replace(f'q[{old_qubit}]', f'q[_{old_qubit}]')

    for old_qubit, new_qubit in mapping.items():
        c = c.replace(f'q[_{old_qubit}]', f'q[{new_qubit}]')

    return c

if __name__ == '__main__':
    chip = nx.grid_2d_graph(12, 6)
    # qubits = ['q' + str(i) for i in range(18)]
    # physicalQ = ['q' + str(i) for i in range(24, 42)]
    # qubits = ['q0', 'q1', 'q2', 'q6', 'q7', 'q8', 'q12', 'q13', 'q14']
    # physicalQ = ['q24', 'q25', 'q26', 'q30', 'q31', 'q32', 'q36', 'q37', 'q38']
    qubits = ['q0', 'q1', 'q2', 'q6', 'q7', 'q8']
    # physicalQ = ['q24', 'q25', 'q26', 'q30', 'q31', 'q32']
    physicalQ = ['q0', 'q1', 'q2', 'q6', 'q7', 'q8']

    for node in chip.nodes:
        chip.nodes[node]['coord'] = node

    relabel_map = dict(zip(chip.nodes, ['q' + str(i) for i in range(len(chip.nodes))]))
    chip = nx.relabel_nodes(chip, relabel_map)

    patternids = (0, 1)
    batchNum = 30

    for patternid in patternids:
        for batch in range(batchNum):
            maxParallelCZs = max_Algsubgraph(chip, patternid)
            for maxParallelCZ in maxParallelCZs:
                for qs in maxParallelCZ:
                    # if not(qs == ('q6', 'q7') or qs == ('q0', 'q1')):
                        # continue
                    if not(qs == ('q0', 'q1')):
                        continue
                    # for lengthId, length in enumerate([1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120]):
                    for lengthId, length in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]):
                    # for lengthId, length in enumerate([6, 7, 8, 9, 10, 12, 14, 16, 18, 20]):
                        c = circuit(length, qubits, maxParallelCZ, qs, parallel=True)
                        if c:
                            c_sim, c_exp = c
                        else:
                            continue

                        # exp
                        mapping = dict(zip([int(qid[1:]) for qid in qubits], [int(qid[1:]) for qid in physicalQ]))

                        c_exp = remapping(c_exp, mapping)
                        # 修改前面的部分
                        c_exp = re.sub(r'QINIT \d+\nCREG \d+', 'QINIT 50\nCREG 20', c_exp, 1)

                        taskid = originq.submit_task([c_exp], shots=1000, auto_mapping=False, measurement_amend=True, circuit_optimize=False)
                        print(qs, length, batch, maxParallelCZs.index(maxParallelCZ))

                        while 1:
                            finished, task_count = originq.query_all_task()
                            if finished == task_count:
                                print('oh, good!')
                                break

                        taskid = originq.get_last_taskid()
                        results = originq.query_by_taskid_sync(taskid)

                        results = qpandalite.convert_originq_result(
                                    results, 
                                    style='list', 
                                    prob_or_shots='shots',
                                    key_style='dec'
                        )

                        expProb = np.real(np.array(results[0]))
                        print(expProb)

                        # sim
                        sim = CircuitSimulator(c_sim, mode='density_matrix_simulator')
                        zero_state = basis(2, 0)
                        # for _ in range(len(qubits) - 1):
                        for _ in range(1):
                            zero_state = tensor(zero_state, basis(2, 0))
                        rho = sim.run(zero_state).get_final_states()[0]

                        simProb = np.real(np.diagonal(rho))
                        print(simProb)
                        print(round(np.sum((4 * simProb - 1) * expProb), 4))
                        with open('xeb_file3.txt', 'a', encoding='utf-8') as file:
                            file.write('pattern' + str(patternid) + 'qs' + str(qs) + 'length' + str(length) + 'batch' + str(batch) + ':' + str(round(np.sum((4 * simProb - 1) * expProb), 4)) + '\n')
        