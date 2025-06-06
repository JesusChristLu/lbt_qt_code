import matplotlib.pyplot as plt
import tensorcircuit as tc
import tensorflow as tf
import numpy as np
import networkx as nx
from pychemiq import Molecules
from pychemiq.Transform.Mapping import jordan_wigner
import scipy.optimize as optimize

def generate_geometry(dist):
    # return f"H 0 0 0, Li 0 0 {dist}"
    return f"H 0 0 0, H 0 0 {dist}"
    # return f"N 0 0 0, N 0 0 {dist}"

def max_Algsubgraph(chip):
    dualChip = nx.Graph()
    dualChip.add_nodes_from(list(chip.edges))
    for coupler1 in dualChip.nodes:
        for coupler2 in dualChip.nodes:
            if coupler1 == coupler2 or set(coupler1).isdisjoint(set(coupler2)):
                continue
            else:
                dualChip.add_edge(coupler1, coupler2)
    maxParallelCZs = [[], [], [], []]
    for edge in chip.edges:
        if sum(chip.nodes[edge[0]]['coord']) < sum(chip.nodes[edge[1]]['coord']):
            start = chip.nodes[edge[0]]['coord']
            end = chip.nodes[edge[1]]['coord']
        else:
            start = chip.nodes[edge[1]]['coord']
            end = chip.nodes[edge[0]]
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

def variational_circuit(params, layer, n_qubits, maxParallelCZs, gamma_phase=0.1, gamma_amp=0.1):
    c = tc.Circuit(n_qubits)
    for l in range(layer):
        for i in range(n_qubits):
            # 在每层的旋转门前应用相位阻尼和振幅阻尼通道
            c.rx(i, theta=params[l * n_qubits + i])
            c.rz(i, theta=params[l * n_qubits + i])
            c.rx(i, theta=params[l * n_qubits + i])
            
        # 在旋转门和CNOT门之间应用相位阻尼和振幅阻尼通道
        # apply_decoherence_channels(c, n_qubits, gamma_phase, gamma_amp)
        
        for maxParallelCZ in maxParallelCZs:
            for qs in maxParallelCZ:
                if not(int(qs[0][1:]) in range(n_qubits)) or not(int(qs[1][1:]) in range(n_qubits)):
                    continue
                c.cnot(int(qs[0][1:]), int(qs[1][1:]))
        # # 在每层的CNOT门后应用相位阻尼和振幅阻尼通道
        # apply_decoherence_channels(c, n_qubits, gamma_phase, gamma_amp)

    # 在每层的CNOT门后应用相位阻尼和振幅阻尼通道
    # apply_decoherence_channels(c, n_qubits, gamma_phase, gamma_amp)
    
    return c

def amplitude_damping(gamma):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return K0, K1

def apply_decoherence_channels(circuit, n_qubits, gamma_phase, gamma_amp):
    for qubit in range(n_qubits):
        K0, K1 = tc.circuit.channels.phasedampingchannel(gamma_phase)
        circuit.general_kraus([K0, K1], qubit)
        # K0, K1 = amplitude_damping(gamma_amp)
        # circuit.general_kraus([K0, K1], qubit)
    return circuit

def grad_descent(params, i):
    val, grad = energy_val_grad_jit(params)
    params = opt.update(grad, params)
    if i % 10 == 0:
        print(f"i={i}, energy={val}")
    return params, val

if __name__ == '__main__':

    chip = nx.grid_2d_graph(5, 5)

    for node in chip.nodes:
        chip.nodes[node]['coord'] = node

    relabel_map = dict(zip(chip.nodes, ['q' + str(i) for i in range(len(chip.nodes))]))
    chip = nx.relabel_nodes(chip, relabel_map)

    maxParallelCZs = max_Algsubgraph(chip)

    geom = generate_geometry(1.1)

    # 创建分子对象
    molecule = Molecules(geom, charge=0, multiplicity=1, basis="sto-3g")
    n_qubits = molecule.n_qubits
    n_elec = molecule.n_electrons

    # 获取分子哈密顿量
    hamiltonian = molecule.get_molecular_hamiltonian()

    # 使用 Jordan-Wigner 变换将费米子哈密顿量转换为量子比特哈密顿量
    qubit_hamiltonian = jordan_wigner(hamiltonian)

    # 打印qubit_hamiltonian对象的内容及其方法或属性
    print(qubit_hamiltonian)

    # 将生成的哈密顿量转换为列表形式，以便在 TensorCircuit 中使用
    tc_hamiltonian = []

    # 遍历 PauliOperator 对象的项
    for term, coeff in qubit_hamiltonian.data():
        ops = []
        for index, op in term[0].items():
            ops.append((index, op))
        tc_hamiltonian.append((coeff, ops))

    # 打印转换后的哈密顿量
    print(tc_hamiltonian)
    layer = 5

    def energy(params):
        c = variational_circuit(params, layer, n_qubits, maxParallelCZs)
        expectation = 0
        for pouliString in tc_hamiltonian:
            coeff = np.real(pouliString[0])
            ops = pouliString[1]
            if ops == []:
                # expectation += coeff
                continue
            x = []
            y = []
            z = []
            for op in ops:
                if op[1] == 'X':
                    x.append(op[0])
                elif op[1] == 'Y':
                    y.append(op[0])
                else:
                    z.append(op[0])
            expectation += coeff * tc.backend.real(c.expectation_ps(x=x, y=y, z=z))
        return K.real(expectation)
    
    K = tc.set_backend('tensorflow')
    energy_val_grad = K.value_and_grad(energy)
    energy_val_grad_jit = K.jit(energy_val_grad)

    learning_rate = 2e-2
    opt = K.optimizer(tf.keras.optimizers.SGD(learning_rate))

    params = K.implicit_randn(layer * n_qubits)
    energys = []
    for i in range(1000):
        params, E = grad_descent(params, i)
        if i % 20 == 0:
            energys.append(E)

    plt.plot(energys)
    plt.show()