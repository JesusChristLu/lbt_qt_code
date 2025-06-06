from pychemiq import Molecules, ChemiQ, QMachineType
from pychemiq.Transform.Mapping import jordan_wigner, MappingType
from pychemiq.Optimizer import vqe_solver
from pychemiq.Circuit.Ansatz import UserDefine
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pychemiq.Circuit.Ansatz import UCC

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

def circuit_generate(maxParallelCZs, qubits):
    circuit = ''
    for _ in range(5):
        for q in range(len(qubits)):
            para = str(np.random.random() * np.pi)[:6]
            circuit += 'RX q[' + str(q) + '],(' + para + ') '
            circuit += 'RZ q[' + str(q) + '],(' + para + ') '
            circuit += 'RX q[' + str(q) + '],(' + para + ') '
        for maxParallelCZ in maxParallelCZs:
            for qs in maxParallelCZ:
                if not(qs[0] in qubits) or not(qs[1] in qubits):
                    continue
                circuit += 'CNOT q[' + str(int(qubits.index(qs[0]))) + '], q[' + str(int(qubits.index(qs[1]))) + '] '
    return circuit

def generate_geometry(dist):
    # return f"H 0 0 0, H 0 0 {dist}"
    # return f"H 0 0 0, H 0 0 {dist}, H 0 0 {2 * dist}, H 0 0 {3 * dist}"
    return f"H 0 0 0, Li 0 0 {dist}"
    # return f"N 0 0 0, N 0 0 {dist}"
    # return f"H {dist} {dist} {dist}, H {-dist} {-dist} {dist}, H {-dist} {dist} {-dist}, H {dist} {-dist} {-dist}"
    # return f"O {dist} {dist} {dist}, H {-dist} {-dist} {dist}, H {-dist} {dist} {-dist}"
    # return f"N {dist} {dist} {dist}, H {-dist} {-dist} {dist}, H {-dist} {dist} {-dist}, H {dist} {-dist} {-dist}"
    # return f"C {dist} {dist} {dist}, O {dist} {-dist} {dist}, O {dist} {dist} {-dist}"

if __name__ == '__main__':

    chip = nx.grid_2d_graph(12, 6)

    for node in chip.nodes:
        chip.nodes[node]['coord'] = node

    relabel_map = dict(zip(chip.nodes, ['q' + str(i) for i in range(len(chip.nodes))]))
    chip = nx.relabel_nodes(chip, relabel_map)
    # qubits = ['q0', 'q1', 'q2', 'q3', 'q6', 'q7', 'q8', 'q9', 'q12', 'q13', 'q14', 'q15']
    qubits = ['q0', 'q1', 'q2', 'q3', 'q6', 'q7', 'q8', 'q9']

    patternid = 0
    maxParallelCZs = max_Algsubgraph(chip, patternid)

    multiplicity = 2
    charge = 1
    basis = "sto-3g"
    distances = np.linspace(0.18, 3.5, 50)
    # distances = [1]
    energies = []
    energiesucc = []

    for dist in distances:
        geom = generate_geometry(dist)
        mol = Molecules(
            geometry=geom,
            basis=basis,
            multiplicity=multiplicity,
            charge=charge)
        fermion = mol.get_molecular_hamiltonian()
        pauli = jordan_wigner(fermion)
        
        chemiq = ChemiQ()
        machine_type = QMachineType.CPU_SINGLE_THREAD
        mapping_type = MappingType.Jordan_Wigner
        pauli_size = len(pauli.data())
        n_qubits = mol.n_qubits
        n_elec = mol.n_electrons

        print(n_qubits, n_elec)

        chemiq.prepare_vqe(machine_type, mapping_type, n_elec, pauli_size, n_qubits)

        circuit = circuit_generate(maxParallelCZs, qubits)
        ansatz = UserDefine(n_elec, circuit=circuit, chemiq=chemiq)

        # ansatzucc = UCC("UCCSD", n_elec, mapping_type, chemiq=chemiq)

        method = "Gradient-Descent"

        init_para = np.zeros(ansatz.get_para_num())
        solver = vqe_solver(
            method=method,
            pauli=pauli,
            chemiq=chemiq,
            ansatz=ansatz,
            init_para=init_para)
        
        print(solver.para)

        energies.append(solver.fun_val)

        # init_para = np.zeros(ansatzucc.get_para_num())
        # solverucc = vqe_solver(
        #     method=method,
        #     pauli=pauli,
        #     chemiq=chemiq,
        #     ansatz=ansatzucc,
        #     init_para=init_para)
        
        # energiesucc.append(solverucc.fun_val)

    for dist, energy in zip(distances, energies):
        print(f"Distance: {dist:.2f}, Energy: {energy:.6f}")

    # Plotting the potential energy surface

    print(energies)
    print(np.min(energies))

    plt.plot(distances, energies, label='theory')
    rs = np.linspace(0.18, 3.5, 5)

    potential1 = [-1.2, -5.1, -5.2, -5.4, -5.26]
    potential2 = [-2.5, -6.3, -6.6, -6.73, -6.68]
    potential3 = [-1.8, -4.7, -5.29, -5.6, -5.6]
    potential4 = [-2.9, -6.9, -7.2, -7.3, -7.25]
    plt.scatter(rs, potential1, label='ABCD Rand. Config', marker='<', color='red')
    plt.scatter(rs, potential2, label='ABCD Opt. Config', marker='o', color='red')

    plt.scatter(rs, potential3, label='EFGH Rand. Config', marker='<', color='blue')
    plt.scatter(rs, potential4, label='EFGH Opt. Config', marker='o', color='blue')

    plt.xlabel('Interatomic Distance (Å)', fontsize=20)
    plt.ylabel('Energy', fontsize=20)
    plt.title('Potential Energy Surface', fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=20) # set the x-axis ticks to the group labels
    plt.yticks(fontsize=20) # set the x-axis ticks to the group labels
    plt.tight_layout()
    plt.show()