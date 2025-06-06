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
from multiprocessing import Pool, cpu_count


def generate_geometry(dist):
    # return f"H 0 0 0, H 0 0 {dist}"
    # return f"H 0 0 0, H 0 0 {dist}, H 0 0 {2 * dist}, H 0 0 {3 * dist}"
    return f"H 0 0 0, Li 0 0 {dist}"
    # return f"N 0 0 0, N 0 0 {dist}"
    # return f"H {dist} {dist} {dist}, H {-dist} {-dist} {dist}, H {-dist} {dist} {-dist}, H {dist} {-dist} {-dist}"

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

def variational_circuit(params, layer, qubits, maxParallelCZs, noise=False):
    c = QubitCircuit(N=len(qubits))
    def noiseCNOt():
        # dj = np.random.random() * (0.1 - 1e-4) + 1e-4
        dj = np.random.random() * (0.5 - 0.09) + 0.09
        phi = np.pi * (1 + dj / (2 * np.sqrt(2)))
        L = 0.5 * ((np.pi / 2) ** 2) * ((dj / (2 * np.sqrt(2))) ** 4)
        return qt.Qobj([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, (1 - L) * np.exp(1.0j * phi)]], dims=[[2, 2], [2, 2]])
        
    c.user_gates = {'noiseCNOT': noiseCNOt}

    for l in range(layer):
        for i in range(len(qubits)):
            # 在每层的旋转门前应用相位阻尼和振幅阻尼通道
            c.add_gate('RX', arg_value=params[l * len(qubits) + i], targets=[i])
            c.add_gate('RZ', arg_value=params[l * len(qubits) + i], targets=[i])
            c.add_gate('RX', arg_value=params[l * len(qubits) + i], targets=[i])

        for maxParallelCZ in maxParallelCZs:
            for qs in maxParallelCZ:
                if not(qs[0] in qubits) or not(qs[1] in qubits):
                    continue
                # c.add_gate('CNOT', targets=[int(qubits.index(qs[0]))], controls=[int(qubits.index(qs[1]))])
                if noise:
                    ncn = Gate('noiseCNOT', targets=[int(qubits.index(qs[0])), int(qubits.index(qs[1]))])
                    c.add_gate(ncn)
                else:
                    c.add_gate('CNOT', targets=[int(qubits.index(qs[0]))], controls=[int(qubits.index(qs[1]))])

    return c

def forward(r, maxParallelCZs, noise):
    geom = generate_geometry(r)

    # 创建分子对象
    # molecule = Molecules(geom, charge=0, multiplicity=1, basis="sto-3g")
    molecule = Molecules(geom, charge=1, multiplicity=2, basis="sto-3g")
    n_qubits = molecule.n_qubits
    qubits = ['q0', 'q1', 'q2', 'q3', 'q6', 'q7', 'q8', 'q9', 'q12', 'q13', 'q14', 'q15']
    # qubits = ['q0', 'q1', 'q2', 'q3', 'q6', 'q7', 'q8', 'q9']
    n_elec = molecule.n_electrons
    # print(n_qubits, n_elec)

    # 获取分子哈密顿量
    hamiltonian = molecule.get_molecular_hamiltonian()

    # 使用 Jordan-Wigner 变换将费米子哈密顿量转换为量子比特哈密顿量
    qubit_hamiltonian = jordan_wigner(hamiltonian)

    # 打印qubit_hamiltonian对象的内容及其方法或属性
    # print(qubit_hamiltonian)

    layer = 5
    zero_state = basis(2, 0)
    for _ in range(n_qubits - 1):
        zero_state = tensor(zero_state, basis(2, 0))

    def energy(params):
        c = variational_circuit(params, layer, qubits, maxParallelCZs, noise)
        sim = CircuitSimulator(c, mode='density_matrix_simulator')
        rho = sim.run(zero_state).get_final_states()[0]

        H = None
        for ops, coeff in qubit_hamiltonian.data():
            x = []
            y = []
            z = []
            ops = ops[0]
            if len(ops) > 0:
                for op in ops:
                    if ops[op] == 'X':
                        x.append(op)
                    elif ops[op] == 'Y':
                        y.append(op)
                    else:
                        z.append(op)
            h = [identity(2)] * n_qubits
            # 构造张量积
            # 在指定位置应用sigmax算符
            for pos in x:
                h[pos] = sigmax()
            # 在指定位置应用sigmay算符
            for pos in y:
                h[pos] = sigmay()
            # 在指定位置应用sigmaz算符
            for pos in z:
                h[pos] = sigmaz()
            h = tensor(h)
            if H == None:
                H = coeff * h
            else:
                H += coeff * h
        meas = np.real((rho * H).tr())
        # print(r, meas)
        return meas
    
    params = np.random.random(size=layer * n_qubits) * (2 * np.pi)
    res = minimize(energy, params, method='nelder-mead')
    # res = minimize(energy, params, method='CG')
    return res

if __name__ == '__main__':

    patternids = [0, 1]
    noises = [True, False]
    patterns = []
    for i in range(2):
        for j in range(2):
            chip = nx.grid_2d_graph(12, 6)

            for node in chip.nodes:
                chip.nodes[node]['coord'] = node

            relabel_map = dict(zip(chip.nodes, ['q' + str(node) for node in range(len(chip.nodes))]))
            chip = nx.relabel_nodes(chip, relabel_map)

            patternid = patternids[i]
            maxParallelCZs = max_Algsubgraph(chip, patternid)
            
            rs = np.linspace(0.18, 3.5, 5)
            potential = []
            paras = []
            potentialNoise = []

            print('start')
            p = Pool(cpu_count)
            res = p.starmap(forward, zip(rs, [maxParallelCZs] * len(rs), [noises[j]] * len(rs)))
            p.close()
            p.join()
            print('stop')

            for r in res:
                potential.append(r.fun)
                paras.append(r.x)

            # print('pattern', patternid, 'dist', paras)
            print('pattern', patternid, 'potential', potential)
            patterns.append(potential)
    
    print(patterns)
    [[1.212010795119146, -4.5928154448693785, -5.429325842853678, -5.545680842235777, -4.8722562794981314], 
     [1.741170116252503, -4.125497539230559, -4.374265789651531, -4.533160092790075, -4.575109401477509], 
     [0.527307533318463, -5.067977290655992, -5.300554154588559, -5.424143337140477, -5.445194007326857], 
     [1.6895129042824693, -4.136793265837744, -4.346232278181424, -4.492054586677392, -4.54787120753259]]
    # scf = np.array([0.06015238712347859, -0.951395618161265, -1.3467653059649876, -1.5032000233449514, -1.556683225634944, -1.5619562578414556, -1.545064522558901, -1.5199068565910958, -1.4939941218029384, -1.4709665253423192, -1.4520965363272524, -1.4373874450774253, -1.4263065740072933, -1.418177343662933, -1.4123474640189402, -1.4082496104515272, -1.4054188632922697, -1.4034916672190163, -1.4021951143413167, -1.401331309255136, -1.4007610418282153, -1.4003877826334734, -1.4001459532997718, -1.3999894125186323, -1.3998935625604263, -1.3998330024142547, -1.3997959994005187, -1.3997741906948606, -1.399761395857274, -1.399754114241834, -1.3997478093482534, -1.3997467137205761, -1.3997461347699194, -1.3997458364889723, -1.3997456866353468, -1.3997456132218846, -1.39974557814624, -1.3997455618234744, -1.399745554398563, -1.3997455511067605, -1.3997455496828293, -1.3997455490817972, -1.399745548509505, -1.3997455487346366, -1.3997455486955626, -1.399745548672419, -1.3997455486749826, -1.3997455486729387, -1.39974554867221, -1.399745548671956])
    # pattern 1 dist [0.13772977195140826, -1.366858693174709, -1.449858642244784, -1.7918602306717129, -1.495986540116644]
    # exp0Mean = np.array([1.82612466, -1.12549587, -1.12277429, -1.20361023, -1.20077862]) 
    # exp0Var = np.array([0.01225732, 0.00743317, 0.00205057, 0.00108407, 0.00244083])
    # exp1Mean = np.array([2.02386862, -1.00826376, -1.0535917,  -1.01510382, -1.03903527])
    # exp1Var = np.array([0.00738289, 0.00366444, 0.00058952, 0.00405666, 0.00124379])

    # plt.scatter(np.linspace(0.18, 3.5, 5), potential0, label='theory 0')
    # plt.scatter(np.linspace(0.18, 3.5, 5), potential1, label='theory 1')
    # plt.errorbar(np.linspace(0.18, 3.5, 5), exp0Mean, exp0Var, fmt='o', elinewidth=2, capsize=5, label='exp pattern 0')
    # plt.errorbar(np.linspace(0.18, 3.5, 5), exp1Mean, exp1Var, fmt='o', elinewidth=2, capsize=5, label='exp pattern 1')
    # plt.plot(np.linspace(0.18, 3.5, 50), scf)
    # # plt.plot(np.abs(exp0 - potential0) / np.abs(potential0) * 100, label='exp pattern 0')
    # # plt.plot(np.abs(exp1 - potential1) / np.abs(potential1) * 100, label='exp pattern 1')
    # plt.plot()
    # plt.legend()
    # plt.show()