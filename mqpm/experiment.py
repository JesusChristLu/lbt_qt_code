# import math tools
import numpy as np

# importing Qiskit
from qiskit import Aer
from qiskit import execute
from qiskit.converters.dag_to_circuit import dag_to_circuit
# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

from xtalk import Xtalk
class Simulator:

    def __init__(self, qubit_belong=None):
        self.qubit_belong = qubit_belong
        

    def forward(self, qc, chip, backendprop, ini_layout, couplingMap, xtalk_matrix):
        backend = Aer.get_backend('qasm_simulator')
        shots = 10000
        basis_gates = ['u1', 'u2', 'u3', 'cx']
        T1s = [qubit[2].value * 1e3 for qubit in backendprop.qubits]
        T2s = [qubit[3].value * 1e3 for qubit in backendprop.qubits]
        time_u3 = 50
        time_cx = 30
        xtalk_data = Xtalk(xtalk_matrix)
        xtalk_data.get_circuit_subgraph(qc, chip, ini_layout)
        cx_edge = [(tuple(gate.qubits)) for gate in backendprop.gates if len(gate.qubits) == 2]
        u3_dep_error = [depolarizing_error(backendprop.gate_error('u3', qubit), 1) for qubit in range(qc.width() // 2)]
        u3_dec_error = [thermal_relaxation_error(T1s[qubit], T2s[qubit], time_u3) for qubit in range(qc.width() // 2)]
        u3_error = [u3_dep_error[qubit].compose(u3_dec_error[qubit]) for qubit in range(qc.width() // 2)]
        cr_dep_error = [depolarizing_error(backendprop.gate_error('cx', edge), 2) for edge in cx_edge]
        cr_dec_error = [thermal_relaxation_error(T1s[edge[0]], T2s[edge[0]], time_cx).\
                        expand(thermal_relaxation_error(T1s[edge[1]], T2s[edge[1]], time_cx)) for edge in cx_edge]
        cr_error = dict([(cx_edge[i], cr_dep_error[i].compose(cr_dec_error[i])) for i in range(len(cx_edge))])
        noise_model = NoiseModel()
        for j in range(qc.width() // 2):
            noise_model.add_quantum_error(u3_error[j], 'u3', [j])
            for k in range(j, qc.width() // 2):
                if (j, k) in cx_edge:
                    noise_model.add_quantum_error(cr_error[(j, k)], 'cx', [j, k])
        qc = dag_to_circuit(qc)
        simulate1 = execute(qc, backend=backend, shots=shots, coupling_map=couplingMap, basis_gates=basis_gates)
        simulate2 = execute(qc, backend=backend, shots=shots, coupling_map=couplingMap, basis_gates=basis_gates,
                            # backend_properties=backendprop, initial_layout=ini_layout)
                            noise_model=noise_model, initial_layout=ini_layout)

        result1 = simulate1.result()
        result2 = simulate2.result()

        count1 = dict(result1.get_counts())
        count1 = dict(sorted(count1.items(), key=lambda x : x[0]))

        count2 = dict(result2.get_counts())
        count2 = dict(sorted(count2.items(), key=lambda x : x[0]))
        
        return self.fidelity(count1, count2)

    def fidelity(self, result1, result2):
        if self.qubit_belong == None:
            length = np.linalg.norm(list(result1.values())) * np.linalg.norm(list(result2.values()))
            keys = set(result1.keys()).union(result2.keys())
            fidelities = 0
            for key in keys:
                if key in result1.keys() and key in result2.keys():
                    fidelities += result1[key] * result2[key]
            fidelities /= length
            fidelities = min(1.0, fidelities)
        else:
            fidelities = []
            for qubits in self.qubit_belong:
                start = self.qubit_belong[qubits][0]
                end = self.qubit_belong[qubits][-1] + 1
                length = end - start
                count1 = {}
                count2 = {}
                for num in range(2 ** length):
                    bin_num = bin(num)[2:]
                    while len(bin_num) < length:
                        bin_num = '0' + bin_num
                    count1[bin_num] = sum([result1[key] for key in result1.keys() if bin_num == key[-start - 1 : -end - 1 : -1][::-1]])
                    count2[bin_num] = sum([result2[key] for key in result2.keys() if bin_num == key[-start - 1 : -end - 1 : -1][::-1]])
                length = np.linalg.norm(list(count1.values())) * np.linalg.norm(list(count2.values()))
                keys = set(count1.keys()).union(count2.keys())
                fidelity = 0
                for key in keys:
                    if key in count1.keys() and key in count2.keys():
                        fidelity += count1[key] * count2[key]  
                fidelity /= length                 
                fidelities.append(min(1.0, fidelity))
        return fidelities