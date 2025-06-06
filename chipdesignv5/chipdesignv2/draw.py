from get_alg import Get_alg
from clustering import Clustering
from prune import Prune
from split import Split
from lattice_chip import Lattice
from compilation import Compilation
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt

# importing Qiskit
from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute

from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

from qiskit.quantum_info import Kraus, SuperOp, Operator, Pauli

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

from qiskit.providers.aer.noise import mixed_unitary_error
from qiskit.providers.aer.noise import coherent_unitary_error
from qiskit.providers.aer.noise import reset_error
from qiskit.providers.aer.noise import phase_amplitude_damping_error
from qiskit.providers.aer.noise import amplitude_damping_error
from qiskit.providers.aer.noise import phase_damping_error
from qiskit.providers.aer.noise import kraus_error

# compilation
from qiskit.transpiler import Layout, CouplingMap
from qiskit.transpiler import PassManager
from qiskit.transpiler import passes
from qiskit.transpiler import preset_passmanagers
from qiskit.transpiler import PropertySet
from qiskit.compiler import transpile


if __name__=="__main__":
    show = True
    path = 'F:\\vs experiment\\chipdesignv5\\chipdesignv2\\compile\\the good guy\\good.txt'
    with open(path, 'r') as fp:
        path_list = fp.read()
    path_list = path_list.split('\n')
    total = 0
    #for file_name in path_list['sym6_145':]:
    for file_name in [path_list[path_list.index('cnt3-5_179.qasm')]]:
        file_name = 'F:\\vs experiment\\chipdesignv2\\ibm_qx_mapping-master\\examples\\' + file_name
        print(file_name)
        alg = Get_alg(os.path.join(path, file_name), show)