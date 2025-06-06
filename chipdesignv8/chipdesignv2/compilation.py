import numpy as np
import networkx as nx

from copy import deepcopy

# We import plotting tools 
import pydot
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# importing Qiskit
from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute

from qiskit.providers.ibmq import least_busy
# from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

from qiskit.quantum_info import Kraus, SuperOp, Operator, Pauli

# Import from Qiskit Aer noise module
# from qiskit.providers.aer.noise import NoiseModel
# from qiskit.providers.aer.noise import QuantumError, ReadoutError
# from qiskit.providers.aer.noise import pauli_error
# from qiskit.providers.aer.noise import depolarizing_error
# from qiskit.providers.aer.noise import thermal_relaxation_error

# from qiskit.providers.aer.noise import mixed_unitary_error
# from qiskit.providers.aer.noise import coherent_unitary_error
# from qiskit.providers.aer.noise import reset_error
# from qiskit.providers.aer.noise import phase_amplitude_damping_error
# from qiskit.providers.aer.noise import amplitude_damping_error
# from qiskit.providers.aer.noise import phase_damping_error
# from qiskit.providers.aer.noise import kraus_error

# compilation
from qiskit.transpiler import Layout, CouplingMap
from qiskit.transpiler import PassManager
from qiskit.transpiler import passes
from qiskit.transpiler import preset_passmanagers
from qiskit.transpiler import PropertySet, PassManagerConfig
from qiskit.compiler import transpile

class Compilation():
    def __init__(self, chip, in_qc, layoutMethod='ibm', routingMethod='ibm', setLayout = None):
        self.couplingMap = self.get_coupling_map(chip)
        ps = PassManager()
        if layoutMethod == 'chip':
            chipLayoutIBM = self.chip_layout_transform(in_qc[0], setLayout)[0]

            ps.append(passes.SetLayout(chipLayoutIBM))
        elif layoutMethod == 'li':
            liLayoutIBM = self.li_layout_transform(in_qc, setLayout)
            ps.append(passes.SetLayout(liLayoutIBM))
        else:
            ps.append(passes.DenseLayout(self.couplingMap))
        ps.append(passes.FullAncillaAllocation(self.couplingMap))
        ps.append(passes.ApplyLayout())

        if isinstance(in_qc, list):
            in_qc = in_qc[0]
        ps.append(passes.SabreSwap(self.couplingMap, heuristic='decay'))
        self.out_qc = ps.run(in_qc)
        self.out_qc_qasm = self.out_qc.qasm()
        print('compiled circuit: ', dict(self.out_qc.count_ops()), ' depth: ', self.out_qc.depth())
        self.additionSwap = dict(self.out_qc.count_ops()).get('swap', 0)
        self.additionDepth = self.out_qc.depth() - in_qc.depth()
        print('add ', self.additionSwap, ' swap ', ' ', self.additionDepth, 'depth')

    def get_coupling_map(self, chip):
        couplingMap = CouplingMap()
        for bit in list(chip.nodes):
        #for bit in range(len(chip.nodes)):
            couplingMap.add_physical_qubit(int(bit))
        for edge in list(chip.edges):
            couplingMap.add_edge(edge[0], edge[1])
        return couplingMap

    def li_layout_transform(self, in_qc, setLayout):
        iniLayoutDict = {}
        for map in setLayout:
            iniLayoutDict[map] = in_qc.qubits[setLayout[map]]
        iniLayout = Layout()
        iniLayout.from_dict(iniLayoutDict)
        return iniLayout

    def chip_layout_transform(self, in_qc, setLayout):
        iniLayoutDict = {}
        iniLayout = {}
        for q in setLayout:
        #for q in range(len(setLayout)):
            iniLayoutDict[setLayout[q][0]] = in_qc.qubits[q] # p to v
            iniLayout[q] = setLayout[q][0] # v to p
        iniLayoutIBM = Layout()
        iniLayoutIBM.from_dict(iniLayoutDict)
        return iniLayoutIBM, iniLayout