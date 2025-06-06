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
from qiskit.transpiler import PropertySet, PassManagerConfig
from qiskit.compiler import transpile

class Compilation():
    def __init__(self, chip, in_qc, layoutMethod='ibm', routingMethod='ibm', setLayout = None):
        print(len(chip.nodes) - len(in_qc.qubits))
        self.couplingMap = self.get_coupling_map(chip)
        ps = PassManager()
        if layoutMethod == 'chip':
            chipLayoutIBM, chipLayout = self.chip_layout_transform(in_qc, setLayout)
            print(chipLayoutIBM)
            ps.append(passes.SetLayout(chipLayoutIBM))
            #ps.append(passes.DenseLayout(self.couplingMap))
            #ps.append(passes.SabreLayout(self.couplingMap))
            #ps.append(passes.CSPLayout(self.couplingMap))
        elif layoutMethod == 'li':
            liLayoutIBM = self.li_layout_transform(in_qc, setLayout)
            print(liLayoutIBM)
            ps.append(passes.SetLayout(liLayoutIBM))
        else:
            #ps.append(passes.TrivialLayout(self.couplingMap))
            ps.append(passes.DenseLayout(self.couplingMap))
            #ps.append(passes.CSPLayout(self.couplingMap))
            #ps.append(passes.SabreLayout(self.couplingMap))
        ps.append(passes.FullAncillaAllocation(self.couplingMap))
        #ps.append(passes.EnlargeWithAncilla())
        ps.append(passes.ApplyLayout())

        #if routingMethod == 'chip':

        #    self.out_qc = self.chip_routing(chip, in_qc, chipLayout, setLayout)
        #    print(len(in_qc))
        #    print(len(self.out_qc))
        #else:
        ps.append(passes.SabreSwap(self.couplingMap))
        #ps.append(passes.BasicSwap(self.couplingMap))
        #ps.append(passes.LookaheadSwap(self.couplingMap))
        #ps.append(passes.StochasticSwap(self.couplingMap))
        self.out_qc = ps.run(in_qc)
        #self.out_qc2 = transpile(in_qc, coupling_map=self.couplingMap, basic_gates=['u1', 'cx', 'h', 'x'], optimization_level=3)
        #print(self.out_qc2.__len__(), self.out_qc2.depth())
        print(in_qc.__len__(), in_qc.depth())
        print(self.out_qc.__len__(), self.out_qc.depth())


    def get_coupling_map(self, chip):
        couplingMap = CouplingMap()
        for bit in list(chip.nodes):
            couplingMap.add_physical_qubit(bit)
        for edge in list(chip.edges):
            couplingMap.add_edge(edge[0], edge[1])
        return couplingMap

    def chip_layout_transform(self, in_qc, setLayout):
        iniLayout = {}
        iniLayoutDict = {}
        iniLayoutIBM = Layout()
        for q in setLayout:
            iniLayout[q] = setLayout[q][0]
            iniLayoutDict[setLayout[q][0]] = in_qc.qubits[q]
        iniLayoutIBM.from_dict(iniLayoutDict)
        return iniLayoutIBM, iniLayout

    def li_layout_transform(self, in_qc, setLayout):
        iniLayoutDict = {}
        for map in setLayout:
            iniLayoutDict[map] = in_qc.qubits[setLayout[map]]
        iniLayout = Layout()
        iniLayout.from_dict(iniLayoutDict)
        return iniLayout

    def chip_routing(self, chip, in_qc, layout, alternativeLayout):
        out_qc = deepcopy(in_qc)
        offset = 0
        for gate in in_qc:
            if len(gate) == 2:
                if not (layout[gate[0]], layout[gate[1]]) in chip.edges:
                    shortestPath = nx.shortest_path(chip, source=layout[gate[0]], target=layout[gate[1]])


                    #for edge in chip.edges:
                    #    if len(set(edge).intersection(set(shortestPath))) == 2:
                    #        chip.remove_edge(edge[0], edge[1])
                    #        chip.add_edge(edge[0], edge[1], color='r')
                    #    else:
                    #        chip.remove_edge(edge[0], edge[1])
                    #        chip.add_edge(edge[0], edge[1], color='black')
                    #colors = [chip[u][v]['color'] for u, v in chip.edges]

                    for bit in range(len(shortestPath) - 1):
                        out_qc.insert(in_qc.index(gate) + offset, [shortestPath[bit], shortestPath[bit + 1]])
                        offset += 1
                    for bit in range(len(shortestPath) - 1, 0, - 1):
                        out_qc.insert(in_qc.index(gate) + offset + 1, [shortestPath[bit], shortestPath[bit - 1]])
                        offset += 1
        return out_qc