# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# 
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Crosstalk mitigation through adaptive instruction scheduling.
The scheduling algorithm is described in:
Prakash Murali, David C. McKay, Margaret Martonosi, Ali Javadi Abhari,
Software Mitigation of Crosstalk on Noisy Intermediate-Scale Quantum Computers,
in International Conference on Architectural Support for Programming Languages
and Operating Systems (ASPLOS), 2020.
Please cite the paper if you use this pass.

The method handles crosstalk noise on two-qubit gates. This includes crosstalk
with simultaneous two-qubit and one-qubit gates. The method ignores
crosstalk between pairs of single qubit gates.

The method assumes that all qubits get measured simultaneously whether or not
they need a measurement. This assumption is based on current device properties
and may need to be revised for future device generations.
"""

import math
import operator
import networkx as nx
import numpy as np
from copy import deepcopy
from itertools import chain, combinations

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import RXGate, CXGate, XGate, ZGate
from qiskit.circuit import Measure
from qiskit.circuit.barrier import Barrier
from qiskit.dagcircuit import DAGOpNode, DAGInNode
from qiskit.utils import optionals as _optionals
import z3

NUM_PREC = 10
TWOQ_XTALK_THRESH = 3
ONEQ_XTALK_THRESH = 2


@_optionals.HAS_Z3.require_in_instance
class CrosstalkAdaptiveSchedule(TransformationPass):
    """Crosstalk mitigation through adaptive instruction scheduling."""

    def __init__(self, chipG, backend_prop, crosstalk_prop, weight_factor=0.5, measured_qubits=None, maxParallelCZs=None):
        """CrosstalkAdaptiveSchedule initializer.

        Args:
            backend_prop (BackendProperties): backend properties object
            crosstalk_prop (dict): crosstalk properties object
                crosstalk_prop[g1][g2] specifies the conditional error rate of
                g1 when g1 and g2 are executed simultaneously.
                g1 should be a two-qubit tuple of the form (x,y) where x and y are physical
                qubit ids. g2 can be either two-qubit tuple (x,y) or single-qubit tuple (x).
                We currently ignore crosstalk between pairs of single-qubit gates.
                Gate pairs which are not specified are assumed to be crosstalk free.

                Example::

                    crosstalk_prop = {(0, 1) : {(2, 3) : 0.2, (2) : 0.15},
                                                (4, 5) : {(2, 3) : 0.1},
                                                (2, 3) : {(0, 1) : 0.05, (4, 5): 0.05}}

                The keys of the crosstalk_prop are tuples for ordered tuples for CX gates
                e.g., (0, 1) corresponding to CX 0, 1 in the hardware.
                Each key has an associated value dict which specifies the conditional error rates
                with nearby gates e.g., ``(0, 1) : {(2, 3) : 0.2, (2) : 0.15}`` means that
                CNOT 0, 1 has an error rate of 0.2 when it is executed in parallel with CNOT 2,3
                and an error rate of 0.15 when it is executed in parallel with a single qubit
                gate on qubit 2.
            weight_factor (float): weight of gate error/crosstalk terms in the objective
                :math:`weight_factor*fidelities + (1-weight_factor)*decoherence errors`.
                Weight can be varied from 0 to 1, with 0 meaning that only decoherence
                errors are optimized and 1 meaning that only crosstalk errors are optimized.
                weight_factor should be tuned per application to get the best results.
            measured_qubits (list): a list of qubits that will be measured in a particular circuit.
                This arg need not be specified for circuits which already include measure gates.
                The arg is useful when a subsequent module such as state_tomography_circuits
                inserts the measure gates. If CrosstalkAdaptiveSchedule is made aware of those
                measurements, it is included in the optimization.
        Raises:
            ImportError: if unable to import z3 solver

        """

        super().__init__()
        self.chipG = chipG
        self.backend_prop = backend_prop
        self.crosstalk_prop = crosstalk_prop
        self.weight_factor = weight_factor
        if measured_qubits is None:
            self.input_measured_qubits = []
        else:
            self.input_measured_qubits = measured_qubits
        self.bp_u1_err = {}
        self.bp_u1_dur = {}
        self.bp_u2_err = {}
        self.bp_u2_dur = {}
        self.bp_u3_err = {}
        self.bp_u3_dur = {}
        self.bp_z_err = {}
        self.bp_z_dur = {}
        self.bp_x_err = {}
        self.bp_x_dur = {}
        self.bp_cx_err = {}
        self.bp_cx_dur = {}
        self.bp_t1_time = {}
        self.bp_t2_time = {}
        self.gate_id = {}
        self.gate_start_time = {}
        self.gate_duration = {}
        self.gate_fidelity = {}
        self.gate_time = {}
        self.overlap_amounts = {}
        self.overlap_indicator = {}
        self.qubit_lifetime = {}
        self.dag_overlap_set = {}
        self.xtalk_overlap_set = {}
        # self.opt = z3.Optimize()
        self.measured_qubits = []
        self.measure_start = None
        self.last_gate_on_qubit = None
        self.first_gate_on_qubit = None
        self.fidelity_terms = []
        self.coherence_terms = []
        self.model = None
        self.dag = None
        self.parse_backend_properties()
        self.qubit_indices = None
        self.maxParallelCZs = maxParallelCZs

    def powerset(self, iterable):
        """
        Finds the set of all subsets of the given iterable
        This function is used to generate constraints for the Z3 optimization
        """
        l_s = list(iterable)
        return chain.from_iterable(combinations(l_s, r) for r in range(len(l_s) + 1))


    def parse_backend_properties(self):
        """
        This function assumes that gate durations and coherence times
        are in seconds in backend.properties()
        This function converts gate durations and coherence times to
        nanoseconds.
        """
        backend_prop = self.backend_prop
        for qid in range(len(backend_prop.qubits)):
            self.bp_t1_time[qid] = int(backend_prop.t1(qid) * 10**9)
            self.bp_t2_time[qid] = int(backend_prop.t2(qid) * 10**9)
            self.bp_z_dur[qid] = int(backend_prop.gate_length("z", qid)) * 10**9
            z_err = backend_prop.gate_error("z", qid)
            if z_err == 1.0:
                z_err = 0.9999
            self.bp_z_err = round(z_err, NUM_PREC)
            self.bp_x_dur[qid] = int(backend_prop.gate_length("x", qid)) * 10**9
            x_err = backend_prop.gate_error("x", qid)
            if x_err == 1.0:
                x_err = 0.9999
            self.bp_x_err[qid] = round(x_err, NUM_PREC)

            self.bp_u1_dur[qid] = int(backend_prop.gate_length("u1", qid)) * 10**9
            u1_err = backend_prop.gate_error("u1", qid)
            if u1_err == 1.0:
                u1_err = 0.9999
            self.bp_u1_err[qid] = round(u1_err, NUM_PREC)
            self.bp_u2_dur[qid] = int(backend_prop.gate_length("u2", qid)) * 10**9
            u2_err = backend_prop.gate_error("u2", qid)
            if u2_err == 1.0:
                u2_err = 0.9999
            self.bp_u2_err[qid] = round(u2_err, NUM_PREC)
            self.bp_u3_dur[qid] = int(backend_prop.gate_length("u3", qid)) * 10**9
            u3_err = backend_prop.gate_error("u3", qid)
            if u3_err == 1.0:
                u3_err = 0.9999
            self.bp_u3_err[qid] = round(u3_err, NUM_PREC)
            
        for ginfo in backend_prop.gates:
            if ginfo.gate == "cx":
                q_0 = ginfo.qubits[0]
                q_1 = ginfo.qubits[1]
                cx_tup = (min(q_0, q_1), max(q_0, q_1))
                self.bp_cx_dur[cx_tup] = int(backend_prop.gate_length("cx", cx_tup)) * 10**9
                cx_err = backend_prop.gate_error("cx", cx_tup)
                if cx_err == 1.0:
                    cx_err = 0.9999
                self.bp_cx_err[cx_tup] = round(cx_err, NUM_PREC)


    def cx_tuple(self, gate):
        """
        Representation for two-qubit gate
        Note: current implementation assumes that the CX error rates and
        crosstalk behavior are independent of gate direction
        """
        physical_q_0 = self.qubit_indices[gate.qargs[0]]
        physical_q_1 = self.qubit_indices[gate.qargs[1]]
        r_0 = min(physical_q_0, physical_q_1)
        r_1 = max(physical_q_0, physical_q_1)
        return (r_0, r_1)


    def singleq_tuple(self, gate):
        """
        Representation for single-qubit gate
        """
        physical_q_0 = self.qubit_indices[gate.qargs[0]]
        tup = (physical_q_0,)
        return tup


    def gate_tuple(self, gate):
        """
        Representation for gate
        """
        if len(gate.qargs) == 2:
            return self.cx_tuple(gate)
        else:
            return self.singleq_tuple(gate)


    def assign_gate_id(self, dag):
        """
        ID for each gate
        """
        idx = 0
        for gate in dag.gate_nodes():
            self.gate_id[gate] = idx
            idx += 1


    def extract_dag_overlap_sets(self, dag):
        """
        Gate A, B are overlapping if
        A is neither a descendant nor an ancestor of B.
        Currenty overlaps (A,B) are considered when A is a 2q gate and
        B is either 2q or 1q gate.
        """
        for gate in dag.two_qubit_ops():
            overlap_set = []
            descendants = dag.descendants(gate)
            ancestors = dag.ancestors(gate)
            for tmp_gate in dag.gate_nodes():
                if tmp_gate == gate:
                    continue
                if tmp_gate in descendants:
                    continue
                if tmp_gate in ancestors:
                    continue
                overlap_set.append(tmp_gate)
            self.dag_overlap_set[gate] = overlap_set

    def extract_gate_times(self, dag, isreturn=False):
        single_gate_time = 30
        two_gate_time = 60
        
        gate_time = dict()
        for qubit in dag.nodes():
            if isinstance(qubit, DAGInNode):
                gate_time[qubit] = (0, 0)
            
        for gate in dag.topological_op_nodes():
            ancestors = dag.ancestors(gate)
            startTime = max([gate_time[ancestor][1] for ancestor in ancestors])
            if gate in dag.two_qubit_ops():
                gate_time[gate] = (startTime, startTime + two_gate_time)
            elif gate.name == 'barrier':
                gate_time[gate] = (startTime, startTime)
            else:
                gate_time[gate] = (startTime, startTime + single_gate_time)
        if isreturn:
            output_gate_time = dict()
            for gate in gate_time:
                if isinstance(gate, DAGOpNode):
                    output_gate_time[gate] = gate_time[gate]
            return output_gate_time
        else:
            for gate in gate_time:
                if isinstance(gate, DAGOpNode):
                    self.gate_time[gate] = gate_time[gate]
        

    def is_significant_xtalk(self, gate1, gate2):
        """
        Given two conditional gate error rates
        check if there is high crosstalk by comparing with independent error rates.
        """
        gate1_tup = self.gate_tuple(gate1)
        if len(gate2.qargs) == 2:
            gate2_tup = self.gate_tuple(gate2)
            independent_err_g_1 = self.bp_cx_err[gate1_tup]
            independent_err_g_2 = self.bp_cx_err[gate2_tup]
            rg_1 = self.crosstalk_prop[gate1_tup][gate2_tup] / independent_err_g_1
            rg_2 = self.crosstalk_prop[gate2_tup][gate1_tup] / independent_err_g_2
            if rg_1 > TWOQ_XTALK_THRESH or rg_2 > TWOQ_XTALK_THRESH:
                return True
        else:
            gate2_tup = self.gate_tuple(gate2)
            independent_err_g_1 = self.bp_cx_err[gate1_tup]
            rg_1 = self.crosstalk_prop[gate1_tup][gate2_tup] / independent_err_g_1
            if rg_1 > ONEQ_XTALK_THRESH:
                return True
        return False


    def extract_crosstalk_relevant_sets(self):
        """
        Extract the set of program gates which potentially have crosstalk noise
        """
        for gate in self.dag_overlap_set:
            self.xtalk_overlap_set[gate] = []
            tup_g = self.gate_tuple(gate)
            if tup_g not in self.crosstalk_prop:
                continue
            for par_g in self.dag_overlap_set[gate]:
                tup_par_g = self.gate_tuple(par_g)
                if tup_par_g in self.crosstalk_prop[tup_g]:
                    if self.is_significant_xtalk(gate, par_g):
                        if par_g not in self.xtalk_overlap_set[gate]:
                            self.xtalk_overlap_set[gate].append(par_g)

    def check_dag_dependency(self, gate1, gate2):
        """
        gate2 is a DAG dependent of gate1 if it is a descendant of gate1
        """
        return gate2 in self.dag.descendants(gate1) or gate1 in self.dag.descendants(gate2)


    def check_xtalk_dependency(self, t_1, t_2):
        """
        Check if two gates have a crosstalk dependency.
        We do not consider crosstalk between pairs of single qubit gates.
        """
        g_1 = t_1[0]
        s_1 = t_1[1]
        f_1 = t_1[2]
        g_2 = t_2[0]
        s_2 = t_2[1]
        f_2 = t_2[2]
        if len(g_1.qargs) == 1 and len(g_2.qargs) == 1:
            return False, ()
        if s_2 <= f_1 and s_1 <= f_2:
            # Z3 says it's ok to overlap these gates,
            # so no xtalk dependency needs to be checked
            return False, ()
        else:
            # Assert because we are iterating in Z3 gate start time order,
            # so if two gates are not overlapping, then the second gate has to
            # start after the first gate finishes
            assert s_2 >= f_1
            # Not overlapping, but we care about this dependency
            if len(g_1.qargs) == 2 and len(g_2.qargs) == 2:
                cx1 = self.cx_tuple(g_1)
                cx2 = self.cx_tuple(g_2)
                if cx1 in self.crosstalk_prop[cx2]:
                    barrier = tuple(sorted([cx1[0], cx1[1], cx2[0], cx2[1]]))
                    return True, barrier
            elif len(g_1.qargs) == 1 and len(g_2.qargs) == 2:
                singleq = self.gate_tuple(g_1)
                cx1 = self.cx_tuple(g_2)
                if singleq in self.crosstalk_prop[cx1]:
                    print(singleq, cx1)
                    barrier = tuple(sorted([singleq, cx1[0], cx1[1]]))
                    return True, barrier
            elif len(g_1.qargs) == 2 and len(g_2.qargs) == 1:
                singleq = self.gate_tuple(g_2)
                cx1 = self.cx_tuple(g_1)
                if singleq in self.crosstalk_prop[cx1]:
                    barrier = tuple(sorted([singleq, cx1[0], cx1[1]]))
                    return True, barrier
            # Not overlapping, and we don't care about xtalk between these two gates
            return False, ()


    def filter_candidates(self, layer, triplet):
        """
        For a gate G and layer L,
        L is a candidate layer for G if no gate in L has a DAG dependency with G,
        and if Z3 allows gates in L and G to overlap.
        """
        curr_start_time = triplet[1]
        curr_end_time = triplet[2]
        prev_start_time = -1
        prev_end_time = -1
        for prev_triplet in layer:
            if prev_triplet[1] < prev_start_time or prev_start_time == -1:
                prev_start_time = prev_triplet[1]
            if prev_triplet[2] > prev_end_time or prev_start_time == -1:
                prev_end_time = prev_triplet[2]
            # is_dag_dep = self.check_dag_dependency(prev_gate, curr_gate)
            if curr_end_time <= prev_start_time or prev_end_time <= curr_start_time:
                return False
            else:
                return True

    def find_layer(self, layers, triplet):
        """
        Find the appropriate layer for a gate
        """
        candidates = False
        for i, layer in enumerate(layers):
            candidates = self.filter_candidates(layer, triplet)
            if candidates:
                break
        if not candidates:
            return len(layers)
            # Open a new layer
        else:
            return i
        
    def generate_barriers(self, layers, isSchedule):
        """
        For each gate g, see if a barrier is required to serialize it with
        some previously processed gate
        """
        xTalkG = nx.Graph()
        # for coupler in self.crosstalk_prop:
            # xTalkG.add_node(coupler)
        for coupler, neighbors in self.crosstalk_prop.items():
            for neighbor in neighbors:
                xTalkG.add_edge(coupler, neighbor)
                xTalkG.edges[(coupler, neighbor)]['w'] = self.crosstalk_prop[coupler][neighbor]
        
        if isSchedule == 'static':
            for edge in xTalkG.edges:
                for maxParallelCz in self.maxParallelCZs:
                    if edge[0] in maxParallelCz and edge[1] in maxParallelCz:
                        xTalkG.remove_edge(edge[0], edge[1])
                        break

        barriers = []
        for _, layer in enumerate(layers):
            tup_gs = dict()
            for gate in layer:
                tup_g = self.gate_tuple(gate[0])

                if len(tup_g) == 2:
                    tup_gs[gate] = tup_g
            if len(tup_gs) > 2:
                subXtalkG = xTalkG.subgraph(tup_gs.values())
                if isSchedule == True or isSchedule == 'static':
                    n_cluster = 2
                    while 1:
                        partitions, labels = self.split_graph(subXtalkG, n_cluster)
                        xtalkSerious = False
                        for partition in partitions:
                            subLayerG = nx.subgraph(subXtalkG, partition)
                            for component in nx.connected_components(subLayerG):
                                if isSchedule == 'static' and len(component) >= 2:
                                    xtalkSerious = True
                                    break
                                elif len(component) > 2:
                                    xtalkSerious = True
                                    break
                                elif len(component) == 1:
                                    continue
                                elif np.max(np.array([[nx.shortest_path_length(self.chipG, i, j) \
                                                       for i in list(component)[0] if not(i == j)] \
                                                      for j in list(component)[1]]).ravel()) > 2:
                                    xtalkSerious = True
                                    break
                            if xtalkSerious:
                                break
                        if xtalkSerious:
                            n_cluster += 1
                        else:
                            break
                elif isSchedule == 'serial':
                    labels = dict()
                    partitions = []
                    for component in nx.connected_components(subXtalkG):
                        label = 0
                        for g in component:
                            labels[g] = label
                            if label + 1 > len(partitions):
                                partitions.append([])
                            partitions[label].append(g)
                            label += 1

                for label in labels:
                    tup_gs[list(tup_gs.keys())[list(tup_gs.values()).index(label)]] = labels[label]
            else:
                for tup_g in tup_gs:
                    tup_gs[tup_g] = 0
            barriers.append(tup_gs)
        return barriers
    

    def split_graph(self, G, n):

        if not isinstance(G, nx.Graph) or not isinstance(n, int) or n < 1:
            raise ValueError("Invalid input")
        
        if n > G.number_of_nodes():
            return

        labels = dict()
        Gcopy = nx.Graph()
        for node in G.nodes():
            Gcopy.add_node(node)
        for edge in G.edges():
            Gcopy.add_edge(edge[0], edge[1])
            Gcopy.edges[edge]['w'] = G.edges[edge]['w']

        nx.set_node_attributes(Gcopy, Gcopy.degree(), "degree")

        partitionNum = 0
        while len(Gcopy.nodes) > 0:
            # 获取最大独立集
            max_independent_set = nx.maximal_independent_set(Gcopy)

            # # 输出最大独立集
            # print("Max independent set:", max_independent_set)

            # 从图中删除最大独立集对应节点
            Gcopy.remove_nodes_from(max_independent_set)

            for node in max_independent_set:
                labels[node] = partitionNum
            partitionNum += 1

        if len(Gcopy.nodes) > 0:
            for node in Gcopy.nodes:
                labels[node] = partitionNum

        if partitionNum > n:
            labels = dict()
            # 计算节点的度
            degree_sorted_nodes = sorted(dict(G.degree).items(), key=lambda x: x[1], reverse=True)

            # 分配节点到子图
            for i, node in enumerate(degree_sorted_nodes):
                labels[node[0]] = i % n

        partition = []
        label = 0
        while label <= max(labels.values()):
            partition.append([])
            for i in labels:
                if labels[i] == label:
                    partition[-1].append(i)
            label += 1
        return partition, labels

    def create_updated_dag(self, layers, barriers):
        """
        Given a set of layers and barriers, construct a new dag
        """
        new_dag = DAGCircuit()
        for qreg in self.dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in self.dag.cregs.values():
            new_dag.add_creg(creg)

        for barrier, layer in zip(barriers, layers):
            haveAlloc = []
            if barrier == {}:
                for gate in layer:
                    new_dag.apply_operation_back(gate[0].op, gate[0].qargs, gate[0].cargs)
            else:
                for label in range(max(lb for lb in barrier.values()) + 1):
                    for gate in layer:
                        if not(gate in haveAlloc):
                            if len(self.gate_tuple(gate[0])) == 2:
                                if barrier[gate] == label:
                                    new_dag.apply_operation_back(gate[0].op, gate[0].qargs, gate[0].cargs)
                                    haveAlloc.append(gate)
                            else:
                                new_dag.apply_operation_back(gate[0].op, gate[0].qargs, gate[0].cargs)
                                haveAlloc.append(gate)
                    if len(haveAlloc) <= len(layer):
                        new_dag.apply_operation_back(Barrier(len(new_dag.qubits)), new_dag.qubits)        
                # new_dag.apply_operation_back(Barrier(len(new_dag.qubits)), new_dag.qubits)        
        for node in self.dag.op_nodes():
            if isinstance(node.op, Measure):
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag


    def enforce_schedule_on_dag(self, input_gate_times, isSchedule):
        """
        Some gates need to be serialized to implement the Z3 schedule.
        This function inserts barriers to implement those serializations
        """
        gate_times = []
        for key in input_gate_times:
            gate_times.append((key, input_gate_times[key][0], input_gate_times[key][1]))
        # Sort gates by start time
        sorted_gate_times = sorted(gate_times, key=operator.itemgetter(1))
        layers = []

        # Construct a set of layers. Each layer has a set of gates that
        for triplet in sorted_gate_times:
            layer_idx = self.find_layer(layers, triplet)
            if layer_idx == len(layers):
                layers.append([triplet])
            else:
                layers[layer_idx].append(triplet)

        # for layer in layers:
        #     for l in layer:
        #         print(l[1], l[2])
        #     print('\n')
        # Insert barriers if necessary to enforce the above layers
        if isSchedule:
            barriers = self.generate_barriers(layers, isSchedule)
            new_dag = self.create_updated_dag(layers, barriers)
        else:
            new_dag = self.dag

        output_gate_times = self.extract_gate_times(new_dag, isreturn=True)

        gate_times = []
        for key in output_gate_times:
            gate_times.append((key, output_gate_times[key][0], output_gate_times[key][1]))
        # sorted_gate_times = sorted(gate_times, key=operator.itemgetter(1))
        layers = []

        for triplet in gate_times:
            layer_idx = self.find_layer(layers, triplet)
            if layer_idx == len(layers):
                layers.append([triplet])
            else:
                layers[layer_idx].append(triplet)

        return new_dag, output_gate_times, layers


    def EPST(self, layers, isSchedule, isDynamic=False):
        xTalkG = nx.Graph()
        # for coupler in self.crosstalk_prop:
            # xTalkG.add_node(coupler)
        for coupler, neighbors in self.crosstalk_prop.items():
            for neighbor in neighbors:
                xTalkG.add_edge(coupler, neighbor)
                xTalkG.edges[(coupler, neighbor)]['w'] = self.crosstalk_prop[coupler][neighbor]

        if isSchedule == 'static':
            for edge in xTalkG.edges:
                for maxParallelCz in self.maxParallelCZs:
                    if edge[0] in maxParallelCz and edge[1] in maxParallelCz:
                        xTalkG.remove_edge(edge[0], edge[1])
                        break

        EPST = 1
        for layer in layers:
            tup_gs = dict()
            for triplet in layer:
                gate = triplet[0]
                tup_g = self.gate_tuple(gate)
                if len(tup_g) == 2:
                    tup_gs[gate] = tup_g
                else:
                    EPST *= 1 - self.bp_u3_err[tup_g[0]]
            subXtalkG = xTalkG.subgraph(tup_gs.values())
            for connectComponent in nx.connected_components(subXtalkG):
                subsubXtalkG = subXtalkG.subgraph(connectComponent)
                if len(connectComponent) >= 2 and isSchedule == 'static':
                    for gate in subsubXtalkG.nodes():
                        EPST *= 1 - (self.bp_cx_err[gate] + sum([xTalkG.edges()[edge]['w'] for edge in subsubXtalkG.edges() if gate in edge]))                    
                elif len(connectComponent) > 2 or \
                    (len(connectComponent) == 2 and \
                    np.max(np.array([[nx.shortest_path_length(self.chipG, i, j) for i in list(connectComponent)[0] if not(i == j)] for j in list(connectComponent)[1]]).ravel()) > 2):
                    for gate in subsubXtalkG.nodes():
                        EPST *= 1 - (self.bp_cx_err[gate] + sum([xTalkG.edges()[edge]['w'] for edge in subsubXtalkG.edges() if gate in edge]))
                else:
                    for gate in subsubXtalkG.nodes():
                        if isDynamic:
                            EPST *= 1 - (self.bp_cx_err[gate] * 10)
                        else:
                            EPST *= 1 - self.bp_cx_err[gate]
        return EPST


    def reset(self):
        """
        Reset variables
        """
        self.gate_id = {}
        self.gate_start_time = {}
        self.gate_duration = {}
        self.gate_fidelity = {}
        self.overlap_amounts = {}
        self.overlap_indicator = {}
        self.qubit_lifetime = {}
        self.dag_overlap_set = {}
        self.xtalk_overlap_set = {}
        self.measured_qubits = []
        self.measure_start = None
        self.last_gate_on_qubit = None
        self.first_gate_on_qubit = None
        self.fidelity_terms = []
        self.coherence_terms = []
        self.model = None


    def run(self, dag, isSchedule, isDynamic=False):
        """
        Main scheduling function
        """
        self.dag = dag

        # process input program
        self.qubit_indices = {bit: idx for idx, bit in enumerate(dag.qubits)}
        self.assign_gate_id(self.dag)
        self.extract_gate_times(self.dag)

        # post-process to insert barriers
        new_dag, newGateTime, layers = self.enforce_schedule_on_dag(self.gate_time, isSchedule)

        EPST = self.EPST(layers, isSchedule, isDynamic)
        maxGateTime = 0
        for gatetime in newGateTime:
            if newGateTime[gatetime][-1] > maxGateTime:
                maxGateTime = newGateTime[gatetime][-1]

        self.reset()

        # for gate in newGateTime:
        #     if len(gate.qargs) == 2:
        #         print(gate.name, gate.qargs[0].index, gate.qargs[1].index, newGateTime[gate])
        #     elif gate.name == 'barrier':
        #         continue
        #     else:
        #         print(gate.name, gate.qargs[0].index, newGateTime[gate])

        # print(maxGateTime)
        # print('\n')

        return new_dag, maxGateTime, EPST