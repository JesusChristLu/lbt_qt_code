from pyqpanda import *
from math import pi
import copy
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time

class Machine():
    def __init__(self, kind, n, noise_model='', precision=''):
        self.qubit_num = n
        if kind == '00':
            self.machine = CPUQVM()
            self.machine.init_qvm()
        elif kind == '01':
            self.machine = NoiseQVM()
            self.machine.init_qvm()
            self.machine.set_rotation_angle_error(precision)
        elif kind == '10':
            self.machine = NoiseQVM()
            self.machine.init_qvm(noise_model)
        else:
            self.machine = NoiseQVM()
            self.machine.init_qvm(noise_model)
            self.machine.set_rotation_angle_error(precision)
        print(self.machine)

class Circuit():
    def __init__(self, beta, hamiltonian, qlist, p=1):
        self.hamiltonian = hamiltonian
        self.qlist = qlist
        self.p = p
        self.circuit = self.circuit_f(self.qlist, self.hamiltonian, self.p, beta)

    def update(self, beta):
        self.circuit = self.circuit_f(self.qlist, self.hamiltonian, self.p, beta)
    '''
    def old_circuit(self, qlist, Hamiltonian, step, beta):
        qc = QCircuit()
        for i in qlist:
            qc.insert(RX(i, pi))
            qc.insert(RY(i, pi * 0.5))
        for layer in range(step):
            for i in range(len(Hamiltonian)):
                tmp_vec=[]
                item=Hamiltonian[i]
                dict_p = item[0]
                for iter in dict_p:
                    if 'Z'!= dict_p[iter]:
                        pass
                    tmp_vec.append(qlist[iter])

                coef = item[1]

                if 2 != len(tmp_vec):
                    pass
                qc.insert(RX(tmp_vec[1], pi))
                qc.insert(RY(tmp_vec[1], pi * 0.5))
                qc.insert(CZ(tmp_vec[0], tmp_vec[1]))
                qc.insert(RX(tmp_vec[1], pi))
                qc.insert(RY(tmp_vec[1], pi))
                qc.insert(RX(tmp_vec[1], beta[layer][0] * coef))
                qc.insert(RY(tmp_vec[1], pi * 1.5))
                qc.insert(RX(tmp_vec[1], pi))
                qc.insert(RY(tmp_vec[1], pi * 0.5))
                qc.insert(CZ(tmp_vec[0], tmp_vec[1]))
                qc.insert(RX(tmp_vec[1], pi))
                qc.insert(RY(tmp_vec[1], pi * 0.5))

            for j in qlist:
                qc.insert(RX(j, 2 * beta[layer][1]))
        return qc
    '''
    def circuit_f(self, qlist, Hamiltonian, step, beta):
        qc = QCircuit()
        for i in qlist:
            qc.insert(H(i))
        for layer in range(step):
            for i in range(len(Hamiltonian)):
                tmp_vec=[]
                item=Hamiltonian[i]
                dict_p = item[0]
                for iter in dict_p:
                    if 'Z'!= dict_p[iter]:
                        pass
                    tmp_vec.append(qlist[iter])

                coef = item[1]

                if 2 != len(tmp_vec):
                    pass
                qc.insert(RZ(tmp_vec[0], beta[layer][0] * coef))
                qc.insert(RZ(tmp_vec[1], beta[layer][0] * coef))
                qc.insert(CR(tmp_vec[0], tmp_vec[1], -2 * beta[layer][0] * coef))

            for j in qlist:
                qc.insert(RY(j, -np.pi / 2))
                qc.insert(RZ(j, 2 * beta[layer][1]))
                qc.insert(RY(j, np.pi / 2))
        return qc

class QProgram():
    def __init__(self, machine_kind, shots, problem, beta,
                 noise_model='', precision='', circuit=''):
        self.hp = PauliOperator(problem).to_hamiltonian(1)
        self.qubit_num = PauliOperator(problem).getMaxIndex()
        self.shots = shots
        self.beta = beta
        self.step = len(beta)
        self.machine = Machine(machine_kind, self.qubit_num, 
                               noise_model=noise_model, precision=precision)
        self.qlist = self.machine.machine.qAlloc_many(self.qubit_num)
        self.clist = self.machine.machine.cAlloc_many(self.qubit_num)
        if circuit == '':
            print('QAOA')
            self.circuit = Circuit(beta, 
                                   self.hp, self.qlist, p=self.step)
        self.max_loss = self.get_max_loss()

    def get_max_loss(self):
        max_loss = 0
        for i in range(len(self.hp)):
            item = self.hp[i]
            max_loss += item[1]
        return max_loss

    def get_Hp_mat(self):
        Hp_mat = cp.zeros((1 << self.qubit_num, 1 << self.qubit_num), dtype = 'float64')
        for i in range(len(self.hp)):
            item = self.hp[i]
            Hp_mat -= 0.5 * cp.eye((1 << self.qubit_num)) * item[1]
            dict_p = item[0]
            z1z2 = cp.eye(1 << self.qubit_num)
            for iter in dict_p:
                temp = cp.array([[1, 0],[0, -1]], dtype = 'float64')
                temp = cp.kron(cp.eye(1 << iter, dtype = 'float64'), temp)
                temp = cp.kron(temp, cp.eye(1 << (self.qubit_num - 1 - iter), dtype = 'float64'))
                z1z2 = cp.dot(temp, z1z2)
            Hp_mat += z1z2 * 0.5 * item[1]
        return Hp_mat

    def get_loss_mat(self):
        prog = QProg()
        prog.insert(self.circuit.circuit)
        self.machine.machine.directly_run(prog)
        psi_out = cp.array(self.machine.machine.get_qstate())
        psi_out = psi_out.reshape(1, len(psi_out))
        loss = cp.dot(cp.dot(psi_out, 
                       self.get_Hp_mat()), 
                       cp.conj(psi_out).T)
        return loss.real / self.max_loss
    
    def get_prob_shot(self):
        prog = QProg()
        prog.insert(self.circuit.circuit)
        prog.insert(measure_all(self.qlist, self.clist))
        config = {'shots': self.shots}
        result = self.machine.machine.run_with_configuration(prog, self.clist, config)
        for state in result:
            result[state] = result[state] / self.shots
        return result

    def get_prob_mat(self):
        prog = QProg()
        prog.insert(self.circuit.circuit)
        self.machine.machine.directly_run(prog)
        psi_out = cp.array(self.machine.machine.get_qstate())
        psi_out = psi_out.reshape(1, len(psi_out))
        prob_mat = (psi_out * cp.conj(psi_out).T).real
        return prob_mat

    def get_loss(self):
        hamiltonian = self.hp
        #start = time.time()
        result = self.get_prob_shot()
        #end = time.time()
        #duration = end - start
        #print(duration)
        loss = 0
        for state in result:
            prob = result[state]
            state_list = []
            for bit in state:
                state_list.append(int(bit))
            for i in range(len(hamiltonian)):
                item = hamiltonian[i]
                tmp_vec = []
                dict_p = item[0]
                for iter in dict_p:
                    tmp_vec.append(iter)
                coef = item[1]
                z1z2_bool = state_list[tmp_vec[0]] ^ state_list[tmp_vec[1]]
                if z1z2_bool:
                    z1z2 = -coef
                else:
                    z1z2 = coef
                loss += z1z2 / 2 * prob
        loss -= len(hamiltonian) / 2
        return loss / self.max_loss

    def program_update(self, beta):
        self.circuit.update(beta)
