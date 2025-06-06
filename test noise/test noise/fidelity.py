from pyqpanda import *
from math import pi
import copy
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
def oneCircuit(qlist, Hamiltonian, step, beta, epsilon = 0.0):
    beta_copy = copy.deepcopy(beta)
    if epsilon:
        for ii in range(len(beta_copy)):
            for jj in range(len(beta_copy[ii])):
                beta_copy[ii, jj] = cp.round_(beta_copy[ii, jj] / epsilon) * epsilon
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
            qc.insert(RX(tmp_vec[1], 2 * beta_copy[layer][0] * coef))
            qc.insert(RY(tmp_vec[1], pi * 1.5))
            qc.insert(RX(tmp_vec[1], pi))
            qc.insert(RY(tmp_vec[1], pi * 0.5))
            qc.insert(CZ(tmp_vec[0], tmp_vec[1]))
            qc.insert(RX(tmp_vec[1], pi))
            qc.insert(RY(tmp_vec[1], pi * 0.5))

        for j in qlist:
            qc.insert(RX(j, 2 * beta_copy[layer][1]))
    return qc

def get_Hp_mat(Hamiltonian, qnum):
    Hp_mat = cp.zeros((1 << qnum, 1 << qnum), dtype = 'float64')
    for i in range(len(Hamiltonian)):
        item = Hamiltonian[i]
        Hp_mat -= 0.5 * cp.eye((1 << qnum)) * item[1]
        dict_p = item[0]
        z1z2 = cp.eye(1 << qnum)
        for iter in dict_p:
            temp = cp.array([[1, 0],[0, -1]], dtype = 'float64')
            temp = cp.kron(cp.eye(1 << iter, dtype = 'float64'), temp)
            temp = cp.kron(temp, cp.eye(1 << (qnum - 1 - iter), dtype = 'float64'))
            z1z2 = cp.dot(temp, z1z2)
        Hp_mat += z1z2 * 0.5 * item[1]
    return Hp_mat

def get_prob_standard(machine, beta, Hp, qlist, epsilon = 0.0):
    step = len(beta)
    prog = QProg()
    prog.insert(oneCircuit(qlist, Hp.toHamiltonian(1), step, beta, epsilon))
    machine.directly_run(prog)
    psi_out = cp.array(machine.get_qstate())
    prob = []
    for states in range(len(psi_out)):
        prob.append(cp.asnumpy(psi_out[states] * cp.conj(psi_out[states])).real)
    return prob

def get_prob_shot(machine, beta, Hp, qlist, clist, shots, epsilon = 0.0):
    step = len(beta)
    prog = QProg()
    prog.insert(oneCircuit(qlist, Hp.toHamiltonian(1), step, beta, epsilon))
    prog.insert(measure_all(qlist, clist))
    config = {'shots': shots}
    result = machine.run_with_configuration(prog, clist, config)
    for state in result:
        result[state] = result[state] / shots
    return result

def get_loss_mat(machine, beta, Hp, qlist, epsilon = 0.0):
    step = len(beta)
    prog = QProg()
    prog.insert(oneCircuit(qlist, Hp.toHamiltonian(1), step, beta, epsilon))
    machine.directly_run(prog)
    psi_out = cp.array(machine.get_qstate())
    psi_out = psi_out.reshape(1, len(psi_out))
    loss = cp.dot(cp.dot(psi_out, 
                   get_Hp_mat(Hp.toHamiltonian(1), len(qlist))), 
                   cp.conj(psi_out).T)
    return loss.real

def get_loss(machine, beta, Hp, qlist, clist, shots, epsilon = 0.0):
    hamiltonian = Hp.to_hamiltonian(1)
    result = get_prob_shot(machine, beta, Hp, qlist, clist, shots, epsilon)
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
                z1z2 = -1
            else:
                z1z2 = 1
            loss += z1z2 / 2 * prob
    loss -= len(hamiltonian) / 2
    return loss

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

def biggest(result):
    indx = []
    name = list(result.keys())
    num = list(result.values())
    for i in range(2):
        indx.append(num.index(max(num)))
        for id in indx:
            num[id] = 0
    for i in range(len(name)):
        if not(i in indx):
            name[i] = ''
    return name, list(result.values())

def compare_loss(machine, beta, Hp, qlist, clist, is_noise = False, epsilon = 0.0):
    '''
    print(machine)
    x = []
    x = list(range(1000, 101000, 1000))
    #x = [11000]
    for shots in range(len(x)):
        differences = abs(cp.asnumpy(get_loss(machine, beta, Hp, qlist, clist, x[shots], epsilon) - 
                                     get_loss(machine, beta, Hp, qlist, clist, x[shots], epsilon)))
        print(x[shots], differences)
        if is_noise and epsilon:
            write('noise epsilon difference compare.txt', [[x[shots], differences]])
        elif is_noise:
            write('noise difference compare.txt', [[x[shots], differences]])
        elif epsilon:
            write('epsilon difference compare.txt', [[x[shots], differences]])
        else:
            write('standard difference compare.txt', [[x[shots], differences]])
    '''
    if is_noise and epsilon:
        data = cp.array(read('noise epsilon difference compare.txt'))
    elif is_noise:
        data = cp.array(read('noise difference compare.txt'))
    elif epsilon:
        data = cp.array(read('epsilon difference compare.txt'))
    else:
        data = cp.array(read('standard difference compare.txt'))
    differences = data[:, 1]
    log_differences = cp.log10(differences)
    x = data[:, 0]
    log_x = cp.log10(x)
    log_x_draw = cp.log10(x)
    #x_draw = x
    x_mean = cp.mean(log_x)
    #x_mean = cp.mean(x)
    y_mean = cp.mean(log_differences)
    m1 = 0
    m2 = 0
    for x_i, y_i in zip(log_x, log_differences):
    #for x_i, y_i in zip(x, log_differences):
        m1 += (x_i - x_mean) * (y_i - y_mean)
        m2 += (x_i - x_mean) ** 2
    a = m1 / m2
    b = y_mean - a * x_mean
    #y_line = a * x_draw + b
    y_line = a * log_x_draw + b
    plt.title('The relationship between sample times and dispersion', font)
    plt.scatter(cp.asnumpy(log_x), cp.asnumpy(log_differences).transpose())
    #plt.scatter(cp.asnumpy(x), cp.asnumpy(log_differences).transpose())
    plt.plot(cp.asnumpy(log_x_draw), cp.asnumpy(y_line), label = 'y=' + str(a)[:7] + 'x+' + str(b)[:6], color = 'r')
    #plt.plot(cp.asnumpy(x_draw), cp.asnumpy(y_line), label = 'y=' + str(a)[:7] + 'x+' + str(b)[:6], color = 'r')
    plt.xlabel('lg10(shots)', font)
    #plt.xlabel('shots', font)
    plt.ylabel('lg10(difference)', font)
    plt.legend()
    plt.show()

def compare_kl(machine, beta, Hp, qlist, clist, is_noise = False, epsilon = 0.0):
    print(machine)
    state_number = 1 << len(qlist)
    kl_divergence = []
    x = list(range(1000, 101000, 1000))
    prob_shot = get_prob_shot(machine, beta, Hp, qlist, clist, x[0], epsilon)
    prob_shot_cp = cp.zeros(state_number) + 1e-50
    for state in prob_shot:
        prob_shot_cp[int(state, 2)] = prob_shot[state]
    old_shot = copy.deepcopy(prob_shot_cp)
    for shots in x[1:]:
        prob_shot = get_prob_shot(machine, beta, Hp, qlist, clist, shots, epsilon)
        prob_shot_cp = cp.zeros(state_number) + 1e-50
        for state in prob_shot:
            prob_shot_cp[int(state, 2)] = prob_shot[state]
        kl_divergence = cp.sum(prob_shot_cp * cp.log(prob_shot_cp) - prob_shot_cp * cp.log(old_shot))
        old_shot = copy.deepcopy(prob_shot_cp)
        print(shots, kl_divergence)
        if is_noise and epsilon:
            write('noise epsilon kl compare.txt', [[shots, kl_divergence]])
        elif is_noise:
            write('noise kl compare.txt', [[shots, kl_divergence]])
        elif epsilon:
            write('epsilon kl compare.txt', [[shots, kl_divergence]])
        else:
            write('standard kl compare.txt', [[shots, kl_divergence]])

    noise_kl = np.array(read('noise kl compare.txt'))
    standard_kl = np.array(read('standard kl compare.txt'))
    epsilon_kl = np.array(read('epsilon kl compare.txt'))
    noise_epsilon_kl = np.array(read('noise epsilon kl compare.txt'))
    plt.plot(noise_kl[:, 0], noise_kl[:, 1], linewidth = '2', label = "decoherence noise kl", color = 'r')
    plt.plot(standard_kl[:, 0], standard_kl[:, 1], linewidth = '2', label = "standard kl", color = 'b')
    plt.plot(epsilon_kl[:, 0], epsilon_kl[:, 1], linewidth = '2', label = "unitary noise kl", color = 'g')
    plt.plot(noise_epsilon_kl[:, 0], epsilon_kl[:, 1], linewidth = '2', label = "decoherence and unitary", color = 'black')
    plt.xlabel("shot", font)
    plt.ylabel("kl divervgemce", font)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('relationship between kl divergence and shot', font)
    plt.legend()
    plt.show()

def compare_pdf(beta, Hp, machine, qlist, clist, noise_machine, noise_qlist, noise_clist, shots, epsilon = 0.0):
    state_number = 1 << len(qlist)

    standard_prob_mat = cp.array(get_prob_standard(machine, beta, Hp, qlist, 0))

    #noise_prob_shot = get_prob_shot(noise_machine, beta, Hp, noise_qlist, noise_clist, shots, 0)
    #noise_prob_shot_cp = cp.zeros(state_number) + 1e-50
    #for state in noise_prob_shot:
    #    noise_prob_shot_cp[int(state, 2)] = noise_prob_shot[state]

    epsilon_prob_shot = get_prob_shot(machine, beta, Hp, qlist, clist, shots, epsilon)
    epsilon_prob_shot_cp = cp.zeros(state_number) + 1e-50
    for state in epsilon_prob_shot:
        epsilon_prob_shot_cp[int(state, 2)] = epsilon_prob_shot[state]

    noise_epsilon_prob_shot = get_prob_shot(noise_machine, beta, Hp, noise_qlist, noise_clist, shots, epsilon)
    noise_epsilon_prob_shot_cp = cp.zeros(state_number) + 1e-50
    for state in noise_epsilon_prob_shot:
        noise_epsilon_prob_shot_cp[int(state, 2)] = noise_epsilon_prob_shot[state]

    #noise_fidelity = cp.sqrt(cp.sum(standard_prob_mat * noise_prob_shot_cp))

    epsillon_fidelity = cp.sqrt(cp.sum(standard_prob_mat * epsilon_prob_shot_cp))

    noise_epsilon_fidelity = cp.sqrt(cp.sum(standard_prob_mat * noise_epsilon_prob_shot_cp))

    #standard_loss_mat = get_loss_mat(machine, beta, Hp, qlist, epsilon)

    #noise_loss_shot = get_loss(noise_machine, beta, Hp, noise_qlist, noise_clist, shots, 0)

    epsilon_loss_shot = get_loss(machine, beta, Hp, qlist, clist, shots, epsilon)

    noise_epsilon_loss_shot = get_loss(noise_machine, beta, Hp, noise_qlist, noise_clist, shots, epsilon)

    name_list = list(range(state_number))
    #plt.plot(name_list, cp.asnumpy(standard_prob_mat), label = str(standard_loss_mat[0][0])[:6])
    #plt.xlabel("state")
    #plt.ylabel("probability")
    #plt.title('noise free probability')
    #plt.legend()
    #plt.show()
    #plt.plot(name_list, cp.asnumpy(noise_prob_shot_cp), label = str(noise_loss_shot)[:6])
    #plt.xlabel("state")
    #plt.ylabel("probability")
    #plt.title('deooherence noise shot')
    #plt.legend()
    #plt.show()
    plt.plot(name_list, cp.asnumpy(epsilon_prob_shot_cp), label = str(epsilon_loss_shot)[:6])
    plt.xlabel("state")
    plt.ylabel("probability")
    plt.title('unitary noise shott')
    plt.legend()
    plt.show()
    plt.plot(name_list, cp.asnumpy(noise_epsilon_prob_shot_cp), label = str(noise_epsilon_loss_shot)[:6])
    plt.xlabel("state")
    plt.ylabel("probability")
    plt.title('noise shot')
    plt.legend()
    plt.show()
    #return noise_fidelity, epsillon_fidelity, noise_epsilon_fidelity
    return epsillon_fidelity, noise_epsilon_fidelity

def write(fn, data):
    with open(fn, 'a+') as fp:
        for i in range(len(data)):
            fp.write(str(data[i][0]) + ' ' + str(data[i][1]))
            fp.write('\n')
    fp.close()

def read(fn):
    with open(fn, "r") as f:  
        data = []
        while True:
            data_txt = f.readline()
            if data_txt == '':
                break
            number_txt = data_txt.split(' ')
            data_raw = []
            for i in number_txt:
                data_raw.append(float(i))
            data.append(data_raw)
    return data

if __name__=="__main__":
    print('Parent process %s.' % os.getpid())
    problem = {'Z0 Z1': 1,'Z1 Z2': 1,'Z2 Z3': 1,'Z3 Z4': 1,'Z4 Z5': 1,
               'Z5 Z6': 1,'Z6 Z7': 1,'Z7 Z8': 1,'Z8 Z9': 1, 'z9 z0': 1}
    Hp = PauliOperator(problem)
    qubit_num = Hp.getMaxIndex()
    state_number = 1 << qubit_num
    clause_num = len(Hp.toHamiltonian(2))
    step = 1
    epsilon = pi / 32
    shots = 10000########################################################################################
    beta1 = np.array([[0.39309590329727323, 0.3920708368256456]])

    beta2 = np.array([[0.3276756841011983, 0.6215754986564492],
                     [0.621604313685337, 0.32846209327296916]])

    beta3 = np.array([[0.2962307601128539, 0.682664477183167],
                      [0.5773206569611757, 0.5785617705029669], 
                      [0.6825345676680731, 0.29693065913753247]])

    beta4 = np.array([[0.2764748262723683, 0.7063221571235917],
                      [0.5542378047033222, 0.6521604100403179],
                      [0.652916125816355, 0.5533943464088232],
                      [0.7087375350123197, 0.2753418823880171]])

    beta5 = np.array([[0.47926354281578215, 0.6183906005544298],
                      [0.6434937103828292, 0.6673708173608746],
                      [0.6671348595825479, 0.6674198984673613],
                      [0.666139751361548, 0.6442983775052746],
                      [0.6188064471285891, 0.47919647808973304]])

    machine = CPUQVM()
    machine.init_qvm()

    qlist = machine.qAlloc_many(qubit_num)
    clist = machine.cAlloc_many(qubit_num)
    
    standard_prob_mat = cp.array(get_prob_standard(machine, beta1, Hp, qlist, 0))
    standard_prob_mat_cp = standard_prob_mat + 1e-50
    standard_loss_mat = get_loss_mat(machine, beta1, Hp, qlist, epsilon)

    noise_machine = NoiseQVM()
    t1 = 0.06
    t2 = 0.03
    T_1 = 5.0
    T_2 = 2.0
    

    different_t1 = np.arange(0.01, 0.5, 0.001)
    different_t2 = np.arange(0.01, 0.5, 0.001)
    different_precision = np.arange(0.001, pi / 64, 0.001)
    
    fidelity_different= []
    #loss_different = []
    
    for j in different_precision:
        epsilon = j
        doc = {'noisemodel': {'RX': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t1], 
                              'RZ': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t1],                            
                              'CZ': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t2]}}
        noise_machine.init_qvm(doc)
        noise_machine.set_rotation_angle_error(epsilon)
        noise_qlist = noise_machine.qAlloc_many(qubit_num)
        noise_clist = noise_machine.cAlloc_many(qubit_num)
        noise_prob_shot = get_prob_shot(noise_machine, beta1, Hp, noise_qlist, noise_clist, shots, epsilon)
        noise_prob_shot_cp = cp.zeros(state_number) + 1e-50
        for state in noise_prob_shot:
            noise_prob_shot_cp[int(state, 2)] = noise_prob_shot[state]
    #    noise_loss_shot = get_loss(noise_machine, beta1, Hp, noise_qlist, noise_clist, shots, epsilon)        
        fidelity_different.append(cp.sum(cp.sqrt(standard_prob_mat * noise_prob_shot_cp)))
    #    loss_different.append(abs(noise_loss_shot - standard_loss_mat)[0][0])

    
    write('measurement2\\epsilon_fidelity.txt', list(zip(different_precision, fidelity_different)))
    #write('measurement2\\epsilon_loss.txt', list(zip(different_t1, loss_different)))
    '''
    fidelity_different= []
    #loss_different = []
    for j in np.arange(0.001, 0.1, 0.001):
        t1 = j
        doc = {'noisemodel': {'RX': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t1], 
                              'RZ': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t1],                            
                              'CZ': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t2]}}
        noise_machine.init_qvm(doc)
        noise_machine.set_rotation_angle_error(1e-50)
        noise_qlist = noise_machine.qAlloc_many(qubit_num)
        noise_clist = noise_machine.cAlloc_many(qubit_num)
        noise_prob_shot = get_prob_shot(noise_machine, beta1, Hp, noise_qlist, noise_clist, shots, epsilon)
        noise_prob_shot_cp = cp.zeros(state_number) + 1e-50
        for state in noise_prob_shot:
            noise_prob_shot_cp[int(state, 2)] = noise_prob_shot[state]
    #    noise_loss_shot = get_loss(noise_machine, beta1, Hp, noise_qlist, noise_clist, shots, epsilon)        
        fidelity_different.append(cp.sum(cp.sqrt(standard_prob_mat * noise_prob_shot_cp)))
    #    loss_different.append(abs(noise_loss_shot - standard_loss_mat)[0][0])

    
    write('measurement2\\t1_fidelity.txt', list(zip(different_t1, fidelity_different)))
    #write('measurement2\\t1_loss.txt', list(zip(different_t1, loss_different)))
    
    fidelity_different= []
    #loss_different = []
    for j in np.arange(0.001, 0.1, 0.001):
        t2 = j
        doc = {'noisemodel': {'RX': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t1], 
                              'RZ': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t1],                            
                              'CZ': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t2]}}
        noise_machine.init_qvm(doc)
        noise_machine.set_rotation_angle_error(1e-50)
        noise_qlist = noise_machine.qAlloc_many(qubit_num)
        noise_clist = noise_machine.cAlloc_many(qubit_num)
        noise_prob_shot = get_prob_shot(noise_machine, beta1, Hp, noise_qlist, noise_clist, shots, epsilon)
        noise_prob_shot_cp = cp.zeros(state_number) + 1e-50
        for state in noise_prob_shot:
            noise_prob_shot_cp[int(state, 2)] = noise_prob_shot[state]
    #    noise_loss_shot = get_loss(noise_machine, beta1, Hp, noise_qlist, noise_clist, shots, epsilon)        
        fidelity_different.append(cp.sum(cp.sqrt(standard_prob_mat * noise_prob_shot_cp)))
    #    loss_different.append(abs(noise_loss_shot - standard_loss_mat)[0][0])
    #    kl_different.append(cp.sum(standard_prob_mat_cp * cp.log(standard_prob_mat_cp / noise_prob_shot_cp)))

    
    write('measurement2\\t2_fidelity.txt', list(zip(different_t2, fidelity_different)))
    #write('measurement2\\t2_loss.txt', list(zip(different_t2, loss_different)))
    #write('measurement2\\t2_kl.txt', list(zip(different_t2, kl_different)))
   
    T_1_list = np.arange(5.0, 160.0, 50.0)
    
    
    for T_1 in T_1_list:       
        fidelity_different= []
    #    loss_different = []
        for T_2 in np.arange(2, 2 * T_1, (2 * T_1 - 2) / 40):
            doc = {'noisemodel': {'RX': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t1], 
                                  'RZ': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t1],                            
                                  'CZ': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t2]}}
            noise_machine.init_qvm(doc)
            noise_machine.set_rotation_angle_error(1e-50)
            noise_qlist = noise_machine.qAlloc_many(qubit_num)
            noise_clist = noise_machine.cAlloc_many(qubit_num)
            noise_prob_shot = get_prob_shot(noise_machine, beta1, Hp, noise_qlist, noise_clist, shots, epsilon)
            noise_prob_shot_cp = cp.zeros(state_number) + 1e-50
            for state in noise_prob_shot:
                noise_prob_shot_cp[int(state, 2)] = noise_prob_shot[state]
    #        noise_loss_shot = get_loss(noise_machine, beta1, Hp, noise_qlist, noise_clist, shots, epsilon)        
            fidelity_different.append(cp.sum(cp.sqrt(standard_prob_mat * noise_prob_shot_cp)))
    #        loss_different.append(abs(noise_loss_shot - standard_loss_mat)[0][0])

    
        write('measurement2\\T_2' + str(T_1) + ' fidelity.txt', list(zip(list(np.arange(2, 2 * T_1, (2 * T_1 - 2) / 40)), fidelity_different)))
    #    write('measurement2\\T_2' + str(T_1) + ' loss.txt', list(zip(list(np.arange(2, 2 * T_1, (2 * T_1 - 2) / 40)), loss_different)))

    
    T_2_list = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 2.0, 62.0, 122.0, 182.0]
    for T_2 in T_2_list:
        fidelity_different= []
    #    loss_different = []
        for T_1 in np.arange(T_2 / 2, 160, (160 - T_2 / 2) / 40):
            doc = {'noisemodel': {'RX': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t1], 
                                  'RZ': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t1],                            
                                  'CZ': [NoiseModel.DECOHERENCE_KRAUS_OPERATOR, T_1, T_2, t2]}}
            noise_machine.init_qvm(doc)
            noise_machine.set_rotation_angle_error(1e-50)
            noise_qlist = noise_machine.qAlloc_many(qubit_num)
            noise_clist = noise_machine.cAlloc_many(qubit_num)
            noise_prob_shot = get_prob_shot(noise_machine, beta1, Hp, noise_qlist, noise_clist, shots, epsilon)
            noise_prob_shot_cp = cp.zeros(state_number) + 1e-50
            for state in noise_prob_shot:
                noise_prob_shot_cp[int(state, 2)] = noise_prob_shot[state]
    #        noise_loss_shot = get_loss(noise_machine, beta1, Hp, noise_qlist, noise_clist, shots, epsilon)        
            fidelity_different.append(cp.sum(cp.sqrt(standard_prob_mat * noise_prob_shot_cp)))
    #        loss_different.append(abs(noise_loss_shot - standard_loss_mat)[0][0])

    
        write('measurement2\\T_1' + str(T_2) + ' fidelity.txt', list(zip(list(np.arange(T_2 / 2, 160, (160 - T_2 / 2) / 40)), fidelity_different)))
    #    write('measurement2\\T_1' + str(T_2) + ' loss.txt', list(zip(list(np.arange(T_2 / 2, 160, (160 - T_2 / 2) / 40)), loss_different)))
    
    colors = ['red', 'green', 'blue', 'black', 'peru', 'c', 'm', 'teal', 'y', 'gold']
    markers = ['>', '<', '*', 'o', '^', 'D', 'H', '+', '2', 'v']
    T_1_list = np.arange(5.0, 160.0, 50.0)
       
    T_2_list = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 2.0, 62.0, 122.0, 182.0]

    for i in range(10):
    
        fidelity_different = read('measurement2\\T_1' + str(T_2_list[i]) + ' fidelity.txt')
            
        plt.plot(np.array(fidelity_different)[:, 0], np.array(fidelity_different)[:, 1], color=colors[i], marker=markers[i], label='T_2=' + str(T_2_list[i]))
    plt.xlabel("T_1(mus)")
    plt.ylabel("fidelity")
    plt.title('relationship between T_1 and fidelity')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.show()

    #for i in range(10):

    #    loss_different = read('measurement2\\T_1' + str(T_2_list[i]) + ' loss.txt')
    
    #    plt.plot(np.array(loss_different)[:, 0], np.array(loss_different)[:, 1], color=colors[i], marker=markers[i], label='T_2=' + str(T_2_list[i]))
    #plt.xlabel("T_1(mus)")
    #plt.ylabel("loss different")
    #plt.title('relationship between T_1 and loss different')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    #plt.show()

    

    for i in range(4):
    
        fidelity_different = read('measurement2\\T_2' + str(T_1_list[i]) + ' fidelity.txt')
            
        plt.plot(np.array(fidelity_different)[:, 0], np.array(fidelity_different)[:, 1], color=colors[i], marker=markers[i], label='T_1=' + str(T_1_list[i]))
    plt.xlabel("T_2(mus)")
    plt.ylabel("fidelity")
    plt.title('relationship between T_2 and fidelity')
    plt.legend()

    plt.show()

    #for i in range(4):

    #    loss_different = read('measurement2\\T_2' + str(T_1_list[i]) + ' loss.txt')
    
    #    plt.plot(np.array(loss_different)[:, 0], np.array(loss_different)[:, 1], color=colors[i], marker=markers[i], label='T_1=' + str(T_1_list[i]))
    #plt.xlabel("T_2(mus)")
    #plt.ylabel("loss different")
    #plt.title('relationship between T_2 and loss different')
    #plt.legend()

    #plt.show()

    fidelity_different = read('measurement2\\t2_fidelity.txt')
    #loss_different = read('measurement2\\t2_loss.txt')
    
    plt.plot(np.array(fidelity_different)[:, 0], np.array(fidelity_different)[:, 1], color='g', marker='o', label='fidelity')
    plt.xlabel("t2(mus)")
    plt.ylabel("fidelity")
    plt.title('relationship between t2 and fidelity')
    plt.legend()
    plt.show()
    
    #plt.plot(np.array(loss_different)[:, 0], np.array(loss_different)[:, 1], color='r', marker='>', label='loss different')
    #plt.xlabel("t2(mus)")
    #plt.ylabel("loss different")
    #plt.title('relationship between t2 and loss different')
    #plt.legend()
    #plt.show()


    fidelity_different = read('measurement2\\t1_fidelity.txt')
    #loss_different = read('measurement2\\t1_loss.txt')
   
    plt.plot(np.array(fidelity_different)[:, 0], np.array(fidelity_different)[:, 1], color='g', marker='o', label='fidelity')
    plt.xlabel("t1(mus)")
    plt.ylabel("fidelity")
    plt.title('relationship between t1 and fidelity')
    plt.legend()
    plt.show()
    
    #plt.plot(np.array(loss_different)[:, 0], np.array(loss_different)[:, 1], color='r', marker='>', label='loss different')
    #plt.xlabel("t1(mus)")
    #plt.ylabel("loss different")
    #plt.title('relationship between t1 and loss different')
    #plt.legend()
    #plt.show()
    '''

    fidelity_different = read('measurement2\\epsilon_fidelity.txt')
    #loss_different = read('measurement2\\epsilon_loss.txt')
   
    plt.plot(np.array(fidelity_different)[:, 0], np.array(fidelity_different)[:, 1], color='g', marker='o', label='fidelity')
    plt.xlabel("precision")
    plt.ylabel("fidelity")
    plt.title('relationship between precision and fidelity')
    plt.legend()
    plt.show()
    
    #plt.plot(np.array(loss_different)[:, 0], np.array(loss_different)[:, 1], color='r', marker='>', label='loss different')
    #plt.xlabel("epsilon")
    #plt.ylabel("loss different")
    #plt.title('relationship between t1 and loss different')
    #plt.legend()
    #plt.show()

    #plt.plot(np.array(fidelity_different)[:, 0], np.array(fidelity_different)[:, 1], color='g', marker='o', label='fidelity x 40')
    #plt.plot(np.array(loss_different)[:, 0], np.array(loss_different)[:, 1], color='r', marker='>', label='loss different')
    #plt.xlabel('epsilon')
    #plt.title('')
    #plt.legend()
    #plt.show()
    