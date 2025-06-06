# import

import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from qutip import *
from qutip.qip.operations import rz
from qutip.qip.operations import cphase
from scipy.optimize import minimize, minimize_scalar
from multiprocessing import Pool, cpu_count

# 升降算符

def getaad(energyLevel, bitNum):
    a, aDag = destroy(energyLevel), create(energyLevel)
    I = qeye(energyLevel)
    IenergyLevel = tensorOperator(energyLevel, I, 0, bitNum)
    aqList, aqDagList = [], []
    acList, acDagList = [], []
    sxList, syList, szList = [], [], []
    for b in range(bitNum):
        if b % 2 == 0:
            aq = tensorOperator(energyLevel, a, b, bitNum)
            aqDag = tensorOperator(energyLevel, aDag, b, bitNum)
            aqList.append(aq)
            aqDagList.append(aqDag)
            sxList.append(aq + aqDag)
            syList.append(1j * (aqDag - aq))
            szList.append(IenergyLevel - 2 * aqDag * aq)
        else:
            acList.append(tensorOperator(energyLevel, a, b, bitNum))
            acDagList.append(tensorOperator(energyLevel, aDag, b, bitNum))
    return aqList, aqDagList, acList, acDagList, sxList, syList, szList


# 一个把张量张起来的函数

def tensorOperator(energyLevel, op, qubit, qubitNum):
    I = qeye(energyLevel)
    for dim in range(qubitNum):
        if dim == qubit:
            tempOp = op
        else:
            tempOp = I
        if dim == 0:
            tensorOp = tempOp
        else:
            tensorOp = tensor(tempOp, tensorOp)
    return tensorOp

# 哈密顿量

def H(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm):
    aqList, aqDagList, acList, acDagList, _, _, _ = \
    getaad(energyLevel, bitNum)
    qNum = (bitNum + 1) // 2
    cNum = bitNum - qNum
    noH = True
    for freQ in range(len(omegaq)):
        if noH:
            H0 = omegaq[freQ] * aqDagList[freQ] * aqList[freQ] + 0.5 * anharm['q' + str(freQ)] * aqDagList[freQ] * aqDagList[freQ] * aqList[freQ] * aqList[freQ]
            noH = False
        else:
            H0 += omegaq[freQ] * aqDagList[freQ] * aqList[freQ] + 0.5 * anharm['q' + str(freQ)] * aqDagList[freQ] * aqDagList[freQ] * aqList[freQ] * aqList[freQ]
    for freC in range(len(omegac)):
        H0 += omegac[freC] * acDagList[freC] * acList[freC] + 0.5 * anharm['c' + str(freC)] * acDagList[freC] * acDagList[freC] * acList[freC] * acList[freC]
    noH = True
    for g in g_qc:
        q, c = int(g[1]), int(g[3])
        if q < qNum and c < cNum:
            if noH:
                Hi = g_qc[g] * np.sqrt(omegaq[q] * omegac[c]) * (aqDagList[q] + aqList[q]) * (acList[c] + acDagList[c])
                noH = False
            else:
                Hi += g_qc[g] * np.sqrt(omegaq[q] * omegac[c]) * (aqDagList[q] + aqList[q]) * (acList[c] + acDagList[c])
    for g in g_qq:
        q0, q1 = int(g[1]), int(g[3])
        if q0 < qNum and q1 < qNum:
            Hi += g_qq[g] * np.sqrt(omegaq[q0] *  omegaq[q1]) * (aqDagList[q0] + aqList[q0]) * (aqList[q1] + aqDagList[q1])
    return H0, Hi

# 本征态按照顺序排列

def eigensolve(H0, H, energyLevel, bitNum):

    energy_info = {'dim' : [energyLevel] * bitNum, 'exci_num': energyLevel * bitNum, 'bas_list':[]}

    for bas in state_number_enumerate(energy_info['dim'], excitations=energy_info['exci_num']):
        energy_info['bas_list'].append(state_number_qobj(energy_info['dim'], bas))

    ei_states = H.eigenstates()
    ei_energy = ei_states[0]
    ei_vector = ei_states[1]
    
    ei_states0 = H0.eigenstates()
    ei_energy0 = ei_states0[0]
    ei_vector0 = ei_states0[1]
    
    states_order = ei_vector.copy()
    states0_order = ei_vector.copy()
    energy_order = ei_energy.copy()
    energy0_order = ei_energy.copy()
    for n, vector in enumerate(ei_vector0):
        try:
            index = energy_info['bas_list'].index(vector)
            states_order[index] = ei_vector[n]
            states0_order[index] = ei_vector0[n]
            energy_order[index] = ei_energy[n]
            energy0_order[index] = ei_energy0[n]            
        except:
            pass
    return states_order, states0_order, energy_order, energy0_order

# zz耦合

def zzcoupling(g_qc, g_qq, energyLevel, bitNum, omegaq, omegac, anharm):

    H0, Hi = H(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm)
    _, _, energy, _ = eigensolve(H0, H0 + Hi, energyLevel, bitNum)
    zeta = energy[state_number_index([energyLevel] * bitNum, [1, 0, 1])] - energy[state_number_index([energyLevel] * bitNum, [1, 0, 0])] - \
            energy[state_number_index([energyLevel] * bitNum, [0, 0, 1])] + energy[state_number_index([energyLevel] * bitNum, [0, 0, 0])]
    return zeta

# 脉冲函数

def drive_pulseX0(t, args):  
    tg = args['gate time']
    amp = args['q0 amp']
    w_d = args['q0 drive frequency']
    alpha = args['q0 anharm']
    detune = args['q0 detune']
    lambda0 = args['drag weight']
    phi = args['q0 phi']
    I = np.cos(phi)
    Q = np.sin(phi)
    X0 = amp * (1 - np.cos(2 * np.pi * t / tg)) / 2 
    Y0 = -amp * np.pi / tg * np.sin(2 * np.pi * t / tg) / alpha * lambda0
    X = (X0 * I + Y0 * Q) * np.cos((w_d + detune) * t) 
    return X

def drive_pulseY0(t, args):  
    tg = args['gate time']
    amp = args['q0 amp']
    w_d = args['q0 drive frequency']
    alpha = args['q0 anharm']
    detune = args['q0 detune']
    lambda0 = args['drag weight']
    phi = args['q0 phi']
    I = np.cos(phi)
    Q = np.sin(phi)
    X0 = amp * (1 - np.cos(2 * np.pi * t / tg)) / 2 
    Y0 = -amp * np.pi / tg * np.sin(2 * np.pi * t / tg) / alpha * lambda0
    X = (Y0 * I - X0 * Q) * np.cos((w_d + detune) * t) 
    return X

def drive_pulseX1(t, args):  
    tg = args['gate time']
    amp = args['q1 amp']
    w_d = args['q1 drive frequency']
    alpha = args['q1 anharm']
    detune = args['q1 detune']
    lambda1 = args['drag weight']
    phi = args['q1 phi']
    I = np.cos(phi)
    Q = np.sin(phi)
    X0 = amp * (1 - np.cos(2 * np.pi * t / tg)) / 2 
    Y0 = -amp * np.pi / tg * np.sin(2 * np.pi * t / tg) / alpha * lambda1
    X = (X0 * I + Y0 * Q) * np.cos((w_d + detune) * t) 
    return X

def drive_pulseY1(t, args):  
    tg = args['gate time']
    amp = args['q1 amp']
    w_d = args['q1 drive frequency']
    alpha = args['q1 anharm']
    detune = args['q1 detune']
    lambda1 = args['drag weight']
    phi = args['q1 phi']
    I = np.cos(phi)
    Q = np.sin(phi)
    X0 = amp * (1 - np.cos(2 * np.pi * t / tg)) / 2 
    Y0 = -amp * np.pi / tg * np.sin(2 * np.pi * t / tg) / alpha * lambda1
    X = (Y0 * I - X0 * Q) * np.cos((w_d + detune) * t) 
    return X

# 保真度函数

def Fidelity_X(UEff):
    d = UEff.data.shape[0]
    bitNum = int(np.log2(d))
    phi = []
    bound = []
    for i in range(bitNum):
        phi.append(np.pi)
        bound.append((0, 2 * np.pi))
        if i == 0:
            XIdeal = sigmax()
        else:
            XIdeal = tensor(sigmax(), sigmax())
    res = minimize(fidelity, phi, args=(UEff, XIdeal, d), method='brent')
    return res.fun

def fidelity(phi, UEff, UIdeal, d):
    start = 0
    for i in phi:
        if start == 0:
            Rz = rz(-i)
            start = 1
        else:
            Rz = tensor(rz(-i), Rz)
    f = -((np.trace(np.dot(UEff.dag(), UEff)) + 
                        np.abs(np.trace(np.dot(Rz * UEff.dag(), UIdeal))) ** 2) / (d * (d + 1)))
    return f

def evolution_q0(pulse_paras, pulse_const, H, states, n, energyLevel, bitNum, anharm):
    
    _, _, _, _, sxList, syList, _ = \
    getaad(energyLevel, bitNum)

    H_x0 = [sxList[0], drive_pulseX0]
    H_y0 = [syList[0], drive_pulseY0]
    Ht = [H, H_x0, H_y0]
    args = dict()

    args['gate time'] = pulse_const[0] #  [tg, wd0, wd1, drag]
    args['q0 drive frequency'] = pulse_const[1]
    args['drag weight'] = pulse_const[3]

    args['q0 anharm'] = anharm['q0']
    args['q0 amp'] = pulse_paras[1]
    args['q0 detune'] = pulse_paras[0]
    args['q0 phi'] = -pulse_paras[0] * args['gate time'] / 2

    tList = np.arange(0, args['gate time'], 30 / 300)
    U_full = propagator(Ht, tList, args=args)[-1]
    U=np.zeros([2, 2], dtype='complex128')
    for i in range(2):
        for j in range(2):
            U[i][j] = (states[state_number_index([energyLevel] * bitNum, [n, 0, i])].dag() * U_full * 
                        states[state_number_index([energyLevel] * bitNum, [n, 0, j])]).full()[0][0]
    F = -Fidelity_X(Qobj(U))
    error = 1 - F
    return np.real(error)

def evolution_q1(pulse_paras, pulse_const, H, states, n, energyLevel, bitNum, anharm):

    _, _, _, _, sxList, syList, _ = \
    getaad(energyLevel, bitNum)

    H_x1 = [sxList[1], drive_pulseX1]
    H_y1 = [syList[1], drive_pulseY1]
    Ht = [H, H_x1, H_y1]
    args = dict()

    args['gate time'] = pulse_const[0] #  [tg, wd0, wd1, drag]
    args['q1 drive frequency'] = pulse_const[2]
    args['drag weight'] = pulse_const[3]

    args['q1 anharm'] = anharm['q1']
    args['q1 amp'] = pulse_paras[1]
    args['q1 detune'] = pulse_paras[0]
    args['q1 phi'] = -pulse_paras[0] * args['gate time'] / 2

    tList = np.arange(0, args['gate time'], 30 / 300)
    U_full = propagator(Ht, tList, args=args)[-1]
    U=np.zeros([2, 2], dtype='complex128')
    for i in range(2):
        for j in range(2):
            U[i][j] = (states[state_number_index([energyLevel] * bitNum, [i, 0, n])].dag() * U_full * 
                        states[state_number_index([energyLevel] * bitNum, [j, 0, n])]).full()[0][0]
    F = -Fidelity_X(Qobj(U))
    error = 1 - F
    return np.real(error)

# 单比特门哈密顿量演化

def evolution_q0q1(pulse_paras, pulse_const, H, states, n, energyLevel, bitNum, anharm):

    _, _, _, _, sxList, syList, _ = getaad(energyLevel, bitNum)

    H_x0 = [sxList[0], drive_pulseX0]
    H_y0 = [syList[0], drive_pulseY0]
    H_x1 = [sxList[1], drive_pulseX1]
    H_y1 = [syList[1], drive_pulseY1]
    Ht = [H, H_x0, H_y0, H_x1 ,H_y1]

    args = dict()

    args['gate time'] = pulse_const[0] # [tg, wd0, wd1, drag]
    args['q0 drive frequency'] = pulse_const[1]
    args['q1 drive frequency'] = pulse_const[2]
    args['drag weight'] = pulse_const[3]

    pulse_paras1 = [pulse_paras[0], pulse_paras[1]]
    pulse_paras2 = [pulse_paras[2], pulse_paras[3]]
    args['q0 anharm'] = anharm['q0']
    args['q0 detune'] = pulse_paras1[0]    
    args['q0 amp'] = pulse_paras1[1]
    args['q0 phi'] = -pulse_paras1[0] * args['gate time'] / 2
    args['q1 anharm'] = anharm['q1']
    args['q1 detune'] = pulse_paras2[0]
    args['q1 amp'] = pulse_paras2[1]
    args['q1 phi'] = -pulse_paras2[0] * args['gate time'] / 2

    tList = np.arange(0, args['gate time'], 30 / 300)
    U_full=propagator(Ht, tList, args = args)[-1]
    U=np.zeros([4, 4],dtype='complex128')
    for i in range(2):
        for j in range(2):
            for m in range(2):
                for n in range(2):
                    ii = i * 2 + j
                    jj = m * 2 + n
                    U[ii][jj]=(states[state_number_index([energyLevel] * bitNum, [i, 0, j])].dag() * U_full *
                    states[state_number_index([energyLevel] * bitNum, [m, 0, n])]).full()[0][0]
    F = -Fidelity_X(Qobj(U))
    error = 1 - F
    return np.real(error)

# 校准参数的函数

def par_X0(g_qc, g_qq, energyLevel, bitNum, omegaq, omegac, tg, drag, anharm):    
    wd0 = omegaq[0]
    wd1 = omegaq[1]
    pulseConst = [tg, wd0, wd1, drag]
    H0, Hi = H(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm)
    states, _, energy, energy0 = eigensolve(H0, H0 + Hi, energyLevel, bitNum)
    
    detune = (energy[state_number_index([energyLevel] * bitNum, [1, 0, 1])] - \
            energy[state_number_index([energyLevel] * bitNum, [1, 0, 0])] - \
            energy0[state_number_index([energyLevel] * bitNum, [1, 0, 1])] + \
            energy0[state_number_index([energyLevel] * bitNum, [1, 0, 0])])
    xIni = [detune, 0.21]
    
    bounds = ((xIni[0] - 10e-3 * 2 * np.pi, xIni[0] + 10e-3 * 2 * np.pi), (xIni[1] - 0.07, xIni[1] + 0.07))
    
    result0 = minimize(evolution_q0, xIni, args=(pulseConst, H0 + Hi, states, 0, energyLevel, bitNum, anharm), 
    bounds=bounds, method='SLSQP', options={'ftol' : 1e-06})
    err_0 = result0.fun
    err_1 = evolution_q0(result0.x, pulseConst, H0 + Hi, states, 1, energyLevel, bitNum, anharm)
    print(err_0, err_1)
    return err_0, err_1

def par_X0X1(g_qc, g_qq, energyLevel, bitNum, omegaq, omegac, tg, drag, anharm):    
    wd0 = omegaq[0]
    wd1 = omegaq[1]
    pulseConst = [tg, wd0, wd1, drag]
    H0, Hi = H(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm)
    states, _, energy, energy0 = eigensolve(H0, H0 + Hi, energyLevel, bitNum)
    
    detune0 = (energy[state_number_index([energyLevel] * bitNum, [1, 0, 1])] - \
            energy[state_number_index([energyLevel] * bitNum, [1, 0, 0])] - \
            energy0[state_number_index([energyLevel] * bitNum, [1, 0, 1])] + \
            energy0[state_number_index([energyLevel] * bitNum, [1, 0, 0])])
    
    detune1 = (energy[state_number_index([energyLevel] * bitNum, [1, 0, 1])] - \
            energy[state_number_index([energyLevel] * bitNum, [0, 0, 1])] - \
            energy0[state_number_index([energyLevel] * bitNum, [1, 0, 1])] + \
            energy0[state_number_index([energyLevel] * bitNum, [0, 0, 1])])

    amp = 0.21

    xIni = [detune0, amp, detune1, amp]
    
    bounds = ((xIni[0] - 10e-3 * 2 * np.pi, xIni[0] + 10e-3 * 2 * np.pi), (xIni[1] - 0.07, xIni[1] + 0.07), 
                (xIni[2] - 10e-3 * 2 * np.pi, xIni[2] + 10e-3 * 2 * np.pi), (xIni[3] - 0.07, xIni[3] + 0.07))
    
    result0 = minimize(evolution_q0, xIni[:2], args=(pulseConst, H0 + Hi, states, 0, energyLevel, bitNum, anharm), 
                        bounds=bounds[:2], method='SLSQP', options={'ftol' : 1e-06})

    result1 = minimize(evolution_q1, xIni[2:], args=(pulseConst, H0 + Hi, states, 0, energyLevel, bitNum, anharm), 
                        bounds=bounds[2:], method='SLSQP', options={'ftol' : 1e-06})
    err_0 = result0.fun

    x = [result0.x[0], result0.x[1], result1.x[0], result1.x[1]]

    err_1 = evolution_q0q1(x, pulseConst, H0 + Hi, states, 1, energyLevel, bitNum, anharm)
    print(err_0, err_1)
    return err_0, err_1

# 写数据

def write_data(f, data, xyzs):
    xyzs = np.array(list(xyzs)).T
    dimension = len(xyzs)
    with open(f, 'w') as fp:
        for d in data:
            fp.write(str(d) + ' ')
        fp.write('\n')
        for xyz in range(dimension):
            for x in xyzs[xyz]:
                fp.write(str(x) + ' ')
            fp.write('\n')



if __name__ == '__main__':

    # 参数设置

    anharm = {'q0' : -2 * np.pi * 201 * 1e-3, 'c0' : -2 * np.pi * 200 * 1e-3, 'q1' : -2 * np.pi * 204 * 1e-3, 'c1' : -2 * np.pi * 206 * 1e-3, 'q2' : -2 * np.pi * 208 * 1e-3}
    C_q = [97, 99, 97]
    C_c = [95, 100]
    C_ic = {'q0c0' : 3.8, 'q1c0' : 4.1, 'q1c1' : 4.3, 'q2c1' : 4}
    C_12 = {'q0q1' : 0.27, 'q1q2' : 0.15}

    g_qc = dict()
    for k in C_ic:
        q, c = int(k[1]), int(k[3])
        g_qc[k] = 0.5 * C_ic[k] / np.sqrt(C_q[q] * C_c[c])

    g_qq = dict()
    for k in C_12:
        q0, q1 = int(k[1]), int(k[3])
        C_1cC_2c = 1
        for kk in C_ic:
            if ('q' + str(q0) in kk and 'q' + str(q1) + kk[2:] in C_ic) or ('q' + str(q1) in kk and 'q' + str(q0) + kk[2:] in C_ic):
                C_1cC_2c *= C_ic[kk]
                ck = int(kk[3])
        g_qq[k] = 0.5 * (1 + C_1cC_2c / (C_12[k] * C_c[ck])) * C_12[k] / np.sqrt(C_q[q0] * C_q[q1])

    energyLevel = 3
    bitNum = 3

    # 扫一个zz耦合热力图

    # omega2List = np.arange(4, 6, 2 / 50)
    # omegacList = np.arange(4, 8, 4 / 50)
    # g_qcs = [g_qc] * len(omega2List) * len(omegacList)
    # g_qqs = [g_qq] * len(omega2List) * len(omegacList)
    # bitNums = [bitNum] * len(omega2List) * len(omegacList)
    # energyLevels = [energyLevel] * len(omega2List) * len(omegacList)
    # anharms = [anharm] * len(omega2List) * len(omegacList)


    # omegaqs = []
    # omegacs = []
    # zeta_list = []
    # for omega2 in omega2List:
    #     for omegac in omegacList:
    #         omegaqs.append(np.array([5, omega2]) * 2 * np.pi)
    #         omegacs.append(np.array([omegac]) * 2 * np.pi)
    # tStart = time.time()
    # print(cpu_count())
    # p = Pool(cpu_count())
    # zeta_list = p.starmap(zzcoupling, zip(g_qcs, g_qqs, energyLevels, bitNums, omegaqs, omegacs, anharms))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # write_data('zz coupling.txt', zeta_list, zip(omega2List, omegacList))

    # 扫一个误差热力图

    # omega0 = 5
    # omega1List = np.arange(4, 6, 2 / 50)
    # omegacList = np.arange(4, 8, 4 / 50)
    # g_qcs = [g_qc] * len(omega1List) * len(omegacList)
    # g_qqs = [g_qq] * len(omega1List) * len(omegacList)
    # bitNums = [bitNum] * len(omega1List) * len(omegacList)
    # energyLevels = [energyLevel] * len(omega1List) * len(omegacList)
    # anharms = [anharm] * len(omega1List) * len(omegacList)
    # gatetimes = [30] * len(omega1List) * len(omegacList)
    # dragWeights = [0.5] * len(omega1List) * len(omegacList)

    # omegaqs = []
    # omegacs = []
    # for omega1 in omega1List:
    #     for omegac in omegacList:
    #         omegaqs.append(np.array([omega0, omega1]) * 2 * np.pi)
    #         omegacs.append(np.array([omegac]) * 2 * np.pi)
    # tStart = time.time()
    # print(cpu_count())
    # p = Pool(cpu_count())
    # err_list = p.starmap(par_X0, zip(g_qcs, g_qqs, energyLevels, bitNums, 
    #             omegaqs, omegacs, gatetimes, dragWeights, anharms))

    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)
    # err0_list = []
    # err1_list = []
    # for i in err_list:
    #     err0_list.append(i[0])
    #     err1_list.append(i[1])

    # write_data('error0.txt', err0_list, zip(omega1List, omegacList))
    # write_data('error1.txt', err1_list, zip(omega1List, omegacList))

    # 并行两比特门工作点热力图

    omega0 = 5
    omega1List = np.arange(4, 6, 2 / 3)
    omegacList = np.arange(4, 8, 4 / 3)
    g_qcs = [g_qc] * len(omega1List) * len(omegacList)
    g_qqs = [g_qq] * len(omega1List) * len(omegacList)
    bitNums = [bitNum] * len(omega1List) * len(omegacList)
    energyLevels = [energyLevel] * len(omega1List) * len(omegacList)
    anharms = [anharm] * len(omega1List) * len(omegacList)
    gatetimes = [30] * len(omega1List) * len(omegacList)
    dragWeights = [0.5] * len(omega1List) * len(omegacList)

    omegaqs = []
    omegacs = []
    for omega1 in omega1List:
        for omegac in omegacList:
            omegaqs.append(np.array([omega0, omega1]) * 2 * np.pi)
            omegacs.append(np.array([omegac]) * 2 * np.pi)
    tStart = time.time()
    print(cpu_count())
    p = Pool(cpu_count())
    err_list = p.starmap(par_X0X1, zip(g_qcs, g_qqs, energyLevels, bitNums, 
                omegaqs, omegacs, gatetimes, dragWeights, anharms))

    p.close()
    p.join()
    t = time.time() - tStart
    print(t)
    err0_list = []
    err1_list = []
    for i in err_list:
        err0_list.append(i[0])
        err1_list.append(i[1])
