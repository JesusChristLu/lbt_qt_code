# import
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cmath
from copy import deepcopy
from qutip import *
from qutip.qip.operations import rz
from qutip.qip.operations import cphase
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import interp1d
from scipy.special import erf
from multiprocessing import Pool, cpu_count
from draw import draw_heat_map

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

# 不含时哈密顿量

def H(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm):
    aqList, aqDagList, acList, acDagList, _, _, _ = getaad(energyLevel, bitNum)
    qNum = (bitNum + 1) // 2
    cNum = bitNum - qNum
    noH = True
    for freQ in range(qNum):
        if noH:
            H0 = 2 * np.pi * omegaq[freQ] * aqDagList[freQ] * aqList[freQ] + np.pi * anharm['q' + str(freQ)] * aqDagList[freQ] * aqDagList[freQ] * aqList[freQ] * aqList[freQ]
            noH = False
        else:
            H0 += 2 * np.pi * omegaq[freQ] * aqDagList[freQ] * aqList[freQ] + np.pi * anharm['q' + str(freQ)] * aqDagList[freQ] * aqDagList[freQ] * aqList[freQ] * aqList[freQ]
    for freC in range(cNum):
        H0 += 2 * np.pi * omegac[freC] * acDagList[freC] * acList[freC] + np.pi * anharm['c' + str(freC)] * acDagList[freC] * acDagList[freC] * acList[freC] * acList[freC]
    noH = True
    for g in g_qc:
        q, c = int(g[1]), int(g[3])
        if q < qNum and c < cNum:
            if noH:
                Hi = 2 * np.pi * g_qc[g] * np.sqrt(omegaq[q] * omegac[c]) * (aqDagList[q] + aqList[q]) * (acList[c] + acDagList[c])
                noH = False
            else:
                Hi += 2 * np.pi * g_qc[g] * np.sqrt(omegaq[q] * omegac[c]) * (aqDagList[q] + aqList[q]) * (acList[c] + acDagList[c])
    for g in g_qq:
        q0, q1 = int(g[1]), int(g[3])
        if q0 < qNum and q1 < qNum:
            Hi += 2 * np.pi * g_qq[g] * np.sqrt(omegaq[q0] *  omegaq[q1]) * (aqDagList[q0] + aqList[q0]) * (aqList[q1] + aqDagList[q1])
    return H0, Hi

# 两比特门含时哈密顿量

# 绝热H

def H_ad(tList, args):
    g_qc = args['g qc']
    g_qq = args['g qq']
    freqqMax = args['freq q max']
    freqcMax = args['freq c max']
    lambdas = args['lambdas']
    bitNum = args['bitNum']
    energyLevel = args['energy level']
    anharm = args['anharm']

    aqList, aqDagList, acList, acDagList, _, _, _ = getaad(energyLevel, bitNum)

    qNum = (bitNum + 1) // 2
    cNum = bitNum - qNum

    lambdaList = []
    for i in range(cNum):
        lambList = []
        for m in range(len(lambdas)):
            lambList.append(lambdas[len(lambdas) * i + m])
        lambdaList.append(lambList)

    HAnharm = []
    H0 = []
    Hi = []
    for q in range(qNum):
        HAnharm.append(np.pi * anharm['q' + str(q)] * aqDagList[q] * aqDagList[q] * aqList[q] * aqList[q])
        H0.append(np.pi * 2 * freqqMax[q] * aqDagList[q] * aqList[q])
    for c in range(cNum):
        HAnharm.append(np.pi * anharm['c' + str(c)] * acDagList[c] * acDagList[c] * acList[c] * acList[c])
        if sum(lambdaList[c]) == 0:
            H0.append(np.pi * 2 * freqcMax[c] * acDagList[c] * acList[c])
        else:
            freqList = freqcMax[c] + freq_pulse(tList, lambdaList[c])
            H0.append([acDagList[c] * acList[c], freqList * np.pi * 2])
    for gkey in g_qc:
        q, c = int(gkey[1]), int(gkey[3])
        if q < qNum and c < cNum:
            if sum(lambdaList[c]) == 0:
                g = g_qc[gkey] * np.sqrt(freqcMax[q] * freqcMax[c])
                Hi.append(g * np.pi * 2 * (aqDagList[q] + aqList[q]) * (acList[c] + acDagList[c]))
            else:
                freqcList = freqcMax[c] + freq_pulse(tList, lambdaList[c])
                g = g_qc[gkey] * np.sqrt(freqqMax[q] * freqcList)
                Hi.append([(aqDagList[q] + aqList[q]) * (acList[c] + acDagList[c]), g * np.pi * 2])
    for gkey in g_qq:
        q0, q1 = int(gkey[1]), int(gkey[3])
        if q0 < qNum and q1 < qNum:
            g = g_qq[gkey] * np.sqrt(freqqMax[q0] * freqqMax[q1])
            Hi.append(g * np.pi * 2 * (aqDagList[q0] + aqList[q0]) * (aqList[q1] + aqDagList[q1]))
    return [*H0, *HAnharm, *Hi]

# 非绝热H

def H_no_ad(tList, args):
    g_qc = args['g qc']
    g_qq = args['g qq']
    freqqMax = args['freq q max']
    freqcMax = args['freq c max']
    freqqWork = [freqqMax[0], args['lambdas'][0]]
    freqcWork = [args['lambdas'][1]]
    sigma = args['lambdas'][2:]
    bitNum = args['bitNum']
    energyLevel = args['energy level']
    anharm = args['anharm']

    aqList, aqDagList, acList, acDagList, _, _, _ = getaad(energyLevel, bitNum)

    qNum = (bitNum + 1) // 2
    cNum = bitNum - qNum

    HAnharm = []
    H0 = []
    Hi = []
    for q in range(qNum):
        HAnharm.append(np.pi * anharm['q' + str(q)] * aqDagList[q] * aqDagList[q] * aqList[q] * aqList[q])
        if freqqMax[q] == freqqWork[q]:
            H0.append(np.pi * 2 * freqqMax[q] * aqDagList[q] * aqList[q])
        else:
            freqList = gaus_flat(tList, tList[-1], sigma[0], freqqWork[q], freqqMax[q])
            H0.append([aqDagList[q] * aqList[q], freqList * np.pi * 2])
    for c in range(cNum):
        HAnharm.append(np.pi * anharm['c' + str(c)] * acDagList[c] * acDagList[c] * acList[c] * acList[c])
        if freqcMax[c] == freqcWork[c]:
            H0.append(np.pi * 2 * freqcMax[c] * acDagList[c] * acList[c])
        else:
            freqcList = gaus_flat(tList, tList[-1], sigma[1], freqcWork[c], freqcMax[c])
            H0.append([acDagList[c] * acList[c], freqcList * np.pi * 2])

    for gkey in g_qc:
        q, c = int(gkey[1]), int(gkey[3])
        if q < qNum and c < cNum:
            if freqqMax[q] == freqqWork[q]:
                freqqList = freqqWork[q]
            else:
                freqqList = gaus_flat(tList, tList[-1], sigma[0], freqqWork[q], freqqMax[q])
            if freqcMax[c] == freqcWork[c]:
                freqcList = freqcWork[c]
            else:
                freqcList = gaus_flat(tList, tList[-1], sigma[1], freqcWork[c], freqcMax[c])
            g = g_qc[gkey] * np.sqrt(freqqList * freqcList)
            if isinstance(g, float):
                Hi.append(np.pi * 2 * g * (aqDagList[q] + aqList[q]) * (acList[c] + acDagList[c]))
            else:
                Hi.append([(aqDagList[q] + aqList[q]) * (acList[c] + acDagList[c]), g * np.pi * 2])
    for gkey in g_qq:
        q0, q1 = int(gkey[1]), int(gkey[3])
        if q0 < qNum and q1 < qNum:
            if freqqWork[q0] == freqqMax[q0]:
                freqq1List = gaus_flat(tList, tList[-1], sigma[0], freqqWork[q1], freqqMax[q1])
                g = g_qq[gkey] * np.sqrt(freqqMax[q0] * freqq1List)
            else:
                freqq0List = gaus_flat(tList, tList[-1], sigma[0], freqqWork[q0], freqqMax[q0])
                g = g_qq[gkey] * np.sqrt(freqq0List * freqqMax[q1])
            Hi.append([(aqDagList[q0] + aqList[q0]) * (aqList[q1] + aqDagList[q1]), g * np.pi * 2])
    return [*H0, *HAnharm, *Hi]


# 求解上述哈密顿本征态并按照顺序排列

def eigensolve(H0, H, energyLevel, bitNum):

    energy_info = {'dim' : [energyLevel] * bitNum, 'exci_num': energyLevel * bitNum, 'bas_list':[]}

    for bas in state_number_enumerate(energy_info['dim'], excitations=energy_info['exci_num']):
        energy_info['bas_list'].append(state_number_qobj(energy_info['dim'], bas))
    
    ei_states0 = H0.eigenstates()
    ei_energy0 = ei_states0[0]
    ei_vector0 = ei_states0[1]
    
    if H0 == H:
        ei_energy = ei_energy0
        ei_vector = ei_vector0
    else:
        ei_states = H.eigenstates()
        ei_energy = ei_states[0]
        ei_vector = ei_states[1]

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
    return states_order, states0_order, energy_order / (np.pi * 2), energy0_order / (np.pi * 2)
    # return states_order, states0_order, energy_order, energy0_order

# zz耦合

def zzcoupling(g_qc, g_qq, energyLevel, bitNum, omegaq, omegac, anharm):

    H0, Hi = H(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm)
    _, _, energy, _ = eigensolve(H0, H0 + Hi, energyLevel, bitNum)
    zeta = energy[state_number_index([energyLevel] * bitNum, [1, 0, 1])] - energy[state_number_index([energyLevel] * bitNum, [1, 0, 0])] - \
            energy[state_number_index([energyLevel] * bitNum, [0, 0, 1])] + energy[state_number_index([energyLevel] * bitNum, [0, 0, 0])]

    g1c = g_qc['q0c0'] * np.sqrt(omegaq[0] * omegac[0])
    g2c = g_qc['q0c0'] * np.sqrt(omegaq[1] * omegac[0])
    g12 = g_qq['q0q1'] * np.sqrt(omegaq[0] * omegaq[1])
    g = g12 + 0.5 * g1c * g2c * (1 / (omegaq[0] - omegac[0]) + 1 / (omegaq[1] - omegac[0]) - 
                                1 / (omegaq[0] + omegac[0]) - 1 / (omegaq[1] + omegac[0]))

    print(omegaq[1], omegac[0], zeta, g, g12, g1c, g2c)
    return zeta, g, g12, g1c, g2c

# 计算非绝热跃迁率

def state_deriv(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm, computeBasName, stateOrder, computBas, allBas):
    # omegac时的特征向量

    H00, Hi0 = H(g_qc, g_qq, [omegaq[0], omegaq[1]], [omegac], bitNum, energyLevel, anharm)
    dressState0, bareState0, dressEnergy0, _ = eigensolve(H00, H00 + Hi0, energyLevel, bitNum)

    stateEnergy0 = []
    for i in range(len(bareState0)):
        if bareState0[i] in allBas:
            stateEnergy0.append([dressEnergy0[i], dressState0[i]])

    stateEnergy0 = sorted(stateEnergy0, key=lambda x : x[0], reverse=True)

    stateDict0 = dict()
    energyDict0 = dict()

    for state in stateOrder:
        stateIndex = stateOrder.index(state)
        energyDict0[state] = stateEnergy0[stateIndex][0]
        stateDict0[state] = stateEnergy0[stateIndex][1]

    # omegac+domegac 时的特征向量
    domegac = 1e-6
    omegac1 = omegac + domegac
    H01, Hi1 = H(g_qc, g_qq, [omegaq[0], omegaq[1]], [omegac1], bitNum, energyLevel, anharm)
    dressState1, bareState1, dressEnergy1, _ = eigensolve(H01, H01 + Hi1, energyLevel, bitNum)

    stateEnergy1 = []

    for i in range(len(bareState1)):
        if bareState1[i] in allBas:
            stateEnergy1.append([dressEnergy1[i], dressState1[i]])
    
    stateEnergy1 = sorted(stateEnergy1, key=lambda x : x[0], reverse=True)

    stateDict1 = dict()

    for state in stateOrder:
        stateIndex = stateOrder.index(state)
        stateDict1[state] = stateEnergy1[stateIndex][1]

    stateDeriv = dict()
    for state in computeBasName:
        deriv0 = np.max(np.abs(np.array(stateDict0[state]) - np.array(stateDict1[state])))
        deriv1 = np.max(np.abs(np.array(stateDict0[state]) + np.array(stateDict1[state])))
        if deriv0 < deriv1:
            stateDeriv[state] = (stateDict0[state] - stateDict1[state]) / domegac
        else:
            stateDeriv[state] = (stateDict0[state] + stateDict1[state]) / domegac
    return stateDeriv, stateDict0, energyDict0

def D_factor(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm, computeBasName, stateOrder, computBas, allBas):
    stateDeriv, stateDict, energyDict = \
    state_deriv(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm, computeBasName, stateOrder, computBas, allBas)
    D = 0
    for s in computeBasName:
        for l in stateOrder:
            if s == l:
                continue
            D += np.abs(np.array(stateDict[l].dag() * stateDeriv[s])[0][0] / (energyDict[s] - energyDict[l] + 1e-9))
    return D

# 共振频率

def energy_ren(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm):
    H0, Hi = H(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm)
    _, _, energy, _ = eigensolve(H0, H0 + Hi, energyLevel, bitNum)
    ren1 = energy[state_number_index([energyLevel] * bitNum, [0, 0, 2])] - energy[state_number_index([energyLevel] * bitNum, [1, 0, 1])]
    ren2 = energy[state_number_index([energyLevel] * bitNum, [2, 0, 0])] - energy[state_number_index([energyLevel] * bitNum, [1, 0, 1])] 
    print(omegaq[1], omegac[0], ren1, ren2)
    return ren1, ren2

# 单比特脉冲

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

# 保真度

def Fidelity_X(UEff, gateLen=1):
    d = UEff.shape[0]
    phi = []

    for i in range(gateLen):
        phi.append(np.pi)
    if i == 0:
        XIdeal = sigmax()
    else:
        XIdeal = tensor(sigmax(), sigmax())
    XIdeal = np.array(XIdeal)
    if gateLen == 1:
        res = minimize(fidelity, phi, args=(UEff, XIdeal, d), method='Nelder-Mead')
        return -res.fun
    else:
        res1 = minimize(fidelity, phi, args=(UEff, XIdeal, d), method='Nelder-Mead')
        print('opt phi', res1.x[0], res1.x[1])
        Rz = tensor(rz(res1.x[1] + np.pi), rz(res1.x[0] + np.pi))
        res2 = np.real((np.trace(np.dot(np.dot(Rz, UEff).conjugate(), np.dot(Rz, UEff))) + 
                np.abs(np.trace(np.dot(np.dot(Rz, UEff).conjugate(), XIdeal))) ** 2) / (d * (d + 1))) 
        res1 = -res1.fun
        if res1 > res2:
            return res1
        else:
            return res2
            
def Fidelity_CZ(UEff):
    d = UEff.shape[0]
    phi = [np.pi, np.pi]
    U_CZ = cphase(theta = np.pi)
    res1 = minimize(fidelity, phi, args=(UEff, U_CZ, d), method='Nelder-Mead')
    Rz1 = tensor(rz(res1.x[1]), rz(res1.x[0]))
    Rz2 = tensor(rz(res1.x[1] + np.pi), rz(res1.x[0] + np.pi))
    res2 = np.real((np.trace(np.dot(np.dot(Rz2, UEff).conjugate(), np.dot(Rz2, UEff))) + 
            np.abs(np.trace(np.dot(np.dot(Rz2, UEff).conjugate(), U_CZ))) ** 2) / (d * (d + 1))) 
    res1 = -res1.fun
    if res1 > res2:
        return res1, np.dot(Rz1, UEff)
    else:
        return res2, np.dot(Rz2, UEff)

def fidelity(phi, UEff, UIdeal, d):
    start = 0
    for i in phi:
        if start == 0:
            Rz = rz(i)
            start = 1
        else:
            Rz = tensor(rz(i), Rz)
    Rz = np.array(Rz)
    U = np.dot(Rz, UEff)
    f = -((np.trace(np.dot(U.conjugate(), U)) + np.abs(np.trace(np.dot(U.conjugate(), UIdeal))) ** 2) / (d * (d + 1)))
    return np.real(f)

# 单比特门哈密顿量演化

def evolutionX0(pulse_paras, pulse_const, H, states, energyLevel, bitNum, anharm, neighbor):

    _, _, _, _, sxList, syList, _ = getaad(energyLevel, bitNum)

    H_x0 = [sxList[0], drive_pulseX0]
    H_y0 = [syList[0], drive_pulseY0]
    Ht = [H, H_x0, H_y0]

    args = dict()

    args['gate time'] = pulse_const[0]
    args['q0 drive frequency'] = pulse_const[1] * np.pi * 2
    args['q1 drive frequency'] = pulse_const[2] * np.pi * 2
    args['drag weight'] = pulse_const[3]

    args['q0 anharm'] = anharm['q0']
    args['q0 detune'] = pulse_paras[0] * np.pi * 2
    args['q0 amp'] = pulse_paras[1]
    args['q0 phi'] = -pulse_paras[0] * args['gate time'] * np.pi 

    tList = np.arange(0, args['gate time'], args['gate time'] / 300)
    U_full=propagator(Ht, tList, args=args)[-1]
    U=np.zeros([2, 2], dtype='complex128')
    for i in range(2):
        for j in range(2):
            U[i][j] = (states[state_number_index([energyLevel] * bitNum, [neighbor, 0, i])].dag() * U_full *
                states[state_number_index([energyLevel] * bitNum, [neighbor, 0, j])]).full()[0][0]
    F = Fidelity_X(U)
    error = 1 - F
    return error


def evolutionX1(pulse_paras, pulse_const, H, states, energyLevel, bitNum, anharm, neighbor):

    _, _, _, _, sxList, syList, _ = getaad(energyLevel, bitNum)

    H_x1 = [sxList[1], drive_pulseX1]
    H_y1 = [syList[1], drive_pulseY1]
    Ht = [H, H_x1, H_y1]

    args = dict()

    args['gate time'] = pulse_const[0]
    args['q0 drive frequency'] = pulse_const[1] * np.pi * 2
    args['q1 drive frequency'] = pulse_const[2] * np.pi * 2
    args['drag weight'] = pulse_const[3]

    args['q1 anharm'] = anharm['q1']
    args['q1 detune'] = pulse_paras[0] * np.pi * 2
    args['q1 amp'] = pulse_paras[1]
    args['q1 phi'] = -pulse_paras[0] * args['gate time'] * np.pi 

    tList = np.arange(0, args['gate time'], args['gate time'] / 300)
    U_full=propagator(Ht, tList, args=args)[-1]
    U=np.zeros([2, 2], dtype='complex128')
    for i in range(2):
        for j in range(2):
                U[i][j] = (states[state_number_index([energyLevel] * bitNum, [i, 0, neighbor])].dag() * U_full *
                states[state_number_index([energyLevel] * bitNum, [j, 0, neighbor])]).full()[0][0]
    F = Fidelity_X(U)
    error = 1 - F
    return error


def evolutionXX(pulse_paras, pulse_const, H, states, energyLevel, bitNum, anharm):

    _, _, _, _, sxList, syList, _ = getaad(energyLevel, bitNum)

    H_x0 = [sxList[0], drive_pulseX0]
    H_y0 = [syList[0], drive_pulseY0]
    H_x1 = [sxList[1], drive_pulseX1]
    H_y1 = [syList[1], drive_pulseY1]
    Ht = [H, H_x0, H_y0, H_x1, H_y1]

    args = dict()

    args['gate time'] = pulse_const[0]
    args['q0 drive frequency'] = pulse_const[1] * np.pi * 2
    args['q1 drive frequency'] = pulse_const[2] * np.pi * 2
    args['drag weight'] = pulse_const[3]

    pulse_paras1 = [pulse_paras[0], pulse_paras[1]]
    pulse_paras2 = [pulse_paras[2], pulse_paras[3]]
    args['q0 anharm'] = anharm['q0']
    args['q0 detune'] = pulse_paras1[0] * np.pi * 2
    args['q0 amp'] = pulse_paras1[1]
    args['q0 phi'] = -pulse_paras1[0] * args['gate time'] * np.pi 
    args['q1 anharm'] = anharm['q1']
    args['q1 detune'] = pulse_paras2[0] * np.pi * 2
    args['q1 amp'] = pulse_paras2[1]
    args['q1 phi'] = -pulse_paras2[0] * args['gate time'] * np.pi 

    tList = np.arange(0, args['gate time'], args['gate time'] / 300)
    U_full=propagator(Ht, tList, args=args)[-1]
    U=np.zeros([4, 4], dtype='complex128')
    for i in range(2):
        for j in range(2):
            for m in range(2):
                for n in range(2):
                    ii = i * 2 + j
                    jj = m * 2 + n
                    U[ii][jj] = (states[state_number_index([energyLevel] * bitNum, [i, 0, j])].dag() * U_full *
                                states[state_number_index([energyLevel] * bitNum, [m, 0, n])]).full()[0][0]
    F = Fidelity_X(U, 2)
    error = 1 - F
    return error

# 校准单比特门参数

def par_X(g_qc, g_qq, energyLevel, bitNum, omegaq, omegac, tg, drag, anharm):    
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
    
    result0 = minimize(evolutionX0, xIni[:2], args=(pulseConst, H0 + Hi, states, energyLevel, bitNum, anharm, 0), 
                        method='Nelder-Mead', options={'ftol' : 1e-06})
    errix0 = result0.fun
    result1 = minimize(evolutionX1, xIni[2:], args=(pulseConst, H0 + Hi, states, energyLevel, bitNum, anharm, 0), 
                        method='Nelder-Mead', options={'ftol' : 1e-06})
    x = [result0.x[0], result0.x[1], result1.x[0], result1.x[1]]

    errix1 = evolutionX0(x, pulseConst, H0 + Hi, states, energyLevel, bitNum, anharm, 1) 
    errxx = evolutionXX(x, pulseConst, H0 + Hi, states, energyLevel, bitNum, anharm)
    print('opt args', x[0], x[1], x[2], x[3])
    print('omega1', omegaq[1], 'omegac', omegac[0])
    print(errix0, errix1, errxx)
    return np.abs(errix0), np.abs(errix1), np.abs(errxx)

# 不同脉冲参数下的单比特门保真度

def pars_X(g_qc, g_qq, energyLevel, bitNum, omegaq, omegac, tg, drag, anharm, detune, amp):
    wd0 = omegaq[0]
    wd1 = omegaq[1]
    pulseConst = [tg, wd0, wd1, drag]
    pulseParas = [detune, amp]
    H0, Hi = H(g_qc, g_qq, omegaq, omegac, bitNum, energyLevel, anharm)
    states, _, _, _ = eigensolve(H0, H0 + Hi, energyLevel, bitNum)
    err = evolutionX0(pulseParas, pulseConst, H0 + Hi, states, energyLevel, bitNum, anharm, 0)
    print(err)
    return err

# 不同参数下的两比特门

def evolution_CZ(lambdas, tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm, isAd=False):
    tList = np.arange(0, tg, tg / 10000)

    H0, Hi = H(g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm)
    states, _, _, _ = eigensolve(H0, H0 + Hi, energyLevel, bitNum)

    args = dict()
    args['g qc'] = g_qc
    args['g qq'] = g_qq
    args['freq q max'] = freqqMax
    args['freq c max'] = freqcMax
    args['lambdas'] = lambdas
    args['bitNum'] = bitNum
    args['energy level'] = energyLevel
    args['anharm'] = anharm

    if isAd:
        Htwo = H_no_ad(tList, args=args)
    else:
        Htwo = H_ad(tList, args=args)

    U_full = propagator(Htwo, tList)[-1]

    U = np.zeros([4, 4], dtype='complex128')
    for i in range(2):
        for j in range(2):
            for m in range(2):
                for n in range(2):
                    ii = i * 2 + j
                    jj = m * 2 + n
                    U[ii][jj] = (states[state_number_index([energyLevel] * bitNum, [i, 0, j])].dag() * U_full *
                                states[state_number_index([energyLevel] * bitNum, [m, 0, n])]).full()[0][0]            
    F, U = Fidelity_CZ(U)
    error = 1 - F
    # print(lambdas, error)
    return error

# 不同参数下的泄漏

def evolution_leak(lambdas, tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm, isAd=False):
    tList = np.arange(0, tg, tg / 10000)

    H0, Hi = H(g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm)
    states, _, _, _ = eigensolve(H0, Hi + H0, energyLevel, bitNum)

    args = dict()
    args['g qc'] = g_qc
    args['g qq'] = g_qq
    args['freq q max'] = freqqMax
    args['freq c max'] = freqcMax
    args['lambdas'] = lambdas
    args['bitNum'] = bitNum
    args['energy level'] = energyLevel
    args['anharm'] = anharm

    if isAd:
        Htwo = H_no_ad(tList, args=args)
    else:
        Htwo = H_ad(tList, args=args)

    basis000 = states[state_number_index([energyLevel] * bitNum, [0, 0, 0])]
    basis001 = states[state_number_index([energyLevel] * bitNum, [0, 0, 1])]
    basis100 = states[state_number_index([energyLevel] * bitNum, [1, 0, 0])]
    basis101 = states[state_number_index([energyLevel] * bitNum, [1, 0, 1])]

    psi = mesolve(Htwo, 0.5 * (basis000 + basis001 + basis100 + basis101), tList).states
    finalPsi = psi[-1]
    leakage = 1 - (np.abs(finalPsi.overlap(basis000)) ** 2 + np.abs(finalPsi.overlap(basis001)) ** 2 + \
        np.abs(finalPsi.overlap(basis100)) ** 2 + np.abs(finalPsi.overlap(basis101)) ** 2)

    return leakage

# 不同参数下的受控相位0

def evolution_cphase(lambdas, tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm, isAd=False):
    tList = np.arange(0, tg, tg / 10000)

    H0, Hi = H(g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm)
    states, _, _, _ = eigensolve(H0, H0 + Hi, energyLevel, bitNum)

    args = dict()
    args['g qc'] = g_qc
    args['g qq'] = g_qq
    args['freq q max'] = freqqMax
    args['freq c max'] = freqcMax
    args['lambdas'] = lambdas
    args['bitNum'] = bitNum
    args['energy level'] = energyLevel
    args['anharm'] = anharm

    if isAd:
        Htwo = H_no_ad(tList, args=args)
    else:
        Htwo = H_ad(tList, args=args)

    U_full = propagator(Htwo, tList)[-1]

    U = np.zeros([4, 4], dtype='complex128')
    for i in range(2):
        for j in range(2):
            for m in range(2):
                for n in range(2):
                    ii = i * 2 + j
                    jj = m * 2 + n
                    U[ii][jj] = (states[state_number_index([energyLevel] * bitNum, [i, 0, j])].dag() * U_full *
                                states[state_number_index([energyLevel] * bitNum, [m, 0, n])]).full()[0][0]            
    _, U = Fidelity_CZ(U)
    return cmath.phase(U[3, 3] / U[2, 2])

def par_CZ(g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm):
    tg = 120

    intervalHigh = max(freqqMax)
    intervalLow = min(2 * freqqMax[0] + anharm['q0'], 2 * freqqMax[1] + anharm['q1']) - min(freqqMax)
    amp = (freqcMax[0] - (intervalHigh + intervalLow) * 0.5) * 0.5
    lambdaIni = [amp, amp / 10, amp / 100]

    res = minimize(evolution_CZ, lambdaIni, args=(tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm), 
                                            method='Nelder-Mead', options={'maxiter' : 500})
    error = res.fun
    lambdas = res.x
    leak = evolution_leak(lambdas, tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm)
    phi = evolution_cphase(lambdas, tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm)
    print('omega1', freqqMax, 'omegac', freqcMax, 'anharm', anharm['c0'], 
        'lambda0', lambdas, 'err', error, 'leak', leak, 'phi', phi)
    return error, leak, phi

def par_CZ_no_ad(g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm):
    tg = 120

    if freqqMax[0] < freqqMax[1]:
        freqqWork = [freqqMax[0] - anharm['q1']]
    else:
        freqqWork = [freqqMax[0] + anharm['q0']]
    freqcWork = [5]                                      
    sigma = [0.8, 0.8]
    lambdasIni = [*freqqWork, *freqcWork, *sigma]
    res = minimize(evolution_CZ, lambdasIni, args=(tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm, True),
                                                method='Nelder-Mead', options={'maxiter' : 500})
    error = res.fun
    lambdas = res.x
    leak = evolution_leak(lambdas, tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm, isAd=True)
    phi = evolution_cphase(lambdas, tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm, isAd=True)
    print('omega1', freqqMax, 'omegac', freqcMax, 'anharm', anharm['c0'], 
        'omega1Work', lambdas[0], 'omegacWork', lambdas[1], 'sigma1', lambdas[2], 'sigma2', lambdas[3], 
        'err', error, 'leak', leak, 'phi', phi)
    return error, leak, phi

# 频率磁通转换函数

def qubit_spectrum(fluxList, fqMax, d=0):
    return fqMax * np.sqrt(np.cos(fluxList) ** 2 + (d * np.sin(fluxList)) ** 2)

def flux2freq(flux, fqMax):
    fluxList = np.arange(0, 1, 1 / 1000)
    freqList = qubit_spectrum(fluxList, fqMax)
    func_interp = interp1d(fluxList, freqList, kind='cubic')
    if isinstance(flux, (int, float)):
        return float(func_interp(flux))
    else:
        return func_interp(flux)

def freq2flux(freq, fqMax):
    fluxList = np.arange(0, 1, 1 / 1000)
    freqList = qubit_spectrum(fluxList, fqMax)
    func_interp = interp1d(freqList, fluxList, kind='cubic')
    if isinstance(freq, (int, float)):
        return float(func_interp(freq))
    else:
        return func_interp(freq)

# 脉冲波形

def freq_pulse(tList, lambdas):
    n = 1
    freqList = np.zeros(len(tList))
    for i in lambdas:
        freqList += i * (np.cos(2 * np.pi * n * tList / tList[-1]) - 1)
    return freqList

def gaus_flat(tList, pulseLen, sigma, freqWork, freqMax):
    flattopStart = 3 * sigma
    flattopEnd = pulseLen - 3 * sigma

    freqList = 0.5 * (freqWork - freqMax) * (erf((tList - flattopStart) / np.sqrt(2) * sigma) - \
            erf((tList - flattopEnd) / np.sqrt(2) * sigma)) + freqMax
    return freqList

# 能级排序

def sort_energy(result, basList, basNameList, interestKey):
    state0 = result[1]
    tempEnergys0Dict = dict()
    tempEnergyStates1Dict = dict()

    for s in state0:
        stateindex = basList.index(s)
        if basNameList[stateindex] in interestKey:
            tempEnergys0Dict[basNameList[stateindex]] = result[3][stateindex]
            tempEnergyStates1Dict[basNameList[stateindex]] = (result[2][stateindex], result[0][stateindex])
    
    sortedEnergy1Dict = dict(sorted(tempEnergyStates1Dict.items(), key=lambda x : x[1][0], reverse=True))
    stateEnergy = list(sortedEnergy1Dict.values())

    return tempEnergys0Dict, stateEnergy

# 写数据

def write_data(f, data):
    with open(f, 'w') as fp:
        for d in data:
            fp.write(str(d) + '\n')

if __name__ == '__main__':

    # 并行设置
    nn = int(os.path.basename(__file__)[-4])
    parallelNum = 5

    # 参数设置

    anharm = {'q0' : -200 * 1e-3, 'c0' : -200 * 1e-3, 'q1' : -200 * 1e-3, 'c1' : 200 * 1e-3, 'q2' : -200 * 1e-3}
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

    basNameList = []
    basList = []
    for bas in state_number_enumerate([energyLevel] * bitNum, excitations=bitNum * (energyLevel - 1)):
        basList.append(state_number_qobj([energyLevel] * bitNum, bas))
        basNameList.append(''.join(map(str, bas)))
    print(basNameList)

    omega0 = 5

    delta12Low, delta12High = -1, 1
    delta12step = 50
    delta12s = np.arange(delta12Low, delta12High, (delta12High - delta12Low) / delta12step)
    omega1List = omega0 - delta12s
    # order = 0

    alphacLow, alphacHigh = -0.5, 1
    alphacStep = 1
    alphacList = np.arange(alphacLow, alphacHigh, (alphacHigh - alphacLow) / alphacStep)
    
    clow, chigh = 4, 7
    cstep = 50
    omegacList = np.arange(clow, chigh, (chigh - clow) / cstep)
    omegacList = omegacList[::-1]

    len_delta12 = len(delta12s)
    len_alphac = len(alphacList)
    len_omegac = len(omegacList)

    energys = dict()
    for i in basNameList:
        if int(i[0]) + int(i[1]) + int(i[2]) > 2:
            continue
        else:
            energys[i] = [np.zeros(len_delta12 * len_alphac * len_omegac), 
                        np.zeros(len_delta12 * len_alphac * len_omegac)]

    stateEnergys = []

    interestKey = ['020', '011', '110', '002', '101', '200', '010', '001', '100', '000']
    computKey = ['101', '001', '100', '000']

    DFactorPicname = 'D factor alphac=' + str(alphacLow)
    xiPicname = 'zz coupling alphac=' + str(alphacLow)

    xlabel = 'detune12(MHz)'
    ylabel = 'omegac(GHz)'

    xstick = delta12s * 1e3
    ystick = omegacList

    H0_list = []
    H_list = []
    stateOrders = []
    anharms = []
    omegacs = []
    omegaqs = []

    computeBasName = []
    computeBas = []

    allBas = []

    g_qqs = [g_qq] * len_delta12 * len_alphac * len_omegac
    g_qcs = [g_qc] * len_delta12 * len_alphac * len_omegac
    bitNums = [bitNum] * len_delta12 * len_alphac * len_omegac
    energyLevels = [energyLevel] * len_delta12 * len_alphac * len_omegac

    for omega1 in omega1List:
        for alphac in alphacList:
            for omegac in omegacList:     

                omegaqs.append([omega0, omega1])
                anhar = deepcopy(anharm)
                anhar['c0'] = alphac
                anharms.append(anhar)

                omegacs.append(omegac)

                H0, Hi = H(g_qc, g_qq, [omega0, omega1], [omegac], bitNum, energyLevel, anhar)

                H0_list.append(H0)
                H_list.append(H0 + Hi)

                if omegac == max(omegacList):
                    _, state0, _, energy0 = eigensolve(H0, H0, energyLevel, bitNum)
                    tempEnergys0Dict = dict()
                    for s in state0:
                        stateindex = basList.index(s)
                        if basNameList[stateindex] in interestKey:
                            tempEnergys0Dict[basNameList[stateindex]] = energy0[stateindex]
                    sortedEnergy0Dict = dict(sorted(tempEnergys0Dict.items(), key=lambda x : x[1], reverse=True))
                    stateOrders.append(list(sortedEnergy0Dict.keys()))
                else:
                    stateOrders.append(stateOrders[-1])

                computBasName = []
                computBas = []

                alBas = []     

                for s in stateOrders[-1]:
                    alBas.append(basList[basNameList.index(s)])
                    if s in computKey:
                        computBasName.append(s)
                        computBas.append(basList[basNameList.index(s)])
                computeBas.append(computBas)
                computeBasName.append(computBasName)
                allBas.append(alBas)

    # 求坠饰态本征向量

    tStart = time.time()
    print(cpu_count())
    p = Pool(cpu_count())
    result_list = p.starmap(eigensolve, zip(H0_list, H_list, energyLevels, bitNums))
    p.close()
    p.join()
    t = time.time() - tStart
    print(t / 60)

    tStart = time.time()

    print(cpu_count())
    p = Pool(cpu_count())
    sortEnergys = p.starmap(sort_energy, 
    zip(result_list, [basList] * len(result_list), [basNameList] * len(result_list), [interestKey] * len(result_list)))
    p.close()
    p.join()

    od = 0
    for tempEnergys0Dict, stateEnergy in sortEnergys:
        for s in stateOrders[od]:
            energys[s][0][od] = tempEnergys0Dict[s]
            energys[s][1][od] = stateEnergy[list(stateOrders[od]).index(s)][0]
        od += 1

    t = time.time() - tStart
    print(t / 60)

    # zz coupling

    xi = np.abs(np.array(energys['101'][1]) - np.array(energys['100'][1]) - np.array(energys['001'][1]) + np.array(energys['000'])[1]) * 1e3

    # 非绝热跃迁

    tStart = time.time()
    print(cpu_count())
    p = Pool(cpu_count())
    DList = p.starmap(D_factor, 
    zip(g_qcs, g_qqs, omegaqs, omegacs, bitNums, energyLevels, anharms, computeBasName, stateOrders, computeBas, allBas))
    p.close()
    p.join()
    t = time.time() - tStart
    print(t / 60)

    # zz D 随着delta/alpha和omegac的变化

    draw_heat_map(xstick, ystick, xi, xiPicname, xiPicname, xlabel, ylabel, 'abs')

    for i in range(len(DList)):
        if np.abs(DList[i]) > 200:
            DList[i] = 200

    draw_heat_map(xstick, ystick, DList, DFactorPicname, DFactorPicname, xlabel, ylabel, 'abs')

    zzOverD = xi / DList
    zzOverD = zzOverD.reshape(len_delta12, len_alphac, len_omegac)

    xii = np.zeros((len_delta12, len_alphac))
    xi = xi.reshape(len_delta12, len_alphac, len_omegac)
    DListt = np.zeros((len_delta12, len_alphac))
    DList = np.array(DList).reshape(len_delta12, len_alphac, len_omegac)

    # zz D 随着delta和alpha变化

    for i in range(len_delta12):
        for j in range(len_alphac):
            ocd = 0
            hasBest = False
            for oc in zzOverD[i, j]:
                if ocd == 0 or ocd == cstep - 1:
                    ocd += 1
                    continue
                elif oc > 1 and zzOverD[i, j, ocd] > zzOverD[i, j, ocd - 1] and zzOverD[i, j, ocd] > zzOverD[i, j, ocd + 1]:
                    hasBest = True
                    break
                ocd += 1
            if hasBest:
                xii[i, j] = xi[i, j, ocd]
                DListt[i, j] = DList[i, j, ocd]
            else:
                xii[i, j] = max(xi[i, j])
                DListt[i, j] = max(DList[i, j])

    xii = xii.reshape(len_delta12 * len_alphac)
    write_data(xiPicname + '.txt', xii)

    draw_heat_map(xstick, ystick, xii, xiPicname, xiPicname, xlabel, ylabel, 'abs')

    DListt = DListt.reshape(len_delta12 * len_alphac)
    write_data(DFactorPicname + '.txt', DListt)

    draw_heat_map(xstick, ystick, DListt, DFactorPicname, DFactorPicname, xlabel, ylabel, 'abs')

    # 画能级图

    drawKey = ['020', '011', '110', '002', '101', '200']
    colors = ['deepskyblue', 'grey', 'peru', 'chartreuse', 'red', 
            'mediumblue', 'pink', 'green', 'brown', 'hotpink']

    id = 0
    plt.figure(figsize=(7, 8))
    ax = plt.subplot(211)
    plt.title('delta=' + str(delta12Low * 1e3)[:5] + '(MHz) alphac=' + str(alphacLow * 1e3)[:5] + '(MHz)')
    ax.set_ylabel('eigen frequency(GHz)')
    for i in stateOrders[0]:
        if i in drawKey:
            ax.plot(omegacList, energys[i][0], linestyle='--', color=colors[id])
            ax.plot(omegacList, energys[i][1], label='|' + i + '>', color=colors[id])
            id += 1
    ax.plot(omegacList, energys['001'][1] + energys['100'][1] - 2 * energys['000'][1], label='|100>+|001>', color=colors[id + 1])
    ax.set_xlim(clow, chigh)
    plt.legend(loc=4)

    ax = plt.subplot(212)
    xi = np.abs(np.array(energys['101'][1]) - np.array(energys['100'][1]) - np.array(energys['001'][1]) + np.array(energys['000'][1])) * 1e3

    lin1 = ax.plot(omegacList, xi, label='E_101-E_100-E_001', color=colors[0])
    lin2 = ax.plot(omegacList, DList[0][0], label='D factor', color=colors[1])
    ax.set_ylabel('frequency(MHz)')
    ax.set_xlabel('c frequency(GHz)')
    ax.set_xlim(clow, chigh)

    ax_twinx = ax.twinx()
    # lin2 = ax_twinx.semilogy(omegacList, xi / DList, label='D/zz', color=colors[2])
    lin3 = ax_twinx.plot(omegacList, xi / DList[0][0], label='D/zz', color=colors[2])
    ax_twinx.set_ylabel('zz/D')
    ax_twinx.set_xlim(clow, chigh)

    lns = lin1 + lin2 + lin3
    labs = [l.get_label() for l in lns]
    ax_twinx.legend(lns, labs)

    plt.savefig('energy level diagram delta=' + str(round((omega0 - omega1) * 1e3, 3)) + 
                ' alphac=' + str(round(anhar['c0'] * 1e3, 3)) + '.pdf', dpi=500)

    # 计算单比特门残留耦合情况

    # omega1 omegac

    omega0 = 4.5
    # qlow = max(omega0 - anharm['q1'], omega0 + anharm['q0'])
    qlow = 3.9
    qhigh = 5.1
    clow = 5
    # clow = 6
    chigh = 7
    step = 50
    omega1List = np.arange(qlow, qhigh, (qhigh - qlow) / step)
    omegacList = np.arange(clow, chigh, (chigh - clow) / step)

    omega0s = [omega0] * len(omegacList) * len(omegacList)
    g_qcs = [g_qc] * len(omegacList) * len(omegacList)
    bitNums = [bitNum] * len(omegacList) * len(omegacList)
    energyLevels = [energyLevel] * len(omegacList) * len(omegacList)
    anharms = [anharm] * len(omegacList) * len(omegacList)
    g_qqs = [g_qq] * len(omegacList) * len(omegacList)

    omegaqs = []
    omegacs = []
    for w1 in omega1List:
        for wc in omegacList:
            omegaqs.append([omega0, w1])
            omegacs.append([wc])

    tStart = time.time()
    print(cpu_count())
    p = Pool(cpu_count())
    data_list = p.starmap(zzcoupling, zip(g_qcs, g_qqs, energyLevels, bitNums, omegaqs, omegacs, anharms))
    p.close()
    p.join()
    t = time.time() - tStart
    print(t)


    zeta_list = []
    g_list = []
    g12_list = []
    g1c_list = []
    g2c_list = []
    for i in data_list:
        zeta_list.append(i[0])
        g_list.append(i[1])
        g12_list.append(i[2])
        g1c_list.append(i[3])
        g2c_list.append(i[4])
    zeta_list = np.array([zeta_list]).reshape(len(omega1List), len(omegacList)) * 1e3
    g_list = np.array([g_list]).reshape(len(omega1List), len(omegacList)) * 1e3
    draw_heat_map(omega1List, omegacList, zeta_list, 'zz', 'zz', 'omega1(GHz)', 'omegac(GHz)', 'logabs')
    draw_heat_map(omega1List, omegacList, g_list, 'g', 'g', 'omega1(GHz)', 'omegac(GHz)',  'logabs')

    # g omegac

    g_qqList = []
    c01List = np.arange(1e-4, 0.3, 0.3 / 50)
    gq0q1 = []
    for c01 in c01List:
        C_12 = {'q0q1' : c01, 'q1q2' : 0.15}
        g_qq = dict()
        for k in C_12:
            q0, q1 = int(k[1]), int(k[3])
            C_1cC_2c = 1
            for kk in C_ic:
                if ('q' + str(q0) in kk and 'q' + str(q1) + kk[2:] in C_ic) or ('q' + str(q1) in kk and 'q' + str(q0) + kk[2:] in C_ic):
                    C_1cC_2c *= C_ic[kk]
                    ck = int(kk[3])
            g_qq[k] = 0.5 * (1 + C_1cC_2c / (C_12[k] * C_c[ck])) * C_12[k] / np.sqrt(C_q[q0] * C_q[q1])
        g_qqList.append(g_qq)
        gq0q1.append(g_qq['q0q1'])

    # delta12s = [1e-5, 0.1, 0.199999, 0.200001, 0.25, 0.3, 0.4, 0.5]
    delta12s = [1e-5, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5]
    for delta12 in delta12s:
        # omegacList = np.arange(5, 7, 2 / 50)
        omegacList = [6]
        omega1s = [4.5] * len(c01List) * len(omegacList)
        omega2s = [4.5 - delta12] * len(c01List) * len(omegacList)
        omegaqs = list(zip(omega1s, omega2s))
        g_qcs = [g_qc] * len(c01List) * len(omegacList)
        bitNums = [bitNum] * len(c01List) * len(omegacList)
        energyLevels = [energyLevel] * len(c01List) * len(omegacList)
        anharms = [anharm] * len(c01List) * len(omegacList)
        omegacs = []
        g_qqs = []
        for g_qq in g_qqList:
            for omegac in omegacList:
                g_qqs.append(g_qq)
                omegacs.append(np.array([omegac]))

        tStart = time.time()
        print(cpu_count())
        p = Pool(cpu_count())
        data_list = p.starmap(zzcoupling, zip(g_qcs, g_qqs, energyLevels, bitNums, omegaqs, omegacs, anharms))
        p.close()
        p.join()
        t = time.time() - tStart
        print(t)

        zeta_list = []
        g_list = []
        g12_list = []
        g1c_list = []
        g2c_list = []
        for i in data_list:
            zeta_list.append(i[0])
            g_list.append(i[1])
            g12_list.append(i[2])
            g1c_list.append(i[3])
            g2c_list.append(i[4])
        zeta_list = np.array([zeta_list]).reshape(len(g_qqList), len(omegacList)) * 1e3
        g_list = np.array([g_list]).reshape(len(g_qqList), len(omegacList)) * 1e3
        draw_heat_map(gq0q1, omegacList, zeta_list, 'zz' + str(delta12), 'zz' + str(delta12), 'g12', 'omegac(GHz)', 'logabs')
        draw_heat_map(gq0q1, omegacList, g_list, 'g' + str(delta12), 'g' + str(delta12), 'g12', 'omegac(GHz)',  'logabs')
        plt.plot(g_list, zeta_list, label=str(delta12))
    plt.xlabel('g(MHz)')
    plt.ylabel('zeta(MHz)')
    plt.legend()
    plt.show()

    # 扫一个单比特门最优参数图

    omega0 = 4.5
    omega1 = 4
    omegac = 4.75
    
    detunelow = -0.1
    detunelow = -0.15 # 倒数第二
    detunelow = -0.2 # 倒数第一
    detunehigh = 0
    amplow = 0.1
    amphigh = 0.35
    step = 50
    detuneList = np.arange(detunelow, detunehigh, (detunehigh - detunelow) / step)
    ampList = np.arange(amplow, amphigh, (amphigh - amplow) / step)

    g_qcs = [g_qc] * step * step
    g_qqs = [g_qq] * step * step
    energyLevels = [energyLevel] * step * step
    bitNums = [bitNum] * step * step
    omegaqs = [[2 * np.pi * omega0, 2 * np.pi * omega1]] * step * step
    omegacs = [[2 * np.pi * omegac]] * step * step
    gatetimes = [30] * step * step
    dragWeights = [0.5] * step * step
    anharms = [anharm] * step * step

    detunes = []
    amps = []
    for i in range(step):
        for j in range(step):
            detunes.append(detuneList[i])
            amps.append(ampList[j])

    tStart = time.time()
    print(cpu_count())
    p = Pool(cpu_count())
    err_list = p.starmap(pars_X, zip(g_qcs, g_qqs, energyLevels, bitNums, 
                    omegaqs, omegacs, gatetimes, dragWeights, anharms, detunes, amps))
    p.close()
    p.join()
    t = time.time() - tStart
    print(t)
    write_data('errampdetune' + str(omega1 - omega0)[:4] + ' ' + str(omegac - omega0)[:4] + '.txt', err_list)                    

    # 频率磁通转换图

    fluxList = np.arange(0, 1, 1 / 100)
    fluxList = np.arange(0, 1, 1 / 100)
    plt.plot(fluxList, qubit_spectrum(fluxList, 6.5) * 0.5 / np.pi)
    plt.plot(fluxList, qubit_spectrum(fluxList, 4.5) * 0.5 / np.pi)
    plt.show()

    omega0 = 4.5
    # qlow = max(omega0 - anharm['q1'], omega0 + anharm['q0'])
    qlow = 5
    qhigh = 5.1
    # clow = 5
    clow = 6
    chigh = 7
    step = 1
    # omega1List = np.arange(qlow, qhigh, (qhigh - qlow) / step)[int(nn * step / parallelNum) : int((nn + 1) * step / parallelNum)]
    omega1List = np.arange(qlow, qhigh, (qhigh - qlow) / step)
    omegacList = np.arange(clow, chigh, (chigh - clow) / step)
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
            omegaqs.append(np.array([omega0, omega1]))
            omegacs.append(np.array([omegac]))


    # 单比特门工作点

    tStart = time.time()
    print(cpu_count())
    p = Pool(cpu_count())
    err_list = p.starmap(par_X, zip(g_qcs, g_qqs, energyLevels, bitNums, 
                omegaqs, omegacs, gatetimes, dragWeights, anharms))
    p.close()
    p.join()
    t = time.time() - tStart
    print(t)
    err0_list = []
    err1_list = []
    errxx_list = []
    for i in err_list:
        err0_list.append(i[0])
        err1_list.append(i[1])
        errxx_list.append(i[2])
    write_data('errori0 ' + str(anharm['q0'])  + str(anharm['c0']) + ' ' + str(anharm['q1']) + ' ' + str(nn) + '.txt', err0_list)
    write_data('errori1 ' + str(anharm['q0'])  + str(anharm['c0']) + ' ' + str(anharm['q1']) + ' ' + str(nn) + '.txt', err1_list)
    write_data('errorxx ' + str(anharm['q0'])  + str(anharm['c0']) + ' ' + str(anharm['q1']) + ' ' + str(nn) + '.txt', errxx_list)

    # 非绝热门工作点

    cphasePicname = 'noadcphase ' + str(anharm['q0'])  + str(anharm['c0']) + ' ' +  str(anharm['q1']) + str(nn)
    leakPicname = 'noadleak ' + str(anharm['q0'])  + str(anharm['c0']) + ' ' +  str(anharm['q1']) + str(nn)
    czPicname = 'noadcz ' + str(anharm['q0'])  + str(anharm['c0']) + ' ' +  str(anharm['q1']) + str(nn)

    tStart = time.time()
    print(cpu_count())
    p = Pool(cpu_count())
    # p = Pool(1)
    err_list = p.starmap(par_CZ_no_ad, zip(g_qcs, g_qqs, omegaqs, omegacs,
                                            bitNums, energyLevels, anharms))
    p.close()
    p.join()
    t = time.time() - tStart
    print(t)

    errList = []
    leakList = []
    cphaseList = []

    for i in err_list:
        errList.append(i[0])
        leakList.append(i[1])
        cphaseList.append(i[2])

    write_data(czPicname + '.txt', errList)
    write_data(leakPicname + '.txt', leakList)
    write_data(cphasePicname + '.txt', cphaseList)
