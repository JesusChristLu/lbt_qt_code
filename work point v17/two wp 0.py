# import
from asyncore import write
from cProfile import label
import time
import os
from tkinter import Grid
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
# from draw import draw_heat_map

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

def H(omegaq, bitNum, energyLevel, anharm, g_qq, g_qc, omegac):
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

# 非绝热H

def H_no_ad(tList, args):
    g_qc = args['g qc']
    g_qq = args['g qq']
    freqqMax = args['freq q max']
    freqcMax = args['freq c max']
    bitNum = args['bitNum']
    energyLevel = args['energy level']
    anharm = args['anharm']
    freqcWork = [args['lambdas'][1]]
    sigma = args['lambdas'][2:]
    if freqqMax[0] < freqqMax[1]:
        freqqWork = [freqqMax[0], args['lambdas'][0]]
    else:
        freqqWork = [args['lambdas'][0], freqqMax[1]]

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
    

# zz耦合

def zzcoupling(g_qc, g_qq, energyLevel, bitNum, omegaq, omegac, anharm):
    H0, Hi = H(omegaq, bitNum, energyLevel, anharm, g_qq, g_qc, omegac)
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

def evolutionX(pulse_paras, pulse_const, H, states, energyLevel, anharm, q1Freq=None, mu=None):
    bitNum = 1
    _, _, _, _, sxList, syList, _ = getaad(energyLevel, bitNum)

    H_x0 = [sxList[0], drive_pulseX0]
    H_y0 = [syList[0], drive_pulseY0]
    Ht = [H, H_x0, H_y0]

    args = dict()

    args['gate time'] = pulse_const[0]
    args['q0 drive frequency'] = pulse_const[1] * np.pi * 2
    args['drag weight'] = pulse_const[2]

    args['q0 anharm'] = anharm['q0']
    args['q0 detune'] = pulse_paras[0] * np.pi * 2
    args['q0 amp'] = pulse_paras[1]
    args['q0 phi'] = -pulse_paras[0] * args['gate time'] * np.pi 

    if not(q1Freq == None):
        H_cx0 = [sxList[0], drive_pulseX1]
        H_cx1 = [syList[0], drive_pulseY1]

        args['q1 drive frequency'] = q1Freq * np.pi * 2

        args['q1 anharm'] = anharm['q1']
        args['q1 detune'] = 0
        args['q1 amp'] = mu * pulse_paras[1]
        args['q1 phi'] = 0
        Ht = [*Ht, H_cx0, H_cx1]


    tList = np.arange(0, args['gate time'], args['gate time'] / 300)
    U_full=propagator(Ht, tList, args=args)[-1]
    U=np.zeros([2, 2], dtype='complex128')
    for i in range(2):
        for j in range(2):
            U[i][j] = (states[state_number_index([energyLevel] * bitNum, [i])].dag() * U_full *
                states[state_number_index([energyLevel] * bitNum, [j])]).full()[0][0]
    F = Fidelity_X(U)
    error = 1 - F
    return error

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
    H0, Hi = H(omegaq, bitNum, energyLevel, anharm, g_qq, g_qc, omegac)
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

def pars_X(energyLevel, bitNum, omegaq, tg, drag, anharm, pulseParas, q1Freq=None, mu=None):
    pulseConst = [tg, *omegaq, drag]
    H0, _ = H(omegaq, bitNum, energyLevel, anharm)
    states, _, _, _ = eigensolve(H0, H0, energyLevel, bitNum)
    err = evolutionX(pulseParas, pulseConst, H0, states, energyLevel, anharm, q1Freq, mu)
    if not(q1Freq == None):
        print(q1Freq, mu, err)
    else:    
        print(pulseParas, err)
    return err

# 不同参数下的两比特门


def evolution_CZ(lambdas, tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm):
    tList = np.arange(0, tg, tg / 300)

    H0, Hi = H(freqqMax, bitNum, energyLevel, anharm, g_qq, g_qc, freqcMax)
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

    Htwo = H_no_ad(tList, args=args)

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
    print(lambdas, error)
    return error

def par_CZ_no_ad(tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm):
    if freqqMax[0] < freqqMax[1]:
        freqqWork = [freqqMax[0] - anharm['q1']]
    else:
        freqqWork = [freqqMax[1] - anharm['q0']]
    freqcWork = freqcMax                                      
    sigma = [0.8, 0.8]
    lambdasIni = [*freqqWork, *freqcWork, *sigma]
    res = minimize(evolution_CZ, lambdasIni, args=(tg, g_qc, g_qq, freqqMax, freqcMax, bitNum, energyLevel, anharm),
                                                method='Nelder-Mead', options={'ftol' : 1e-4})
    error = res.fun
    return error

def spectator_CZ(tg, g_qq, freqqMax, bitNum, energyLevel, anharm, sq, lambdas=None):
    print()
    gq = list(set([0, 1, 2]) - set([sq]))
    bounds = [(3.6, 4), (0, 1)]
    res = minimize(evolution_CZ, lambdas, args=(tg, g_qq, freqqMax, bitNum, energyLevel, anharm, gq, 0), bounds=bounds, method='SLSQP', options={'maxiter' : 200})

    lambdas = res.x
    error0 = res.fun
    print('err0', error0)
    
    error1 = evolution_CZ(lambdas, tg, g_qq, freqqMax, bitNum, energyLevel, anharm, gq, 1)

    print('sw', freqqMax[sq], 'err0', error0, 'err1', error1)

    return error0, error1

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

# 读写数据

def write_data(f, data):
    with open(f, 'w') as fp:
        for d in data:
            fp.write(str(d) + '\n')

def read_data(f):
    with open(f, 'r') as fp:
        data = fp.read()
        data = data.split('\n')
        if '' in data:
            data.remove('')
    f = []
    for d in data:
        if '(' in d:
            d = d[1:-2]
        f.append(float(d))
    return f

if __name__ == '__main__':

    # 01 能级偏置图
    # E_J = 20
    # E_C = 0.35
    # e = 1.6e-19
    # h = 6.6e-34
    # phi_e = np.arange(-0.99, 0.99, 2 / 1000)
    # omega_01 = np.sqrt(8 * E_J * E_C * np.cos(np.pi * phi_e / 2))
    # T_2 = 1 / (1e9 * np.sqrt(E_C * E_J / 2 * np.abs(np.sin(np.pi * phi_e / 2) * np.tan(np.pi * phi_e / 2))) * 1e-5 * np.pi) * 1e6

    # colors = ['deepskyblue', 'grey', 'peru', 'chartreuse', 'red', 
    #     'mediumblue', 'pink', 'green', 'brown', 'hotpink']
    # ax1 = plt.subplot(111)
    # lin1 = ax1.plot(phi_e, omega_01, label='omega 01', color=colors[0])
    # ax1.set_ylabel('frequency(GHz)')
    # ax1.set_xlabel('phi_e/phi_0')
    # ax2 = ax1.twinx()
    # lin2 = ax2.plot(phi_e, T_2, label='T2', color=colors[1])
    # ax2.set_ylabel('T2(ms)')
    # ax2.set_ylim([0, 100])
    # lns = lin1 + lin2
    # labs = [l.get_label() for l in lns]
    # ax2.legend(lns, labs)
    # plt.show()

    # 并行设置
    nn = int(os.path.basename(__file__)[-4])
    parallelNum = 10

    # 参数设置
    anharm = {'q0' : -220 * 1e-3, 'c0' : -200 * 1e-3, 'q1' : -220 * 1e-3, 'c1' : -200 * 1e-3, 'q2' : -220 * 1e-3}

    g_qc = {'q0c0' : 0.038452, 'q1c0' : 0.038452, 'q1c1' : 0.038452, 'q2c1' : 0.038452}
    g_qq = {'q0q1' : 0.004272, 'q1q2' : 0.004272}


    # 电容与 rho
    # C_q = [100, 100, 100]
    # C_c = [100, 100]
    # C_ic = {'q0c0' : 4, 'q1c0' : 4, 'q1c1' : 4, 'q2c1' : 4}
    # C_12 = {'q0q1' : 0.25, 'q1q2' : 0.25}

    # g_qc = dict()
    # for k in C_ic:
    #     q, c = int(k[1]), int(k[3])
    #     g_qc[k] = 0.5 * C_ic[k] / np.sqrt(C_q[q] * C_c[c])

    # g_qq = dict()
    # for k in C_12:
    #     q0, q1 = int(k[1]), int(k[3])
    #     C_1cC_2c = 1
    #     for kk in C_ic:
    #         if ('q' + str(q0) in kk and 'q' + str(q1) + kk[2:] in C_ic) or ('q' + str(q1) in kk and 'q' + str(q0) + kk[2:] in C_ic):
    #             C_1cC_2c *= C_ic[kk]
    #             ck = int(kk[3])
    #     g_qq[k] = 0.5 * (1 + C_1cC_2c / (C_12[k] * C_c[ck])) * C_12[k] / np.sqrt(C_q[q0] * C_q[q1])
    
    # print('pause')
    # 单比特门

    # omega = 4.5
    # bitNum = 1
    # energyLevel = 3
    # tg = 30
    
    # 单比特门参数空间

    # detuneLow, detuneHigh = -0.1, 0.1
    # detuneStep = 50
    # detunes = np.arange(detuneLow, detuneHigh, (detuneHigh - detuneLow) / detuneStep)
    # ampLow, ampHigh = 0.01, 0.4
    # ampStep = 50
    # amps = np.arange(ampLow, ampHigh, (ampHigh - ampLow) / ampStep)

    # bitNums = [bitNum] * len(detunes) * len(detunes) 
    # energyLevels = [energyLevel] * len(detunes) * len(detunes) 
    # anharms = [anharm] * len(detunes) * len(detunes) 
    # gatetimes = [tg] * len(detunes) * len(detunes) 
    # omegaqs = [[omega]] * len(detunes) * len(detunes) 
    # drags = [0.5] * len(detunes) * len(detunes) 
    # anharms = [anharm] * len(detunes) * len(detunes) 
    # paras = []
    # for detune in detunes:
    #     for amp in amps:
    #         paras.append([detune, amp])

    # tStart = time.time()
    # p = Pool(cpu_count())
    # print(cpu_count())
    # err_list = p.starmap(pars_X, zip(energyLevels, bitNums, 
    #                             omegaqs, gatetimes, drags, anharms, paras))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # draw_heat_map(detunes * 1e3, amps, err_list, 'x gate para', 'x gate para', 'detune(MHz)', 'amplitude', 'logabs')

    # 单比特门xy串扰
    # lambdasIni = [-0.001, 0.2]
    # pulse_const = [tg, omega, 0.5]
    # H0, _ = H([omega], bitNum, energyLevel, anharm)
    # states, _, _, _ = eigensolve(H0, H0, energyLevel, bitNum)
    # res = minimize(evolutionX, lambdasIni, args=(pulse_const, H0, states, energyLevel, anharm),
    #                 method='Nelder-Mead', options={'ftol' : 1e-5})
    # err = res.fun
    # lambdas = res.x
    # print(lambdas, err)

    # detuneLow, detuneHigh = 0, 0.5
    # detuneStep = 50
    # detunes = np.arange(detuneLow, detuneHigh, (detuneHigh - detuneLow) / detuneStep)
    # muLow, muHigh = 0.01, 0.4
    # muStep = 50
    # mus = np.arange(muLow, muHigh, (muHigh - muLow) / muStep)
    
    # lambdas = [lambdas] * len(detunes) * len(detunes) 
    # bitNums = [bitNum] * len(detunes) * len(detunes) 
    # energyLevels = [energyLevel] * len(detunes) * len(detunes) 
    # anharms = [anharm] * len(detunes) * len(detunes)
    # gatetimes = [tg] * len(detunes) * len(detunes)
    # omegaqs = [[omega]] * len(detunes) * len(detunes) 
    # drags = [0.5] * len(detunes) * len(detunes) 
    # anharms = [anharm] * len(detunes) * len(detunes) 

    # q1Freqs = []
    # xtalkps = []
    # for detune in detunes:
    #     for mu in mus:
    #         q1Freqs.append(omega - detune)
    #         xtalkps.append(mu)

    # tStart = time.time()
    # p = Pool(cpu_count())
    # print(cpu_count())
    # err_list = p.starmap(pars_X, zip(energyLevels, bitNums, 
    #                             omegaqs, gatetimes, drags, anharms, lambdas, q1Freqs, xtalkps))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)
    # draw_heat_map(detunes * 1e3, mus, err_list, 'x gate fidelity', 'x gate fidelity', 'detune(MHz)', 'mu', 'logabs')

    # zz残留耦合与保真度关系

    # omega0 = 4.5
    # bitNum = 3
    # energyLevel = 3

    # zz-(omega1,omegac)

    # qlow = 3.9
    # qhigh = 5.1
    # clow = 5
    # chigh = 7.5
    # step = 50
    # omega1List = np.arange(qlow, qhigh, (qhigh - qlow) / step)[:step]
    # omegacList = np.arange(clow, chigh, (chigh - clow) / step)[:step]

    # omega0s = [omega0] * len(omegacList) * len(omegacList)
    # g_qcs = [g_qc] * len(omegacList) * len(omegacList)
    # bitNums = [bitNum] * len(omegacList) * len(omegacList)
    # energyLevels = [energyLevel] * len(omegacList) * len(omegacList)
    # anharms = [anharm] * len(omegacList) * len(omegacList)
    # g_qqs = [g_qq] * len(omegacList) * len(omegacList)

    # omegaqs = []
    # omegacs = []
    # for w1 in omega1List:
    #     for wc in omegacList:
    #         omegaqs.append([omega0, w1])
    #         omegacs.append([wc])

    # tStart = time.time()
    # print(cpu_count())
    # p = Pool(cpu_count())
    # data_list = p.starmap(zzcoupling, zip(g_qcs, g_qqs, energyLevels, bitNums, omegaqs, omegacs, anharms))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)


    # zeta_list = []
    # g_list = []
    # g12_list = []
    # g1c_list = []
    # g2c_list = []
    # for i in data_list:
    #     zeta_list.append(i[0])
    #     g_list.append(i[1])
    #     g12_list.append(i[2])
    #     g1c_list.append(i[3])
    #     g2c_list.append(i[4])
    # zeta_list = np.array([zeta_list]).reshape(len(omega1List), len(omegacList)) * 1e3
    # draw_heat_map(omega1List, omegacList, zeta_list, 'zz', 'zz', 'omega1(GHz)', 'omegac(GHz)', 'logabs', threshold=-1.1)# 1.1 0.6

    # zz-(gqc,gqq)

    # omega1 = 4.1

    # step = 50

    # rho0cLow, rho0cHigh = 1e-4, 0.08
    # rhoqcList = np.arange(rho0cLow, rho0cHigh, (rho0cHigh - rho0cLow) / step)[:step]

    # rho01Low, rho01High = 1e-4, 0.015
    # rhoqqList = np.arange(rho01Low, rho01High, (rho01High - rho01Low) / step)[:step]

    # clow = 5
    # chigh = 7
    # cstep = 50
    # omegacList = np.arange(clow, chigh, (chigh - clow) / cstep)[:cstep]

    # omegaqs = [[omega0, omega1]] * cstep * step * step
    # bitNums = [bitNum] * cstep * step * step
    # energyLevels = [energyLevel] * cstep * step * step
    # anharms = [anharm] * cstep * step * step

    # rho01s = []
    # rho0cs = []
    # omegacs = []
    # for rho_qc in rhoqcList:
    #     for rho_qq in rhoqqList:
    #         for omegac in omegacList:
    #             rho0cs.append({'q0c0' : rho_qc, 'q1c0' : rho_qc, 'q1c1' : rho_qc, 'q2c1' : rho_qc})
    #             rho01s.append({'q0q1' : rho_qq, 'q1q2' : rho_qq})
    #             omegacs.append([omegac])

    # tStart = time.time()
    # print(cpu_count())
    # p = Pool(cpu_count())
    # data_list = p.starmap(zzcoupling, zip(rho0cs, rho01s, energyLevels, bitNums, omegaqs, omegacs, anharms))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # zeta_list = []
    # g_list = []
    # g12_list = []
    # g1c_list = []
    # g2c_list = []
    # for i in data_list:
    #     zeta_list.append(i[0])
    #     g_list.append(i[1])
    #     g12_list.append(i[2])
    #     g1c_list.append(i[3])
    #     g2c_list.append(i[4])
    # zeta_list = np.abs(np.array([zeta_list]).reshape((step, step, cstep)) * 1e3)

    # zz_list = np.zeros((step, step))
    # zz_list_write = []
    # for i in range(step):
    #     for j in range(step):
    #         zz_list[i, j] = np.min(zeta_list[i, j, :])
    #         zz_list_write.append(zz_list[i, j])

    # write_data('zz rho.txt', zz_list_write)

    # fidelity-(omega1,omegac)

    # tg = 30
    # qlow, qhigh = 3.9, 5.1
    # clow, chigh = 5, 7
    # step = 50
    # omega1List = np.arange(qlow, qhigh, (qhigh - qlow) / step)[int(nn * step / parallelNum) : int((nn + 1) * step / parallelNum)]
    # # omega1List = np.arange(qlow, qhigh, (qhigh - qlow) / step)
    # omegacList = np.arange(clow, chigh, (chigh - clow) / step)
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
    #         omegaqs.append(np.array([omega0, omega1]))
    #         omegacs.append(np.array([omegac]))

    # tStart = time.time()
    # print(cpu_count())
    # p = Pool(cpu_count())
    # err_list = p.starmap(par_X, zip(g_qcs, g_qqs, energyLevels, bitNums, 
    #             omegaqs, omegacs, gatetimes, dragWeights, anharms))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)
    # err0_list = []
    # err1_list = []
    # errxx_list = []
    # for i in err_list:
    #     err0_list.append(i[0])
    #     err1_list.append(i[1])
    #     errxx_list.append(i[2])
    # write_data('errori0 ' + str(anharm['q0'])  + str(anharm['c0']) + ' ' + str(anharm['q1']) + ' ' + str(nn) + '.txt', err0_list)
    # write_data('errori1 ' + str(anharm['q0'])  + str(anharm['c0']) + ' ' + str(anharm['q1']) + ' ' + str(nn) + '.txt', err1_list)
    # write_data('errorxx ' + str(anharm['q0'])  + str(anharm['c0']) + ' ' + str(anharm['q1']) + ' ' + str(nn) + '.txt', errxx_list)

    # 两比特门

    # omega0, omega1 = 4.5, 4.1
    # omegac = 6.5
    # bitNum = 3
    # energyLevel = 3
    # tg = 60

    # path = os.getcwd()
    # datanames = os.listdir(path)
    # for i in datanames:
    #     if '.txt' in i and 'zz rho' in i:
    #         fileName = i
    #         break

    # step = 50
    # zz_list = read_data(fileName)
    # zz_list = np.array(zz_list).reshape((step, step))

    # rho0cLow, rho0cHigh = 1e-4, 0.08
    # rhoqcList = np.arange(rho0cLow, rho0cHigh, (rho0cHigh - rho0cLow) / step)[:step]

    # rho01Low, rho01High = 1e-4, 0.015
    # rhoqqList = np.arange(rho01Low, rho01High, (rho01High - rho01Low) / step)[:step]

    # clow = 5
    # chigh = 6.5
    # cstep = 50
    # cList = np.arange(clow, chigh, (chigh - clow) / cstep)[:cstep]

    # qlow = omega1 - anharm['q0'] - 0.04
    # qhigh = omega1 - anharm['q0'] + 0.04
    # qstep = 50
    # qList = np.arange(qlow, qhigh, (qhigh - qlow) / qstep)[:qstep]

    # cz_err = []
    # cz_sta = []
    # cz_amp = []

    # tgs = [tg] * qstep * cstep
    # freqqMaxs = [[omega0, omega1]] * qstep * cstep
    # freqcMaxs = [[omegac]] * qstep * cstep
    # bitNums = [bitNum] * qstep * cstep
    # energyLevels = [energyLevel] * qstep * cstep
    # anharms = [anharm] * qstep * cstep

    # for qc in range(step)[int(nn * step / parallelNum) : int((nn + 1) * step / parallelNum)]:
    #     for qq in range(step):
    #         print(rhoqcList[qc], rhoqqList[qq])
    #         if zz_list[qc, qq] > 1e-1:
    #             cz_err.append(1)
    #             cz_sta.append(0)
    #             cz_amp.append(1)
    #         else:
    #             rho_qc = rhoqcList[qc]
    #             rho_qq = rhoqqList[qq]
    #             rho0cs = {'q0c0' : rho_qc, 'q1c0' : rho_qc, 'q1c1' : rho_qc, 'q2c1' : rho_qc}
    #             rho01s = {'q0q1' : rho_qq, 'q1q2' : rho_qq}
    #             g_qcs = [rho0cs] * qstep * cstep
    #             g_qqs = [rho01s] * qstep * cstep
    #             lambdas = []
    #             for q in qList:
    #                 for c in cList:
    #                     lambdas.append([q, c, 0.8, 0.8])
    #             tStart = time.time()
    #             print(cpu_count())
    #             p = Pool(cpu_count())
    #             err_list = p.starmap(evolution_CZ, 
    #             zip(lambdas, tgs, g_qcs, g_qqs, freqqMaxs, freqcMaxs, bitNums, energyLevels, anharms))
    #             p.close()
    #             p.join()
    #             t = time.time() - tStart
    #             print(t)
    #             err_list = np.array(err_list).reshape(qstep, cstep)
    #             cz_err.append(np.min(err_list))
    #             amp = 6.5
    #             cz_sta.append(0)
    #             for i in range(qstep):
    #                 for j in range(cstep):
    #                     if err_list[i, j] < cz_err[-1] + 1e-2:
    #                         cz_sta[-1] += 1
    #                         if 6.5 - cList[j] < amp:
    #                             amp = 6.5 - cList[j]
    #             cz_sta[-1] /= qstep * cstep
    #             cz_amp.append(amp)
    #             print(cz_err[-1], cz_sta[-1], cz_amp[-1])
    # write_data('czerr rho' + str(nn) + '.txt', cz_err)
    # write_data('czsta rho' + str(nn) + '.txt', cz_sta)
    # write_data('czamp rho' + str(nn) + '.txt', cz_amp)

    # 选定耦合系数后，两比特门保真度

    omega0, omega1 = 4.5, 4.1
    omegac = 7
    gateTime = 60
    bitNum = 3
    energyLevel = 5

    # 优化

    # error = par_CZ_no_ad(gateTime, g_qc, g_qq, [omega0, omega1], [omegac], bitNum, energyLevel, anharm)

    # 扫描搜索

    qlow = omega1 - anharm['q0'] - 0.04
    qhigh = omega1 - anharm['q0'] + 0.04
    qstep = 50
    qList = np.arange(qlow, qhigh, (qhigh - qlow) / qstep)[:qstep][int(nn * qstep / parallelNum) : int((nn + 1) * qstep / parallelNum)]

    clow = 5
    chigh = 7
    cstep = 50
    cList = np.arange(clow, chigh, (chigh - clow) / cstep)[:cstep]

    tgs = [gateTime] * qstep * cstep
    freqqMaxs = [[omega0, omega1]] * qstep * cstep
    freqcMaxs = [[omegac]] * qstep * cstep
    bitNums = [bitNum] * qstep * cstep
    energyLevels = [energyLevel] * qstep * cstep
    anharms = [anharm] * qstep * cstep
    g_qcs = [g_qc] * qstep * cstep
    g_qqs = [g_qq] * qstep * cstep

    lambdas = []
    for q in qList:
        for c in cList:
            lambdas.append([q, c, 0.8, 0.8])
    tStart = time.time()
    print(cpu_count())
    p = Pool(cpu_count())
    err_list = p.starmap(evolution_CZ, 
    zip(lambdas, tgs, g_qcs, g_qqs, freqqMaxs, freqcMaxs, bitNums, energyLevels, anharms))
    p.close()
    p.join()
    t = time.time() - tStart
    print(t)

    write_data('cz_err' + str(nn) + '.txt', err_list)
