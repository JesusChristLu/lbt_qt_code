# import
import time
import cmath
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.qip.operations import rz
from qutip.qip.operations import cphase
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from scipy.interpolate import interp1d
from scipy.special import erf
# from pyswarm import pso
from multiprocessing import Pool, cpu_count
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

def generate_chip(w, h, omegaH):
    chip = nx.grid_2d_graph(w, h)
    for qubit in chip:
        if (qubit[0] + qubit[1]) % 2:
            chip.nodes[qubit]['type'] = 1
        else:
            chip.nodes[qubit]['type'] = 0
    xtalkG = nx.Graph()
    for qubit in chip:
        if not chip.nodes[qubit]['type']:
            for i in chip[qubit]:
                for j in chip[qubit]:
                    if not i == j:
                        xtalkG.add_edge(i, j)
        else:
            for i in chip[qubit]:
                if not chip.nodes[i].get('freq', False):
                    chip.nodes[i]['freq'] = 0.05 * (np.random.random() - 0.5) + omegaH
                    while 1:
                        crowd = False
                        for j in chip[qubit]:
                            if i == j:
                                continue
                            if chip.nodes[j].get('freq', False):
                                if np.abs(chip.nodes[i]['freq'] - chip.nodes[j]['freq']) < 0.01:
                                    crowd = True
                                    chip.nodes[i]['freq'] = 0.05 * (np.random.random() - 0.5) + omegaH
                        if not crowd:
                            break
    return xtalkG, chip


def gen_pos(chip):
    wStep = 1
    hStep = 1
    pos = dict()
    for qubit in chip:
        pos[qubit] = [qubit[0] * wStep, qubit[1] * hStep]
    return pos

def getaad(energyLevel, bitNum):
    a, aDag = destroy(energyLevel), create(energyLevel)
    I = qeye(energyLevel)
    IenergyLevel = tensorOperator(energyLevel, I, 0, bitNum)
    aList, aDagList = [], []
    sxList, syList, szList = [], [], []
    for b in range(bitNum):
        aq = tensorOperator(energyLevel, a, b, bitNum)
        aqDag = tensorOperator(energyLevel, aDag, b, bitNum)
        aList.append(aq)
        aDagList.append(aqDag)
        sxList.append(aq + aqDag)
        syList.append(1j * (aqDag - aq))
        szList.append(IenergyLevel - 2 * aqDag * aq)
    return aList, aDagList, sxList, syList, szList

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

def H(inducedChip, energyLevel, anharmq, anharmc, g_qq, g_qc):
    qNum = len(inducedChip.nodes)
    cNum = len(inducedChip.edges)
    qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    aList, aDagList, _, _, _ = getaad(energyLevel, qNum + cNum)
    noH = True
    for q in range(qNum):
        if noH:
            H0 = 2 * np.pi * inducedChip.nodes[qubitList[q]]['freq'] * aDagList[q] * aList[q] + np.pi * anharmq* aDagList[q] * aDagList[q] * aList[q] * aList[q]
            noH = False
        else:
            H0 += 2 * np.pi * inducedChip.nodes[qubitList[q]]['freq'] * aDagList[q] * aList[q] + np.pi * anharmq * aDagList[q] * aDagList[q] * aList[q] * aList[q]
        
    for c in range(qNum, qNum + cNum):
        H0 += 2 * np.pi * inducedChip.edges[qubitList[c]]['freq'] * aDagList[c] * aList[c] + np.pi * anharmc * aDagList[c] * aDagList[c] * aList[c] * aList[c]
        
    if cNum == 0:
        return H0

    noH = True
    for edge in inducedChip.edges:
        q0 = qubitList.index(edge[0])
        q1 = qubitList.index(edge[1])
        c = qubitList.index(edge)
        if noH:
            Hi = 2 * np.pi * g_qc * np.sqrt(inducedChip.nodes[qubitList[q0]]['freq'] * inducedChip.edges[qubitList[c]]['freq']) * (aDagList[q0] * aList[c] + aList[q0] * aDagList[c])
            Hi += 2 * np.pi * g_qc * np.sqrt(inducedChip.nodes[qubitList[q1]]['freq'] * inducedChip.edges[qubitList[c]]['freq']) * (aDagList[q1] * aList[c] + aList[q1] * aDagList[c])
            Hi += 2 * np.pi * g_qq * np.sqrt(inducedChip.nodes[qubitList[q0]]['freq'] *  inducedChip.nodes[qubitList[q1]]['freq']) * \
            (aDagList[q0] * aList[q1] + aList[q0] * aDagList[q1])
            noH = False
        else:
            Hi += 2 * np.pi * g_qc * np.sqrt(inducedChip.nodes[qubitList[q0]]['freq'] * inducedChip.edges[qubitList[c]]['freq']) * (aDagList[q0] * aList[c] + aList[q0] * aDagList[c])
            Hi += 2 * np.pi * g_qc * np.sqrt(inducedChip.nodes[qubitList[q1]]['freq'] * inducedChip.edges[qubitList[c]]['freq']) * (aDagList[q1] * aList[c] + aList[q1] * aDagList[c])
            Hi += 2 * np.pi * g_qq * np.sqrt(inducedChip.nodes[qubitList[q0]]['freq'] *  inducedChip.nodes[qubitList[q1]]['freq']) * \
            (aDagList[q0] * aList[q1] + aList[q0] * aDagList[q1])
    return H0, Hi

def H_two(tList, args):
    inducedChip = args['induced chip']
    subStatesId = args['sub states id']
    qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    targetCoupler = qubitList.index(args['target edge'])
    g_qc = args['g qc']
    g_qq = args['g qq']
    energyLevel = args['energy level']
    anharmq, anharmc = args['anharm'][0], args['anharm'][1]
    freqcWork = args['lambdas'][0]
    sigma = args['lambdas'][1:]
    qNum = len(inducedChip.nodes)
    cNum = len(inducedChip.edges)

    aList, aDagList, _, _, _ = getaad(energyLevel, qNum + cNum)

    HAnharm = []
    H0 = []
    Hi = []
    for q in range(qNum):
        mat = find_sub_matrix(aDagList[q] * aDagList[q] * aList[q] * aList[q], subStatesId)
        HAnharm.append(np.pi * anharmq * mat)
        mat = find_sub_matrix(aDagList[q] * aList[q], subStatesId)
        H0.append(np.pi * 2 * inducedChip.nodes[qubitList[q]]['freq'] * mat)
    for c in range(qNum, qNum + cNum):
        mat = find_sub_matrix(aDagList[c] * aDagList[c] * aList[c] * aList[c], subStatesId)
        HAnharm.append(np.pi * anharmc * mat)
        if not(c == targetCoupler):
            mat = find_sub_matrix(aDagList[c] * aList[c], subStatesId)
            H0.append(np.pi * 2 * inducedChip.edges[qubitList[c]]['freq'] * mat)
        else:
            freqcList = pulse_fun(tList, tList[-1], sigma, freqcWork, inducedChip.edges[qubitList[c]]['freq'])
            mat = find_sub_matrix(aDagList[c] * aList[c], subStatesId)
            H0.append([mat, freqcList * np.pi * 2])

    for edge in inducedChip.edges:
        q0 = qubitList.index(edge[0])
        q1 = qubitList.index(edge[1])
        c = qubitList.index(edge)
        mat0c = find_sub_matrix((aDagList[q0] * aList[c] + aList[q0] *aDagList[c]), subStatesId)
        mat1c = find_sub_matrix((aDagList[q1] * aList[c] + aList[q1] *aDagList[c]), subStatesId)
        mat01 = find_sub_matrix((aDagList[q0] * aList[q1] + aList[q0] *aDagList[q1]), subStatesId)
        if c == targetCoupler:
            freqcList = pulse_fun(tList, tList[-1], sigma, freqcWork, inducedChip.edges[qubitList[c]]['freq'])
        else:
            freqcList = inducedChip.edges[qubitList[c]]['freq']
        freq0List = inducedChip.nodes[qubitList[q0]]['freq']
        freq1List = inducedChip.nodes[qubitList[q1]]['freq']
        g_q0c = g_qc * np.sqrt(freq0List * freqcList)
        g_q1c = g_qc * np.sqrt(freq1List * freqcList)
        g_q0q1 = g_qq * np.sqrt(freq0List * freq1List)

        if isinstance(g_q0c, float):
            Hi.append(np.pi * 2 * g_q0c * mat0c)
        else:
            Hi.append([mat0c, np.pi * 2 * g_q0c])
        if isinstance(g_q1c, float):
            Hi.append(np.pi * 2 * g_q1c * mat1c)
        else:
            Hi.append([mat1c, np.pi * 2 * g_q1c])
        Hi.append(np.pi * 2 * g_q0q1 * mat01)
    return [*H0, *HAnharm, *Hi]

def pulse_fun(tList, pulseLen, sigma, freqWork, freqMax):
    freqList = freqMax + (freqWork - freqMax) * (1 - np.sum(sigma)) * (1 - np.cos(tList / pulseLen * (len(sigma) + 1) * 2 * np.pi)) / 2
    w = 1
    for i in sigma:
        freqList += (freqWork - freqMax) * i * (1 - np.cos(tList / pulseLen * w * 2 * np.pi)) / 2
        w += 1
    
    return freqList

def find_sub_space(states, energyLevel, bitNum):
    subStates = []
    subStatesId = dict()
    id = 0
    for state in states:
        for bas in state_number_enumerate([energyLevel] * bitNum, energyLevel * bitNum):
            if sum(bas) == sum(state) and not bas in subStates:
                subStates.append(bas)
                subStatesId[state_number_index([energyLevel] * bitNum, bas)] = id
                id += 1
    return subStatesId

def find_sub_matrix(mat, subStateId):
    npmat = np.zeros((len(subStateId), len(subStateId)), dtype=complex)
    for i in subStateId:
        for j in subStateId:
            npmat[subStateId[i], subStateId[j]] = mat[i, j]
    return Qobj(npmat)

def eigensolve(H0, H):

    H0diag = H0.diag()

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
    states0_order = ei_vector0.copy()
    energy_order = ei_energy.copy()
    energy0_order = ei_energy0.copy()
    assignedIndex = []
    for n, energy in enumerate(ei_energy0):
        indexs = [i for i in range(len(H0diag)) if H0diag[i] == energy]
        for index in indexs:
            if not (index in assignedIndex):
                states0_order[index] = ei_vector0[n]
                energy0_order[index] = ei_energy0[n]            
                assignedIndex.append(index)
                break         
    # assignedIndex = []
    for n, energy in enumerate(ei_energy0):
        maxindex = 0
        maxProd = np.abs((ei_vector[maxindex].dag() * states0_order[n]).full()[0][0])
        for index in range(1, len(energy0_order)):
            # if index in assignedIndex:
                # continue
            prod = np.abs((ei_vector[index].dag() * states0_order[n]).full()[0][0])
            if prod > maxProd:
                maxProd = prod
                maxindex = index
        states_order[n] = ei_vector[maxindex]
        energy_order[n] = ei_energy[maxindex]
        # assignedIndex.append(maxindex)
    return states_order, states0_order, energy_order / (np.pi * 2), energy0_order / (np.pi * 2)
    
def drive_pulseXt(t, args):  
    tg = args['gate time']
    amp = args['qt amp']
    w_d = args['qt drive frequency']
    alpha = args['qt anharm']
    detune = args['qt detune']
    lambda0 = args['drag weight']
    phi = args['qt phi']
    I = np.cos(phi)
    Q = np.sin(phi)
    X0 = amp * (1 - np.cos(2 * np.pi * t / tg)) / 2 
    Y0 = -amp * np.pi / tg * np.sin(2 * np.pi * t / tg) / alpha * lambda0
    X = (X0 * I + Y0 * Q) * np.cos((w_d + detune) * t) 
    return X

def drive_pulseYt(t, args):  
    tg = args['gate time']
    amp = args['qt amp']
    w_d = args['qt drive frequency']
    alpha = args['qt anharm']
    detune = args['qt detune']
    lambda0 = args['drag weight']
    phi = args['qt phi']
    I = np.cos(phi)
    Q = np.sin(phi)
    X0 = amp * (1 - np.cos(2 * np.pi * t / tg)) / 2 
    Y0 = -amp * np.pi / tg * np.sin(2 * np.pi * t / tg) / alpha * lambda0
    X = (Y0 * I - X0 * Q) * np.cos((w_d + detune) * t) 
    return X

def xyxtalk(t, args):
    tg = args['gate time']
    amp = args['qt amp']
    w_d = args['qt drive frequency']
    mu = args['mu']
    X0 = mu * amp * (1 - np.cos(2 * np.pi * t / tg)) / 2 
    X = X0 * np.cos(w_d * t) 
    return X

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
    f = -((np.trace(np.dot(U.T.conjugate(), U)) + np.abs(np.trace(np.dot(U.T.conjugate(), UIdeal))) ** 2) / (d * (d + 1)))
    return np.real(f)

def evolutionX(pulse_paras, pulse_const, H, states, subStatesId, energyLevel, anharm, tq, neighborzz, neighborxy=None):
    bitNum = len(neighborzz)
    _, _, sxList, syList, _ = getaad(energyLevel, bitNum)

    npsx = np.zeros((len(subStatesId), len(subStatesId)), dtype=complex)
    npsy = np.zeros((len(subStatesId), len(subStatesId)), dtype=complex)

    npsx = find_sub_matrix(sxList[tq], subStatesId)
    npsy = find_sub_matrix(syList[tq], subStatesId)

    H_xt = [npsx, drive_pulseXt]
    H_yt = [npsy, drive_pulseYt]
    Ht = [H, H_xt, H_yt]

    args = dict()
    
    args['gate time'] = pulse_const[0]
    args['qt drive frequency'] = pulse_const[1] * np.pi * 2
    args['drag weight'] = pulse_const[2]
    args['qt anharm'] = anharm
    args['qt detune'] = pulse_paras[0] * np.pi * 2
    args['qt amp'] = pulse_paras[1]
    args['qt phi'] = -pulse_paras[0] * args['gate time'] * np.pi 

    tList = np.arange(0, args['gate time'], args['gate time'] / 100)

    U_full=propagator(Ht, tList, args=args)[-1]
    U=np.zeros([2, 2], dtype='complex128')
    for i in range(2):
        for j in range(2):
            idbra = deepcopy(neighborzz)
            idket = deepcopy(neighborzz)
            idbra[tq] = i
            idket[tq] = j
            idbra = subStatesId[state_number_index([energyLevel] * bitNum, idbra[::-1])]
            idket = subStatesId[state_number_index([energyLevel] * bitNum, idket[::-1])]
            U[i][j] = (states[idbra].dag() * U_full * states[idket]).full()[0][0]
    
    F = Fidelity_X(U)
    error = 1 - F
    return error

def evolutionCZ(lambdas, states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, neighborzz, neighborxy=None, last=False):
    
    bitNum = len(neighborzz)
    tg = 60
    tList = np.arange(0, tg, tg / (tg * 2))

    args = dict() 
    args['induced chip'] = inducedChip
    args['sub states id'] = subStatesId
    args['target edge'] = tq
    args['g qc'] = g_qc
    args['g qq'] = g_qq
    args['energy level'] = energyLevel
    args['anharm'] = [anharmq, anharmc]
    args['lambdas'] = lambdas
    HTwo = H_two(tList, args=args)

    U_full = propagator(HTwo, tList)[-1]
    qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    q1 = min([qubitList.index(tq[0]),qubitList.index(tq[1])])
    q2 = max([qubitList.index(tq[0]),qubitList.index(tq[1])])
    U = np.zeros([4, 4], dtype=complex)
    for i in range(2):
        for j in range(2):
            for m in range(2):
                for n in range(2):
                    ii = i * 2 + j
                    jj = m * 2 + n
                    idbra = deepcopy(neighborzz)
                    idket = deepcopy(neighborzz)
                    idbra[q1], idbra[q2] = j, i
                    idket[q1], idket[q2] = n, m
                    idbra = subStatesId[state_number_index([energyLevel] * bitNum, idbra[::-1])]
                    idket = subStatesId[state_number_index([energyLevel] * bitNum, idket[::-1])]
                    U[ii][jj] = (states[idbra].dag() * U_full * states[idket]).full()[0][0]    
    F, U = Fidelity_CZ(U)
    error = 1 - F
    if last:
        print(lambdas, error, cmath.phase(U[3, 3] / U[2, 2]), 1 - 0.25 * (np.abs(U[0, 0]) + np.abs(U[1, 1]) + np.abs(U[2, 2]) + np.abs(U[3, 3])))
        # print(np.abs(U[0, 0]), np.abs(U[1, 1]), np.abs(U[2, 2]), np.abs(U[3, 3]))
        # print(U)
    return error

def calibration_single_qubit_gate(inducedChip, tq, energyLevel, anharmq, anharmc, g_qq, g_qc):
    calibratedQ = tq
    qubitNum = len(inducedChip.nodes) + len(inducedChip.edges)
    pulseConst = [30, inducedChip.nodes[tq]['freq'], 0.5]
    tq = list(inducedChip.nodes).index(tq)
    nbState = list(np.zeros(qubitNum, dtype=int))
    s1 = deepcopy(nbState)
    s0 = deepcopy(nbState)
    s1[tq] = 1
    s1 = s1[::-1]

    H0, Hi = H(inducedChip, energyLevel, anharmq, anharmc, g_qq, g_qc)

    subStatesId = find_sub_space([s0, s1], energyLevel, qubitNum)
    H0 = find_sub_matrix(H0, subStatesId)
    Hi = find_sub_matrix(Hi, subStatesId)
    states, _, energys, energys0 = eigensolve(H0, H0 + Hi)

    s1Id = subStatesId[state_number_index([energyLevel] * qubitNum, s1)]
    s0Id = subStatesId[state_number_index([energyLevel] * qubitNum, s0)]
    detuneIni = (energys[s1Id] - energys[s0Id] - energys0[s1Id] + energys0[s0Id])
    xIni = [detuneIni, 0.2]
    result = minimize(evolutionX, xIni, args=(pulseConst, H0 + Hi, states, subStatesId, energyLevel, anharmq, tq, s0), 
                        method='Nelder-Mead', options={'maxiter' : 100})
    errdict = dict()
    if result.fun < 1e-4:
        errdict[0] = 0
    else:
        errdict[0] = result.fun
    for bas in state_number_enumerate([2] * (len(inducedChip.nodes) - 1), len(inducedChip.nodes) - 1):
        # if 1:
        if not sum(bas) == 0:
            id = 0
            zzbas = list(np.zeros(qubitNum, dtype=int))
            for i in range(len(inducedChip.nodes)):
                if not i == tq:
                    zzbas[i] = bas[id]
                    id += 1
            errdict[str(bas[::-1])] = par_single_qubit_gate(inducedChip, calibratedQ, energyLevel, anharmq, anharmc, g_qq, g_qc, result.x, zzbas)
    print('calibtration of gate', calibratedQ, errdict)
    return calibratedQ, result.x[0], result.x[1], errdict

def par_single_qubit_gate(inducedChip, tq, energyLevel, anharmq, anharmc, g_qq, g_qc, lambdas, nbState, xyxtalk=None):
    Q = tq
    qubitNum = len(inducedChip.nodes) + len(inducedChip.edges)
    pulseConst = [30, inducedChip.nodes[tq]['freq'], 0.5]
    tq = list(inducedChip.nodes).index(tq)
    s1 = deepcopy(nbState)
    s0 = deepcopy(nbState)
    s1[tq] = 1
    s0 = s0[::-1]
    s1 = s1[::-1]
    H0, Hi = H(inducedChip, energyLevel, anharmq, anharmc, g_qq, g_qc)
    subStatesId = find_sub_space([s0, s1], energyLevel, qubitNum)
    H0 = find_sub_matrix(H0, subStatesId)
    Hi = find_sub_matrix(Hi, subStatesId)
    states, _, energys, energys0 = eigensolve(H0, H0 + Hi)

    s1Id = subStatesId[state_number_index([energyLevel] * qubitNum, s1)]
    s0Id = subStatesId[state_number_index([energyLevel] * qubitNum, s0)]
    err = evolutionX(lambdas, pulseConst, H0 + Hi, states, subStatesId, energyLevel, anharmq, tq, nbState, xyxtalk)
    if err < 1e-4:
        err = 0
    print(Q, (energys[s1Id] - energys[s0Id] - energys0[s1Id] + energys0[s0Id]), nbState[::-1], err)
    return err

def calibration_two_qubit_gate(inducedChip, tq, energyLevel, anharmq, anharmc, g_qq, g_qc):
    calibratedG = tq
    qubitNum = len(inducedChip.nodes) + len(inducedChip.edges)
    qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    g1 = qubitList.index(tq[0])
    g2 = qubitList.index(tq[1])
    nbState = list(np.zeros(qubitNum, dtype=int))
    s00 = deepcopy(nbState)
    s01 = deepcopy(nbState)
    s10 = deepcopy(nbState)
    s11 = deepcopy(nbState)
    s01[g1] = 1
    s10[g2] = 1
    s11[g1], s11[g2] = 1, 1
    freq0, freq1 = inducedChip.nodes[tq[0]]['freq'], inducedChip.nodes[tq[1]]['freq']

    s00 = s00[::-1]
    s01 = s01[::-1]
    s10 = s10[::-1]
    s11 = s11[::-1]

    H0, Hi = H(inducedChip, energyLevel, anharmq, anharmc, g_qq, g_qc)
    subStatesId = find_sub_space([s00, s01, s10, s11], energyLevel, qubitNum)
    
    H0 = find_sub_matrix(H0, subStatesId)
    Hi = find_sub_matrix(Hi, subStatesId)
    
    states, _, _, _ = eigensolve(H0, H0 + Hi)

    # freqWork = 4.6
    # bounds = (4.6, 5)

    freqWork = 4.3
    bounds = (4, 4.3)

    # lambdas = [freqWork, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # result = minimize(evolutionCZ, lambdas, args=(states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, nbState), 
    # method='Nelder-Mead', bounds=(bounds, (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4)), options={'ftol' : 1e-4})
    # lambdas = [freqWork, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # result = minimize(evolutionCZ, lambdas, args=(states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, nbState), 
    # method='Nelder-Mead', bounds=(bounds, (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4)), options={'ftol' : 1e-4})
    # lambdas = [freqWork, 1, 0, 0, 0, 0, 0, 0, 0]
    # result = minimize(evolutionCZ, lambdas, args=(states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, nbState), 
    # method='Nelder-Mead', bounds=(bounds, (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4)), options={'ftol' : 1e-4})
    # lambdas = [freqWork, 1, 0, 0, 0, 0, 0, 0]
    # result = minimize(evolutionCZ, lambdas, args=(states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, nbState), 
    # method='Nelder-Mead', bounds=(bounds, (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4)), options={'ftol' : 1e-4})
    # lambdas = [freqWork, 1, 0, 0, 0, 0, 0]
    # result = minimize(evolutionCZ, lambdas, args=(states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, nbState), 
    # method='Nelder-Mead', bounds=(bounds, (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4)), options={'ftol' : 1e-4})
    # lambdas = [freqWork, 1, 0, 0, 0, 0]
    # result = minimize(evolutionCZ, lambdas, args=(states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, nbState), 
    # method='Nelder-Mead', bounds=(bounds, (-2, 4), (-2, 4), (-2, 4), (-2, 4), (-2, 4)), options={'ftol' : 1e-4})
    # lambdas = [freqWork, 1, 0, 0, 0]
    # result = minimize(evolutionCZ, lambdas, args=(states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, nbState), 
    # method='Nelder-Mead', bounds=(bounds,(-2, 4), (-2, 4), (-2, 4), (-2, 4)), options={'ftol' : 1e-4})
    lambdas = [freqWork, 1, 0, 0]
    result = minimize(evolutionCZ, lambdas, args=(states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, nbState), 
    method='Nelder-Mead', bounds=(bounds, (-2, 4), (-2, 4), (-2, 4)), options={'ftol' : 1e-4})
    # lambdas = [freqWork, 1, 0]
    # result = minimize(evolutionCZ, lambdas, args=(states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, nbState), 
    # method='Nelder-Mead', bounds=(bounds, (-2, 4), (-2, 4)), options={'ftol' : 1e-4})
    # lambdas = [freqWork, 1]
    # result = minimize(evolutionCZ, lambdas, args=(states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, nbState), 
    # method='Nelder-Mead', bounds=(bounds, (-2, 4)), options={'ftol' : 1e-4})

    errdict = dict()
    # if result.fun < 1e-4:
        # errdict[0] = 0
    # else:
        # errdict[0] = result.fun
    errdict[0] = result.fun
    if len(inducedChip.nodes) - 2 > 0:
        for bas in state_number_enumerate([2] * (len(inducedChip.nodes) - 2), 1 * (len(inducedChip.nodes) - 2)):
            if not sum(bas) == 0:
                id = 0
                zzbas = list(np.zeros(qubitNum, dtype=int))
                for i in range(len(inducedChip.nodes)):
                    if not(i in [g1, g2]):
                        zzbas[i] = bas[id]
                        id += 1
                errdict[str(bas[::-1])] = par_two_qubit_gate(inducedChip, tq, energyLevel, anharmq, anharmc, g_qq, g_qc, result.x, zzbas)
    print('calibtration of gate', calibratedG, errdict)
    return calibratedG, list(result.x), errdict

def par_two_qubit_gate(inducedChip, tq, energyLevel, anharmq, anharmc, g_qq, g_qc, lambdas, nbState, xyxtalk=None):
    G = tq
    qubitNum = len(inducedChip.nodes) + len(inducedChip.edges)
    qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    g1 = qubitList.index(tq[0])
    g2 = qubitList.index(tq[1])
    s00 = deepcopy(nbState)
    s01 = deepcopy(nbState)
    s10 = deepcopy(nbState)
    s11 = deepcopy(nbState)
    s01[g1] = 1
    s10[g2] = 1
    s11[g1], s11[g2] = 1, 1
    s00 = s00[::-1]
    s01 = s01[::-1]
    s10 = s10[::-1]
    s11 = s11[::-1]

    H0, Hi = H(inducedChip, energyLevel, anharmq, anharmc, g_qq, g_qc)
    subStatesId = find_sub_space([s00, s01, s10, s11], energyLevel, qubitNum)
    
    H0 = find_sub_matrix(H0, subStatesId)
    Hi = find_sub_matrix(Hi, subStatesId)
    
    states, _, _, _ = eigensolve(H0, H0 + Hi)

    err = evolutionCZ(lambdas, states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, g_qq, g_qc, nbState, xyxtalk, last=True)
    # if err < 1e-4:
    #     err = 0
    print(G, nbState[::-1], err)
    return err

def zzcoupling(inducedChip, tq, rho_qq, rho_qc, energyLevel, anharmq, anharmc):
    qubitNum = len(inducedChip.nodes) + len(inducedChip.edges)
    qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    g = qubitList.index(tq)
    s00 = np.zeros(qubitNum, dtype=int)
    s01 = np.zeros(qubitNum, dtype=int)
    s10 = np.zeros(qubitNum, dtype=int)
    s11 = np.zeros(qubitNum, dtype=int)
    s01[g] = 1
    s11[g] = 1

    s10[qubitList.index(list(dict(inducedChip[tq]).keys())[0])] = 1
    s11[qubitList.index(list(dict(inducedChip[tq]).keys())[0])] = 1

    s01 = s01[::-1]
    s10 = s10[::-1]
    s11 = s11[::-1]
    H0, Hi = H(inducedChip, energyLevel, anharmq, anharmc, rho_qq, rho_qc)
    subStatesId = find_sub_space([s00, s01, s10, s11], energyLevel, qubitNum)
    
    H0 = find_sub_matrix(H0, subStatesId)
    Hi = find_sub_matrix(Hi, subStatesId)
    
    _, _, energys, _ = eigensolve(H0, H0 + Hi)

    shift = np.abs((energys[subStatesId[state_number_index([energyLevel] * qubitNum, s11)]] - \
                energys[subStatesId[state_number_index([energyLevel] * qubitNum, s10)]]) - \
                (energys[subStatesId[state_number_index([energyLevel] * qubitNum, s01)]] - \
                energys[subStatesId[state_number_index([energyLevel] * qubitNum, s00)]]))
    
    # omega1 = inducedChip.nodes[qubitList[0]]['freq']
    # omega2 = inducedChip.nodes[qubitList[1]]['freq']
    # omegac = inducedChip.edges[qubitList[2]]['freq']
    # g12 = rho_qq * np.sqrt(omega1 * omega2)
    # gic = (rho_qc ** 2) * np.sqrt(omega1 * omega2) * omegac * \
    #     (1 / (omega1 - omegac) + 1 / (omega2 - omegac)) * 0.5
    # g = g12 + gic
    return shift#, g
    
def write_data(f, data, type='w'):
    with open(f, type) as fp:
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

def ini_object(omegas, tq, neighbor, chip, anharmq):
    cost = []

    inducedChip = nx.Graph()
    inducedChip.add_node(tq)

    id = 0
    if chip.nodes[tq].get('freq', False):
        inducedChip.nodes[tq]['freq'] = chip.nodes[tq]['freq']
    else:
        inducedChip.nodes[tq]['freq'] = omegas[0]
        id += 1
    for nb in neighbor:
        inducedChip.add_node(nb)
        if chip.nodes[nb].get('freq', False):
            inducedChip.nodes[nb]['freq'] = chip.nodes[nb]['freq']
        else:
            inducedChip.nodes[nb]['freq'] = omegas[id]
            id += 1

    bases = list(state_number_enumerate([energyLevel] * len(inducedChip.nodes), 1 * len(inducedChip.nodes)))
    subStatesId = find_sub_space(bases, energyLevel, len(inducedChip.nodes))
    H0 = H(inducedChip, energyLevel, anharmq, 0, 0, 0)
    H0 = find_sub_matrix(H0, subStatesId)
    _, _, _, energys0 = eigensolve(H0, H0)
    for i in range(1, len(inducedChip.nodes) + 1):
        computeBas = []
        leakBas = []
        for bas in bases:
            if sum(bas) == i:
                if max(bas) < 2:
                    computeBas.append(energys0[subStatesId[state_number_index([energyLevel] * len(inducedChip.nodes), bas)]])
                else:
                    leakBas.append(energys0[subStatesId[state_number_index([energyLevel] * len(inducedChip.nodes), bas)]])
        for e1 in range(len(computeBas) - 1):
            for e2 in range(e1 + 1, len(computeBas)):
                cost.append(np.abs(computeBas[e1] - computeBas[e2]))
        for e1 in range(len(computeBas)):
            for e2 in range(len(leakBas)):
                cost.append(np.abs(computeBas[e1] - leakBas[e2]))
    print(omegas, tq, neighbor, -min(cost))
    return -min(cost)

def second_object(omegas, chip, rho_qq, rho_qc, energyLevel, anharmq, anharmc):
    cost = 0
    for coupler in chip.edges:
        chip.edges[coupler]['freq'] = omegas[0]
    tq = list(chip.nodes)[0]
    zz = np.abs(zzcoupling(chip, tq, rho_qq, rho_qc, energyLevel, anharmq, anharmc) * 1e3)
    print(omegas, zz)
    cost += zz
    return cost


# 画热力图

def draw_heat_map(xx, yy, mat, title, picname, xlabel, ylabel, drawtype=None, threshold=None):
    mat = np.array([mat]).reshape(len(xx), len(yy))
 
    if drawtype == None:
        mat = mat.T
    elif drawtype == 'log':
        mat = np.log10(mat.T)
    elif drawtype == 'abs':
        mat = np.abs(mat.T)
    elif 'log' in drawtype and 'abs' in drawtype:
        mat = np.log10(np.abs(mat.T) + 1e-6)

    xx,yy = np.meshgrid(xx, yy)
    font2 = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 20}
    fig, ax = plt.subplots()

    if threshold == None:
        cs = plt.contour(xx, yy, mat, colors="r", linewidths=0.5) 
    else:
        cs = plt.contour(xx, yy, mat, [threshold], colors="r", linewidths=0.5) 

    plt.clabel(cs, fontsize=12, inline=True) 
    plt.contourf(xx, yy, mat, 200, cmap=plt.cm.jet)

    plt.colorbar()

    plt.xlabel(xlabel, font2)
    plt.ylabel(ylabel, font2)
    plt.title(title)
    plt.savefig(picname + '.pdf', dpi=300)

if __name__ == '__main__':
    w = 3
    h = 4
    energyLevel = 3
    anharmq = -0.22

    # 搜索最优化耦合系数
    # freq1 = 4.5
    # freq2 = 4.4

    # cFreqLow = 4.6
    # cFreqHigh = 6.5
    # # cFreqLow = 2.9
    # # cFreqHigh = 4.3

    # rho_qcs = np.arange(0.005, 0.03, 0.001)
    # rho_qqs = np.arange(0.001, 0.003, 0.0001)
    # # rho_qcs = np.arange(0.005, 0.03, 0.001)
    # # rho_qqs = np.arange(-0.001, -0.003, -0.0001)

    # anharmcs = np.arange(-0.1, -0.3, -0.005)

    # cFreqs = np.arange(cFreqLow, cFreqHigh, 0.02)
    # chips = []
    # couplerOffFreqs = []
    # rho_qcss = []
    # rho_qqss = []
    # anharmcss = []

    # for rho_qc in rho_qcs:
    #     for rho_qq in rho_qqs:
    #         for anharmc in anharmcs:            
    #             for cFreq in cFreqs:
    #                 chip = nx.Graph()
    #                 chip.add_nodes_from([(0, 0), (0, 1)])
    #                 chip.nodes[(0, 0)]['freq'] = 4.5
    #                 chip.nodes[(0, 0)]['type'] = 0
    #                 chip.nodes[(0, 1)]['freq'] = 4.4
    #                 chip.nodes[(0, 1)]['type'] = 1
    #                 chip.add_edge((0, 0), (0, 1))
    #                 chip.edges[((0, 0), (0, 1))]['freq'] = cFreq
    #                 chips.append(chip)
    #                 rho_qcss.append(rho_qc)
    #                 rho_qqss.append(rho_qq)
    #                 anharmcss.append(anharmc)

    # energyLevels = [energyLevel] * len(rho_qcs) * len(rho_qqs) * len(anharmcs) * len(cFreqs)
    # anharmqs = [anharmq] * len(rho_qcs) * len(rho_qqs) * len(anharmcs) * len(cFreqs)
    # tqs = [(0, 0)] * len(rho_qcs) * len(rho_qqs) * len(anharmcs) * len(cFreqs)

    # tStart = time.time()
    # print('start')
    # p = Pool(10)
    # shift = p.starmap(zzcoupling, zip(chips, tqs, rho_qqss, rho_qcss, energyLevels, anharmqs, anharmcss))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # data_list = np.array(shift).reshape(len(rho_qcs), len(rho_qqs), len(anharmcs), len(cFreqs)) * 1e3

    # write_data('rho_data p.txt', data_list.reshape(len(rho_qcs) * len(rho_qqs) * len(anharmcs) * len(cFreqs)))
    # # write_data('rho_data n.txt', data_list.reshape(len(rho_qcs) * len(rho_qqs) * len(anharmcs) * len(cFreqs)))

    # rho_qc = 0.027
    # rho_qq = 0.0021
    # anharmc = -0.1
    # # rho_qc = 0.027
    # # rho_qq = -0.0016                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    # # anharmc = -0.14

    # 计算shift
    # sFreqLow = 4.5 - 0.4
    # sFreqHigh = 4.5 + 0.4
    
    # cFreqLow = 4.6
    # cFreqHigh = 6.5
    # # cFreqLow = 2.9
    # # cFreqHigh = 4.3

    # sFreqs = np.arange(sFreqLow, sFreqHigh, 0.002)
    # cFreqs = np.arange(cFreqLow, cFreqHigh, 0.02)

    # chips = []
    # couplerOffFreqs = []
    # for sFreq in sFreqs:
    #     for cFreq in cFreqs:
    #         chip = nx.Graph()
    #         chip.add_nodes_from([(0, 0), (0, 1)])
    #         chip.nodes[(0, 0)]['freq'] = 4.5
    #         chip.nodes[(0, 0)]['type'] = 0
    #         chip.nodes[(0, 1)]['freq'] = sFreq
    #         chip.nodes[(0, 1)]['type'] = 1
    #         chip.add_edge((0, 0), (0, 1))
    #         chip.edges[((0, 0), (0, 1))]['freq'] = cFreq
    #         chips.append(chip)
    # energyLevels = [energyLevel] * len(sFreqs) * len(cFreqs)
    
    # anharmqs = [anharmq] * len(sFreqs) * len(cFreqs)
    # anharmcs = [anharmc] * len(sFreqs) * len(cFreqs)
    # rho_qqs = [rho_qq] * len(sFreqs) * len(cFreqs)
    # rho_qcs = [rho_qc] * len(sFreqs) * len(cFreqs)
    # tqs = [(0, 0)] * len(sFreqs) * len(cFreqs)

    # tStart = time.time()
    # p = Pool(50)
    # singleShift = p.starmap(zzcoupling, zip(chips, tqs, rho_qqs, rho_qcs, energyLevels, anharmqs, anharmcs))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # for i in range(len(singleShift)):
    #     singleShift[i] = singleShift[i] * 1e3

    # write_data('single shift p.txt', singleShift) 
    # # write_data('single shift n.txt', singleShift) 

    # 计算单比特门串扰

    # sFreqLow = 4.5 - 0.4
    # sFreqHigh = 4.5 + 0.4
    # cFreqLow = 4.6
    # cFreqHigh = 6.5
    # # cFreqLow = 2.9
    # # cFreqHigh = 4.3

    # sFreqs = np.arange(sFreqLow, sFreqHigh, 0.002)
    # cFreqs = np.arange(cFreqLow, cFreqHigh, 0.02)
    # chips = []
    # couplerOffFreqs = []
    # for sFreq in sFreqs:
    #     for cFreq in cFreqs:
    #         chip = nx.Graph()
    #         chip.add_nodes_from([(0, 0), (0, 1)])
    #         chip.nodes[(0, 0)]['freq'] = 4.5
    #         chip.nodes[(0, 0)]['type'] = 0
    #         chip.nodes[(0, 1)]['freq'] = sFreq
    #         chip.nodes[(0, 1)]['type'] = 1
    #         chip.add_edge((0, 0), (0, 1))
    #         chip.edges[((0, 0), (0, 1))]['freq'] = cFreq
    #         chips.append(chip)

    # energyLevels = [energyLevel] * len(sFreqs) * len(cFreqs)

    # anharmqs = [anharmq] * len(sFreqs) * len(cFreqs)
    # anharmcs = [anharmc] * len(sFreqs) * len(cFreqs)
    # rho_qqs = [rho_qq] * len(sFreqs) * len(cFreqs)
    # rho_qcs = [rho_qc] * len(sFreqs) * len(cFreqs)
    # tqs = [(0, 0)] * len(sFreqs) * len(cFreqs)
    # tStart = time.time()
    # p = Pool(50)
    # calibratedSinglePara = p.starmap(calibration_single_qubit_gate, zip(chips, tqs, energyLevels, anharmqs, anharmcs, rho_qqs, rho_qcs))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)
    # errordict = []
    # for i in calibratedSinglePara:
    #     errordict.append(i[3])
    # write_data('sing err p-3.txt', errordict)
    # # write_data('sing err n-3.txt', errordict)

    # 次近邻串扰

    # freq0 = 4.5
    # freq1 = 4.4
    # coupler1 = ((0, 0), (0, 1))
    # coupler2 = ((0, 1), (0, 2))
    # chips = []
    # tqs = []
    # sFreqs = np.arange(4.49 + anharmq, 4.51 - anharmq, 0.001)
    # for freq in sFreqs:
    #     chip = nx.Graph()
    #     chip.add_edge(coupler1[0], coupler1[1])
    #     chip.nodes[coupler1[0]]['freq'] = freq0
    #     chip.nodes[coupler1[1]]['freq'] = freq1
    #     chip.add_edge(coupler2[0], coupler2[1])
    #     chip.nodes[coupler2[1]]['freq'] = freq
    #     for coupler in chip.edges:
    #         inducedChip = nx.Graph()
    #         inducedChip.add_edge(coupler[0], coupler[1])
    #         inducedChip.nodes[coupler[0]]['freq'] = chip.nodes[coupler[0]]['freq']
    #         inducedChip.nodes[coupler[1]]['freq'] = chip.nodes[coupler[1]]['freq']
    #         res = minimize(second_object, [6], args=(inducedChip, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #         bounds=((4.6, 6.5),), method='Nelder-Mead', options={'maxiter' : 50})
    #         # res = minimize(second_object, [3], args=(inducedChip, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #         # bounds=((2.9, 4),), method='Nelder-Mead', options={'maxiter' : 50})
    #         chip.edges[(coupler[0], coupler[1])]['freq'] = res.x[0]
    #     chips.append(chip)
    # energyLevels = [energyLevel] * len(sFreqs)
    
    # anharmqs = [anharmq] * len(sFreqs)
    # anharmcs = [anharmc] * len(sFreqs)
    # rho_qqs = [rho_qq] * len(sFreqs)
    # rho_qcs = [rho_qc] * len(sFreqs)
    # tqs = [(0, 0)] * len(sFreqs)
    # tStart = time.time()
    # p = Pool(50)
    # calibratedSinglePara = p.starmap(calibration_single_qubit_gate, zip(chips, tqs, energyLevels, anharmqs, anharmcs, rho_qqs, rho_qcs))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)
    # errordict = []
    # for i in calibratedSinglePara:
    #     errordict.append(i[3])
    # write_data('sing err2 p.txt', errordict)
    # # write_data('sing err2 n.txt', errordict)

    # 计算两比特门能级图
    # # cFreqs = np.arange(4.6, 6.5, 0.01)
    # cFreqs = np.arange(2.9, 4.3, 0.01)
    # freq0 = 4.5
    # freq1 = 4.4
    # tq = ((0, 0), (0, 1))
    # chip = nx.Graph()
    # chip.add_nodes_from([tq[0], tq[1]])
    # chip.nodes[tq[0]]['freq'] = freq0
    # chip.nodes[tq[0]]['type'] = 0
    # chip.nodes[tq[1]]['freq'] = freq1
    # chip.nodes[tq[1]]['type'] = 1
    # chip.add_edge(tq[0], tq[1])
    # chip.edges[tq]['freq'] = 0
    # qubitList = list(chip.nodes) + list(chip.edges)
    # g1 = qubitList.index(tq[0])
    # g2 = qubitList.index(tq[1])
    # nbState = list(np.zeros(len(qubitList), dtype=int))
    # s00 = deepcopy(nbState)
    # s01 = deepcopy(nbState)
    # s10 = deepcopy(nbState)
    # s11 = deepcopy(nbState)
    # s01[g1] = 1
    # s10[g2] = 1
    # s11[g1], s11[g2] = 1, 1
    # s00 = s00[::-1]
    # s01 = s01[::-1]
    # s10 = s10[::-1]
    # s11 = s11[::-1]
    # subStatesId = find_sub_space([s00, s01, s10, s11], energyLevel, len(qubitList))
    # allEnergys = None
    # allStates = None
    # energyDict = dict()
    # energy0Dict = dict()
    # for cFreq in cFreqs:
    #     inducedChip = deepcopy(chip)
    #     inducedChip.edges[tq]['freq'] = cFreq
    #     H0, Hi = H(inducedChip, energyLevel, anharmq, anharmc, rho_qq, rho_qc)
    #     subStatesId = find_sub_space([s00, s01, s10, s11], energyLevel, len(qubitList))
        
    #     H0 = find_sub_matrix(H0, subStatesId)
    #     Hi = find_sub_matrix(Hi, subStatesId)
        
    #     _, states0, energys, energy0s = eigensolve(H0, H0 + Hi)
    #     for state0 in states0:
    #         for st in range(len(np.array(state0))):
    #             if np.real(state0[st][0][0]) == 1:
    #                 break
    #         lab = list(subStatesId.keys())[list(subStatesId.values()).index(st)]
    #         lab = state_index_number([energyLevel] * len(qubitList), lab)
    #         strlab = ''.join([str(i) for i in lab])
    #         if not(strlab in energyDict.keys()):
    #             energyDict[strlab] = [energys[list(states0).index(state0)]]
    #             energy0Dict[strlab] = [energy0s[list(states0).index(state0)]]
    #         else:
    #             energyDict[strlab].append(energys[list(states0).index(state0)])
    #             energy0Dict[strlab].append(energy0s[list(states0).index(state0)])

    # colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'pink', 'olive']
    # i = 0
    # for lab in energyDict:
    #     if lab == '000':
    #         continue
    #     plt.plot(cFreqs, energyDict[lab], label=lab, color=colors[::-1][i])
    #     # plt.plot(cFreqs, energy0Dict[lab], linestyle='--', color=colors[i])
    #     i += 1
    # # plt.axhline(y=10 ** (1.7), xmin=0.05, xmax=0.9, color='red', linestyle='--')
    # # plt.axhline(y=1e-2, xmin=0.05, xmax=0.9, color='red', linestyle='--')
    # plt.legend()
    # plt.show()

    # 非共振驱动xy串扰

    # 单比特门工作点分配

    # qlow = 4.28
    # qhigh = 4.72
    # # clow = 5.5
    # # chigh = 6.5
    # clow = 2.9
    # chigh = 3.5
    # chip = nx.grid_2d_graph(w, h)

    # labelDict = dict([(i, i) for i in chip.nodes])
    # pos = gen_pos(chip)
    # nx.draw_networkx_labels(chip, pos, labelDict, font_size=10, font_color="red")
    # nx.draw_networkx_edges(chip, pos, edgelist=chip.edges)
    # nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes)
    # plt.axis('off')
    # plt.savefig(str(w) + str(h) + 'chip.pdf', dpi=300)
    # plt.close()

    # 第一轮

    # qubitList = list(chip.nodes)
    # for qubit in qubitList:
    #     if qubit == qubitList[0]:
    #         chip.nodes[qubit]['freq'] = 4.5
    #         continue

    #     iniOmega = []
    #     bounds = []
    #     if not chip.nodes[qubit].get('freq', False):
    #         iniOmega = [qlow + (qhigh - qlow) * np.random.random()]
    #         bounds = ((qlow, qhigh),)

    #     neighbor = []
    #     for nb in list(chip[qubit]):
    #         if chip.nodes[nb].get('freq', False):
    #             neighbor.append(nb)
                
    #     if len(iniOmega) > 0:
    #         res = minimize(ini_object, iniOmega, args=(qubit, neighbor, chip, anharmq), 
    #         bounds=bounds , method='Powell', options={'maxiter' : 50})
    #         id = 0
    #         if not chip.nodes[qubit].get('freq', False):
    #             chip.nodes[qubit]['freq'] = res.x[0] + np.random.random() * 5e-2
    #             id += 1

    # 第二轮

    # bounds = []
    # couplerList = list(chip.edges)
    # for coupler in couplerList:
    #     inducedChip = nx.Graph()
    #     inducedChip.add_edge(coupler[0], coupler[1])
    #     inducedChip.nodes[coupler[0]]['freq'] = chip.nodes[coupler[0]]['freq']
    #     inducedChip.nodes[coupler[1]]['freq'] = chip.nodes[coupler[1]]['freq']
    #     iniOmega = [(clow + chigh) / 2]
    #     # lb = [clow]
    #     # ub = [chigh]
    #     lb = clow
    #     ub = chigh
    #     bounds = [[lb, ub]]
    #     res = minimize(second_object, iniOmega, args=(inducedChip, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #     bounds=bounds, method='Powell', options={'maxiter' : 100})
    #     print(list(inducedChip.nodes), res.x, res.fun)
    #     # res = pso(second_object, lb, ub, maxiter=50, swarmsize=50, 
    #     # args=(inducedChip, rho_qq, rho_qc, energyLevel, anharmq, anharmc))
    #     for e in inducedChip.edges:
    #         chip.edges[e]['freq'] = res.x[0]
    #     # for e in inducedChip.edges:
    #     #     chip.edges[e]['freq'] = res[0][len(inducedChip.nodes) + list(inducedChip.edges).index(e)]

    # freqList = [chip.nodes[qubit]['freq'] for qubit in chip.nodes]
    # edgeList = [chip.edges[coupler]['freq'] for coupler in chip.edges]
    # freqDict = dict([(i, round(chip.nodes[i]['freq'], 3)) for i in chip.nodes])
    # edgeDict = dict([(i, round(chip.edges[i]['freq'], 3)) for i in chip.edges])
    # pos = gen_pos(chip)
    # nx.draw_networkx_labels(chip, pos, freqDict, font_size=10, font_color="black")
    # nx.draw_networkx_edges(chip, pos, edgelist=chip.edges, edge_color=edgeList, edge_cmap=plt.cm.Reds_r)
    # nx.draw_networkx_nodes(chip, pos, nodelist=chip.nodes, node_color=freqList, cmap=plt.cm.Reds_r)
    # plt.axis('off')
    # plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=qlow, vmax=chigh), cmap=plt.cm.Reds_r))
    # plt.savefig(str(w) + str(h) + 'chip freq.pdf', dpi=300)
    # plt.close()
    # with open(str(w) + str(h) + 'chip freq.txt', 'w') as fp:
    #     for d in freqDict:
    #         fp.write(str(d) + ':' + str(freqDict[d]) + '\n')
    # with open(str(w) + str(h) + 'chip freq.txt', 'a+') as fp:
    #     for d in edgeDict:
    #         fp.write(str(d) + ':' + str(edgeDict[d]) + '\n')

    # 读频率

    # chip = nx.grid_2d_graph(w, h)
    # freqList = list(chip.nodes)
    # edgeList = list(chip.edges)
    # feList = freqList + edgeList
    # with open(str(w) + str(h) + 'chip freq.txt', 'r') as fp:
    #     data = fp.read()
    #     data = data.split('\n')
    #     if '' in data:
    #         data.remove('')
    #     id = 0
    #     for d in data:
    #         if id < len(freqList):
    #             chip.nodes[feList[id]]['freq'] = float(d.split(':')[-1])
    #         else:
    #             chip.edges[feList[id][0], feList[id][1]]['freq'] = float(d.split(':')[-1])
    #         id += 1 
    
    # removeNode = list(chip.nodes)
    # removeNode.remove((1, 1))
    # removeNode.remove((1, 2))
    # removeNode.remove((1, 3))
    # removeNode.remove((0, 2))
    # removeNode.remove((2, 2))
    # # removeNode.remove((1, 0))
    # # removeNode.remove((0, 1))
    # removeNode.remove((2, 1))
    # chip.remove_nodes_from(removeNode)
    # print('(1, 3)(0, 2)(2, 2)(2, 1)')

    # surface code 构型下，单比特门串扰

    # inducedChips = []
    # couplerOffFreqs = []
    # tqs = []
    # for qubit in chip.nodes:
    #     inducedChip = nx.Graph()
    #     inducedChip.add_node(qubit)
    #     inducedChip.nodes[qubit]['freq'] = chip.nodes[qubit]['freq']
    #     for nb in chip[qubit]:
    #         inducedChip.add_node(nb)
    #         inducedChip.add_edge(qubit, nb)
    #         inducedChip.nodes[nb]['freq'] = chip.nodes[nb]['freq']
    #         inducedChip.edges[(qubit, nb)]['freq'] = chip.edges[(qubit, nb)]['freq']
    #     inducedChips.append(inducedChip)
    #     tqs.append(list(inducedChip.nodes)[0])
    # energyLevels = [energyLevel] * len(chip)
    # anharmqs = [anharmq] * len(chip)
    # anharmcs = [anharmc] * len(chip)
    # rho_qqs = [rho_qq] * len(chip)
    # rho_qcs = [rho_qc] * len(chip)
    # tStart = time.time()
    # p = Pool(w * h)
    # calibratedSinglePara = p.starmap(calibration_single_qubit_gate, zip(inducedChips, tqs, energyLevels, anharmqs, anharmcs, rho_qqs, rho_qcs))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)
    # write_data('single Q para p15.txt', calibratedSinglePara)

    # 固定比特两比特门

    # 不同detune下的最小保真度

    # # ini = [6]
    # # bounds = (5.5, 6.5)
    # ini = [3]
    # bounds = (2.5, 3.5)

    # freq0 = 4.5
    # freq1 = np.arange(4.31, 4.47, 0.005)
    # coupler = ((0, 0), (0, 1))
    # inducedChips = []
    # tqs = []
    # for freq in freq1:
    #     inducedChip = nx.Graph()
    #     inducedChip.add_edge(coupler[0], coupler[1])
    #     inducedChip.nodes[coupler[0]]['freq'] = freq0
    #     inducedChip.nodes[coupler[1]]['freq'] = freq
    #     res = minimize(second_object, ini, args=(inducedChip, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #     bounds=(bounds,), method='Powell', options={'maxiter' : 50})
    #     inducedChip.edges[(coupler[0], coupler[1])]['freq'] = res.x[0]

    #     inducedChips.append(inducedChip)
    #     tqs.append(coupler)

    # energyLevels = [energyLevel] * len(freq1)
    # anharmqs = [anharmq] * len(freq1)
    # anharmcs = [anharmc] * len(freq1)
    # rho_qqs = [rho_qq] * len(freq1)
    # rho_qcs = [rho_qc] * len(freq1)
    # tStart = time.time()
    # p = Pool(50)
    # calibratedTwoPara = p.starmap(calibration_two_qubit_gate, zip(inducedChips, tqs, energyLevels, anharmqs, anharmcs, rho_qqs, rho_qcs))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # write_data('two Q para detune n.txt', calibratedTwoPara, 'a+')

    # 串扰

    # 单个串扰比特
    
    # freq0 = 4.4
    # freq1 = 4.5
    # freqs = np.arange(min(freq0, freq1) + anharmq, max(freq0, freq1) - anharmq, 0.005)

    # coupler1 = ((0, 0), (0, 1))
    # coupler2 = ((0, 1), (0, 2))
    # inducedChips = []
    # couplerOffFreqs = []
    # tqs = []
    # for freq in freqs:
    #     chip1 = nx.Graph()
    #     chip1.add_edge(coupler1[0], coupler1[1])
    #     chip1.nodes[coupler1[0]]['freq'] = freq0
    #     chip1.nodes[coupler1[1]]['freq'] = freq1
    #     res = minimize(second_object, [3], args=(chip1, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #         bounds=((2.9, 4.3),), method='Powell', options={'maxiter' : 50})
    #     chip1.edges[coupler1]['freq'] = res.x[0]
    #     chip2 = nx.Graph()
    #     chip2.add_edge(coupler2[0], coupler2[1])
    #     chip2.nodes[coupler2[0]]['freq'] = freq1
    #     chip2.nodes[coupler2[1]]['freq'] = freq
    #     res = minimize(second_object, [3], args=(chip2, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #         bounds=((2.9, 4.3),), method='Powell', options={'maxiter' : 50})
    #     chip2.edges[coupler2]['freq'] = res.x[0]
    #     inducedChip = nx.Graph()
    #     inducedChip.add_edges_from([coupler1, coupler2])
    #     inducedChip.nodes[coupler1[0]]['freq'] = freq0
    #     inducedChip.nodes[coupler1[1]]['freq'] = freq1
    #     inducedChip.nodes[coupler2[1]]['freq'] = freq
    #     inducedChip.edges[coupler1]['freq'] = chip1.edges[coupler1]['freq']
    #     inducedChip.edges[coupler2]['freq'] = chip2.edges[coupler2]['freq']
    #     inducedChips.append(inducedChip)

    # tqs = [coupler1] * len(freqs)
    # energyLevels = [energyLevel] * len(freqs)
    # anharmqs = [anharmq] * len(freqs)
    # anharmcs = [anharmc] * len(freqs)
    # rho_qqs = [rho_qq] * len(freqs)
    # rho_qcs = [rho_qc] * len(freqs)
    # tStart = time.time()
    # p = Pool(25)
    # calibratedTwoPara = p.starmap(calibration_two_qubit_gate, zip(inducedChips, tqs, energyLevels, anharmqs, anharmcs, rho_qqs, rho_qcs))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # write_data('two Q para s4.5.txt', calibratedTwoPara)

    # 两个串扰比特

    # freq0 = 4.4
    # freq1 = 4.5
    # # freqs0s = np.arange(freq0 - 0.22, freq0 + 0.22, 0.005)
    # # freqs1s = np.arange(freq1 - 0.22, freq1 + 0.22, 0.005)
    # freqs0s = [4.52]
    # freqs1s = [4.39]

    # coupler0 = ((1, 0), (0, 0))
    # coupler1 = ((0, 0), (0, 1))
    # coupler2 = ((0, 1), (1, 1))
    # # coupler3 = ((1, 1), (1, 0))

    # inducedChips = []
    # couplerOffFreqs = []
    # tqs = []
    # for freqs0 in freqs0s:
    #     for freqs1 in freqs1s:
    #         chip0 = nx.Graph()
    #         chip0.add_edge(coupler0[0], coupler0[1])
    #         chip0.nodes[coupler0[0]]['freq'] = freqs0
    #         chip0.nodes[coupler0[1]]['freq'] = freq0
    #         res = minimize(second_object, [3], args=(chip0, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #             bounds=((1.5, 4.5),), method='trust-constr', options={'maxiter' : 50})
    #         chip0.edges[coupler0]['freq'] = res.x[0]
    #         chip1 = nx.Graph()
    #         chip1.add_edge(coupler1[0], coupler1[1])
    #         chip1.nodes[coupler1[0]]['freq'] = freq0
    #         chip1.nodes[coupler1[1]]['freq'] = freq1
    #         res = minimize(second_object, [3], args=(chip1, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #             bounds=((1.5, 4.5),), method='trust-constr', options={'maxiter' : 50})
    #         chip1.edges[coupler1]['freq'] = res.x[0]
    #         chip2 = nx.Graph()
    #         chip2.add_edge(coupler2[0], coupler2[1])
    #         chip2.nodes[coupler2[0]]['freq'] = freq1
    #         chip2.nodes[coupler2[1]]['freq'] = freqs1
    #         res = minimize(second_object, [3], args=(chip2, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #             bounds=((1.5, 4.5),), method='trust-constr', options={'maxiter' : 50})
    #         chip2.edges[coupler2]['freq'] = res.x[0]

    #         # chip3 = nx.Graph()
    #         # chip3.add_edge(coupler3[0], coupler3[1])
    #         # chip3.nodes[coupler3[0]]['freq'] = freqs1
    #         # chip3.nodes[coupler3[1]]['freq'] = freqs0
    #         # res = minimize(second_object, [3], args=(chip3, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #         #     bounds=((1.5, 4.5),), method='trust-constr', options={'maxiter' : 50})
    #         # # chip3.edges[coupler3]['freq'] = res.x[0]
    #         # chip3.edges[coupler3]['freq'] = 3.6

    #         inducedChip = nx.Graph()

    #         # inducedChip.add_edges_from([coupler0, coupler1, coupler2, coupler3])
    #         # inducedChip.add_edges_from([coupler1, coupler2, coupler3])
    #         inducedChip.add_edges_from([coupler0, coupler1, coupler2])

    #         inducedChip.nodes[coupler0[0]]['freq'] = freqs0
    #         # inducedChip.nodes[coupler3[1]]['freq'] = freqs0
    #         inducedChip.nodes[coupler1[0]]['freq'] = freq0
    #         inducedChip.nodes[coupler1[1]]['freq'] = freq1
    #         inducedChip.nodes[coupler2[1]]['freq'] = freqs1
    #         inducedChip.edges[coupler0]['freq'] = chip0.edges[coupler0]['freq']
    #         inducedChip.edges[coupler1]['freq'] = chip1.edges[coupler1]['freq']
    #         inducedChip.edges[coupler2]['freq'] = chip2.edges[coupler2]['freq']
    #         # inducedChip.edges[coupler3]['freq'] = chip3.edges[coupler3]['freq']
    #         inducedChips.append(inducedChip)

    # tqs = [coupler1] * len(freqs0s) * len(freqs1s)
    # energyLevels = [energyLevel] * len(freqs0s) * len(freqs1s)
    # anharmqs = [anharmq] * len(freqs0s) * len(freqs1s)
    # anharmcs = [anharmc] * len(freqs0s) * len(freqs1s)
    # rho_qqs = [rho_qq] * len(freqs0s) * len(freqs1s)
    # rho_qcs = [rho_qc] * len(freqs0s) * len(freqs1s)
    # tStart = time.time()
    # p = Pool(20)
    # calibratedTwoPara = p.starmap(calibration_two_qubit_gate, zip(inducedChips, tqs, energyLevels, anharmqs, anharmcs, rho_qqs, rho_qcs))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # write_data('two Q para 2s.txt', calibratedTwoPara)

    # 校准 

    # inducedChips = []
    # tqs = []
    # # for coupler in chip.edges:
    # for coupler in [((1, 1), (1, 2))]:
    #     inducedChip = nx.Graph()
    #     inducedChip.add_edge(coupler[0], coupler[1])
    #     inducedChip.nodes[coupler[0]]['freq'] = chip.nodes[coupler[0]]['freq']
    #     inducedChip.nodes[coupler[1]]['freq'] = chip.nodes[coupler[1]]['freq']
    #     inducedChip.edges[(coupler[0], coupler[1])]['freq'] = chip.edges[(coupler[0], coupler[1])]['freq']

    #     for b in coupler:
    #         for nb in chip[b]:
    #             if not(nb in coupler):
    #                 inducedChip.add_edge(b, nb)
    #                 inducedChip.nodes[nb]['freq'] = chip.nodes[nb]['freq']
    #                 inducedChip.edges[(b, nb)]['freq'] = chip.edges[(b, nb)]['freq']

    #     # nbs = []
    #     # for b in coupler:
    #     #     for nb in chip[b]:
    #     #         if not(nb in coupler):
    #     #             nbs.append(nb) 
    #     #             inducedChip.add_edge(b, nb)
    #     #             inducedChip.nodes[nb]['freq'] = chip.nodes[nb]['freq']
    #     #             inducedChip.edges[(b, nb)]['freq'] = chip.edges[(b, nb)]['freq']

    #     # for i in range(len(nbs) - 1):
    #     #     for j in range(i + 1, len(nbs)):
    #     #         if (nbs[i], nbs[j]) in chip.edges:
    #     #             inducedChip.add_edge(nbs[i], nbs[j])
    #     #             inducedChip.edges[(nbs[i], nbs[j])]['freq'] = chip.edges[(nbs[i], nbs[j])]['freq']

    #     inducedChips.append(inducedChip)
    #     tqs.append(coupler) 

    # energyLevels = [energyLevel] * len(chip.edges)
    # anharmqs = [anharmq] * len(chip.edges)
    # anharmcs = [anharmc] * len(chip.edges)
    # rho_qqs = [rho_qq] * len(chip.edges)
    # rho_qcs = [rho_qc] * len(chip.edges)
    # tStart = time.time()
    # p = Pool(len(chip.edges))
    # calibratedTwoPara = p.starmap(calibration_two_qubit_gate, zip(inducedChips, tqs, energyLevels, anharmqs, anharmcs, rho_qqs, rho_qcs))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)
    # write_data('pure two Q para.txt', calibratedTwoPara)
    # write_data('1s two Q para.txt', calibratedTwoPara)
    # write_data('2s two Q para.txt', calibratedTwoPara)
    # write_data('3s two Q para.txt', calibratedTwoPara)
    # write_data('4s two Q para.txt', calibratedTwoPara)
    # write_data('5s two Q para.txt', calibratedTwoPara)
    # write_data('6s two Q para.txt', calibratedTwoPara)

    # 两比特门并行

    # inducedChip = nx.Graph()
    # bits = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # couplers = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((1, 0), (1, 1))]
    # for coupler in couplers:
    #     inducedChip.add_edge(coupler[0], coupler[1])
    #     inducedChip.edges[coupler]['freq'] = chip.edges[coupler]['freq']
    # for bit in bits:
    #     inducedChip.nodes[bit]['freq'] = chip.nodes[bit]['freq']

    # workFreq = [4.3, 4.3, 4.3, 4.3]
    # workSigma1 = [0.6898631426955542, 0.8547612352716976, 1.0222906836681385, 1.0194337632777746]
    # workSigma2 = [-0.1288946875845103, -0.05433494065985109, -0.01447847702188370, 0]
    # workSigma3 = [0.33119552493065535, 0.130256738932715, -0.0007144343853733758, 0.0001293808109563816]


    # # tq = couplers[0]
    # # nbq = couplers[3]

    # # tq = couplers[3]
    # # nbq = couplers[0]

    # # tq = couplers[2]
    # # nbq = couplers[1]

    
    # tq = couplers[1] 
    # nbq = couplers[2]

    # tg = 60
    # tList = np.arange(0, tg, 1 / (2 * tg))

    # pulse = pulse_fun(tList, tg, [workSigma1[couplers.index(tq)], workSigma2[couplers.index(tq)], workSigma3[couplers.index(tq)]],
    # workFreq[couplers.index(tq)], inducedChip.edges[nbq]['freq'])
    # inducedChip.edges[nbq]['freq'] = np.max(pulse)

    # lambdas = [workFreq[couplers.index(tq)], workSigma1[couplers.index(tq)], workSigma2[couplers.index(tq)], workSigma3[couplers.index(tq)]]

    # errdict = dict()
    # qubitNum = len(inducedChip.nodes) + len(inducedChip.edges)
    # qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    # g1 = qubitList.index(tq[0])
    # g2 = qubitList.index(tq[1])
    # if len(inducedChip.nodes) - 2 > 0:
    #     for bas in state_number_enumerate([2] * (len(inducedChip.nodes) - 2), 1 * (len(inducedChip.nodes) - 2)):
    #             id = 0
    #             zzbas = list(np.zeros(qubitNum, dtype=int))
    #             for i in range(len(inducedChip.nodes)):
    #                 if not(i in [g1, g2]):
    #                     zzbas[i] = bas[id]
    #                     id += 1
    #             errdict[str(bas[::-1])] = par_two_qubit_gate(inducedChip, tq, energyLevel, anharmq, anharmc, rho_qq, rho_qc, lambdas, zzbas)
    # for key in errdict:
    #     if errdict[key] < 1e-4:
    #         errdict[key] = 0
    # print('target gate', tq, errdict)
    # print('neighbor gate', nbq, inducedChip.edges[nbq]['freq'])