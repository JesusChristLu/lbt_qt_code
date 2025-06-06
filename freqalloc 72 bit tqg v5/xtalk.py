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
import matplotlib

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

def H(inducedChip, energyLevel, anharmq, anharmc=0, g_qq=0, g_qc=0):
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
        return H0, H0 - H0

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


def H_time(args):
    inducedChip = args['induced chip']
    subStatesId = args['sub states id']
    qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    energyLevel = args['energy level']
    anharmq = args['qt anharm']
    qNum = len(inducedChip.nodes)

    aList, aDagList, _, _, _ = getaad(energyLevel, qNum)

    HAnharm = []
    H0 = []
    for q in range(qNum):
        mat = find_sub_matrix(aDagList[q] * aDagList[q] * aList[q] * aList[q], subStatesId)
        HAnharm = np.pi * anharmq * mat
        mat = find_sub_matrix(aDagList[q] * aList[q], subStatesId)
        H0 = inducedChip.nodes[qubitList[q]]['freq'] * np.pi * 2 * mat
    return H0 + HAnharm

def pulse_fun(tList, pulseLen, sigma, freqWork, freqMax):
    if freqWork == freqMax:
        return freqWork
    else:
        # freqList = freqMax + (freqWork - freqMax) * (1 - np.sum(sigma)) * (1 - np.cos(tList / pulseLen * (len(sigma) + 1) * 2 * np.pi)) / 2
        # w = 1
        # for i in sigma:
        #     freqList += (freqWork - freqMax) * i * (1 - np.cos(tList / pulseLen * w * 2 * np.pi)) / 2
        #     w += 1
        
        # return freqList
        
        flattop_start = 3 * sigma[0]
        flattop_end = pulseLen - 3 * sigma[0]


        freqList = (freqWork - freqMax) * 1 / 2 * (erf((tList - flattop_start) / (np.sqrt(2) * sigma[0])) - \
                                    erf((tList - flattop_end) / (np.sqrt(2) * sigma[0]))) + freqMax
        
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
    # subStates = []
    # subStatesId = dict()
    # for bas in state_number_enumerate([energyLevel] * bitNum, energyLevel * bitNum):
    #     subStates.append(bas)
    #     subStatesId[state_number_index([energyLevel] * bitNum, bas)] = state_number_index([energyLevel] * bitNum, bas)
    # return subStatesId

def find_sub_matrix(mat, subStateId):
    npmat = np.zeros((len(subStateId), len(subStateId)), dtype=complex)
    for i in subStateId:
        for j in subStateId:
            npmat[subStateId[i], subStateId[j]] = mat[i, j]
    return Qobj(npmat)
    # return mat
    
def eigensolve(H0, H, sort=True):

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

    if sort:
        assignedIndex = []
        for n, energy in enumerate(ei_energy0):
            indexs = [i for i in range(len(H0diag)) if H0diag[i] == energy]
            for index in indexs:
                if not (index in assignedIndex):
                    states0_order[index] = ei_vector0[n]
                    energy0_order[index] = ei_energy0[n]            
                    assignedIndex.append(index)
                    break        
        assignedIndex = []
        for n, energy in enumerate(ei_energy0):
            maxindex = 0
            maxProd = np.abs((ei_vector[maxindex].dag() * states0_order[n]).full()[0][0])
            for index in range(1, len(energy0_order)):
                if index in assignedIndex:
                    continue
                prod = np.abs((ei_vector[index].dag() * states0_order[n]).full()[0][0])
                if prod > maxProd:
                    maxProd = prod
                    maxindex = index
            assignedIndex.append(maxindex)
            states_order[n] = ei_vector[maxindex]
            energy_order[n] = ei_energy[maxindex]
    return states_order, states0_order, energy_order / (np.pi * 2), energy0_order / (np.pi * 2)
   
def drive_pulseXt(tList, args):  
    tg = args['gate time']
    amp = args['qt amp']
    w_d = args['qt drive frequency']
    alpha = args['qt anharm']
    detune = args['qt detune']
    lambda0 = args['drag weight']
    phi = args['qt phi']
    I = np.cos(phi)
    Q = np.sin(phi)
    X0 = amp * (1 - np.cos(2 * np.pi * tList / tg)) / 2 
    Y0 = -amp * np.pi / tg * np.sin(2 * np.pi * tList / tg) / alpha * lambda0
    X = (X0 * I + Y0 * Q) * np.cos((w_d * np.pi * 2 + detune) * tList)
    return X

def drive_pulseYt(tList, args):  
    tg = args['gate time']
    amp = args['qt amp']
    w_d = args['qt drive frequency']
    alpha = args['qt anharm']
    detune = args['qt detune']
    lambda0 = args['drag weight']
    phi = args['qt phi']
    I = np.cos(phi)
    Q = np.sin(phi)
    X0 = amp * (1 - np.cos(2 * np.pi * tList / tg)) / 2 
    Y0 = -amp * np.pi / tg * np.sin(2 * np.pi * tList / tg) / alpha * lambda0
    X = (Y0 * I - X0 * Q) * np.cos((w_d * np.pi * 2 + detune) * tList)
    return X

def xtalk_pulseXt(tList, args):  
    tg = args['gate time']
    amp = args['qt amp']
    w_d = args['xtalk freq']
    alpha = args['qt anharm']
    detune = args['qt detune']
    lambda0 = args['drag weight']
    phi = args['qt phi']
    mu = args['mu']
    I = np.cos(phi)
    Q = np.sin(phi)
    X0 = mu * amp * (1 - np.cos(2 * np.pi * tList / tg)) / 2 
    Y0 = -mu * amp * np.pi / tg * np.sin(2 * np.pi * tList / tg) / alpha * lambda0
    X = (X0 * I + Y0 * Q) * np.cos((w_d * np.pi * 2 + detune) * tList)
    return X

def xtalk_pulseYt(tList, args):  
    tg = args['gate time']
    amp = args['qt amp']
    w_d = args['xtalk freq']
    alpha = args['qt anharm']
    detune = args['qt detune']
    lambda0 = args['drag weight']
    phi = args['qt phi']
    mu = args['mu']
    I = np.cos(phi)
    Q = np.sin(phi)
    X0 = mu * amp * (1 - np.cos(2 * np.pi * tList / tg)) / 2 
    Y0 = -mu * amp * np.pi / tg * np.sin(2 * np.pi * tList / tg) / alpha * lambda0
    X = (Y0 * I - X0 * Q) * np.cos((w_d * np.pi * 2 + detune) * tList)
    return X

def fidelity(phi, UEff, UIdeal, computeBas=None, allBas=None):
    d = len(UIdeal)
    start = 0
    for i in phi:
        if start == 0:
            Rz = rz(i)
            start = 1
        else:
            Rz = tensor([rz(i), Rz])
    Rz = np.array(Rz)
    if not(len(Rz) == len(UIdeal)):
        subStatesId = dict()
        for ab in allBas:
            for cpb in computeBas:
                if ab == cpb[-len(phi):]:
                    subStatesId[allBas.index(ab)] = computeBas.index(cpb)
                    break
        Rz = find_sub_matrix(Rz, subStatesId)
    U = np.dot(Rz, UEff)
    f = -((np.trace(np.dot(U.T.conjugate(), U)) + np.abs(np.trace(np.dot(U.T.conjugate(), UIdeal))) ** 2) / (d * (d + 1)))
    return np.real(f)

def evolution(lambdas, states, subStatesId, inducedChip, energyLevel, anharmq, excitedN, tg, fn=None, mu=None):
    lambdas = list(lambdas[::-1])
    qubitList = list(inducedChip.nodes)
    qubitNum = len(qubitList)

    tList = np.linspace(0, tg, 300)
    # tList = np.linspace(0, tg, 500)

    args = dict() 
    args['induced chip'] = inducedChip
    args['sub states id'] = subStatesId
    args['energy level'] = energyLevel
    args['qt anharm'] = anharmq
    args['gate time'] = 30
    args['drag weight'] = 0.5
    if not(mu == None):
        args['xtalk freq'] = fn
        args['mu'] = mu

    args['sigmac'] = []
    args['sigmaq'] = []
    args['freqcWork'] = []
    args['freqqWork'] = []
    args['qt drive frequency'] = []
    args['qt detune'] = []
    args['qt amp'] = []

    _, _, sxList, syList, _ = getaad(energyLevel, qubitNum)

    args['freqqWork'] = inducedChip.nodes[qubitList[0]]['freq']
    args['qt drive frequency'] = inducedChip.nodes[qubitList[0]]['freq']
    args['qt detune'] = lambdas.pop()
    args['qt amp'] = lambdas.pop()

    npsx = find_sub_matrix(sxList[0], subStatesId)
    npsy = find_sub_matrix(syList[0], subStatesId)

    args['qt phi'] = -np.array(args['qt detune']) * args['gate time'] * np.pi

    xt = drive_pulseXt(tList, args=args)
    yt = drive_pulseYt(tList, args=args)

    Ht = H_time(args=args)
    H_xt = [npsx, xt]
    H_yt = [npsy, yt]
    if not(mu == None):
        xt = xtalk_pulseXt(tList, args=args)
        yt = xtalk_pulseYt(tList, args=args)
        H_xtalk_xt = [npsx, xt]
        H_xtalk_yt = [npsy, yt]
        Ht = [Ht, H_xt, H_yt, H_xtalk_xt, H_xtalk_yt]
    else:
        Ht = [Ht, H_xt, H_yt]

    # U_fulls = propagator(Ht, tList, parallel=True, num_cpus=4)
    U_fulls = propagator(Ht, tList, parallel=False)

    U_full = U_fulls[-1]

    gIdeal = None
    oneBit = []
    for gate in inducedChip.nodes:
        if gIdeal == None:
            gIdeal = tensorOperator(2, sigmax(), qubitList.index(gate), qubitNum)
        else:
            gIdeal = tensorOperator(2, sigmax(), qubitList.index(gate), qubitNum) * gIdeal
        oneBit.append(gate)
                
    kets = []
    idealKets = []
    computeBas = []
    for bas in state_number_enumerate([2] * qubitNum, excitedN):
        leakBas = False
        for b in range(len(bas)):
            leakFlag = not(qubitList[b] in oneBit)
            if bas[::-1][b] == 1 and leakFlag:
                leakBas = True
                break
        if not leakBas:
            ket = states[subStatesId[state_number_index([energyLevel] * qubitNum, bas)]]
            if not(np.max(np.array(ket)) == np.max(np.abs(np.array(ket)))):
                ket = -ket
            kets.append(ket)
            idealKets.append(state_number_qobj([2] * qubitNum, bas))
            computeBas.append(bas)

    UIdeal = np.zeros([len(kets), len(kets)], dtype=complex)
    U = np.zeros([len(kets), len(kets)], dtype=complex)
    for i in range(len(kets)):
        for j in range(len(kets)):
            U[i, j] = (kets[i].dag() * U_full * kets[j]).full()[0][0]
            UIdeal[i, j] = (idealKets[i].dag() * gIdeal * idealKets[j]).full()[0][0]

    phi = []
    for _ in qubitList:
        phi.append(0)
    allBas = list(state_number_enumerate([2] * len(phi), len(phi)))
    F = minimize(fidelity, phi, args=(U, UIdeal, computeBas, allBas), method='BFGS')
    phi = F.x
    F = -F.fun
    error = 1 - F
    # print(para, error)
    return error
    
def par_evolution(inducedChip, tq, tg, energyLevel, anharmq, lambdas, fn=0, mu=0):
    calibratedG = tq
    qubitNum = len(inducedChip.nodes) + len(inducedChip.edges)
    excitedN = len(inducedChip.nodes)
    xNum = 0
    for gg in calibratedG:
        if gg in inducedChip.nodes:
            xNum += 1

    states = []
    for n in range(excitedN + xNum + 1):
        for exciteds in state_number_enumerate([energyLevel] * qubitNum, n):
            states.append(exciteds)

    H0, Hi = H(inducedChip, energyLevel, anharmq)
    subStatesId = find_sub_space(states, energyLevel, qubitNum)

    H0 = find_sub_matrix(H0, subStatesId)
    Hi = find_sub_matrix(Hi, subStatesId)         

    states, _, _, _ = eigensolve(H0, H0 + Hi)
    err = evolution(lambdas, states, subStatesId, inducedChip, energyLevel, anharmq, excitedN, tg, fn, mu)
    print(calibratedG, err)
    return err

def calibration(inducedChip, tq, tg, energyLevel, anharmq):
    qubitNum = len(inducedChip.nodes) + len(inducedChip.edges)
    excitedN = len(inducedChip.nodes)
    xNum = 0
    for gg in tq:
        if gg in inducedChip.nodes:
            xNum += 1

    states = []
    for n in range(excitedN + xNum + 1):
        for exciteds in state_number_enumerate([energyLevel] * qubitNum, n):
            states.append(exciteds)

    H0, Hi = H(inducedChip, energyLevel, anharmq)
    subStatesId = find_sub_space(states, energyLevel, qubitNum)

    H0 = find_sub_matrix(H0, subStatesId)
    Hi = find_sub_matrix(Hi, subStatesId)         

    states, _, _, _ = eigensolve(H0, H0 + Hi)

    bounds = []
    lambdas = []
    for qubit in inducedChip.nodes:
        if qubit in tq:
            lambdas.append(0)
            bounds.append((-1, 1))
            lambdas.append(0.5)
            bounds.append((0, 2))

    methods = ['BFGS', 'CG', 'Nelder-Mead', 'Powell']
    result = minimize(evolution, lambdas, 
    args=(states, subStatesId, inducedChip, energyLevel, anharmq, excitedN, tg), 
        method=methods[2], bounds=bounds, options={'ftol' : 1e-3})
    lambdas = result.x
    return lambdas
    # err = evolution(lambdas, states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, cq, g_qq, g_qc, 
    # excitedN, state_index_number([2] * qubitNum, 0), last=True)
    # return lambdas, err

def zzcoupling(inducedChip, tq, rho_qq, rho_qc, energyLevel, anharmq, anharmc, ada=False):
    qubitNum = len(inducedChip.nodes) + len(inducedChip.edges)
    qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    g = qubitList.index(tq)
    s00 = np.zeros(qubitNum, dtype=int)
    s01 = np.zeros(qubitNum, dtype=int)
    s10 = np.zeros(qubitNum, dtype=int)
    s11 = np.zeros(qubitNum, dtype=int)

    s02 = np.zeros(qubitNum, dtype=int)
    s12 = np.zeros(qubitNum, dtype=int)

    s01[g] = 1
    s11[g] = 1

    s02[g] = 2
    s12[g] = 2

    s10[qubitList.index(list(dict(inducedChip[tq]).keys())[0])] = 1
    s11[qubitList.index(list(dict(inducedChip[tq]).keys())[0])] = 1

    s12[qubitList.index(list(dict(inducedChip[tq]).keys())[0])] = 1

    s01 = s01[::-1]
    s10 = s10[::-1]
    s11 = s11[::-1]

    s02 = s02[::-1]
    s12 = s12[::-1]

    H0, Hi = H(inducedChip, energyLevel, anharmq, anharmc, rho_qq, rho_qc)

    subStatesId = find_sub_space([s00, s01, s10, s11], energyLevel, qubitNum)
    # subStatesId = find_sub_space([s00, s01, s10, s11, s02, s12], energyLevel, qubitNum)
    
    H0 = find_sub_matrix(H0, subStatesId)
    Hi = find_sub_matrix(Hi, subStatesId)

    if ada:
        bareInducedChip = deepcopy(inducedChip)
        if inducedChip.edges[tq, list(inducedChip[tq])[0]]['freq'] > inducedChip.nodes[tq]['freq']:
            bareInducedChip.edges[tq, list(inducedChip[tq])[0]]['freq'] = 7
        else:
            bareInducedChip.edges[tq, list(inducedChip[tq])[0]]['freq'] = 2.5

        H0bare, Hibare = H(bareInducedChip, energyLevel, anharmq, anharmc, rho_qq, rho_qc)
        H0bare = find_sub_matrix(H0bare, subStatesId)
        Hibare = find_sub_matrix(Hibare, subStatesId)
        _, states0, energys, _ = eigensolve(H0bare, H0bare + Hibare, sort=False)
        index11 = subStatesId[state_number_index([energyLevel] * qubitNum, s11)]
        index10 = subStatesId[state_number_index([energyLevel] * qubitNum, s10)]
        index01 = subStatesId[state_number_index([energyLevel] * qubitNum, s01)]
        for state0 in states0:
            if np.real((state0[index11])) == 1:
                order11 = list(states0).index(state0)
            elif np.real((state0[index10])) == 1:
                order10 = list(states0).index(state0)
            elif np.real((state0[index01])) == 1:
                order01 = list(states0).index(state0)
        order00 = 0
        _, _, energys, _ = eigensolve(H0, H0 + Hi, sort=False)
    else:
        order11 = subStatesId[state_number_index([energyLevel] * qubitNum, s11)]
        order10 = subStatesId[state_number_index([energyLevel] * qubitNum, s10)]
        order01 = subStatesId[state_number_index([energyLevel] * qubitNum, s01)]
        order00 = 0
        _, _, energys, _ = eigensolve(H0, H0 + Hi)


    shift1 = np.abs((energys[order11] - \
                energys[order10]) - \
                (energys[order01] - \
                energys[order00]))
    
    # omega1 = inducedChip.nodes[qubitList[0]]['freq']
    # omega2 = inducedChip.nodes[qubitList[1]]['freq']
    # omegac = inducedChip.edges[qubitList[2]]['freq']
    # g12 = rho_qq * np.sqrt(omega1 * omega2)
    # gic = (rho_qc ** 2) * np.sqrt(omega1 * omega2) * omegac * \
    #     (1 / (omega1 - omegac) + 1 / (omega2 - omegac)) * 0.5
    # g = g12 + gic

    return shift1

def xy_fun(fq, anharm, tg, fn=None, mu=None, lambdas=None):
    chip = nx.Graph()
    chip.add_node((0, 0))
    chip.nodes[(0, 0)]['freq'] = fq
    tq = [(0, 0)]
    if lambdas is None:
        lambdas = calibration(chip, tq, tg, energyLevel=3, anharmq=anharm)
        return lambdas
    if isinstance(fn, (float, int)):
        err = par_evolution(chip, tq, tg, 3, anharm, lambdas, fn, mu)
    else:
        err = []
        for f in fn:
            err.append(par_evolution(chip, tq, tg, 3, anharm, lambdas, f, mu))
    return err

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

    fq = 4.5
    anharm = -0.2
    detune = np.linspace(-1, 1, 1000)
    fn = fq - detune
    mu = np.linspace(0.001, 1, 1000)
    tq = 30
    lambdas = xy_fun(fq, anharm, tq)
    fns = []
    mus = []
    for f in fn:
        for m in mu:
            fns.append(f)
            mus.append(m)
    fqs = [fq] * len(fn) * len(mu)
    anharms = [anharm] * len(fn) * len(mu)
    tqs = [tq] * len(fn) * len(mu)
    lambdases = [lambdas] * len(fn) * len(mu)
    p = Pool(20)
    err = p.starmap(xy_fun, zip(fqs, anharms, tqs, fns, mus, lambdases))
    p.close()
    p.join()
    with open('xy data.txt', 'w') as fp:
        for d in err:
            fp.write(str(d) + '\n')
    draw_heat_map(detune, mu, err, 'xy data.pdf', 'xy data.pdf', xlabel='detune(GHz)', ylabel='mu', drawtype='logabs')