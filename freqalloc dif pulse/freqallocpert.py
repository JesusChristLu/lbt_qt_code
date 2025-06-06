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

def H_time(tList, args):
    inducedChip = args['induced chip']
    subStatesId = args['sub states id']
    qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    g_qc = args['g qc']
    g_qq = args['g qq']
    energyLevel = args['energy level']
    anharmq, anharmc = args['qt anharm'][0], args['qt anharm'][1]
    freqcWork = args['freqcWork']
    freqqWork = args['freqqWork']
    sigmac = args['sigmac']
    sigmaq = args['sigmaq']
    qNum = len(inducedChip.nodes)
    cNum = len(inducedChip.edges)


    aList, aDagList, _, _, _ = getaad(energyLevel, qNum + cNum)

    HAnharm = []
    H0 = []
    Hi = []
    for q in range(qNum):
        mat = find_sub_matrix(aDagList[q] * aDagList[q] * aList[q] * aList[q], subStatesId)
        HAnharm.append(np.pi * anharmq * mat)
        freqqList = pulse_fun(tList, tList[-1], sigmaq[q], freqqWork[q], inducedChip.nodes[qubitList[q]]['freq'])
        mat = find_sub_matrix(aDagList[q] * aList[q], subStatesId)
        if isinstance(freqqList, (int, float)):
            H0.append(freqqList * np.pi * 2 * mat)
        else:
            H0.append([mat, freqqList * np.pi * 2])
    for c in range(qNum, qNum + cNum):
        mat = find_sub_matrix(aDagList[c] * aDagList[c] * aList[c] * aList[c], subStatesId)
        HAnharm.append(np.pi * anharmc * mat)
        freqcList = pulse_fun(tList, tList[-1], sigmac[c - qNum], freqcWork[c - qNum], inducedChip.edges[qubitList[c]]['freq'])
        mat = find_sub_matrix(aDagList[c] * aList[c], subStatesId)
        if isinstance(freqcList, (int, float)):
            H0.append(freqcList * np.pi * 2 * mat)
        else:
            H0.append([mat, freqcList * np.pi * 2])

    for edge in inducedChip.edges:
        q0 = qubitList.index(edge[0])
        q1 = qubitList.index(edge[1])
        c = qubitList.index(edge)
        mat0c = find_sub_matrix((aDagList[q0] * aList[c] + aList[q0] *aDagList[c]), subStatesId)
        mat1c = find_sub_matrix((aDagList[q1] * aList[c] + aList[q1] *aDagList[c]), subStatesId)
        mat01 = find_sub_matrix((aDagList[q0] * aList[q1] + aList[q0] *aDagList[q1]), subStatesId)
        freqcList = pulse_fun(tList, tList[-1], sigmac[c - qNum], freqcWork[c - qNum], inducedChip.edges[qubitList[c]]['freq'])
        freq0List = pulse_fun(tList, tList[-1], sigmaq[q0], freqqWork[q0], inducedChip.nodes[qubitList[q0]]['freq'])
        freq1List = pulse_fun(tList, tList[-1], sigmaq[q1], freqqWork[q1], inducedChip.nodes[qubitList[q1]]['freq'])
        g_q0c = g_qc * np.sqrt(freq0List * freqcList)
        g_q1c = g_qc * np.sqrt(freq1List * freqcList)
        g_q0q1 = g_qq * np.sqrt(freq0List * freq1List)

        if isinstance(g_q0c, (int, float)):
            Hi.append(np.pi * 2 * g_q0c * mat0c)
        else:
            Hi.append([mat0c, np.pi * 2 * g_q0c])
        if isinstance(g_q1c, (int, float)):
            Hi.append(np.pi * 2 * g_q1c * mat1c)
        else:
            Hi.append([mat1c, np.pi * 2 * g_q1c])
        if isinstance(g_q0q1, (int, float)):
            Hi.append(np.pi * 2 * g_q0q1 * mat01)
        else:
            Hi.append([mat01, np.pi * 2 * g_q0q1])
    return [*H0, *HAnharm, *Hi]

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
    alpha = args['qt anharm'][0]
    detune = args['qt detune']
    lambda0 = args['drag weight']
    phi = args['qt phi']
    X = []
    for q in range(len(w_d)):
        I = np.cos(phi[q])
        Q = np.sin(phi[q])
        t1 = []
        t2 = []
        for t in tList:
            if t < tg:
                t1.append(t)
            else:
                t2.append(tg)
        tList = np.array(t1 + t2)
        X0 = amp[q] * (1 - np.cos(2 * np.pi * tList / tg)) / 2 
        Y0 = -amp[q] * np.pi / tg * np.sin(2 * np.pi * tList / tg) / alpha * lambda0
        X.append((X0 * I + Y0 * Q) * np.cos((w_d[q] * np.pi * 2 + detune[q]) * tList))
    return X

def drive_pulseYt(tList, args):  
    tg = args['gate time']
    amp = args['qt amp']
    w_d = args['qt drive frequency']
    alpha = args['qt anharm'][0]
    detune = args['qt detune']
    lambda0 = args['drag weight']
    phi = args['qt phi']
    X = []
    for q in range(len(w_d)):
        I = np.cos(phi[q])
        Q = np.sin(phi[q])
        t1 = []
        t2 = []
        for t in tList:
            if t < tg:
                t1.append(t)
            else:
                t2.append(tg)
        tList = np.array(t1 + t2)
        X0 = amp[q] * (1 - np.cos(2 * np.pi * tList / tg)) / 2 
        Y0 = -amp[q] * np.pi / tg * np.sin(2 * np.pi * tList / tg) / alpha * lambda0
        X.append((Y0 * I - X0 * Q) * np.cos((w_d[q] + detune[q]) * tList))
    return X

def xyxtalk(t, args):
    tg = args['gate time']
    amp = args['qt amp']
    w_d = args['qt drive frequency']
    mu = args['mu']
    X0 = mu * amp * (1 - np.cos(2 * np.pi * t / tg)) / 2 
    X = X0 * np.cos(w_d * t) 
    return X
  
def fidelity(phi, UEff, UIdeal, computeBas=None, allBas=None):
    # print(phi)
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

def evolution(lambdas, states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, cq, g_qq, g_qc, excitedN, 
              xyXtalk=None, nbState=None, last=False, draw=False):
    para = deepcopy(lambdas)
    lambdas = list(lambdas[::-1])
    qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    qubitNum = len(qubitList)

    tg = 30
    for g in tq:
        if g in inducedChip.edges:
            tg = 60
            break
    tList = np.arange(0, tg, 0.1)

    args = dict() 
    args['induced chip'] = inducedChip
    args['sub states id'] = subStatesId
    args['target edge'] = tq
    args['g qc'] = g_qc
    args['g qq'] = g_qq
    args['energy level'] = energyLevel
    args['qt anharm'] = [anharmq, anharmc]
    args['gate time'] = 30
    args['drag weight'] = 0.5

    args['sigmac'] = []
    args['sigmaq'] = []
    args['freqcWork'] = []
    args['freqqWork'] = []
    args['qt drive frequency'] = []
    args['qt detune'] = []
    args['qt amp'] = []

    _, _, sxList, syList, _ = getaad(energyLevel, qubitNum)

    H_xt = []
    H_yt = []

    npsx = []
    npsy = []

    for q in inducedChip.nodes:
        args['freqqWork'].append(inducedChip.nodes[q]['freq'])
        args['sigmaq'].append(0)
        if q in tq:
            args['qt drive frequency'].append(inducedChip.nodes[q]['freq'])
            args['qt detune'].append(lambdas.pop())
            args['qt amp'].append(lambdas.pop())

            npsx.append(find_sub_matrix(sxList[qubitList.index(q)], subStatesId))
            npsy.append(find_sub_matrix(syList[qubitList.index(q)], subStatesId))

    for c in inducedChip.edges:
        if c in tq or c[::-1] in tq:
            args['freqcWork'].append(lambdas.pop())
            sig = []
            for _ in range(1):
                sig.append(lambdas.pop())
            args['sigmac'].append(tuple(sig))
            if inducedChip.nodes[c[0]]['freq'] > inducedChip.nodes[c[1]]['freq']:
                highFreqQ = qubitList.index(c[0])
            else:
                highFreqQ = qubitList.index(c[1])
            args['freqqWork'][highFreqQ] = lambdas.pop()
            sig = []
            for _ in range(1):
                sig.append(lambdas.pop())
            args['sigmaq'][highFreqQ] = tuple(sig)
        
        elif not(c in cq or c[::-1] in cq):
            args['freqcWork'].append(lambdas.pop())
            args['sigmac'].append(tuple([lambdas.pop()]))
            
        else:
            args['sigmac'].append(0)
            args['freqcWork'].append(inducedChip.edges[c]['freq'])

    args['qt phi'] = list(-np.array(args['qt detune']) * args['gate time'] * np.pi)

    xt = drive_pulseXt(tList, args=args)
    yt = drive_pulseYt(tList, args=args)

    Ht = H_time(tList, args=args)
    xtid = 0
    for q in inducedChip.nodes:
        if q in tq:
            H_xt.append([npsx[xtid], xt[xtid]])

            H_yt.append([npsy[xtid], yt[xtid]])
            xtid += 1
    Ht = [*Ht, *H_xt, *H_yt]

    U_fulls = propagator(Ht, tList, parallel=True, num_cpus=20)
    # U_fulls = propagator(Ht, tList, parallel=False)

    U_full = U_fulls[-1]

    gIdeal = None
    oneBit = []
    for gate in tq:
        if gate in inducedChip.nodes:
            if gIdeal == None:
                gIdeal = tensorOperator(2, sigmax(), qubitList.index(gate), qubitNum)
            else:
                gIdeal = tensorOperator(2, sigmax(), qubitList.index(gate), qubitNum) * gIdeal
            oneBit.append(gate)
        else:
            q1 = qubitList.index(gate[0])
            q2 = qubitList.index(gate[1])

            if gIdeal == None:
                gIdeal = cphase(theta=np.pi, N=qubitNum, 
                control=qubitList[::-1].index(gate[0]), 
                target=qubitList[::-1].index(gate[1]))
            else:
                gIdeal = gIdeal * cphase(theta=np.pi, N=qubitNum, 
                control=qubitList[::-1].index(gate[0]), 
                target=qubitList[::-1].index(gate[1]))

            if not(q1 in oneBit):
                oneBit.append(qubitList[q1])
                
            if not(q2 in oneBit):
                oneBit.append(qubitList[q2])
                
    kets = []
    idealKets = []
    computeBas = []
    for bas in state_number_enumerate([2] * qubitNum, excitedN):
        leakBas = False
        for b in range(len(bas)):
            if last:
                leakFlag = not(qubitList[b] in inducedChip.nodes)
            else:
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
    print(para, error)

    if last:
        amppU = np.zeros(shape=U.shape)
        angleU = np.zeros(shape=U.shape)
        start = 0
        for i in phi:
            if start == 0:
                Rz = rz(i)
                start = 1
            else:
                Rz = tensor([rz(i), Rz])
        Rz = np.array(Rz)
        subStatessId = dict()
        for ab in allBas:
            for cpb in computeBas:
                if ab == cpb[-len(phi):]:
                    subStatessId[allBas.index(ab)] = computeBas.index(cpb)
                    break
        Rz = find_sub_matrix(Rz, subStatessId)
        pU = np.dot(Rz, U) 
        # pU = U
        for ui in range(pU.shape[0]):
            for uj in range(pU.shape[1]):
                if np.abs(pU[ui, uj]) < 1e-2:
                    amppU[ui, uj] = 0
                    angleU[ui, uj] = 0
                else:
                    amppU[ui, uj] = round(np.abs(pU[ui, uj]), 4)
                    angleU[ui, uj] = round(cmath.phase(pU[ui, uj]), 4)
        print('amp')
        print(amppU.diagonal())
        # for uu in amppU:
            # print(uu)
        print('u angle')
        print(angleU.diagonal())
        # for uu in angleU:
            # print(uu)
        print('u ideal')
        print(UIdeal.diagonal())
        # print(UIdeal)
        print('this is the last par evolution', para, error)

    if draw:
        qgate = []
        for gate in tq:
            if gate in inducedChip.nodes:
                qgate.append(gate)
            else:
                if not(gate[0] in qgate):
                    qgate.append(gate[0])
                if not(gate[1] in qgate):
                    qgate.append(gate[1])

        qubitList = list(inducedChip.nodes) + list(inducedChip.edges)

        for computebas in state_number_enumerate([2] * len(qgate), 1 * len(qgate)):
            if nbState is None:
                s = np.zeros(qubitNum, dtype=int)
            else:
                s = list(nbState)[::-1]
            id = 0
            for i in qgate:
                s[qubitList.index(i)] = computebas[::-1][id]
                id += 1

            plt.figure()
            idket = states[subStatesId[state_number_index([energyLevel] * qubitNum, s[::-1])]]
            amp_sum = np.zeros(len(tList))
            for bas in state_number_enumerate([energyLevel] * qubitNum, energyLevel * qubitNum):
                if (state_number_index([energyLevel] * qubitNum, bas)) in subStatesId:
                    amp = []
                    idbra = states[subStatesId[state_number_index([energyLevel] * qubitNum, bas)]].dag()
                    lab = ''
                    for b in bas:
                        lab += str(b)
                    for U_full in U_fulls:
                        amp.append(np.abs((idbra * U_full * idket).full()[0][0]) ** 2)
                    for gg in tq:
                        if isinstance(gg[0], tuple):
                            gh = qubitList.index(gg[0])
                            gl = qubitList.index(gg[1])
                            g = [gh, gl]
                        else:
                            g = qubitList.index(gg)
                            g = [g]
                    if amp[-1] > 1e-3 or amp[0]> 1e-3:
                        plt.plot(tList, amp, label=lab + 'end leak')
                        print(lab + 'end leak', amp[-1])
                        
                    elif max(amp) - min(amp) > 1e-2:
                        plt.plot(tList, amp, label=lab + 'leak')
                        print(lab + 'leak', max(amp) - min(amp))
                    amp_sum += amp
            plt.plot(tList, amp_sum, label='sum')
            plt.legend()
            plt.savefig(str(tq) + str(s[::-1]) + '.pdf', dpi=300)
            plt.close()
    return error
    
def par_evolution(inducedChip, tq, cq, energyLevel, anharmq, anharmc, g_qq, g_qc, lambdas, nbState, draw=True):
    calibratedG = tq
    print(calibratedG)
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

    H0, Hi = H(inducedChip, energyLevel, anharmq, anharmc, g_qq, g_qc)
    subStatesId = find_sub_space(states, energyLevel, qubitNum)

    H0 = find_sub_matrix(H0, subStatesId)
    Hi = find_sub_matrix(Hi, subStatesId)         

    states, _, _, _ = eigensolve(H0, H0 + Hi)
    err = evolution(lambdas, states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, cq, g_qq, g_qc, 
    excitedN, nbState, last=True, draw=draw)
    print(calibratedG, nbState, err)
    return err

def calibration(inducedChip, tq, cq, energyLevel, anharmq, anharmc, g_qq, g_qc):
    calibratedG = tq
    print(calibratedG)
    qubitNum = len(inducedChip.nodes) + len(inducedChip.edges)
    qubitList = list(inducedChip.nodes) + list(inducedChip.edges)
    excitedN = len(inducedChip.nodes)
    xNum = 0
    for gg in calibratedG:
        if gg in inducedChip.nodes:
            xNum += 1

    states = []
    for n in range(excitedN + xNum + 1):
        for exciteds in state_number_enumerate([energyLevel] * qubitNum, n):
            states.append(exciteds)

    H0, Hi = H(inducedChip, energyLevel, anharmq, anharmc, g_qq, g_qc)
    subStatesId = find_sub_space(states, energyLevel, qubitNum)

    H0 = find_sub_matrix(H0, subStatesId)
    Hi = find_sub_matrix(Hi, subStatesId)         

    states, _, _, _ = eigensolve(H0, H0 + Hi)

    bounds = []
    lambdas = []
    for qubit in qubitList:
        if qubit in inducedChip.nodes:
            if qubit in tq:
                lambdas.append(0)
                bounds.append((-1, 1))
                lambdas.append(0.5)
                bounds.append((0, 2))
        elif qubit in inducedChip.edges:
            if qubit in tq or qubit[::-1] in tq:
                freqqWork = min(inducedChip.nodes[qubit[0]]['freq'], inducedChip.nodes[qubit[1]]['freq']) - anharmq
                freqcWork = freqqWork + 0.5
                lambdas.append(freqcWork)
                bounds.append((4, 6))
                lambdas.append(1)
                bounds.append((-2, 4))
                lambdas.append(freqqWork)
                bounds.append((freqqWork - 0.1, freqqWork + 0.1))
                lambdas.append(1)
                bounds.append((-2, 4))
            elif not(qubit in cq or qubit[::-1] in cq):
                lambdas.append(inducedChip.edges[qubit]['freq'])
                bounds.append((5, 8))
                for _ in range(1):
                    lambdas.append(1)
                    bounds.append((-2, 4))                   

    lambdas = [4.80589049, 0.97697981, 4.33315145, 1.02350428, 6.7, 1.0, 6.7, 1.0, 4.78135607, 1.08824515, 4.32307729, 0.98804162]

    methods = ['BFGS', 'CG', 'Nelder-Mead', 'Powell']
    result = minimize(evolution, lambdas, 
    args=(states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, cq, g_qq, g_qc, excitedN), 
        method=methods[2], bounds=bounds, options={'ftol' : 1e-3})
    lambdas = result.x

    err = evolution(lambdas, states, subStatesId, inducedChip, energyLevel, anharmq, anharmc, tq, cq, g_qq, g_qc, 
    excitedN, state_index_number([2] * qubitNum, 0), last=True)
    return lambdas, err

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

# def D_factor():

    # return
    
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

def second_object(omegas, chip, rho_qq, rho_qc, energyLevel, anharmq, anharmc):
    cost = 0
    for coupler in chip.edges:
        chip.edges[coupler]['freq'] = omegas[0]
    if chip.nodes[list(chip.nodes)[0]]['freq'] > chip.nodes[list(chip.nodes)[1]]['freq']:
        tq = list(chip.nodes)[0]
    else:
        tq = list(chip.nodes)[1]
    zz = np.abs(zzcoupling(chip, tq, rho_qq, rho_qc, energyLevel, anharmq, anharmc) * 1e3)
    # print(omegas, zz)
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

def in_compute_bas(bas, tq, neighbor):
    bas = bas[::-1]
    neighbor = neighbor[::-1]
    for i in range(len(bas)):
        if not(i in tq):
            if not(bas[i] == neighbor[i]):
                return False
        elif bas[i] > 1:
            return False
    return True

if __name__ == '__main__':
    w = 1
    h = 3
    energyLevel = 3
    anharmq = -0.22

    # 搜索最优化耦合系数
    # offDetune = 0.4
    # onDetune = -anharmq
    # nearOnDetune = -anharmq - 0.01

    # freq2 = 5.6
    # freq1 = freq2 + nearOnDetune

    # print(freq1, freq2)

    # # cFreqLow = max([freq1, freq2]) + 0.01
    # # cFreqHigh = 7
    # cFreqLow = 3
    # cFreqHigh = min([freq1, freq2]) - 0.01

    # # rho_qcs = np.arange(0.01, 0.03, 0.001)
    # # rho_qqs = np.arange(0.001, 0.0025, 0.0001)
    # rho_qcs = np.arange(0.01, 0.04, 0.001)
    # rho_qqs = np.arange(-0.001, -0.0025, -0.0001)

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
    #                 chip.nodes[(0, 0)]['freq'] = freq1
    #                 chip.nodes[(0, 0)]['type'] = 0
    #                 chip.nodes[(0, 1)]['freq'] = freq2
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
    # poolNum = 20
    # print('start', poolNum)
    # p = Pool(poolNum)
    # shift = p.starmap(zzcoupling, zip(chips, tqs, rho_qqss, rho_qcss, energyLevels, anharmqs, anharmcss))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # shift11 = []
    # for i in range(len(shift)):
    #     shift11.append(shift[i] * 1e3)

    # # write_data(str(freq1)[:3] + ' ' + str(freq2)[:3] + ' shift11 p.txt', shift11)
    # write_data(str(freq1)[:3] + ' ' + str(freq2)[:3] + ' shift11 n.txt', shift11)

    # BAQ优化参数
    rho_qc = 0.026
    rho_qq = 0.0017
    anharmc = -0.1
    # BAQ优化参数2
    # rho_qc = 0.031
    # rho_qq = 0.0034
    # anharmc = -0.2
    # BBQ优化参数
    # rho_qc = 0.025
    # rho_qq = -0.001
    # anharmc = -0.1

    # 计算shift

    # 改变omegac和omegaq

    # highFreq = 4.45
    # lowFreq = highFreq - 0.4
    # # tqFreq = lowFreq - anharmq
    # tqFreq = highFreq

    # sFreqLow = lowFreq
    # sFreqHigh = highFreq
    
    # # cFreqLow = tqFreq + 0.01
    # cFreqLow = 6
    # cFreqHigh = 7
    # # cFreqLow = 3
    # # cFreqHigh = lowFreq - 0.01

    # sFreqs = np.arange(sFreqLow, sFreqHigh, 0.002)
    # cFreqs = np.arange(cFreqLow, cFreqHigh, 0.02)

    # chips = []
    # for sFreq in sFreqs:
    #     for cFreq in cFreqs:
    #         chip = nx.Graph()
    #         chip.add_nodes_from([(0, 0), (0, 1)])
    #         chip.nodes[(0, 0)]['freq'] = tqFreq
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
    # p = Pool(10)
    # singleShift = p.starmap(zzcoupling, zip(chips, tqs, rho_qqs, rho_qcs, energyLevels, anharmqs, anharmcs))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # shift = []
    # for i in range(len(singleShift)):
    #     shift.append(singleShift[i] * 1e3)
    # write_data(str(tqFreq)[:3] + ' single shift p.txt', shift) 
    # # write_data(str(tqFreq)[:3] + ' single shift n.txt', shift) 

    # 改变omegac和rho12
    # tqFreq = 5
    # spFreq = tqFreq - 0.1
    # # cFreqLow = tqFreq + 0.01
    # # cFreqHigh = 7
    # cFreqLow = 3
    # cFreqHigh = spFreq - 0.01
    # cFreqs = np.arange(cFreqLow, cFreqHigh, 0.02)
    # rho12Low = -0.003
    # rho12High = -0.001
    # rho12s = np.arange(rho12Low, rho12High, 0.0001)
    # chips = []
    # rho_qqs = []
    # for rho12 in rho12s:
    #     for cFreq in cFreqs:
    #         chip = nx.Graph()
    #         chip.add_nodes_from([(0, 0), (0, 1)])
    #         chip.nodes[(0, 0)]['freq'] = tqFreq
    #         chip.nodes[(0, 0)]['type'] = 0
    #         chip.nodes[(0, 1)]['freq'] = spFreq
    #         chip.nodes[(0, 1)]['type'] = 1
    #         chip.add_edge((0, 0), (0, 1))
    #         chip.edges[((0, 0), (0, 1))]['freq'] = cFreq
    #         chips.append(chip)
    #         rho_qqs.append(rho12)
    
    # energyLevels = [energyLevel] * len(rho12s) * len(cFreqs)
    # anharmqs = [anharmq] * len(rho12s) * len(cFreqs)
    # anharmcs = [anharmc] * len(rho12s) * len(cFreqs)
    # rho_qcs = [rho_qc] * len(rho12s) * len(cFreqs)
    # tqs = [(0, 0)] * len(rho12s) * len(cFreqs)

    # tStart = time.time()
    # p = Pool(100)
    # singleShift = p.starmap(zzcoupling, zip(chips, tqs, rho_qqs, rho_qcs, energyLevels, anharmqs, anharmcs))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # g = []
    # shift = []
    # for i in range(len(singleShift)):
    #     shift.append(singleShift[i][0] * 1e3)
    #     g.append(singleShift[i][2] * 1e3)
    # name = 'rho omegac' + str(np.abs(tqFreq - spFreq))[:5] + \
    #  str(rho_qc)[:5] + '.txt'
    # write_data(name, shift)

    # 改变freq1和detune
    # freq1s = np.arange(4, 6, 0.01)
    # detunes = np.arange(0.04, 0.6, 0.01)
    # chips = []
    # energyLevels = []
    
    # anharmcs = []
    # anharmqs = []
    # tqs = []
    # rho_qqs = []
    # rho_qcs = []
    # for freq1 in freq1s:
    #     freq2s = freq1 - detunes
    #     for freq2 in freq2s:
    #         freqcs = np.arange(max(freq1, freq2), 7.1, 0.02)
    #         # freqcs = np.arange(2.9, min(freq1, freq2), 0.02)
    #         for cFreq in freqcs:
    #             chip = nx.Graph()
    #             chip.add_nodes_from([(0, 0), (0, 1)])
    #             chip.nodes[(0, 0)]['freq'] = freq1
    #             chip.nodes[(0, 0)]['type'] = 0
    #             chip.nodes[(0, 1)]['freq'] = freq2
    #             chip.nodes[(0, 1)]['type'] = 1
    #             chip.add_edge((0, 0), (0, 1))
    #             chip.edges[((0, 0), (0, 1))]['freq'] = cFreq
    #             chips.append(chip)
    #             energyLevels.append(energyLevel)
    #             anharmcs.append(anharmc)
    #             anharmqs.append(anharmq)
    #             tqs.append((0, 0))
    #             rho_qqs.append(rho_qq)
    #             rho_qcs.append(rho_qc)

    # tStart = time.time()
    # poolNum = 20
    # print('start', poolNum)
    # p = Pool(poolNum)
    # shift = p.starmap(zzcoupling, zip(chips, tqs, rho_qqs, rho_qcs, energyLevels, anharmqs, anharmcs))
    # p.close()
    # p.join()
    # t = time.time() - tStart
    # print(t)

    # shift11 = []
    # for i in range(len(shift)):
    #     shift11.append(shift[i] * 1e3)

    # write_data('freq shift11 p.txt', shift11)
    # # write_data('freq shift11 n.txt', shift11)

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
    # freq0 = 5
    # freq1 = freq0 - 0.6
    # # cFreqs = np.arange(max([freq0, freq1]), 7, 0.01)
    # cFreqs = np.arange(3, max([freq0, freq1]), 0.01)
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
    #             if np.real(state0[st][0][0]) == 1
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

    # plt.subplot(2, 1, 1)

    # i = 0
    # for lab in list(energyDict.keys())[1:]:
    #     if 1:
    #         plt.plot(cFreqs, np.array(energyDict[lab]) * 1e3, label=lab, color=colors[i])
    #         plt.plot(cFreqs, np.array(energy0Dict[lab]) * 1e3, linestyle='--', color=colors[i])
    #         i += 1
    # plt.legend()

    # plt.subplot(2, 1, 2)

    # # plt.plot(cFreqs, np.abs(np.array(energyDict['011']) - energy0Dict['011']) * 1e3, label='011')
    # # plt.plot(cFreqs, np.abs(np.array(energyDict['001']) - energy0Dict['001']) * 1e3, label='001')
    # # plt.plot(cFreqs, np.abs(np.array(energyDict['010']) - energy0Dict['010']) * 1e3, label='010')
    # # plt.plot(cFreqs, np.abs(np.array(energyDict['101']) - energy0Dict['101']) * 1e3, label='101')
    # # plt.plot(cFreqs, np.abs(np.array(energyDict['110']) - energy0Dict['110']) * 1e3, label='110')
    # # plt.plot(cFreqs, np.abs(np.array(energyDict['200']) - energy0Dict['200']) * 1e3, label='200')
    # plt.plot(cFreqs, np.abs((np.array(energyDict['011'])) - \
    #                         (np.array(energyDict['001'])) - \
    #                         (np.array(energyDict['010']))) * 1e3, label='zz')
    # print((np.array(energyDict['011']) - energy0Dict['011'])[0] * 1e3, 
    #       (np.array(energyDict['001']) - energy0Dict['001'])[0] * 1e3,
    #       (np.array(energyDict['010']) - energy0Dict['010'])[0] * 1e3,
    #       (np.array(energyDict['101']) - energy0Dict['101'])[0] * 1e3,
    #       (np.array(energyDict['110']) - energy0Dict['110'])[0] * 1e3,
    #       (np.array(energyDict['200']) - energy0Dict['200'])[0] * 1e3,
    #       (np.array(energyDict['011']) - energy0Dict['011'])[0] * 1e3 - 
    #       (np.array(energyDict['001']) - energy0Dict['001'])[0] * 1e3 - 
    #       (np.array(energyDict['010']) - energy0Dict['010'])[0] * 1e3)

    # plt.semilogy()
    # plt.legend()
    # plt.show()

    # 非共振驱动xy串扰

    # 单比特门工作点分配

    # qlow = 4.5
    # qhigh = 5.0
    # clow = 5.5
    # chigh = 7
    # # clow = 2.7
    # # chigh = 3.5
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
    #     if sum(qubit) % 2:
    #         chip.nodes[qubit]['freq'] = round(4.5 + 0.05 * np.random.random(), 3)
    #     else:
    #         chip.nodes[qubit]['freq'] = round(4.1 + 0.05 * np.random.random(), 3)
    # while 1:
    #     noColli = True
    #     for qubit in qubitList:
    #         while 1:
    #             neighborFreq = []
    #             colliqubit = []
    #             for nb in chip[qubit]:
    #                 if not(chip.nodes[nb]['freq'] in neighborFreq):
    #                     neighborFreq.append(chip.nodes[nb]['freq'])
    #                 else:
    #                     colliqubit.append(nb)
    #             if len(neighborFreq) == len(chip[qubit]):
    #                 break
    #             else:
    #                 noColli = False
    #                 for cqb in colliqubit:
    #                     if sum(cqb) % 2:
    #                         chip.nodes[cqb]['freq'] = round(4.5 + 0.05 * np.random.random(), 3)
    #                     else:
    #                         chip.nodes[cqb]['freq'] = round(4.1 + 0.05 * np.random.random(), 3)
    #     if noColli:
    #         break

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

    # 单比特门

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

    # 两比特门


    # 观察比特

    # freq0 = 4.0
    # freq1 = 4.5
    # freqs = [4.1]
    # # freqs = np.arange(freq1 - 0.1, freq1 + 0.1, 0.005)
    # # freqs = np.arange(freq1 - anharmq - 0.1, freq1 - anharmq + 0.1, 0.005)
    # # freqs = np.arange(freq0 + anharmq - 0.1, freq0 + anharmq + 0.1, 0.005)
    # # freqs = np.arange(freq0 - 0.1, freq0 + 0.1, 0.005)
    # # freqs = np.arange(freq0 - anharmq - 0.1, freq0 - anharmq + 0.1, 0.005)

    # coupler1 = ((0, 0), (0, 1))
    # coupler2 = ((0, 1), (0, 2))
    # inducedChips = []

    # for freq in freqs:
    #     chip1 = nx.Graph()
    #     chip1.add_edge(coupler1[0], coupler1[1])
    #     chip1.nodes[coupler1[0]]['freq'] = freq0
    #     chip1.nodes[coupler1[1]]['freq'] = freq1
    #     res = minimize(second_object, [7], args=(chip1, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #         bounds=((5, 8),), method='Powell', options={'maxiter' : 50})
    #     chip1.edges[coupler1]['freq'] = res.x[0]
    #     chip2 = nx.Graph()
    #     chip2.add_edge(coupler2[0], coupler2[1])
    #     chip2.nodes[coupler2[0]]['freq'] = freq1
    #     chip2.nodes[coupler2[1]]['freq'] = freq
    #     res2 = minimize(second_object, [7], args=(chip2, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
    #         bounds=((5, 8),), method='Powell', options={'maxiter' : 50})
    #     chip2.edges[coupler2]['freq'] = res2.x[0]
    #     inducedChip = nx.Graph()
    #     inducedChip.add_edges_from([coupler1, coupler2])
    #     inducedChip.nodes[coupler1[0]]['freq'] = freq0
    #     inducedChip.nodes[coupler1[1]]['freq'] = freq1
    #     inducedChip.nodes[coupler2[1]]['freq'] = freq
    #     inducedChip.edges[coupler1]['freq'] = chip1.edges[coupler1]['freq']
    #     inducedChip.edges[coupler2]['freq'] = chip2.edges[coupler2]['freq']
    #     inducedChips.append(inducedChip)

    # tqs = [[coupler1]] * len(freqs)
    # cqs = [[coupler2]] * len(freqs)
    # energyLevels = [energyLevel] * len(freqs)
    # anharmqs = [anharmq] * len(freqs)
    # anharmcs = [anharmc] * len(freqs)
    # rho_qqs = [rho_qq] * len(freqs)
    # rho_qcs = [rho_qc] * len(freqs)

    # calibratedTwoPara = []
    # for inducedChip in inducedChips:
    #     calibratedTwoPara.append(calibration(inducedChip, [coupler1], [coupler2], energyLevel, anharmq, anharmc, rho_qq, rho_qc))
    #     # calibratedTwoPara.append(calibration(inducedChip, [coupler1], [], energyLevel, anharmq, anharmc, rho_qq, rho_qc))

    # write_data('flat gaussian two Q para cq.txt', calibratedTwoPara)
    # # write_data('flat gaussian two Q para noncq.txt', calibratedTwoPara)

    # 校准

    freq0 = 4.1
    freq1 = 4.5
    freq2 = 4.09
    freq3 = 4.51
    
    coupler1 = ((0, 0), (0, 1))
    coupler2 = ((0, 1), (0, 2))
    coupler3 = ((0, 2), (0, 3))
    coupler4 = ((0, 3), (0, 0))
   
    inducedChips = []

    chip1 = nx.Graph()
    chip1.add_edge(coupler1[0], coupler1[1])
    chip1.nodes[coupler1[0]]['freq'] = freq0
    chip1.nodes[coupler1[1]]['freq'] = freq1
    res = minimize(second_object, [7], args=(chip1, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
        bounds=((5, 8),), method='Powell', options={'maxiter' : 50})
    chip1.edges[coupler1]['freq'] = res.x[0]

    chip2 = nx.Graph()
    chip2.add_edge(coupler2[0], coupler2[1])
    chip2.nodes[coupler2[0]]['freq'] = freq1
    chip2.nodes[coupler2[1]]['freq'] = freq2
    res = minimize(second_object, [7], args=(chip2, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
        bounds=((5, 8),), method='Powell', options={'maxiter' : 50})
    chip2.edges[coupler2]['freq'] = res.x[0]

    chip3 = nx.Graph()
    chip3.add_edge(coupler3[0], coupler3[1])
    chip3.nodes[coupler3[0]]['freq'] = freq2
    chip3.nodes[coupler3[1]]['freq'] = freq3
    res = minimize(second_object, [7], args=(chip3, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
        bounds=((5, 8),), method='Powell', options={'maxiter' : 50})
    chip3.edges[coupler3]['freq'] = res.x[0]

    chip4 = nx.Graph()
    chip4.add_edge(coupler4[0], coupler4[1])
    chip4.nodes[coupler4[0]]['freq'] = freq3
    chip4.nodes[coupler4[1]]['freq'] = freq0
    res = minimize(second_object, [7], args=(chip4, rho_qq, rho_qc, energyLevel, anharmq, anharmc), 
        bounds=((5, 8),), method='Powell', options={'maxiter' : 50})
    chip4.edges[coupler4]['freq'] = res.x[0]

    chip = nx.Graph()
    chip.add_edges_from([coupler1, coupler2])
    chip.nodes[coupler1[0]]['freq'] = freq0
    chip.nodes[coupler1[1]]['freq'] = freq1
    chip.nodes[coupler2[1]]['freq'] = freq2  
    chip.edges[coupler1]['freq'] = chip1.edges[coupler1]['freq']
    chip.edges[coupler2]['freq'] = chip2.edges[coupler2]['freq']

    chip.add_edges_from([coupler3, coupler4])
    chip.nodes[coupler3[1]]['freq'] = freq3
    chip.edges[coupler3]['freq'] = chip3.edges[coupler3]['freq']
    chip.edges[coupler4]['freq'] = chip4.edges[coupler4]['freq']

    # tq = [((0, 0), (0, 1))]
    # cq = [((0, 1), (0, 2))]
    # cq = []

    tq = [((0, 0), (0, 1)), ((0, 2), (0, 3))]
    # cq = [((0, 1), (0, 2)), ((0, 3), (0, 0))]
    cq = []
    
    # print(tq, cq)
    
    calibratedData = calibration(chip, tq, cq, energyLevel, anharmq, anharmc, rho_qq, rho_qc)
    write_data(str(tq) + str(cq) + '.txt', calibratedData)
    
    # lambdas = [5.13452517, 0.99880288, 4.32349281, 0.99213819]
    # lambdas = [4.81996711, 0.87224673, 4.33972377, 1.02995432, 6.80336904, 1.18508736]
    # U = par_evolution(chip, tq, cq, energyLevel, anharmq, anharmc, rho_qq, rho_qc, lambdas, [0, 0, 0, 0, 0, 0, 0, 0])

    # lambdas = [4.80589049, 0.97697981, 4.33315145, 1.02350428, 4.78135607, 1.08824515, 4.32307729, 0.98804162]
    # lambdas = [4.81090059, 0.84437136, 4.33251504, 1.12951502, 6.7297056, 0.97520526,
    #             6.75644984, 0.93409463, 4.77512765, 1.15222774, 4.32307592, 1.07210715]
    # U = par_evolution(chip, tq, cq, energyLevel, anharmq, anharmc, rho_qq, rho_qc, lambdas, [0, 0, 0, 0, 0, 0, 0, 0])