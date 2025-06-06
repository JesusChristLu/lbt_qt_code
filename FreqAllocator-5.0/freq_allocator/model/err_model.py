import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .formula import freq2amp_formula, eff_g, freq_var_map, lorentzain

MUTHRESHOLD = 0.01
amp = 1

def T1_err(a, f, tq, t1_spectrum):
    try:
        error = a * tq / t1_spectrum(f)
    except:
        error = 5e-4
    if error < 0:
        error = 5e-4
    return error

def T2_err(a, f, tq, t2_spectrum: dict = None, ac_spectrum_paras: list = None):
    if t2_spectrum:
        freq_list = t2_spectrum['freq']
        t2_list = t2_spectrum['t2']
        func_interp = interp1d(freq_list, t2_list, kind='linear')
        return a * tq * func_interp(f)
    else:
        df_dphi = 0.01 / abs(freq2amp_formula(f, *ac_spectrum_paras, tans2phi=True) -
                freq2amp_formula(f - 0.01, *ac_spectrum_paras, tans2phi=True) + 1e-5)

        error = a * tq * df_dphi
        if np.isnan(error):
            return 5e-4
        else:
            return error

def xy_xtalk_err(a, detune, mu, fxy):
    try:
        error = a * fxy(detune, mu)
        if len(error) > 1:
            return error
        else:
            return error[0]
    except:
        return 0

def singq_residual_err(a, fi, fj, alpha_i, alpha_j):
    return lorentzain(fi, fj, a[0], a[1]) + lorentzain(fi + alpha_i, fj, a[0], a[1]) + lorentzain(fi, fj + alpha_j, a[0], a[1])

def edge_distance(chip, qcq1, qcq2):
    distance = []
    for i in qcq1:
        for j in qcq2:
            if nx.has_path(chip, i, j):
                distance.append(nx.shortest_path_length(chip, i, j))
            else:
                distance.append(100000)
    return min(distance)


def twoq_xtalk_err(
    a, pulse1, pulse2, anharm1, anharm2
):
    return (
        (lorentzain(pulse1, pulse2, a[0], a[1])) +
        (lorentzain(pulse1 + anharm1, pulse2, a[2], a[3])) +
        (lorentzain(pulse1, pulse2 + anharm2, a[2], a[3]))
    )

def inner_leakage(a, pulse1, pulse2):
    if (pulse1[1] - pulse2[1]) * (pulse1[0] - pulse2[0]) < 0:
        # print(a[0], 'inner leakage!')
        return a[0]
    else:
        return a[1]


def twoq_pulse_distort_err(a, fi, fj, ac_spectrum_paras1, ac_spectrum_paras2):
    vi0 = freq2amp_formula(fi[0], *ac_spectrum_paras1)
    vi1 = freq2amp_formula(fi[1], *ac_spectrum_paras1)
    vj0 = freq2amp_formula(fj[0], *ac_spectrum_paras2)
    vj1 = freq2amp_formula(fj[1], *ac_spectrum_paras2)
    return a[0] * (abs(vi0 - vi1) + abs(vj0 - vj1)) + inner_leakage(a[1:], fi, fj)


def err_model(frequencys, xtalk_graph, a, targets=None, isTrain=False):
    if frequencys is None:
        # frequencys = [xtalk_graph.nodes[node]['frequency'] for node in xtalk_graph.nodes if xtalk_graph.nodes[node].get('frequency', False)]
        reOptimizeNodes = []
        targets = [node for node in xtalk_graph.nodes if xtalk_graph.nodes[node].get('frequency', False)]
        error_chip = 0
    else:
        for node in targets:
            xtalk_graph.nodes[node]['frequency'] = xtalk_graph.nodes[node]['allow freq'][frequencys[targets.index(node)]]
    
    errAry = np.zeros(len(targets))
    for target in targets:
        errAry[list(targets).index(target)] = 1e-5
        if isinstance(target, str):
            isolateErr = a[0] * xtalk_graph.nodes[target]['isolated_error'][xtalk_graph.nodes[target]['allow freq'].index(xtalk_graph.nodes[target]['frequency'])]
            errAry[list(targets).index(target)] += isolateErr
            xyErr = 0
            residualErr = 0
            for neighbor in xtalk_graph.nodes:
                if isinstance(neighbor, tuple):
                    continue
                if xtalk_graph.nodes[neighbor].get('frequency', False) and not(neighbor == target) and \
                    neighbor in xtalk_graph.nodes[target]['xy_crosstalk_coef'] and \
                    xtalk_graph.nodes[target]['xy_crosstalk_coef'][neighbor] > MUTHRESHOLD:
                    # plt.plot(np.linspace(-500, 500, 1000), xy_xtalk_err(a[1], np.linspace(-500, 500, 1000),
                    #                     xtalk_graph.nodes[target]['xy_crosstalk_coef'][neighbor], xtalk_graph.nodes[target]['xy_crosstalk_f']))
                    # plt.show()
                    xyErrEachPair = xy_xtalk_err(a[1], 
                                                    xtalk_graph.nodes[neighbor]['frequency'] - xtalk_graph.nodes[target]['frequency'],
                                                    xtalk_graph.nodes[target]['xy_crosstalk_coef'][neighbor],
                                                    xtalk_graph.nodes[target]['xy_crosstalk_f']) # omega_d - omega_q = delta
                    errAry[list(targets).index(target)] += xyErrEachPair
                    xyErr += xyErrEachPair

                if nx.has_path(xtalk_graph, target, neighbor):
                    if nx.shortest_path_length(xtalk_graph, target, neighbor) == 1:
                        if xtalk_graph.nodes[neighbor].get('frequency', False):
                            nResidualErr = singq_residual_err(a[2 : 4],
                                                        xtalk_graph.nodes[neighbor]['frequency'],
                                                        xtalk_graph.nodes[target]['frequency'],
                                                        xtalk_graph.nodes[neighbor]['anharm'],
                                                        xtalk_graph.nodes[target]['anharm'])
                            errAry[list(targets).index(target)] += nResidualErr
                            nResidualErr += nResidualErr
                
                    if nx.shortest_path_length(xtalk_graph, target, neighbor) == 2:
                        if xtalk_graph.nodes[neighbor].get('frequency', False):
                            nnResidualErr = singq_residual_err(a[4 : 6],
                                                xtalk_graph.nodes[neighbor]['frequency'],
                                                xtalk_graph.nodes[target]['frequency'],
                                                xtalk_graph.nodes[neighbor]['anharm'],
                                                xtalk_graph.nodes[target]['anharm'])
                            # plt.plot(np.linspace(-500, 500, 1000), singq_residual_err(a[4 : 6], 
                            #                                                             xtalk_graph.nodes[target]['frequency'] + np.linspace(-500, 500, 1000),
                            #                                                             xtalk_graph.nodes[target]['frequency'],
                            #                                                             xtalk_graph.nodes[neighbor]['anharm'],
                            #                                                             xtalk_graph.nodes[target]['anharm']))
                            # plt.show()
                            errAry[list(targets).index(target)] += nnResidualErr
                            residualErr += nnResidualErr

            if frequencys is None:
                allErr = isolateErr + xyErr + residualErr
                error_chip += allErr

                xtalk_graph.nodes[target]['xy err'] = xyErr
                xtalk_graph.nodes[target]['residual err'] = residualErr
                xtalk_graph.nodes[target]['isolate err'] = isolateErr
                xtalk_graph.nodes[target]['all err'] = allErr

                if allErr > 1.5e-2 and not(target in reOptimizeNodes):
                    reOptimizeNodes.append(target)
                    print(target, allErr, 'single target err')

        else:
            qh, ql = xtalk_graph.nodes[target]['qh'], xtalk_graph.nodes[target]['ql']
            fWork = xtalk_graph.nodes[target]['frequency']
            pulseql = fWork
            pulseqh = fWork - xtalk_graph.nodes[qh]['anharm']

            T1Err1 = T1_err(
                a[6],
                pulseql,
                xtalk_graph.nodes[target]['two tq'],
                xtalk_graph.nodes[ql]['T1 spectra']
            )
            errAry[list(targets).index(target)] += T1Err1
            T1Err2 = T1_err(
                a[6],
                pulseqh,
                xtalk_graph.nodes[target]['two tq'],
                xtalk_graph.nodes[qh]['T1 spectra'],
            )
            errAry[list(targets).index(target)] += T1Err2
            T2Err1 = T2_err(
                a[7],
                pulseql,
                xtalk_graph.nodes[target]['two tq'],
                ac_spectrum_paras=xtalk_graph.nodes[ql]['ac_spectrum'],
            )
            errAry[list(targets).index(target)] += T2Err1
            T2Err2 = T2_err(
                a[7],
                pulseqh,
                xtalk_graph.nodes[target]['two tq'],
                ac_spectrum_paras=xtalk_graph.nodes[qh]['ac_spectrum'],
            )
            errAry[list(targets).index(target)] += T2Err2
            if xtalk_graph.nodes[qh].get('frequency', False) and xtalk_graph.nodes[ql].get('frequency', False):
                twoqDistErr = twoq_pulse_distort_err(
                    a[8 : 11],
                    [pulseqh, xtalk_graph.nodes[qh]['frequency']],
                    [pulseql, xtalk_graph.nodes[ql]['frequency']],
                    ac_spectrum_paras1=xtalk_graph.nodes[qh]['ac_spectrum'],
                    ac_spectrum_paras2=xtalk_graph.nodes[ql]['ac_spectrum'],
                )
                errAry[list(targets).index(target)] += twoqDistErr
            else:
                twoqDistErr = 1e-5


            twoqSpectatorErr = 1e-5
            twoqXyErr = 1e-5
            parallelErr = 1e-5
            for neighbor in xtalk_graph.nodes:
                if xtalk_graph.nodes[neighbor].get('frequency', False) and not(neighbor == target):
                    if isinstance(neighbor, str):
                        twoqSpectatorErrOnce = 1e-5
                        twoqXyErrOnce = 1e-5
                        if neighbor in target:
                            continue
                        if nx.has_path(xtalk_graph, neighbor, target):
                            if nx.shortest_path_length(xtalk_graph, neighbor, ql) == 1:
                                twoqSpectatorErrOnce = twoq_xtalk_err(
                                                                    a[11 : 15],
                                                                    pulseql,
                                                                    xtalk_graph.nodes[neighbor]['frequency'],
                                                                    xtalk_graph.nodes[ql]['anharm'],
                                                                    xtalk_graph.nodes[neighbor]['anharm']
                                                                    )   
                                # twoqSpectatorErrOnce = 0
                                errAry[list(targets).index(target)] += twoqSpectatorErrOnce
                            elif nx.shortest_path_length(xtalk_graph, neighbor, qh) == 1:
                                twoqSpectatorErrOnce = twoq_xtalk_err(
                                                                    a[11 : 15],
                                                                    pulseqh,
                                                                    xtalk_graph.nodes[neighbor]['frequency'],
                                                                    xtalk_graph.nodes[qh]['anharm'],
                                                                    xtalk_graph.nodes[neighbor]['anharm']
                                                                    )
                                # twoqSpectatorErrOnce = 0
                                # plt.plot(np.linspace(-500, 500, 1000), twoq_xtalk_err(
                                #                 a[11 : 15],
                                #                 xtalk_graph.nodes[neighbor]['frequency'] + np.linspace(-500, 500, 1000), 
                                #                 xtalk_graph.nodes[neighbor]['frequency'],
                                #                 xtalk_graph.nodes[qh]['anharm'],
                                #                 xtalk_graph.nodes[neighbor]['anharm']))
                                # plt.show()   
                                errAry[list(targets).index(target)] += twoqSpectatorErrOnce
                            
                        if neighbor in xtalk_graph.nodes[ql]['xy_crosstalk_coef'] and \
                            xtalk_graph.nodes[ql]['xy_crosstalk_coef'][neighbor] > MUTHRESHOLD:
                            twoqXyErrOnce = xy_xtalk_err(a[15], 
                                                            xtalk_graph.nodes[neighbor]['frequency'] - pulseql, 
                                                            xtalk_graph.nodes[ql]['xy_crosstalk_coef'][neighbor], 
                                                            xtalk_graph.nodes[ql]['xy_crosstalk_f'])   
                            # twoqXyErrOnce = 0
                            # omega_d - omega_q = delta 
                            errAry[list(targets).index(target)] += twoqXyErrOnce
                        elif neighbor in xtalk_graph.nodes[qh]['xy_crosstalk_coef'] and \
                            xtalk_graph.nodes[qh]['xy_crosstalk_coef'][neighbor] > MUTHRESHOLD:
                            twoqXyErrOnce = xy_xtalk_err(a[15], 
                                                            xtalk_graph.nodes[neighbor]['frequency'] - pulseqh, 
                                                            xtalk_graph.nodes[qh]['xy_crosstalk_coef'][neighbor], 
                                                            xtalk_graph.nodes[qh]['xy_crosstalk_f'])    
                            # twoqXyErrOnce = 0
                            errAry[list(targets).index(target)] += twoqXyErrOnce

                        twoqSpectatorErr += twoqSpectatorErrOnce
                        twoqXyErr += twoqXyErrOnce

                    else:
                        if neighbor == target:
                            continue
                        if nx.has_path(xtalk_graph, neighbor, target):
                            if nx.shortest_path_length(xtalk_graph, neighbor, target) == 1:
                                qnl, qnh = xtalk_graph.nodes[neighbor]['ql'], xtalk_graph.nodes[neighbor]['qh']
                                pulseqnl = xtalk_graph.nodes[neighbor]['frequency']
                                pulseqnh = xtalk_graph.nodes[neighbor]['frequency'] - xtalk_graph.nodes[qnh]['anharm']
                                if (ql, qnl) in xtalk_graph.edges:
                                    qcq = (ql, qnl)
                                    pulse = pulseql
                                    pulsen = pulseqnl
                                elif (ql, qnh) in xtalk_graph.edges:
                                    qcq = (ql, qnh)
                                    pulse = pulseql
                                    pulsen = pulseqnh
                                elif (qh, qnl) in xtalk_graph.edges:
                                    qcq = (qh, qnl)
                                    pulse = pulseqh
                                    pulsen = pulseqnl
                                elif (qh, qnh) in xtalk_graph.edges:
                                    qcq = (qh, qnh)
                                    pulse = pulseqh
                                    pulsen = pulseqnh
                                parallelErrOnce = twoq_xtalk_err(
                                                                a[16:],
                                                                pulse,
                                                                pulsen,
                                                                xtalk_graph.nodes[qcq[0]]['anharm'],
                                                                xtalk_graph.nodes[qcq[1]]['anharm']
                                                                )

                                errAry[list(targets).index(target)] += parallelErrOnce
                                parallelErr += parallelErrOnce

            if frequencys is None:
                allErr = (
                        twoqSpectatorErr
                        + twoqXyErr
                        + parallelErr
                        + T1Err1
                        + T1Err2
                        + T2Err1
                        + T2Err2
                        + twoqDistErr
                        )
                error_chip += allErr

                xtalk_graph.nodes[target]['spectator err'] = twoqSpectatorErr
                xtalk_graph.nodes[target]['parallel err'] = parallelErr
                xtalk_graph.nodes[target]['T1 err'] = T1Err1 + T1Err2
                xtalk_graph.nodes[target]['T2 err'] = T2Err1 + T2Err2
                xtalk_graph.nodes[target]['distort err'] = twoqDistErr
                xtalk_graph.nodes[target]['xy err'] = twoqXyErr
                xtalk_graph.nodes[target]['all err'] = allErr

                if allErr > 1.5e-2 and not (target in reOptimizeNodes):
                    reOptimizeNodes.append(target)
                    print(target, xtalk_graph.nodes[target]['all err'], 'qcq err')


    if isTrain:
        return errAry
    elif frequencys is None:
        return reOptimizeNodes, xtalk_graph, error_chip / len(targets)
    else:
        cost_average = np.sum(errAry) / len(errAry)
        return cost_average