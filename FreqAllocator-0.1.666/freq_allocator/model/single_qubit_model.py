import numpy as np
from scipy.interpolate import interp1d
from .formula import freq2amp_formula, lorentzain, freq_var_map

MUTHRESHOLD = 0.01

def singq_T1_err(a, tq, f, t1_spectrum):
    try:
        error = a * tq / t1_spectrum(f)
    except:
        error = 5e-4
    if error < 0:
        error = 5e-4
    return error

def singq_T2_err(a, tq, f, t2_spectrum: dict = None, ac_spectrum_paras: list = None):
    if t2_spectrum:
        freq_list = t2_spectrum['freq']
        t2_list = t2_spectrum['t2']
        func_interp = interp1d(freq_list, t2_list, kind='linear')
        return a * tq * func_interp(f)
    else:
        df_dphi = 1 / (
            abs(freq2amp_formula(f, *ac_spectrum_paras, tans2phi=True) -
                freq2amp_formula(f - 0.01, *ac_spectrum_paras, tans2phi=True)) / 0.01
        )
        error = a * tq * df_dphi
        if np.isnan(error):
            return 5e-4
        else:
            return error

def singq_xtalk_err(a, detune, mu, fxy):
    try:
        error = a * fxy(detune, mu)
        return error[0]
    except:
        return 0

def singq_residual_err(a, gamma, fi, fj, alpha_i, alpha_j):
    return lorentzain(fi, fj, a, gamma) + lorentzain(fi + alpha_i, fj, a, gamma) + lorentzain(fi, fj + alpha_j, a, gamma)

# def twoq_dist_bound(fqMax1, fqMax2, fq1, fq2, a):
#     if (fqMax1 - fqMax2) * (fq1 - fq2) < 0:
#         # print('oops!')
#         return 1000
#     else:
#         return a * (fq1 - fq2) ** 2

def single_err_model(frequencys, chip, targets, a, varType):
    if varType == 'double':
        for target in targets:
            chip.nodes[target]['frequency'] = int(round(freq_var_map(frequencys[targets.index(target)], chip.nodes[target]['allow freq'])))
    else:
        for target in targets:
            if frequencys.dtype == np.int32:
                chip.nodes[target]['frequency'] = chip.nodes[target]['allow freq'][frequencys[targets.index(target)]]
            else:
                chip.nodes[target]['frequency'] = chip.nodes[target]['allow freq'][int(round(frequencys[targets.index(target)] * (len(chip.nodes[target]['allow freq']) - 1)))]
        
    cost = 0
    for target in targets:
        if varType == 'double':
            cost += chip.nodes[target]['isolated_error'](chip.nodes[target]['frequency'])
        else:
            cost += chip.nodes[target]['isolated_error'][frequencys[targets.index(target)]]
        
        for neighbor in chip.nodes():
            if chip.nodes[neighbor].get('frequency', False) and not(neighbor == target) and \
                chip.nodes[neighbor]['name'] in chip.nodes[target]['xy_crosstalk_coef'] and \
                chip.nodes[target]['xy_crosstalk_coef'][chip.nodes[neighbor]['name']] > MUTHRESHOLD:
                cost += singq_xtalk_err(a[2], chip.nodes[neighbor]['frequency'] - chip.nodes[target]['frequency'], 
                                        chip.nodes[target]['xy_crosstalk_coef'][chip.nodes[neighbor]['name']], chip.nodes[target]['xy_crosstalk_f'])                
                if (target, neighbor) in chip.edges():
                    cost += singq_residual_err(a[0], a[1],                        
                                        chip.nodes[neighbor]['frequency'],
                                        chip.nodes[target]['frequency'],
                                        chip.nodes[neighbor]['anharm'],
                                        chip.nodes[target]['anharm'])       
                    # cost += twoq_dist_bound(chip.nodes[target]['freq_max'], 
                    #                         chip.nodes[neighbor]['freq_max'],
                    #                         chip.nodes[target]['frequency'],
                    #                         chip.nodes[neighbor]['frequency'], a[-1])
                for nNeighbor in chip[neighbor]:
                    if nNeighbor == target:
                        continue
                    elif chip.nodes[nNeighbor].get('frequency', False):
                        cost += singq_residual_err(a[2], a[3],
                                            chip.nodes[nNeighbor]['frequency'],
                                            chip.nodes[target]['frequency'],
                                            chip.nodes[nNeighbor]['anharm'],
                                            chip.nodes[target]['anharm']) / 2
                            
    cost_average = cost / len(targets)
    return cost_average 

