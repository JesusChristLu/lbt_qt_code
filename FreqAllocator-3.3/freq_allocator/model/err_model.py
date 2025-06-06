import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .formula import freq2amp_formula, eff_g, freq_var_map, lorentzain
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from pathlib import Path
import seaborn as sns
import copy

MUTHRESHOLD = 0.01

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
        error = a * fxy.ev(detune, mu)
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


def err_model(frequencys=[], xtalk_graph=None, a=None, targets=None, isTrain=False, freqIndex=True):

    if frequencys is None:
        # frequencys = [xtalk_graph.nodes[node]['frequency'] for node in xtalk_graph.nodes if xtalk_graph.nodes[node].get('frequency', False)]
        reOptimizeNodes = []
        targets = [node for node in xtalk_graph.nodes if xtalk_graph.nodes[node].get('frequency', False)]
        error_chip = 0
    elif freqIndex:
        for node in targets:
            xtalk_graph.nodes[node]['frequency'] = xtalk_graph.nodes[node]['allow freq'][frequencys[targets.index(node)]]
    else:
        for node in targets:
            xtalk_graph.nodes[node]['frequency'] = frequencys[targets.index(node)] * 1e3
    
    errAry = np.zeros(len(targets))
    for target in targets:
        errAry[list(targets).index(target)] = 1e-5
        if isinstance(target, str):
            # 获取当前频率
            current_frequency = xtalk_graph.nodes[target]['frequency']
            # 获取允许的频率列表
            allowed_frequencies = xtalk_graph.nodes[target]['allow freq']
            # 计算每个频率与当前频率的差值，并找到最小差值的索引
            closest_index = min(range(len(allowed_frequencies)), key=lambda i: abs(allowed_frequencies[i] - current_frequency))

            # 获取 isolated_error 对应的值
            isolated_error_value = xtalk_graph.nodes[target]['isolated_error'][closest_index]
            
            isolateErr = a[0] * isolated_error_value
            # xtalk_graph.nodes[target]['isolated_error'][xtalk_graph.nodes[target]['allow freq'].index(xtalk_graph.nodes[target]['frequency'])]
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

def err_model_fun(a, frequencys, xtalk_graph, errors=None, returnLoss=True):
    
    # print(a)
    
    frequencys = frequencys.numpy()
    xtalk_graphs = [xtalk_graph] * frequencys.shape[0]
    aes = [a] * frequencys.shape[0]
    # targets = [list(xtalk_graph.nodes)] * frequencys.shape[0]
    targets = [[node for node in xtalk_graph.nodes if isinstance(node, str)]] * frequencys.shape[0]
    isTrains = [True] * frequencys.shape[0]
    freqIndex = [False] * frequencys.shape[0]
    
    poolNum = 20
    # print('start', poolNum)
    p = Pool(poolNum)
    pred_errs = p.starmap(err_model, zip(frequencys, xtalk_graphs, aes, targets, isTrains, freqIndex))
    p.close()
    p.join()

    pred_errs = np.array(pred_errs).flatten()

    if returnLoss:
        print(a, pred_errs, errors)
        print(np.mean((errors - pred_errs) ** 2))
        return np.mean((errors - pred_errs) ** 2)
    else:
        return pred_errs
    
def train_a_model(data_loader, xtalk_graph):

    aIni = [1, 1, 
        1, 50, 1, 50,
        0, 0, 1e-3, 1e-5, 0,  
        1, 10, 1, 10, 
        1, 
        1, 10, 1, 10]     
    
    # aIni = np.array(aIni) + (np.random.random(len(aIni)) - 0.5)
    
    aIni[6] = aIni[7] = aIni[10] = 0
        
    a_i = None
    
    losses = []
    
    for batch_data in data_loader:
        frequency = batch_data[0]
        real_err = batch_data[1].numpy().flatten()
                
        # 设置边界，确保所有参数大于等于零
        bounds = [(0, None) for _ in range(len(aIni))]

        F = minimize(err_model_fun, aIni, args=(frequency, xtalk_graph, real_err), bounds=bounds, method='Nelder-Mead',  options={'maxiter': 5})
        
        aIni = F.x
        loss = F.fun
        print(aIni, loss)
        losses.append(loss)
        
        if a_i is None:
            a_i = err_model_fun(aIni, frequency, xtalk_graph, returnLoss=False)
            b_i = real_err
        else:
            a_i = np.hstack((a_i, err_model_fun(aIni, frequency, xtalk_graph, returnLoss=False)))
            b_i = np.hstack((b_i, real_err))

    avglosses = np.mean(losses)
    print(avglosses)
    
    plt.figure(figsize=(6, 4))
    plt.scatter(a_i, b_i, s=1)
    # sns.kdeplot(x=a_i, y=b_i, cmap="Reds", fill=True, bw_adjust=0.5)
    # 生成一条45度的线
    # 获取x轴和y轴的最大值和最小值，以确定45度线的范围
    min_val = min(min(a_i), min(b_i))
    max_val = max(max(a_i), max(b_i))
    x = np.linspace(min_val, max_val, 100)
    y = x
    plt.plot(x, y, color='red', linestyle='--')
    # 添加标题和标签
    # plt.title('二维散点图和45度线')
    plt.xlabel('prediction')
    plt.ylabel('measurement')

    plt.title('train')

    plt.semilogx()
    plt.semilogy()

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'a_train.pdf', dpi=300)
    plt.close()
    
    # 计算 c_i
    c_i = np.abs(a_i - b_i) / b_i

    # 对 c_i 进行排序
    c_i_sorted = np.sort(c_i)
    c_i_median = np.median(c_i_sorted)

    # 计算累积频率
    cum_freq = np.arange(1, len(c_i_sorted) + 1) / len(c_i_sorted)

    # 创建累积频率分布图
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median, color='r', linestyle='--', label='median=' + str(c_i_median * 100)[:4] + '%')

    # 添加标题和标签
    plt.title('train relav')
    plt.semilogx()
    plt.xlabel('relav inacc')
    plt.ylabel('cdf')
    plt.legend()

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'a_train relav.pdf', dpi=300)
    plt.close()

    # 计算 c_i
    c_i = np.abs(a_i - b_i)

    # 对 c_i 进行排序
    c_i_sorted = np.sort(c_i)
    c_i_median = np.median(c_i_sorted)

    # 计算累积频率
    cum_freq = np.arange(1, len(c_i_sorted) + 1) / len(c_i_sorted)

    # 创建累积频率分布图
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median, color='r', linestyle='--', label='median=' + str(c_i_median)[:6])

    # 添加标题和标签
    plt.title('train abs')
    plt.semilogx()
    plt.xlabel('inacc')
    plt.ylabel('cdf')
    plt.legend()

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'a_train abs.pdf', dpi=300)
    plt.close()
    
    return aIni
    

def test_a_model(a, data_loader, xtalk_graph):

    a_i = None

    for batch_data in data_loader:
        frequency = batch_data[0]
        real_err = batch_data[1].numpy().flatten()

        pred_err = err_model_fun(a, frequency, xtalk_graph, returnLoss=False)

        if a_i is None:
            a_i = pred_err
            b_i = real_err
        else:
            a_i = np.hstack((a_i, pred_err))
            b_i = np.hstack((b_i, real_err))

    # 获取x轴和y轴的最大值和最小值，以确定45度线的范围
    min_val = min(min(a_i), min(b_i))
    max_val = max(max(a_i), max(b_i))

    # 生成一条45度的线
    x = np.linspace(min_val, max_val, 100)
    y = x
    plt.plot(x, y, color='red', linestyle='--')

    plt.figure(figsize=(6, 4))
    plt.scatter(a_i, b_i, s=1)
    # sns.kdeplot(x=a_i, y=b_i, cmap="Reds", fill=True, bw_adjust=0.5)
    plt.plot(x, y, color='red', linestyle='--')
    # 添加标题和标签
    # plt.title('二维散点图和45度线')
    plt.xlabel('prediction')
    plt.ylabel('measurement')
    plt.title('test')

    plt.semilogx()
    plt.semilogy()

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'a_test.pdf', dpi=300)
    plt.close()
    
    # 计算 c_i
    c_i = np.abs(a_i - b_i) / b_i

    # 对 c_i 进行排序
    c_i_sorted = np.sort(c_i)
    c_i_median = np.median(c_i_sorted)

    # 计算累积频率
    cum_freq = np.arange(1, len(c_i_sorted) + 1) / len(c_i_sorted)

    # 创建累积频率分布图
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median, color='r', linestyle='--', label='median=' + str(c_i_median * 100)[:4] + '%')

    # 添加标题和标签
    plt.title('test')
    plt.semilogx()
    plt.xlabel('relev inacc')
    plt.ylabel('cdf')
    plt.legend()

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'a_test relav.pdf', dpi=300)
    plt.close()
    
    # 计算 c_i
    c_i = np.abs(a_i - b_i)

    # 对 c_i 进行排序
    c_i_sorted = np.sort(c_i)
    c_i_median = np.median(c_i_sorted)

    # 计算累积频率
    cum_freq = np.arange(1, len(c_i_sorted) + 1) / len(c_i_sorted)

    # 创建累积频率分布图
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median, color='r', linestyle='--', label='median=' + str(c_i_median)[:6])

    # 添加标题和标签
    plt.title('test abs')
    plt.semilogx()
    plt.xlabel('inacc')
    plt.ylabel('cdf')
    plt.legend()

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'a_test abs.pdf', dpi=300)
    plt.close()

    return