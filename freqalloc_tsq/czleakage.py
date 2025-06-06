import sys
package_root = r"F:\onedrive\vs experiment\transmon-qubit"
sys.path.insert(0, package_root)
print(sys.executable)

from copy import deepcopy

from matplotlib import pyplot as plt
import numpy as np
import qutip as qp
import matplotlib as mpl
import scipy
from sklearn.metrics import mean_squared_error
import copy
from chip_hamiltonian.chip_hamiltonian import Chip, ChipDynamic
from solvers.solvers import Solver, SolverDynamic
from experiment import *
from pulse.pulse_lib import *
from functions.plot_tools import PlotTool
from functions.fit_tools import fit_fft
from functions import *
from functions.tools import *
from scipy.optimize import *
from itertools import product

font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}
np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

yaml_path_3q = r'F:\OneDrive\vs experiment\freqalloc_tsq\chip_param_3q.yaml'
yaml_path_2q = r'F:\OneDrive\vs experiment\freqalloc_tsq\chip_param_2q.yaml'


if __name__ == '__main__':

    # 初始化物理体系

    chip = Solver(chip_path=yaml_path_3q, dim=3, flag_g_exact=True)

    # 设置关断点并存入yaml
    # w_off, zz_off = chip.ZZ_min((5, 8), 'Q1', 'Q2', save_flag=True)
    # print(f'w_off: {w_off} GHz  zz_off: {zz_off * 1e6} kHz')
    # w_off, zz_off = chip.ZZ_min((5, 8), 'Q2', 'Q3', save_flag=True)
    # print(f'w_off: {w_off} GHz  zz_off: {zz_off * 1e6} kHz')
    
    # 初始化CZ

    time_step = 0.2
    sigma = 1.5
    buffer = 7.5
    cz = CZ(chip_path=yaml_path_3q,
            dim=3,
            QL='Q1',
            QH='Q2',
            spectator_bit='Q3',
            flag_fig=True,
            time_step=time_step,
            flag_data=False,
            flag_g_exact=True)
    exp_args = {
        'shape': 'FlattopGaussian',
        'arg_type': 'wq',
        'sigma': sigma,
        'buffer': buffer
    }
    cz.exp_args = exp_args
    
    # 优化CZ参数

    bit_name_list = ['Q2', 'C1_2']
    width = 40

    paras_best, _ = cz.fidelity_optimize(bit_name_list, width)
    
    # 设置CZ参数

    paras_best = {'Q2': 4.485237133766358, 'C1_2': 4.965560424149795}

    # 观察比特coupler对CZ误差的影响
        
    # ws_list = np.linspace(chip.q_dic['Q1']['w_idle'] - 0.06, chip.q_dic['Q1']['w_idle'] + 0.06, 50)
    # wc_list = np.linspace(5.9, 6.8, 71)
    # width = 40

    # w_list = [
    #     list(item) for item in product([paras_best['Q2']], [paras_best['C1_2']],
    #                                 ws_list, wc_list)
    # ]

    # error_list = qp.parallel_map(cz.fidelity_calculate,
    #                             w_list,
    #                             task_kwargs={
    #                                 'bit_name_list': ['Q2', 'C1_2', 'Q3', 'C2_3'],
    #                                 'width': width,
    #                             },
    #                             progress_bar=True,
    #                             num_cpus=20)

    # error_arr = np.array(error_list).reshape(len(ws_list), len(wc_list))
    # cz.ploter.plot_heatmap(ws_list,
    #                     wc_list,
    #                     error_arr.T,
    #                     xlabel=r'$\omega_{Q_S}$',
    #                     ylabel=r'$\omega_{C_S}$',
    #                     zlabel='$error$',
    #                     ytype='w',
    #                     title=f'CZ Fidelity',
    #                     norm='log')
    
    # 通过把运行次数变为10，增加误差效应

    ws_list = np.linspace(chip.q_dic['Q1']['w_idle'] - 0.06, chip.q_dic['Q1']['w_idle'] + 0.06, 50)
    wc_list = np.linspace(5.9, 6.8, 70)

    width = 40
    gate_num = 10

    init_state_list = list(qp.state_number_enumerate([2] * 2))
    init_state_list = [(*init_state, 0) for init_state in init_state_list]

    arg_dic_list = [{
        **paras_best,
        **{
            'Q3': ws,
            'C2_3': wc_s
        }
    } for ws, wc_s in product(ws_list, wc_list)]

    results = []
    for init_state in init_state_list:
        evolution_result_list = qp.parallel_map(
            cz.run_evolution,
            arg_dic_list,
            task_kwargs={'parallel_args': ['arg_dic'],
                        'width': width,
                        'gate_num': gate_num,
                        'init_state': init_state},
            progress_bar=True,
            num_cpus=cz.num_cpus)
        results.append(evolution_result_list)
        
    fidelities = []
    for i in range(len(evolution_result_list)):
        resulti = [results[0][i], results[1][i], results[2][i], results[3][i]]
        F = cz.analyze_fidelity(fidelity_result=resulti, type='ave')

        fidelities.append(1 - F)

    error_arr = np.array(fidelities).reshape(len(ws_list), len(wc_list))
    cz.ploter.plot_heatmap(ws_list,
                        wc_list,
                        error_arr.T,
                        xlabel=r'$\omega_{Q_S}$',
                        ylabel=r'$\omega_{C_S}$',
                        zlabel='$error$',
                        ytype='w',
                        title=f'CZ Fidelity',
                        norm='log')
        
    # width = 400
    
    # ws_list = np.linspace(cz.q_dic['Q1']['w_idle'] - 0.1, cz.q_dic['Q1']['w_idle'] + 0.1, 100)

    # arg_dic_list = [{**paras_best,**{'Q3': ws}} for ws in ws_list]

    # evolution_result_list = qp.parallel_map(
    #     cz.run_evolution,
    #     arg_dic_list,
    #     task_kwargs={'parallel_args': ['arg_dic'],
    #                 'width': width,
    #                 'gate_num': 1,},
    #     progress_bar=True,
    #     num_cpus=cz.num_cpus)

    # expec_name = '110'
    # evo_result_list = [
    #     evolution_result.states[-1] for evolution_result in evolution_result_list
    # ]

    # mea_ops = cz.Od(('Q1', 'Q2', 'Q3'), (0, 1, 1))

    # expect_list = qp.expect(mea_ops, evo_result_list)

    # fig, ax = plt.subplots()
    # ax.set_xlabel(r'$\omega_s$ (GHz)')
    # ax.set_ylabel(r'population')
    # # ax.set_yscale('log')
    # label = fr'QL-QH-QS: {expec_name}'
    # ax.plot(ws_list, expect_list, label=label)

    # ax.legend()
    # title = fr'观察比特工作频率对泄漏的影响011'
    # cz.ploter.save_fig(fig, title)

    # peaks = []
    # for i in range(1, len(expect_list) - 1):
    #     if expect_list[i] > expect_list[i - 1] and expect_list[i] > expect_list[i + 1]:
    #         peaks.append((i, expect_list[i]))

    # peaks.sort(key=lambda x: x[1], reverse=True)

    # leakFreqs1 = ws_list[peaks[0][0]]
    # leakFreqs2 = ws_list[peaks[1][0]]

    # g1 = np.abs(leakFreqs1 - leakFreqs2)
    # g2 = 1 / (40 - 15 - 3)
    # print(g1, g2, leakFreqs1, leakFreqs2)

    # # ws_list = [4.604, 4.562]
    # ws_list = [4.265, 4.2206]
    # # ws_list = [leakFreqs1, leakFreqs2]
    # wc_list = np.linspace(5.9, 6.8, 71)
    # # ws_list = np.linspace(4.55, 4.65, 100)
    # # wc_list = np.linspace(6.5, 7.5, 70)

    # arg_dic_list = [{
    #     **paras_best,
    #     **{
    #         'Q3': ws,
    #         'C2_3': wc_s
    #     }
    # } for ws, wc_s in product(ws_list, wc_list)]

    # gs = np.zeros((len(ws_list), len(wc_list)))
    
    # rho2s = cz.rho_map['Q2-Q3']
    # rho2c = cz.rho_map['Q2-C2_3']
    # rhosc = cz.rho_map['Q3-C2_3']
    
    # gt = (ws_list[0] - ws_list[1]) / 2
    # w1 = (ws_list[0] + ws_list[1]) / 2
    # # w1 = ws_list[0]
    
    # j = 0
    # for wc in wc_list:
    #     g2c = cz.rho2g(rho2c, [paras_best['Q2'], wc])
    #     g2s1 = cz.rho2g(rho2s, [paras_best['Q2'], ws_list[0]])
    #     gsc1 = cz.rho2g(rhosc, [ws_list[0], wc])
    #     g2s2 = cz.rho2g(rho2s, [paras_best['Q2'], ws_list[1]])
    #     gsc2 = cz.rho2g(rhosc, [ws_list[1], wc])
        
    #     g1 = np.abs(g2s1 + 0.5 * g2c * gsc1 * (1 / (w1 + gt - wc) + 1 / (ws_list[0] - wc) - 1 / (w1 + wc) - 1 / (wc + ws_list[0] - gt)))
    #     g2 = np.abs(g2s2 + 0.5 * g2c * gsc2 * (1 / (w1 - gt - wc) + 1 / (ws_list[1] - wc) - 1 / (w1 + wc) - 1 / (wc + ws_list[1] + gt)))
    #     gs[0, j] = g1
    #     gs[1, j] = g2
    #     j += 1

    # gate_num = 1
    # width = 600
    
    # evolution_result_list = qp.parallel_map(
    #     cz.run_evolution,
    #     arg_dic_list,
    #     task_kwargs={'parallel_args': ['arg_dic'],
    #                 'width': width,
    #                 'gate_num': gate_num},
    #     progress_bar=True,
    #     num_cpus=32)

    
    # # expec_name_list = ['110', '020', '011']
    # expec_name_list = ['011']
    # evo_result_list = [
    #     evolution_result.states[-1] for evolution_result in evolution_result_list
    # ]
    # # mea_ops = [
    #     # cz.Od(('Q1', 'Q2', 'Q3'), (1, 1, 0)),
    #     # cz.Od(('Q1', 'Q2', 'Q3'), (0, 2, 0)),
    #     # cz.Od(('Q1', 'Q2', 'Q3'), (0, 1, 1)),
    # # ]
    # mea_ops = [
    #     cz.Od(('Q1', 'Q2', 'Q3'), (0, 1, 1)),
    # ]

    # expect_list = qp.expect(mea_ops, evo_result_list)

    # fig, ax = plt.subplots()
    # ax.set_xlabel(r'$\omega_s$ (GHz)')
    # ax.set_ylabel(r'population')
    # ax.set_yscale('log')
    # custom_color = cz.ploter.cmap(np.linspace(0, 1, len(expec_name_list)))
    # for i, expec_name in enumerate(expec_name_list):
    #     label = fr'QL-QH-QS: {expec_name}'
    #     expect_arr = expect_list[i].reshape(len(ws_list), len(wc_list))
    #     ax.plot(wc_list, expect_arr[0], linestyle='dashed', label=label + r'$1^{st}$', color=custom_color[i])
    #     ax.plot(wc_list, expect_arr[1], label=label + r'$2^{nd}$', color=custom_color[i])
    # ax.plot(wc_list, gs[0], label='geff 1')
    # ax.plot(wc_list, gs[1], label='geff 2')
    # ax.legend()
    # title = fr'观察比特工作频率对泄漏的影响'
    # cz.ploter.save_fig(fig, title)
    
    # expect_arr = expect_list[0].reshape(len(ws_list), len(wc_list))
    # leak_arr = 1 - expect_arr
    # cz.ploter.plot_heatmap(ws_list,
    #                     wc_list,
    #                     leak_arr.T,
    #                     xlabel=r'$\omega_{Q_S}$',
    #                     ylabel=r'$\omega_{C_S}$',
    #                     zlabel='$1-P_{011}$',
    #                     ytype='w',
    #                     title=f'CZ Leakage N=4 buffer=7.5 liner',
    #                     )
    


    # ws_list = np.linspace(cz.q_dic['Q1']['w_idle'] - 0.1, cz.q_dic['Q1']['w_idle'] + 0.1, 100)

    # gate_num = 10
    # width = 40
    
    # w_list = [
    #     list(item) for item in product([paras_best['Q2']], [paras_best['C1_2']],
    #                                 ws_list, [cz.q_dic['C2_3']['w_idle']])
    # ]

    # error_list = qp.parallel_map(cz.fidelity_calculate,
    #                             w_list,
    #                             task_kwargs={
    #                                 'bit_name_list': ['Q2', 'C1_2', 'Q3', 'C2_3'],
    #                                 'width': width
    #                             },
    #                             progress_bar=True,
    #                             num_cpus=20)

    # error_arr = np.array(error_list).reshape(len(ws_list))

    # fig, ax = plt.subplots()
    # ax.set_xlabel(r'$\omega_s$ (GHz)')
    # ax.set_ylabel(r'err')
    # # ax.set_yscale('log')
    # label = fr'err'
    # ax.plot(ws_list, error_arr, label=label)

    # ax.legend()
    # title = fr'观察比特工作频率对误差影响'
    # cz.ploter.save_fig(fig, title)


    # # ws_list = [4.604, 4.562]
    # ws_list = [4.265, 4.2206]
    # # ws_list = [leakFreqs1, leakFreqs2]
    # ws_list = np.linspace(min(ws_list) - 0.06, max(ws_list) + 0.06, 50)
    # wc_list = np.linspace(5.9, 6.8, 71)
    # # ws_list = np.linspace(4.55, 4.65, 100)
    # # wc_list = np.linspace(6.5, 7.5, 70)

    # arg_dic_list = [{
    #     **paras_best,
    #     **{
    #         'Q3': ws,
    #         'C2_3': wc_s
    #     }
    # } for ws, wc_s in product(ws_list, wc_list)]

    # gs = np.zeros((len(ws_list), len(wc_list)))
    
    # rho2s = cz.rho_map['Q2-Q3']
    # rho2c = cz.rho_map['Q2-C2_3']
    # rhosc = cz.rho_map['Q3-C2_3']
    
    # gt = (ws_list[0] - ws_list[1]) / 2
    # w1 = (ws_list[0] + ws_list[1]) / 2
    # # w1 = ws_list[0]
    
    # j = 0
    # for wc in wc_list:
    #     g2c = cz.rho2g(rho2c, [paras_best['Q2'], wc])
    #     g2s1 = cz.rho2g(rho2s, [paras_best['Q2'], ws_list[0]])
    #     gsc1 = cz.rho2g(rhosc, [ws_list[0], wc])
    #     g2s2 = cz.rho2g(rho2s, [paras_best['Q2'], ws_list[1]])
    #     gsc2 = cz.rho2g(rhosc, [ws_list[1], wc])
        
    #     g1 = np.abs(g2s1 + 0.5 * g2c * gsc1 * (1 / (w1 + gt - wc) + 1 / (ws_list[0] - wc) - 1 / (w1 + wc) - 1 / (wc + ws_list[0] - gt)))
    #     g2 = np.abs(g2s2 + 0.5 * g2c * gsc2 * (1 / (w1 - gt - wc) + 1 / (ws_list[1] - wc) - 1 / (w1 + wc) - 1 / (wc + ws_list[1] + gt)))
    #     gs[0, j] = g1
    #     gs[1, j] = g2
    #     j += 1