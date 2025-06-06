from typing import Union
from pathlib import Path
import numpy as np
import copy
import pandas as pd
import qutip as qp
import cmath
import matplotlib.pyplot as plt
import time

from experiment.experiment_base import ExpBaseDynamic
from itertools import product
from pulse import *
from functions import *
from qutip.qip.operations.gates import cz_gate
from scipy.optimize import *
from functions.containers import ExpBaseDynamicContainer


class CZ(ExpBaseDynamic):
    def __init__(
        self, QL: str = 'Q1', QH: str = 'Q2', spectator_bit: str = None, **kwargs
    ):
        self.QL = QL
        self.QH = QH
        self.spectator_bit = spectator_bit
        self.scan2d_leak_para = None
        self.scan2d_cphase_para = None
        self.scan2d_fidelity_para = None
        self.tlist = None
        self.cphase = None
        self.evolution_result = None
        self.cphase_result = None
        self.fidelity_result = None
        self.result_leak_2d = None
        self.result_cphase_2d = None
        self.result_fidelity_2d = None

        plot_params = kwargs.get('plot_params', {})
        plot_params.setdefault(
            'units',
            {
                'x-t': 'ns',
                'x-w': 'GHz',
                'x-vol': 'V',
                'y-t': 'ns',
                'y-w': 'GHz',
                'y-vol': 'V',
            },
        )
        kwargs.update({'plot_params': plot_params})
        container = ExpBaseDynamicContainer(**kwargs)
        super().__init__(**container)

    @parallel_allocation
    def run_evolution(
        self,
        arg_dic: dict,
        width: float,
        gate_num: int = 1,
        delay: float = 0,
        init_state: tuple = (1, 1),
        **kwargs,
    ):
        if self.exp_args.get('fix_bit_arg'):
            arg_dic = {**self.exp_args.get('fix_bit_arg'), **arg_dic}
        pulse_dic = {}

        for bit in arg_dic:

            arg = arg_dic[bit]
            arg_idle = (
                self.q_dic[bit]['w_idle'] if self.exp_args['arg_type'] == 'wq' else 0
            )
            if bit[0] == 'Q':
                sigma = self.exp_args['sigma_q']
            else:
                sigma = self.exp_args['sigma_c']

            if self.exp_args['shape'] == 'FlattopGaussian':
                zpulse_base = FlattopGaussian(
                    width,
                    arg,
                    arg_idle=arg_idle,
                    arg_type=self.exp_args['arg_type'],
                    sigma=sigma,
                    buffer=self.exp_args['buffer'],
                    sample_rate=self.sample_rate,
                    q_dic=self.q_dic[bit],
                )
            elif self.exp_args['shape'] == 'PiecewiseConstant':
                zpulse_base = PiecewiseConstant(
                    width,
                    data=arg,
                    sample_rate=self.sample_rate,
                    wR=self.q_dic[bit]['w_idle'],
                )

            zpulse_base()

            if delay != 0:
                delay_base = Constant(
                    delay,
                    arg_idle,
                    arg_type=self.exp_args['arg_type'],
                    sample_rate=self.sample_rate,
                )
                delay_base()
                zpulse = zpulse_base + (delay_base + zpulse_base) * (gate_num - 1)

            else:
                zpulse = zpulse_base * gate_num

            pulse_dic[bit] = {'Z': zpulse}
        if self.spectator_bit is None:
            init_state = self.Kd((self.QL, self.QH), init_state)
        else:
            init_state = self.Kd((self.QL, self.QH, self.spectator_bit), init_state)
        result = self.state_solve(pulse_dic, init_state)
        self.evolution_result = result
        self.tlist = result.times
        return result

    @parallel_allocation
    def run_cphase(self, arg_dic: dict, width: float, **kwargs):

        b_0x = (0, '+')
        b_1x = (1, '+')

        result_0x = self.run_evolution(arg_dic, width, init_state=b_0x)
        result_1x = self.run_evolution(arg_dic, width, init_state=b_1x)

        self.cphase_result = [result_0x, result_1x]
        self.tlist = result_0x.times
        return [result_0x, result_1x]

    @parallel_allocation
    def run_fidelity(self, arg_dic: dict, width: float, type: str = 'ave', **kwargs):
        # pulse_dic = {}
        # for bit in arg_dic:
        #     arg_idle = self.q_dic[bit]['w_idle'] if arg_type == 'wq' else 0
        #     arg = arg_dic[bit]
        #     if shape == 'FlattopGaussian':
        #         zpulse = FlattopGaussian(width, arg, arg_idle=arg_idle, arg_type=arg_type,
        #                                  sample_rate=self.sample_rate, q_dic=self.q_dic[bit])
        #         zpulse.get_pulse()
        #     pulse_dic[bit] = {'Z': zpulse}
        if type == 'ave':
            # self.fidelity_result = self.propagator_solve(pulse_dic, flag_last=True, flag_parallel=flag_parallel)
            init_state_list = list(qp.state_number_enumerate([2] * 2))
            if self.spectator_bit is not None:
                init_state_list = [(*init_state, 1) for init_state in init_state_list]

        elif type == 'qpt':
            init_state_list = qpt_rho_in(cz_gate())
        fidelity_result = [
            self.run_evolution(arg_dic, width, init_state=init_state)
            for init_state in init_state_list
        ]
        self.fidelity_result = fidelity_result
        return fidelity_result

    def scan2d_leak(self, arg_list_dic: dict, width: float, init_state: tuple = (1, 1)):

        self.scan2d_leak_para = arg_list_dic
        arg_list = product(*list(arg_list_dic.values()))
        arg_dic_list = [dict(zip(arg_list_dic.keys(), arg)) for arg in arg_list]

        self.result_leak_2d = qp.parallel_map(
            self.run_evolution,
            arg_dic_list,
            task_args=(width,),
            task_kwargs={'init_state': init_state},
            progress_bar=True,
            num_cpus=self.num_cpus,
        )

    def scan2d_leak_N(
        self,
        arg_list_dic: dict,
        width: float,
        arg_type: str = 'wq',
        shape: str = 'FlattopGaussian',
        init_state: tuple = (1, 1),
        spectator_bit: str = None,
    ):
        print(1)

    def scan2d_cphase(self, arg_list_dic: dict, width: float):
        self.scan2d_cphase_para = arg_list_dic
        arg_list = product(*list(arg_list_dic.values()))
        arg_dic_list = [dict(zip(arg_list_dic.keys(), arg)) for arg in arg_list]

        self.result_cphase_2d = qp.parallel_map(
            self.run_cphase,
            arg_dic_list,
            task_args=(width,),
            progress_bar=True,
            num_cpus=self.num_cpus,
        )

    def scan2d_fidelity(self, arg_list_dic: dict, width: float, type: str = 'ave'):
        self.scan2d_fidelity_para = arg_list_dic
        arg_list = product(*list(arg_list_dic.values()))
        arg_dic_list = [dict(zip(arg_list_dic.keys(), arg)) for arg in arg_list]

        self.result_fidelity_2d = qp.parallel_map(
            self.run_fidelity,
            arg_dic_list,
            task_args=(width,),
            progress_bar=True,
            num_cpus=self.num_cpus,
            type=type,
        )

    def analyze_evolution(self, mea_ops=None, name_list=None):
        if mea_ops is None:
            mea_ops = [
                self.Od((self.QL, self.QH), (1, 1)),
                self.Od((self.QL, self.QH), (0, 2)),
            ]
            name_list = [f'QH-QL: {1}-{1}', f'QH-QL: {2}-{0}']
        expect_list = qp.expect(mea_ops, self.evolution_result.states)

        self.plotter.plot_lines(self.evolution_result.times, expect_list, name_list)

    def analyze_cphase(self, cphase_result: list = None, plot_flag: bool = True):
        if cphase_result is None:
            cphase_result = self.cphase_result
            psi_0x_list = [
                state.transform(list(self.eigvec.values()))
                for state in cphase_result[0].states
            ]
            psi_1x_list = [
                state.transform(list(self.eigvec.values()))
                for state in cphase_result[1].states
            ]
        else:
            psi_0x_list = [
                cphase_result[0].states[-1].transform(list(self.eigvec.values()))
            ]
            psi_1x_list = [
                cphase_result[1].states[-1].transform(list(self.eigvec.values()))
            ]

        phase0x_arr = np.array(
            [
                cmath.phase(psi0x.ptrace(self.q_dic[self.QH]['id'])[1, 0])
                for psi0x in psi_0x_list
            ]
        )
        phase1x_arr = np.array(
            [
                cmath.phase(psi1x.ptrace(self.q_dic[self.QH]['id'])[1, 0])
                for psi1x in psi_1x_list
            ]
        )
        cphase = phase1x_arr - phase0x_arr
        cphase = abs(np.arctan2(np.sin(cphase), np.cos(cphase)))
        self.cphase = cphase
        if plot_flag:
            self.plotter.plot(
                self.evolution_result.times,
                self.cphase,
                y_name='$\phi$',
                xlabel='t (ns)',
                ylabel=r'$\Phi_c$',
                flag_save=self.flag_fig,
            )
        return self.cphase

    def analyze_fidelity(
        self, fidelity_result: Union[qp.Qobj, list] = None, type: str = 'ave'
    ):

        if fidelity_result is None:
            fidelity_result = self.fidelity_result
        if type == 'ave':
            # U_result = fidelity_result.transform(list(self.eigvec.values()))
            index_list = generate_idx(self, [self.QH, self.QL])
            if self.spectator_bit is not None:
                id_list = [
                    self.q_dic[q]['id'] for q in [self.QH, self.QL, self.spectator_bit]
                ]
                id_list.sort()
                substate_list = [
                    (*state, 1) for state in qp.state_number_enumerate([2] * 2)
                ]
                state_list = [
                    tuple(
                        [
                            substate[id_list.index(id)] if id in id_list else 0
                            for id in range(len(self.q_dic))
                        ]
                    )
                    for substate in substate_list
                ]
                index_list = [
                    qp.state_number_index(self.dim, state) for state in state_list
                ]

            state_out_list = [
                result.states[-1].transform(list(self.eigvec.values()))
                for result in fidelity_result
            ]
            U_sub = qp.Qobj(
                np.hstack([rho_out[index_list] for rho_out in state_out_list]),
                dims=cz_gate().dims,
                shape=cz_gate().shape,
            )
            # U_sub = qp.Qobj(U_result[index_list, :][:, index_list], dims=cz_gate().dims, shape=cz_gate().shape)
            error, _ = errorU_cali_phi(U_sub, cz_gate())

        elif type == 'qpt':
            index_list = generate_idx(self, [self.QH, self.QL])

            init_state_list = qpt_rho_in(cz_gate())
            rho_in_list = [
                self.Ob((self.QL, self.QH), init_state)
                for init_state in init_state_list
            ]
            rho_in_sub_list = [
                rho_in[index_list, :][:, index_list] for rho_in in rho_in_list
            ]

            rho_out_list = [
                result.states[-1].transform(list(self.eigvec.values()))
                for result in fidelity_result
            ]
            rho_out_sub_list = [
                qp.ket2dm(rho_out)[index_list, :][:, index_list]
                for rho_out in rho_out_list
            ]  # 这里只保留了非计算比特的0态，与平均保真度一致

            error = qpt(rho_in_sub_list, rho_out_sub_list, cz_gate())

        return error

    def analyze2d_leak(self, scan_name: list, mea_ops=None):
        if mea_ops is None:
            mea_ops = self.Od((self.QL, self.QH), (1, 1))
        result_list = [result_leak.states[-1] for result_leak in self.result_leak_2d]
        expect_list = qp.expect(mea_ops, result_list)
        w0_list = self.scan2d_leak_para[scan_name[0]]
        w1_list = self.scan2d_leak_para[scan_name[1]]
        expect_arr = expect_list.reshape(len(w0_list), len(w1_list))
        leak_arr = 1 - expect_arr

        scan_name = [name.replace('_', '') for name in scan_name]
        self.plotter.plot_heatmap(
            w0_list,
            w1_list,
            leak_arr.T,
            xlabel=fr'$\omega_{{{scan_name[0]}}}$',
            ylabel=fr'$\omega_{{{scan_name[1]}}}$',
            zlabel='1 - $P_{11}$',
            title=f'CZ Leakage',
            norm='log',
        )

    def analyze2d_cphase(self, scan_name: list):
        result_list = self.result_cphase_2d
        cphase_list = [
            self.analyze_cphase(result, plot_flag=False)[-1] for result in result_list
        ]
        w0_list = self.scan2d_cphase_para[scan_name[0]]
        w1_list = self.scan2d_cphase_para[scan_name[1]]
        cphase_arr = np.array(cphase_list).reshape(len(w0_list), len(w1_list))

        scan_name = [name.replace('_', '') for name in scan_name]
        self.plotter.plot_heatmap(
            w0_list,
            w1_list,
            np.pi - cphase_arr.T,
            xlabel=fr'$\omega_{{{scan_name[0]}}}$',
            ylabel=fr'$\omega_{{{scan_name[1]}}}$',
            zlabel=r'$\pi - \Phi_c$',
            title=f'CZ Cphase',
            norm='log',
        )

    def analyze2d_fidelity(self, scan_name: list):
        result_list = self.result_fidelity_2d
        # error_list = qp.parallel_map(self.analyze_fidelity, result_list, progress_bar=True, num_cpus=self.num_cpus)
        error_list = [self.analyze_fidelity(result) for result in result_list]
        w0_list = self.scan2d_fidelity_para[scan_name[0]]
        w1_list = self.scan2d_fidelity_para[scan_name[1]]
        error_arr = np.array(error_list).reshape(len(w0_list), len(w1_list))

        scan_name = [name.replace('_', '') for name in scan_name]
        self.plotter.plot_heatmap(
            w0_list,
            w1_list,
            error_arr.T,
            xlabel=fr'$\omega_{{{scan_name[0]}}}$',
            ylabel=fr'$\omega_{{{scan_name[1]}}}$',
            zlabel=r'error',
            title=f'CZ Fidelity',
            norm='log',
        )

    @parallel_allocation
    def fidelity_calculate(
        self,
        w_list,
        bit_name_list: list,
        width: float,
        type: str = 'ave',
        log_flag: bool = False,
        **kwargs,
    ):
        if isinstance(w_list, float):
            w_list = [w_list]
        arg_dic = dict(zip(bit_name_list, w_list))
        fidelity_result = self.run_fidelity(arg_dic, width, type=type)
        error = self.analyze_fidelity(fidelity_result=fidelity_result, type=type)
        if log_flag:
            print(arg_dic, error)
        return error

    def fidelity_optimize(
        self,
        bit_name_list: list,
        width: float,
        w0_list=None,
        bounds=None,
        type: str = 'ave',
        log_flag: bool = True,
    ):
        """ """
        q_name = [bit_name for bit_name in bit_name_list if bit_name[0] == 'Q'][0]
        c_name = [bit_name for bit_name in bit_name_list if bit_name[0] == 'C'][0]
        if w0_list is None:
            wq_work0, E_diff0 = self.resonate_point(
                q_name,
                (self.q_dic[self.QL]['w_idle'], self.q_dic[self.QH]['w_idle']),
                ((self.QL, self.QH), (1, 1)),
                ((self.QL, self.QH), (0, 2)),
            )
            q_dict_exp = {q_name: {'w_idle': wq_work0}}
            print(q_dict_exp)
            g = 1 / (2 * (width - 2 * self.exp_args['buffer']))
            # g = 1 / (2 * (width))
            wc_work0, g_diff0 = self.resonate_point(
                c_name,
                (self.q_dic[self.QH]['w_idle'], self.q_dic[c_name]['w_idle']),
                (('Q1', 'Q2'), (1, 1)),
                (('Q1', 'Q2'), (0, 2)),
                q_dict_exp=q_dict_exp,
                offset=g * 2,
            )
            q_dict_exp1 = {'C1_2': {'w_idle': wc_work0}}
            print(q_dict_exp1)
            wq_work1, E_diff1 = self.resonate_point(
                q_name,
                (self.q_dic[self.QL]['w_idle'], self.q_dic[self.QH]['w_idle']),
                ((self.QL, self.QH), (1, 1)),
                ((self.QL, self.QH), (0, 2)),
                q_dict_exp=q_dict_exp1,
            )
            dict_work = dict(zip([q_name, c_name], [wq_work1, wc_work0]))
            print(dict_work)
            w0_list = [dict_work.get(key) for key in bit_name_list]
        if bounds is None:
            dict_bounds = dict(
                zip(
                    [q_name, c_name],
                    [
                        (
                            w0_list[bit_name_list.index(q_name)] - 50e-3,
                            w0_list[bit_name_list.index(q_name)] + 50e-3,
                        ),
                        (
                            w0_list[bit_name_list.index(c_name)] - 200e-3,
                            w0_list[bit_name_list.index(c_name)] + 200e-3,
                        ),
                    ],
                )
            )
            bounds = (dict_bounds.get(key) for key in bit_name_list)

        result = minimize(
            self.fidelity_calculate,
            w0_list,
            args=(bit_name_list, width, type, log_flag),
            method='Nelder-Mead',
            bounds=bounds,
        )
        w_list_best = result.x
        paras_best = dict(zip(bit_name_list, w_list_best))
        fidelity_best = result.fun

        if self.flag_data:
            data = {
                'work_point': paras_best,
                'fidelity_best': fidelity_best,
                'chip_dic': self.chip_dic,
                'QL-QH': [self.QL, self.QH],
                'gate_time': width,
                'exp_args': self.exp_args,
            }
            save_data(
                f'cz_best_work_point_{self.QL}-{self.QH}_width={width}ns',
                'qu',
                data,
                root_path=self.root_path,
            )

        return paras_best, fidelity_best


if __name__ == '__main__':
    # pass
    from functions import *
    from functions.plot_tools import PlotTool
    from solvers.solvers import Solver

    yaml_path = r'/home/wangpeng/Simulation jupyter/jupyter/transmon_qubit/观察比特对可调耦合两比特门/chip_param_2q.yaml'
    cz = CZ(
        chip_path=yaml_path,
        dim=3,
        QL='Q1',
        QH='Q2',
        flag_fig=True,
        time_step=0.1,
        flag_data=False,
    )
    # plt.style.use('bmh')
    exp_args = {'shape': 'FlattopGaussian', 'arg_type': 'wq', 'sigma': 1.5, 'buffer': 5}
    cz.exp_args = exp_args
    bit_name_list = ['Q2', 'C1_2']
    width = 40
    cz.fidelity_optimize(bit_name_list, width)
