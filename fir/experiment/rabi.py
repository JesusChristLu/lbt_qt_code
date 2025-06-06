# -*- coding: utf-8 -*-
# @Time     : 2022/10/19 20:02
# @Author   : WTL
# @Software : PyCharm
from typing import Union
from pathlib import Path
import numpy as np
import copy
import qutip as qp
from operator import itemgetter

from experiment.experiment_base import ExpBaseDynamic
from pulse.pulse_base import PulseBase
from functions import *
from functions.containers import ExpBaseDynamicContainer


class Rabi(ExpBaseDynamic):
    def __init__(self, **kwargs):
        kwargs.get('plot_params', {}).setdefault(
            'units', {'x-t': 'ns', 'x-w': 'MHz', 'y-t': 'ns', 'y-w': 'MHz'}
        )
        kwargs.setdefault('flag_init_1q_gates', ('Drag',))

        container = ExpBaseDynamicContainer(**kwargs)
        super().__init__(**container)

    def scan(
        self, qubit: str, scan_type: str, arg_list: np.ndarray, gate: PulseBase = None
    ):
        self.exp_args.update(
            {
                'qubit': qubit,
                'scan_type': scan_type,
                'arg_list': arg_list,
            }
        )

        # 初始化波形
        xpulse0 = self.gate_load[f'X@{qubit}'] if gate is None else gate
        # 根据实验参数生成波形列表
        xpulse_dic_list = []
        for arg in arg_list:
            xpulse = copy.deepcopy(xpulse0)
            if scan_type == 'width':
                xpulse(width=arg)
            elif scan_type == 'amp':
                xpulse(amp=arg)
            else:
                raise ValueError(f'scan_type {scan_type} is not supported.')
            print(xpulse)
            xpulse_dic_list.append({qubit: {'XY': xpulse}})

        # 放入求解器中求解
        init_state = self.exp_args.get('init_state', self.Kd(qubit, 0))

        mea_ops = self.exp_args.get('mea_ops', [self.Od(qubit, 0), self.Od(qubit, 1)])

        self.result = qp.parallel_map(
            self.state_solve,
            xpulse_dic_list,
            task_kwargs={
                'state0': init_state,
                'e_ops': mea_ops,
                'wR': self.exp_args.get('wR'),
            },
        )

        save_data(
            f'{qubit} scan {scan_type} exp_args-x-result',
            'qu',
            self.exp_args,
            self.result,
            root_path=self.root_path,
        )

    def scan_amp_opt(
        self, qubit: str, gate_type: str, amp_list: np.ndarray, N: int = 5
    ):
        self.exp_args.update(
            {'qubit': qubit, 'amp_list': amp_list, 'gate_type': gate_type, 'N': N}
        )

        # 初始化波形
        gate_name = f'{"o".join(gate_type.split("/"))}@{qubit}'
        xpulse0 = self.gate_load[gate_name]
        # 根据实验参数生成波形列表
        xpulse_dic_list = []
        for amp in amp_list:
            xpulse = copy.deepcopy(xpulse0)
            xpulse(amp=amp)
            if gate_type == 'X':
                xpulse *= N
            elif gate_type == 'X/2':
                xpulse *= 2 * N
            else:
                raise ValueError(f'gate_type {gate_type} is not supported.')
            xpulse_dic_list.append({qubit: {'XY': xpulse}})

        # 放入求解器中求解
        init_state = self.exp_args.get('init_state', self.Kd(qubit, 0))

        mea_ops = self.exp_args.get('mea_ops', [self.Od(qubit, 0), self.Od(qubit, 1)])

        self.result = qp.parallel_map(
            self.state_solve,
            xpulse_dic_list,
            task_kwargs={
                'state0': init_state,
                'e_ops': mea_ops,
                'wR': self.exp_args.get('wR'),
            },
        )

        save_data(
            f'{qubit} scan {gate_name} exp_args-result',
            'qu',
            self.exp_args,
            self.result,
            root_path=self.root_path,
        )

    def analyze(self, **kwargs):
        # qubit = self.exp_args['qubit']
        # scan_type = self.exp_args['scan_type']
        qubit, scan_type, arg_list = itemgetter('qubit', 'scan_type', 'arg_list')(
            self.exp_args
        )

        analyze_names = kwargs.get(
            'analyze_names', ('P0', 'P1')
        )  # analy_name_list指定需要拟合的数据
        extra_names = kwargs.get('extra_names', ())  # extra_name_list指定不需要拟合的数据
        res_analy = dict.fromkeys(analyze_names)
        res_extra = dict.fromkeys(extra_names)

        for idx, key in enumerate(analyze_names + extra_names):
            if key in analyze_names:
                res_analy[key] = np.asarray(
                    [res.expect[idx][-1] for res in self.result]
                )
            if key in extra_names:
                res_extra[key] = np.asarray(
                    [res.expect[idx][-1] for res in self.result]
                )

        popt_list = []
        rmse_list = []
        fun_cos_list = []
        for key in analyze_names:
            popt, rmse, fun_cos = fit_cos(arg_list, res_analy[key])
            popt_list.append(popt)
            rmse_list.append(rmse)
            fun_cos_list.append(fun_cos)
        popt = popt_list[np.argmin(rmse_list)]

        if scan_type == 'width':
            freq = popt[0]
            xtype = 't'
            title = f'{qubit} scan {scan_type}(freq={freq*1e3:.3f}MHz)'
        elif scan_type == 'amp':
            x_interp = np.linspace(arg_list[0], arg_list[-1], 501)
            if rmse_list[0] < rmse_list[1]:
                opt_idx = np.argmin(fun_cos_list[0](x_interp))
            else:
                opt_idx = np.argmax(fun_cos_list[1](x_interp))
            opt_amp = x_interp[opt_idx]
            xtype = 'w'
            title = f'{qubit} scan {scan_type}(amp={opt_amp*1e3:.3f}MHz)'

            if self.flag_gate:
                gate = self.gate_load[f'X@{qubit}']
                gate(amp=opt_amp)

                gateo2 = self.gate_load[f'Xo2@{qubit}']
                gateo2(amp=opt_amp / 2)
                self.save_gate([f'X@{qubit}', f'Xo2@{qubit}'], [gate, gateo2])
        else:
            raise ValueError(f'scan_type {scan_type} is not supported.')

        # fit_fun = [fun_cos0, fun_cos1]
        self.plotter.plot_lines_fit(
            arg_list,
            list(res_analy.values()),
            analyze_names,
            fun_cos_list,
            title=title,
            xtype=xtype,
            xlabel=scan_type,
        )

        if extra_names:
            self.plotter.plot_lines(
                arg_list,
                list(res_extra.values()),
                extra_names,
                xtype=xtype,
                xlabel=scan_type,
            )

        if self.flag_data:
            save_data(
                title,
                'dat',
                np.vstack([arg_list, list(res_analy.values())]).T,
                root_path=self.root_path,
            )

    def analyze_amp_opt(self):
        qubit, amp_list, gate_type, N = itemgetter(
            'qubit', 'amp_list', 'gate_type', 'N'
        )(self.exp_args)
        gate_name = f'{"o".join(gate_type.split("/"))}@{qubit}'

        resP0 = np.asarray([res.expect[0][-1] for res in self.result])
        resP1 = np.asarray([res.expect[1][-1] for res in self.result])

        popt0, rmse0, fun_cos0 = fit_cos(amp_list, resP0)
        popt1, rmse1, fun_cos1 = fit_cos(amp_list, resP1)

        x_interp = np.linspace(amp_list[0], amp_list[-1], 501)
        if N % 2 == 0:
            if rmse0 < rmse1:
                opt_idx = np.argmax(fun_cos0(x_interp))
            else:
                opt_idx = np.argmin(fun_cos1(x_interp))
        else:
            if rmse0 < rmse1:
                opt_idx = np.argmin(fun_cos0(x_interp))
            else:
                opt_idx = np.argmax(fun_cos1(x_interp))

        opt_amp = x_interp[opt_idx]
        title = f'{qubit} scan {gate_name} AmpOpt(amp={opt_amp * 1e3:.3f}MHz)'

        fit_fun = [fun_cos0, fun_cos1]
        self.plotter.plot_lines_fit(
            amp_list,
            [resP0, resP1],
            ['P0', 'P1'],
            fit_fun,
            title=title,
            xtype='w',
            xlabel='amp',
        )

        if self.flag_gate:
            gate = self.gate_load[gate_name]
            gate(amp=opt_amp)

            pulse_dic = {qubit: {'XY': gate}}
            Ufull = self.propagator_solve(pulse_dic, wR=self.exp_args.get('wR'))[-1]
            print(f'Ufull: {Ufull}')
            U = np.zeros((2, 2), dtype=np.complex128)
            for r in range(2):
                for c in range(2):
                    U[r, c] = complex(
                        self.Kd(qubit, r).overlap(Ufull * self.Kd(qubit, c))
                    )
            error, phi_cali = errorU_cali_phi(qp.Qobj(U), U_XY(gate_type))

            phi = gate.phi
            gate(phi=phi + phi_cali[0])
            print(gate)
            self.save_gate(gate_name, gate)

        if self.flag_data:
            save_data(
                title, 'dat', np.c_[amp_list, resP0, resP1], root_path=self.root_path
            )
