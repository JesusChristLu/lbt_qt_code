# -*- coding: utf-8 -*-
# @Time     : 2022/10/20 15:12
# @Author   : WTL
# @Software : PyCharm
from typing import Union
from pathlib import Path
import numpy as np
import copy
import qutip as qp
from operator import itemgetter

from experiment.experiment_base import ExpBaseDynamic
from functions import *
from functions.containers import ExpBaseDynamicContainer


class APE(ExpBaseDynamic):
    def __init__(self, **kwargs):
        kwargs.get('plot_params', {}).setdefault(
            'units', {'x-t': 'ns', 'x-w': 'MHz', 'y-t': 'ns', 'y-w': 'MHz'}
        )
        kwargs.setdefault('flag_init_1q_gates', ('Drag',))

        container = ExpBaseDynamicContainer(**kwargs)
        super().__init__(**container)

    def scan_detune(
        self,
        qubit: str,
        detune_list: np.ndarray,
        N_list: np.ndarray,
        gate_type: str = 'X',
    ):
        self.exp_args.update(
            {
                'qubit': qubit,
                'gate_type': gate_type,
                'detune_list': detune_list,
                'N_list': N_list,
            }
        )

        # 初始化波形
        xpulse0 = self.gate_load[f'{"o".join(gate_type.split("/"))}@{qubit}']
        # 根据实验参数生成波形列表
        xpulse_dic_list = []
        for N in N_list:
            for detune in detune_list:
                xpulse_p = copy.deepcopy(xpulse0)
                xpulse_p(phi=0, detu=detune)
                xpulse_m = copy.deepcopy(xpulse0)
                xpulse_m(phi=-np.pi, detu=detune)
                xpulse_APE = (xpulse_p + xpulse_m) * N
                print(xpulse_APE)
                xpulse_dic_list.append({qubit: {'XY': xpulse_APE}})

        # 放入求解器中求解
        init_state = self.exp_args.get('init_state', self.Kd(qubit, 0))

        mea_ops = self.exp_args.get('mea_ops', [self.Od(qubit, 0)])

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
            f'{qubit} APE scan detune-exp_args-result',
            'qu',
            self.exp_args,
            self.result,
            root_path=self.root_path,
        )

    def scan_phi(self, qubit: str):
        pass

    def analyze_detune(self):
        qubit, gate_type, detune_list, N_list = itemgetter(
            'qubit', 'gate_type', 'detune_list', 'N_list'
        )(self.exp_args)

        resP0 = np.asarray([res.expect[0][-1] for res in self.result]).reshape(
            len(N_list), len(detune_list)
        )
        resP1 = np.asarray([res.expect[1][-1] for res in self.result]).reshape(
            len(N_list), len(detune_list)
        )

        fun_cos_list = []
        detune_opt_list = []
        for i, N in enumerate(N_list):
            resP0_N = resP0[i]
            popt, rmse, fun_cos = fit_cos(detune_list, resP0_N)
            fun_cos_list.append(fun_cos)
            x_interp = np.linspace(detune_list[0], detune_list[-1], 501)
            detune_opt = x_interp[np.argmax(fun_cos(x_interp))]
            print(f'N={N}, optimal detune: {detune_opt*1e3:.3f}MHz')
            detune_opt_list.append(detune_opt)

        detune = np.mean(detune_opt_list)
        title = f'{qubit} APE scan {"o".join(gate_type.split("/"))}(detune={detune*1e3:.3f}MHz)'
        self.plotter.plot_lines_fit(
            detune_list,
            resP0,
            [f'N={N}' for N in N_list],
            fun_cos_list,
            xtype='w',
            xlabel='detune',
            title=title,
        )
        self.plotter.plot_lines(
            detune_list,
            resP1,
            [f'Q1-P1-N={N}' for N in N_list],
            xtype='w',
            xlabel='detune',
        )

        if self.flag_gate:
            gate_name = f'{"o".join(gate_type.split("/"))}@{qubit}'
            gate = self.gate_load[gate_name]
            gate(detu=detune)

            pulse_dic = {qubit: {'XY': gate}}
            Ufull = self.propagator_solve(pulse_dic, wR=self.exp_args.get('wR'))[-1]
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
                title,
                'dat',
                np.vstack([detune_list, *resP0]).T,
                root_path=self.root_path,
            )
