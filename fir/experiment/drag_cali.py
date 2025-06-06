# -*- coding: utf-8 -*-
# @Time     : 2022/9/27 17:04
# @Author   : WTL
# @Software : PyCharm
from typing import Union
import numpy as np
import qutip as qp
from scipy.optimize import minimize
from pathlib import Path

from experiment.experiment_base import ExpBaseDynamic
from pulse import *  # noqa
from functions import *
from functions.tools import save_data
from functions.containers import ExpBaseDynamicContainer


class DragCali(ExpBaseDynamic):
    def __init__(self, **kwargs):
        kwargs.get('plot_params', {}).setdefault(
            'units', {'x-t': 'ns', 'x-w': 'MHz', 'y-t': 'ns', 'y-w': 'MHz'}
        )
        kwargs.setdefault('flag_init_1q_gates', ('Drag',))

        container = ExpBaseDynamicContainer(**kwargs)
        super().__init__(**container)

    def run_error(
        self,
        Drag_paras,
        qubit: str,
        width: float,
        gate_type: str = 'X',
        pulse_shape: str = 'Drag',
    ):
        amp, detu = Drag_paras  # noqa
        wd = self.q_dic[qubit]['w_idle']
        if gate_type in ['X', 'X/2']:
            phi = 0  # noqa
        elif gate_type in ['Y', 'Y/2']:
            phi = np.pi / 2  # noqa
        else:
            raise ValueError(f'gate type {gate_type} is not supported.')

        if pulse_shape in ['Drag', 'DragGaussian', 'DragTanh']:
            xpulse = eval(f'{pulse_shape}(width, wd, amp, detu=detu, phi=phi)')
            xpulse.get_pulse()
        else:
            raise ValueError(f'Pulse shape {pulse_shape} is not supported.')

        pulse_dic = {qubit: {'XY': xpulse}}
        self.exp_args = {'width': width, 'wd': wd, 'qubit': qubit}

        Ufull = self.propagator_solve(pulse_dic, wR=self.exp_args.get('wR'))[-1]
        U = np.zeros((2, 2), dtype=np.complex128)
        for r in range(2):
            for c in range(2):
                U[r, c] = complex(self.Kd(qubit, r).overlap(Ufull * self.Kd(qubit, c)))

        error, phi_cali = errorU_cali_phi(qp.Qobj(U), U_XY(gate_type))
        self.exp_args['phi_cali'] = phi_cali[0]
        # temp tests
        self.exp_args['U'] = U
        return error

    def run_opti(
        self,
        qubit: str,
        width: float,
        gate_type: str = 'X',
        pulse_shape: str = 'Drag',
        suffix: str = '',
    ):
        Drag_paras0 = np.array(
            [
                1 / width,
                0,
            ]
        )
        self.result = minimize(
            self.run_error,
            x0=Drag_paras0,
            args=(qubit, width, gate_type, pulse_shape),
            method='Nelder-Mead',
            bounds=((0, 1.1 / width), (None, None)),
        )

        amp, detu, *_ = self.result.x
        wd = self.exp_args['wd']  # noqa
        phi_cali = self.exp_args['phi_cali']  # noqa
        print(f'Drag amp={amp * 1e3}MHz, detu/2*t={detu * width}pi\n{self.result}')
        # temp test
        U_phi_opti = U_rphi(self.exp_args['U'], phi_cali)
        print(f'phi_cali: {phi_cali}')
        SU2_param(U_phi_opti)
        SU2_param(U_XY(gate_type))

        if gate_type in ['X', 'X/2']:
            phi = 0  # noqa
        elif gate_type in ['Y', 'Y/2']:
            phi = np.pi / 2  # noqa
        else:
            raise ValueError(f'gate type {gate_type} is not supported.')

        if pulse_shape in ['Drag', 'DragGaussian', 'DragTanh']:
            drag_opti = eval(
                f'{pulse_shape}(width, wd, amp, detu=detu, phi=phi+phi_cali)'
            )
            drag_opti.get_pulse()
        else:
            raise ValueError(f'{gate_type} is not supported gate type.')

        self.exp_args['drag_opti'] = drag_opti

        if self.flag_gate:
            if pulse_shape != 'Drag':
                suffix += f'[{pulse_shape}]'
            self.save_gate(
                f'{"o".join(gate_type.split("/"))}{suffix}@{qubit}', drag_opti
            )

        return drag_opti

    def analyze(self):
        qubit = self.exp_args['qubit']

        drag_opti = self.exp_args['drag_opti']
        self.plotter.plot_pulse(drag_opti, ylabel=r'$\Omega$', plot_type='both')

        pulse_dic = {qubit: {'XY': drag_opti}}
        init_state = self.Kd(qubit, 0)
        mea_ops = [self.Od(qubit, 0), self.Od(qubit, 1)]
        res = self.state_solve(pulse_dic, init_state, e_ops=mea_ops)
        self.plotter.plot_evolution(
            res.times, res.expect, expec_name_list=[r'$P_{0}$', r'$P_{1}$']
        )

        if self.flag_data:
            save_data(
                f'Drag width-P0-P1',
                'dat',
                np.c_[res.times, res.expect[0], res.expect[1]],
                root_path=self.root_path,
            )


if __name__ == '__main__':
    pass
