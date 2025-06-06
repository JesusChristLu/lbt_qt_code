# -*- coding: utf-8 -*-
# @Time     : 2022/10/4 19:59
# @Author   : WTL
# @Software : PyCharm
import itertools
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import copy
import qutip as qp
import geatpy as ea
from scipy.linalg import fractional_matrix_power
from functools import reduce
import operator
from sklearn.metrics import mean_squared_error
from functools import partial
import time
from datetime import datetime

from experiment.experiment_base import ExpBaseDynamic
from experiment.drag_cali import DragCali
from pulse import *
from pulse.pulse_base import XYPulseBase
from pathlib import Path
from functions import *
from functions.containers import ExpBaseDynamicContainer


class CR(ExpBaseDynamic):
    def __init__(self, **kwargs):
        kwargs.get('plot_params', {}).setdefault(
            'units', {'x-t': 'ns', 'x-w': 'MHz', 'y-t': 'ns', 'y-w': 'MHz'}
        )
        kwargs.setdefault(
            'flag_init_1q_gates',
            (
                'Drag',
                'FlattopGaussian_wq',
                'FlattopGaussian_flux',
                'Constant_wq',
                'Constant_flux',
            ),
        )

        container = ExpBaseDynamicContainer(**kwargs)
        super().__init__(**container)

    def HR(self, Omega: float, QC: str, QT: str, wd: float = None):
        """
        Hsys = H0 + Hqq + Hd
             = w1 * a1d*a1 + eta1/2 * a1d*a1d*a1*a1 + w2 * a2d*a2 + eta2/2 * a2d*a2d*a2*a2 +
               J * (a1d*a2 + a1*a2d) +
               O * cos(wd*t) * (a1d + a1)
               [assume the drive term contains only a drive on X-quadrature of the QC with a constant amplitude O
               wd = w2~]
        Rotation frame: R = exp(1j*wd*t*(a1d*a1 + a2d*a2))
        Hsys_R = R * Hsys * Rd + 1j * R' * Rd
               = R * H0 * Rd + R * (Hqq + Hd) * Rd + 1j * (1j*wd*(a1d*a1 + a2d*a2)) * R * Rd
               = H0 - wd * (a1d*a1 + a2d*a2) + R * (Hqq + Hd) * Rd
               = (w1-wd) * a1d*a1 + (w2-wd) * a2d*a2 + eta1/2 * a1d*a1d*a1*a1 + eta2/2 * a2d*a2d*a2*a2 +
                 exp(1j*wd*t*(a1d*a1 + a2d*a2)) * J * (a1d*a2 + a1*a2d) * exp(-1j*wd*t*(a1d*a1 + a2d*a2))[term1] +
                 exp(1j*wd*t*(a1d*a1 + a2d*a2)) * O(t) * cos(wd*t) * (a1d + a1) * exp(-1j*wd*t*(a1d*a1 + a2d*a2))[term2]
        (BCH lemma: exp(A)*B*exp(-A) = B + [A, B] + 1/2! * [A, [A, B]] + ... + 1/n! * [Cn[A], B])
        For term1: [a1d*a1 + a2d*a2, a1d*a2 + a1*a2d] = 0 -> R * Hqq * Rd = J * (a1d*a2 + a1*a2d)
        For term2: [Cn[a1d*a1 + a2d*a2], a1d] = a1d; [Cn[a1d*a1 + a2d*a2], a1] = (-1)**n * a1;
                   -> R * Hd * Rd = O * [exp(1j*wd*t)+exp(-1j*wd*t)]/2 * [exp(1j*wd*t) * a1d + exp(-1j*wd*t) * a1]
                                  = O/2 * { [exp(2j*wd*t)+1] * a1d + [exp(-2j*wd*t)+1] * a1 }
                                  => O/2 * (a1d + a1) [RWA]
        Hsys_RWA = (w1-wd) * a1d*a1 + (w2-wd) * a2d*a2 + eta1/2 * a1d*a1d*a1*a1 + eta2/2 * a2d*a2d*a2*a2 +
                   J * (a1d*a2 + a1*a2d) +  O/2 * (a1d + a1)
        :param wd:
        :param Omega:
        :param QC:
        :param QT:
        :return:
        """
        self.flag_Hqq_RWA = True
        wd = wd if wd else self.q_dic[QT]['w']
        wtarget = self.q_dic[QT]['w']
        self.q_dic[QC]['w'] = wd
        Dc = self.q_dic[QC]['w'] - wtarget
        Dt = self.q_dic[QT]['w'] - wtarget
        J = self.rho_map[f'{QC}-{QT}'] * np.sqrt(
            self.q_dic[QC]['w'] * self.q_dic[QT]['w']
        )

        H0 = (
            2 * np.pi * (Dc * self.Hz_op[QC] + Dt * self.Hz_op[QT])
            + self.Halpha[QC]
            + self.Halpha[QT]
        )
        HR = (
            H0
            + 2 * np.pi * J * self.Hqq_op[f'{QC}-{QT}']
            + 2 * np.pi * Omega / 2 * self.Hx_op[QC]
        )
        print(f'{self.q_dic}')
        return H0, HR

    # def block_diag_4x4(self, Omega: float, QC: str, QT: str):
    #     H0_exp, Hsys_exp = self.HR(Omega, QC, QT)
    #     # H0_exp = self.H0(self.q_dic)
    #     # Hsys_exp = self.Hsys(self.q_dic, self.rho_map)
    #     ei_energy, ei_vector = Hsys_exp.eigenstates()
    #     ei_energy0, ei_vector0 = H0_exp.eigenstates()
    #     ei_vector_bare, ei_vector_dress = self.vector_sort(ei_vector0, ei_vector)
    #     print(f'ei_vector_dress:\n{ei_vector_dress}')
    #
    #     state_numbers = ([0, 0], [0, 1], [1, 0], [1, 1])  # tuple(qp.state_number_enumerate([2, 2], excitations=2))
    #     state_idxs = tuple(qp.state_number_index(self.dim, state) for state in state_numbers)
    #     ei_vector_4x4 = ei_vector_dress[state_idxs, :][:, state_idxs]
    #     Hsys4x4 = Hsys_exp[state_idxs, :][:, state_idxs]
    #
    #     bd0_idxs = [0, 1]  # tuple(qp.state_number_index([2, 2], state) for state in ([0, 0], [0, 1]))
    #     bd1_idxs = [2, 3]  # tuple(qp.state_number_index([2, 2], state) for state in ([1, 0], [1, 1]))
    #     bd_ax0_idxs = np.repeat(bd0_idxs + bd1_idxs, 2)
    #     bd_ax1_idxs = np.hstack((np.tile(bd0_idxs, 2), np.tile(bd1_idxs, 2)))
    #
    #     print(f'ax0_idxs: {bd_ax0_idxs}, ax1_idxs: {bd_ax1_idxs}')
    #
    #     ei_vector_4x4_bd = np.zeros_like(ei_vector_4x4)
    #     ei_vector_4x4_bd[bd_ax0_idxs, bd_ax1_idxs] = ei_vector_4x4[bd_ax0_idxs, bd_ax1_idxs]
    #     print(f'ei_vector_4x4:\n{ei_vector_4x4}')
    #     print(f'ei_vector_4x4_bd:\n{ei_vector_4x4_bd}')
    #
    #     Tmat = ei_vector_4x4 @ ei_vector_4x4_bd.conj().T @ \
    #            fractional_matrix_power(ei_vector_4x4_bd @ ei_vector_4x4_bd.conj().T, -1 / 2)
    #     print(f'Tmat:\n{Tmat}')
    #     Heff_4x4 = Tmat.conj().T @ Hsys4x4 @ Tmat
    #     return Heff_4x4

    def block_diag(
        self, Omega: float, QC: str, QT: str, wd: float = None, flag_fom: bool = False
    ):
        H0_exp, Hsys_exp = self.HR(Omega, QC, QT, wd)
        # H0_exp = self.H0(self.q_dic)
        # Hsys_exp = self.Hsys(self.q_dic, self.rho_map)
        ei_energy, ei_vector = Hsys_exp.eigenstates()
        ei_energy0, ei_vector0 = H0_exp.eigenstates()
        ei_vector_bare, ei_vector_dress = self.vector_sort(ei_vector0, ei_vector)

        # exp_dic = {'qubits': {QC: {'w': wd}}}
        # ei_vector_bare, ei_vector_dress = self.eigen_solve(chip_dict_exp=)

        control_states = list(range(self.dim[self.q_dic[QC]['id']]))
        target_states = list(range(self.dim[self.q_dic[QT]['id']]))

        bd_space_idxs_sets = []
        for control_state in control_states:
            bd_n_idxs = tuple(
                qp.state_number_index(self.dim, [control_state, target_state])
                for target_state in target_states
            )
            bd_space_idxs_sets.append(bd_n_idxs)
        # 索引分块对角子空间
        bd_ax0_idxs = np.repeat(
            reduce(operator.add, bd_space_idxs_sets), len(target_states)
        )
        bd_ax1_idxs = np.hstack(
            tuple(np.tile(idxs, len(target_states)) for idxs in bd_space_idxs_sets)
        )
        ei_vector_bd = np.zeros_like(ei_vector_dress)
        ei_vector_bd[bd_ax0_idxs, bd_ax1_idxs] = ei_vector_dress[
            bd_ax0_idxs, bd_ax1_idxs
        ]
        # print(f'ei_vector_bd:\n{ei_vector_bd}')
        Tmat = (
            ei_vector_dress
            @ ei_vector_bd.conj().T
            @ fractional_matrix_power(ei_vector_bd @ ei_vector_bd.conj().T, -1 / 2)
        )
        # print(f'Tmat:\n{Tmat}')
        Heff = Tmat.conj().T @ np.array(Hsys_exp) @ Tmat

        if flag_fom:
            if not self.exp_args.get('fom'):
                self.exp_args['fom'] = []
            self.exp_args['fom'].append(self.fom_Heff(ei_vector_bd))
        return np.real(Heff)

    def Pauli_coef(
        self,
        Omega: float,
        wd: float,
        QC: str,
        QT: str,
        flag_fom: bool = False,
        Pauli_ops: Union[list, str] = 'all',
    ):
        Heff = self.block_diag(Omega, QC, QT, wd, flag_fom)
        state_numbers = ([0, 0], [0, 1], [1, 0], [1, 1])
        state_idxs = tuple(
            qp.state_number_index(self.dim, state) for state in state_numbers
        )
        Heff_4x4 = qp.Qobj(Heff[state_idxs, :][:, state_idxs], dims=[[2, 2]] * 2)

        Pauli_coef = {}
        if Pauli_ops == 'all':
            Pauli_ops = ['IX', 'IY', 'IZ', 'ZI', 'ZX', 'ZY', 'ZZ']

        for Pauli in Pauli_ops:
            if Pauli == 'IX':
                IX = qp.tensor(qp.qeye(2), qp.sigmax())
                Pauli_coef['IX/2'] = (Heff_4x4 * IX / 2).tr() / (2 * np.pi)
            elif Pauli == 'IY':
                IY = qp.tensor(qp.qeye(2), qp.sigmay())
                Pauli_coef['IY/2'] = (Heff_4x4 * IY / 2).tr() / (2 * np.pi)
            elif Pauli == 'IZ':
                IZ = qp.tensor(qp.qeye(2), qp.sigmaz())
                Pauli_coef['IZ/2'] = (Heff_4x4 * IZ / 2).tr() / (2 * np.pi)
            elif Pauli == 'ZX':
                ZX = qp.tensor(qp.sigmaz(), qp.sigmax())
                Pauli_coef['ZX/2'] = (Heff_4x4 * ZX / 2).tr() / (2 * np.pi)
            elif Pauli == 'ZY':
                ZY = qp.tensor(qp.sigmaz(), qp.sigmay())
                Pauli_coef['ZY/2'] = (Heff_4x4 * ZY / 2).tr() / (2 * np.pi)
            elif Pauli == 'ZZ':
                ZZ = qp.tensor(qp.sigmaz(), qp.sigmaz())
                Pauli_coef['ZZ/2'] = (Heff_4x4 * ZZ / 2).tr() / (2 * np.pi)
        return Pauli_coef

    def fom_Heff(self, ei_vector_bd):
        return np.trace(ei_vector_bd @ ei_vector_bd.conj().T) / np.prod(self.dim)

    def run_BD(
        self,
        drive_power_list: np.ndarray,
        QC: str,
        QT: str,
        Pauli_ops: Union[str, list] = 'all',
        wd: float = None,
        flag_fom: bool = False,
    ):
        """
        run least-action block diagonalization.
        :param drive_power_list:
        :param QC:
        :param QT:
        :param Pauli_ops:
        :param wd:
        :param flag_fom:
        :return:
        """
        self.x = drive_power_list
        self.exp_args.update({'QC': QC, 'QT': QT, 'fom': []})

        self.result = qp.parallel_map(
            self.Pauli_coef,
            drive_power_list,
            task_kwargs={
                'QC': QC,
                'QT': QT,
                'wd': wd,
                'flag_fom': flag_fom,
                'Pauli_ops': Pauli_ops,
            },
        )

    def _Pauli_coef_2D(
        self,
        arg,
        QC: str,
        QT: str,
        flag_fom: bool = False,
        Pauli_ops: Union[list, str] = 'all',
    ):
        wd, Omega = arg
        return self.Pauli_coef(Omega, wd, QC, QT, flag_fom, Pauli_ops)

    def run_BD_2D(
        self,
        drive_power_list: np.ndarray,
        wcontrol_list: np.ndarray,
        QC: str,
        QT: str,
        Pauli_ops: Union[str, list] = 'all',
        flag_fom: bool = False,
    ):
        self.x = drive_power_list
        self.y = wcontrol_list
        self.exp_args.update({'QC': QC, 'QT': QT, 'fom': []})

        # Pauli_coef_2D = lambda arg: self.Pauli_coef(arg[0], arg[1], QC, QT, flag_fom, Pauli_ops)
        mesh_args = list(itertools.product(self.y, self.x))
        self.result = qp.parallel_map(
            self._Pauli_coef_2D,
            mesh_args,
            task_kwargs={
                'QC': QC,
                'QT': QT,
                'flag_fom': flag_fom,
                'Pauli_ops': Pauli_ops,
            },
        )

    def analyze_BD(self):
        Paulis = self.result[0].keys()
        Pauli_coef = {}
        for key in Paulis:
            Pauli_coef[key] = np.array([res[key] for res in self.result])

        self.plotter.plot_coefs(self.x, Pauli_coef)

        return Pauli_coef

    def analyze_BD_2D(self):
        Paulis = self.result[0].keys()
        Pauli_coef = {}
        for key in Paulis:
            Pauli_coef[key] = np.real(np.array([res[key] for res in self.result]))

        ZX_array = Pauli_coef['ZX/2'].reshape(len(self.y), len(self.x))
        mask = np.argwhere(ZX_array > 20e-3)
        ZX_array[mask] = 0
        self.plotter.plot_heatmap(
            self.x * 1e3,
            self.y,
            ZX_array * 1e3,
            xlabel='Drive power(MHz)',
            ylabel='QT Freq(GHz)',
            zlabel='ZX(MHz)',
            title='ZX coefficient v.s. detuning and drive power',
        )

        # if self.flag_data:
        #     index_name = [r'w\power']
        #     dfZX = pd.DataFrame(ZX_array, index=self.x, columns=self.y)
        #     dfZX.index.names = index_name
        #     save_data('ZX coefficient v.s. detuning and drive power', 'xlsx', dfZX, root_path=self.root_path)

        return Pauli_coef

    def get_CR_pulse(
        self,
        args,
        wd,
        wR,
        QC: str,
        QT: str,
        pi_pulse: XYPulseBase,
        QC_phi: float = 0,
        QT_amp: float = 0,
        QT_phi: float = 0,
        CR_sigma: float = 1.25,
        CR_buffer: float = 5,
        drag_buffer: float = 0,
        flag_echo: bool = True,
    ):
        CR_width, QC_amp = args
        # wd = self.q_dic[QT]['w']
        # wR = self.exp_args.get('wR', {QC: wd, QT: wd})

        if flag_echo:
            pi_pulse(wR=wR[QC])
            pi_offset = Square(drag_buffer, wd=pi_pulse.wd, amp=0)
            pi_offset.get_pulse()
            QC_PI = pi_offset + pi_pulse + pi_offset

            QC_CR1 = FlattopGaussianEnv(
                CR_width / 2,
                wd,
                QC_amp,
                phi=QC_phi,
                sigma=CR_sigma,
                buffer=CR_buffer,
                wR=wR[QC],
            )
            # QC_CR1 = Square(CR_width / 2, wd, QC_amp, phi=QC_phi, wR=wR[QC])
            QC_CR1.get_pulse()
            QC_CR2 = FlattopGaussianEnv(
                CR_width / 2,
                wd,
                -QC_amp,
                phi=QC_phi,
                sigma=CR_sigma,
                buffer=CR_buffer,
                wR=wR[QC],
            )
            # QC_CR2 = Square(CR_width / 2, wd, -QC_amp, phi=QC_phi, wR=wR[QC])
            QC_CR2.get_pulse()
            xpulse = QC_CR1 + QC_PI + QC_CR2 + QC_PI
            # print(f'QC CR width: {xpulse.width}')
            pulse_dic = {QC: {'XY': xpulse}}
            # print(f'xpulse: {xpulse}\nQC CR1:\n{QC_CR1}\nPI1:\n{QC_PI}\nCR2:\n{QC_CR2}\n')

            if QT_amp != 0:
                QT_CR1 = FlattopGaussianEnv(
                    CR_width / 2,
                    wd,
                    QT_amp,
                    QT_phi,
                    sigma=CR_sigma,
                    buffer=CR_buffer,
                    wR=wR[QT],
                )
                QT_CR1.get_pulse()
                QT_wait1 = Square(QC_PI.width, wd, 0, wR=wR[QT])
                QT_wait1.get_pulse()
                QT_CR2 = FlattopGaussianEnv(
                    CR_width / 2,
                    wd,
                    -QT_amp,
                    QT_phi,
                    sigma=CR_sigma,
                    buffer=CR_buffer,
                    wR=wR[QT],
                )
                QT_CR2.get_pulse()
                QT_wait2 = copy.deepcopy(QT_wait1)
                pulse_dic.update({QT: {'XY': QT_CR1 + QT_wait1 + QT_CR2 + QT_wait2}})
        else:
            QC_CR = FlattopGaussianEnv(
                CR_width,
                wd,
                QC_amp,
                phi=QC_phi,
                sigma=CR_sigma,
                buffer=CR_buffer,
                wR=wR[QC],
            )
            # QC_CR = Square(CR_width, wd, QC_amp, phi=QC_phi, wR=wR[QC])
            QC_CR.get_pulse()
            # print(f'QC CR width: {QC_CR.width}')
            pulse_dic = {QC: {'XY': QC_CR}}
        return pulse_dic

    def scan_CR(
        self,
        scan_type: str,
        QC: str,
        QT: str,
        CR_width: Union[np.ndarray, float],
        QC_amp: Union[np.ndarray, float],
        QC_phi: float = 0,
        QT_amp: float = 0,
        QT_phi: float = 0,
        CR_sigma: float = 1.25,
        CR_buffer: float = 5,
        drag_buffer: float = 0,
        init_state=0,
        flag_load_pi: bool = True,
        flag_echo: bool = True,
    ):
        """
        Hamiltonian Tomography
        :param scan_type:
        :param flag_echo:
        :param init_state:
        :param drag_buffer:
        :param CR_sigma:
        :param CR_buffer:
        :param QT_phi:
        :param QT_amp:
        :param QC_phi:
        :param QC_amp:
        :param CR_width:
        :param flag_load_pi:
        :param QC:
        :param QT:
        :return:
        """
        if scan_type == 'width':
            self.x = CR_width
            QC_amp = [QC_amp]
        elif scan_type == 'amp':
            self.x = QC_amp
            CR_width = [CR_width]
        elif scan_type == 'both':
            self.x = CR_width
            self.y = QC_amp

        self.exp_args.update({'QC': QC, 'QT': QT})
        wd = self.exp_args.get('wd', self.q_dic[QT]['w'])
        wR = self.exp_args.get('wR', {QC: wd, QT: wd})
        if flag_load_pi:
            QC_PI = self.gate_load[f'X@{QC}']
        else:
            drag_cali = DragCali(
                self.chip_path,
                self.dim,
                self.time_step,
                self.sample_rate,
                flag_data=self.flag_data,
                flag_gate=self.flag_gate,
                flag_fig=self.flag_fig,
                flag_close=self.flag_close,
                root_path=self.root_path,
                **self.plot_params,
            )
            QC_PI = drag_cali.run_opti(QC, width=self.exp_args.get('drag_width', 20))

        # 根据实验参数生成波形列表
        time_start = time.time()
        print(f'start pulse generate...')
        arg_list = list(itertools.product(CR_width, QC_amp))
        if self.exp_args.get('flag_get_pulse_parallel'):
            pulse_dic_list = qp.parallel_map(
                self.get_CR_pulse,
                arg_list,
                task_args=(
                    wd,
                    wR,
                    QC,
                    QT,
                    QC_PI,
                    QC_phi,
                    QT_amp,
                    QT_phi,
                    CR_sigma,
                    CR_buffer,
                    drag_buffer,
                    flag_echo,
                ),
            )
        else:
            pulse_dic_list = [
                self.get_CR_pulse(
                    arg,
                    wd,
                    wR,
                    QC,
                    QT,
                    QC_PI,
                    QC_phi,
                    QT_amp,
                    QT_phi,
                    CR_sigma,
                    CR_buffer,
                    drag_buffer,
                    flag_echo,
                )
                for arg in arg_list
            ]
            # print(f'pulse_dic_list: {pulse_dic_list}')
        time_end = time.time()
        print(f'end pulse generate... consume {time_end - time_start}s')

        # 放入求解器中求解
        init_state = self.exp_args.get('init_state', self.Kd(QC, init_state))
        mea_ops = self.exp_args.get(
            'mea_ops',
            [
                self.Od((QC, QT), ('I', 'X')),
                self.Od((QC, QT), ('I', 'Y')),
                self.Od((QC, QT), ('I', 'Z')),
            ],
        )
        self.result = qp.parallel_map(
            self.state_solve,
            pulse_dic_list,
            task_kwargs={'state0': init_state, 'e_ops': mea_ops, 'wR': wR},
        )

        # if self.flag_data:
        #     save_data(f'C{QC}-T{QT} CR Rabi exp_args-x-result', 'qu', self.exp_args, self.x, self.result,
        #               root_path=self.root_path)

    def scan_CR_Mz(
        self,
        QC: str,
        QT: str,
        CR_width_list: np.ndarray,
        QC_amp: float,
        QC_phi: float = 0,
        QT_amp: float = 0,
        QT_phi: float = 0,
        CR_sigma: float = 1.25,
        CR_buffer: float = 5,
        drag_buffer: float = 0,
        init_state=0,
        flag_load_pi: bool = True,
        flag_echo: bool = True,
    ):
        self.x = CR_width_list
        self.exp_args.update({'QC': QC, 'QT': QT})
        wd = self.q_dic[QT]['w']
        wR = self.exp_args.get('wR', {QC: wd, QT: wd})
        if flag_load_pi:
            QC_PI = self.gate_load[f'X@{QC}']
            QT_X2 = self.gate_load[f'Xo2@{QT}']
            QT_mY2 = copy.deepcopy(QT_X2)
            phi_X2 = QT_X2.phi
            QT_mY2(phi=phi_X2 - np.pi / 2)
        else:
            raise ValueError(f'not supported now.')

        # 根据实验参数生成波形列表
        time_start = time.time()
        print(f'start pulse generate...')
        arg_list = list(itertools.product(CR_width_list, [QC_amp]))
        pulse_dic_list = [
            self.get_CR_pulse(
                arg,
                wd,
                wR,
                QC,
                QT,
                QC_PI,
                QC_phi,
                QT_amp,
                QT_phi,
                CR_sigma,
                CR_buffer,
                drag_buffer,
                flag_echo,
            )
            for arg in arg_list
        ]
        for idx, width in enumerate(CR_width_list):
            pulse_dic = pulse_dic_list[idx]
            QC_CR = pulse_dic[QC]['XY']
            zero_QT = Square(QC_CR.width, QT_X2.wd, 0)
            zero_QT.get_pulse()
            zero_QC = Square(QT_X2.width, QC_PI.wd, 0)
            zero_QC.get_pulse()

            # QC_pulse = QC_CR + zero_QC
            # QT_pulse = zero_QT + QT_mY2
            # print(f'QC pulse width: {QC_pulse.width}, QT pulse width: {QT_pulse.width}')
            pulse_dic = {QC: {'XY': QC_CR + zero_QC}, QT: {'XY': zero_QT + QT_mY2}}
            pulse_dic_list.append(pulse_dic)

        for idx, width in enumerate(CR_width_list):
            pulse_dic = pulse_dic_list[idx]
            QC_CR = pulse_dic[QC]['XY']
            zero_QT = Square(QC_CR.width, QT_X2.wd, 0)
            zero_QT.get_pulse()
            zero_QC = Square(QT_X2.width, QC_PI.wd, 0)
            zero_QC.get_pulse()

            # QC_pulse = QC_CR + zero_QC
            # QT_pulse = zero_QT + QT_X2
            # print(f'QC pulse width: {QC_pulse.width}, QT pulse width: {QT_pulse.width}')
            pulse_dic = {QC: {'XY': QC_CR + zero_QC}, QT: {'XY': zero_QT + QT_X2}}
            pulse_dic_list.append(pulse_dic)
        # self.exp_args['pulse_list'] = pulse_dic_list
        time_end = time.time()
        print(f'end pulse generate... consume {time_end - time_start}s')
        self.exp_args['pulse_schedule'] = pulse_dic_list[-1]

        # 放入求解器中求解
        init_state = self.exp_args.get('init_state', self.Kd(QC, init_state))
        mea_ops = self.exp_args.get('mea_ops', [self.Od((QC, QT), ('I', 'Z'))])
        self.result = qp.parallel_map(
            self.state_solve,
            pulse_dic_list,
            task_kwargs={'state0': init_state, 'e_ops': mea_ops, 'wR': wR},
        )

        # if self.flag_data:
        #     save_data(f'C{QC}-T{QT} CR Rabi exp_args-x-result', 'qu', self.exp_args, self.x, self.result,
        #               root_path=self.root_path)

    def scan_Hamil_Tomo(
        self,
        scan_type: str,
        QC: str,
        QT: str,
        CR_width0: Union[np.ndarray, float],
        CR_width1: Union[np.ndarray, float],
        QC_amp: Union[np.ndarray, float],
        QC_phi: float = 0,
        QT_amp: float = 0,
        QT_phi: float = 0,
        CR_sigma: float = 1.25,
        CR_buffer: float = 5,
        drag_buffer: float = 0,
        init_state=0,
        flag_load_pi: bool = True,
        flag_echo: bool = True,
    ):
        self.scan_CR(
            scan_type,
            QC,
            QT,
            CR_width0,
            QC_amp,
            QC_phi,
            QT_amp,
            QT_phi,
            CR_sigma,
            CR_buffer,
            drag_buffer,
            init_state=0,
            flag_load_pi=flag_load_pi,
            flag_echo=flag_echo,
        )
        result_QC0 = self.result
        self.exp_args.update({'x0': self.x})

        self.scan_CR(
            scan_type,
            QC,
            QT,
            CR_width1,
            QC_amp,
            QC_phi,
            QT_amp,
            QT_phi,
            CR_sigma,
            CR_buffer,
            drag_buffer,
            init_state=1,
            flag_load_pi=flag_load_pi,
            flag_echo=flag_echo,
        )
        result_QC1 = self.result
        self.exp_args.update({'x1': self.x, 'QC_amp': QC_amp})

        self.result = [*result_QC0, *result_QC1]

        # if self.flag_data:
        #     save_data(f'C{QC}-T{QT} CR Rabi exp_args-x-result', 'qu', self.exp_args, self.x, self.result,
        #               root_path=self.root_path)

    def analyze_CR_width(self, **kwargs):
        QC = self.exp_args['QC']
        QT = self.exp_args['QT']

        analy_name_list0 = kwargs.get(
            'analyze_name_list', ['<QT X>', '<QT Y>', '<QT Z>']
        )  # analy_name_list指定需要拟合的数据
        if isinstance(analy_name_list0[0], str):
            analy_name_list = analy_name_list0
            analy_name_list0 = [tuple(analy_name_list0)]
        else:
            analy_name_list = list(itertools.chain(*analy_name_list0))

        extra_name_list0 = kwargs.get(
            'extra_name_list', []
        )  # extra_name_list指定不需要拟合的数据
        if extra_name_list0 and isinstance(extra_name_list0[0], str):
            extra_name_list = extra_name_list0
            extra_name_list0 = [tuple(extra_name_list0)]
        else:
            extra_name_list = list(itertools.chain(*extra_name_list0))

        res_analy = dict.fromkeys(analy_name_list)
        cos_analy = dict.fromkeys(analy_name_list)
        popt_analy = dict.fromkeys(analy_name_list)
        res_extra = dict.fromkeys(extra_name_list)
        for idx, key in enumerate(analy_name_list + extra_name_list):
            if key in analy_name_list:
                res_analy[key] = np.asarray(
                    [res.expect[idx][-1] for res in self.result]
                )
                popt_analy[key], rmse, cos_analy[key] = fit_cos(self.x, res_analy[key])
            if key in extra_name_list:
                res_extra[key] = np.asarray(
                    [res.expect[idx][-1] for res in self.result]
                )

        for analy_name_set in analy_name_list0:
            self.plotter.plot_lines_fit(
                self.x,
                [res_analy[k] for k in analy_name_set],
                analy_name_set,
                [cos_analy[k] for k in analy_name_set],
                title=f'CR Rabi(freq='
                f'{[round(popt_analy[k][0] * 1e3, 3) for k in analy_name_set]}M)'
                f'(QC={QC}, QT={QT})',
            )

        for extra_name_set in extra_name_list0:
            self.plotter.plot_lines(
                self.x,
                [res_extra[k] for k in extra_name_set],
                extra_name_set,
                title=f'CR Rabi(QC={QC}, QT={QT})',
            )

        self.exp_args['res'] = {**res_analy, **res_extra}
        if self.flag_data:
            save_data(
                f'C{QC}-T{QT} CR Rabi exp_args-x-result',
                'qu',
                self.exp_args,
                self.x,
                self.result,
                root_path=self.root_path,
            )

    def analyze_CR_Mz(self, **kwargs):
        QC = self.exp_args['QC']
        QT = self.exp_args['QT']

        analy_name_list = ['<QT X>', '<QT Y>', '<QT Z>']  # analy_name_list指定需要拟合的数据
        # if isinstance(analy_name_list0[0], str):
        #     analy_name_list = analy_name_list0
        #     analy_name_list0 = [tuple(analy_name_list0)]
        # else:
        #     analy_name_list = list(itertools.chain(*analy_name_list0))

        extra_name_list0 = kwargs.get(
            'extra_name_list', []
        )  # extra_name_list指定不需要拟合的数据
        if extra_name_list0 and isinstance(extra_name_list0[0], str):
            extra_name_list = extra_name_list0
            extra_name_list0 = [tuple(extra_name_list0)]
        else:
            extra_name_list = list(itertools.chain(*extra_name_list0))

        node = len(self.result) // 3
        res_analy = dict.fromkeys(analy_name_list)
        res_extra = dict.fromkeys(extra_name_list)
        res_analy['<QT Z>'] = np.asarray(
            [res.expect[0][-1] for res in self.result[:node]]
        )
        res_analy['<QT X>'] = np.asarray(
            [res.expect[0][-1] for res in self.result[node : 2 * node]]
        )
        res_analy['<QT Y>'] = np.asarray(
            [res.expect[0][-1] for res in self.result[2 * node :]]
        )

        ylim = kwargs.get('ylim', (-1.05, 1.05))
        self.plotter.plot_lines(
            self.x,
            [res_analy['<QT Z>'], res_analy['<QT X>'], res_analy['<QT Y>']],
            ['<QT Z>', '<QT X>', '<QT Y>'],
            title=f'CR Rabi Oscillations(QC={QC}, QT={QT})',
            ylim=ylim,
        )

        self.exp_args['res'] = {**res_analy, **res_extra}
        if self.flag_data:
            save_data(
                f'C{QC}-T{QT} CR Rabi exp_args-x-result',
                'qu',
                self.exp_args,
                self.x,
                self.result,
                root_path=self.root_path,
            )

    def _analyze_HT(self, res: dict, **kwargs):
        QC = self.exp_args['QC']
        QT = self.exp_args['QT']
        x0 = self.exp_args['x0']
        x1 = self.exp_args['x1']
        QC_amp = self.exp_args['QC_amp']

        self.exp_args['res'] = res
        self.x = x0
        popt0, rmse0, bloch_fun0 = self.fit_bloch_flat(QC_state=0)
        self.x = x1
        popt1, rmse1, bloch_fun1 = self.fit_bloch_flat(QC_state=1)
        OX0, OY0, DZ0 = popt0
        OX1, OY1, DZ1 = popt1
        IX = (OX0 + OX1) / 2
        IY = (OY0 + OY1) / 2
        IZ = (DZ0 + DZ1) / 2
        ZX = (OX0 - OX1) / 2
        ZY = (OY0 - OY1) / 2
        ZZ = (DZ0 - DZ1) / 2

        ylim = kwargs.get('ylim', (-1.05, 1.05))
        for k0, k1, v0, v1 in zip(
            ['X0', 'Y0', 'Z0'], ['X1', 'Y1', 'Z1'], [IX, IY, IZ], [ZX, ZY, ZZ]
        ):
            fig, ax = plt.subplots()
            color, color2 = plt.rcParams.get('axes.prop_cycle')[0:2]
            # flag_fig = self.flag_fig
            # self.plotter.plot_lines_fit(x0, [res[k0]], [f'<{k0}>'], [partial(bloch_fun0, oper_type='X')],
            #                                 fig=fig, ax=ax)
            self.plotter.scatter(
                x0,
                res[k0],
                label=f'<{k0}>',
                xtype='t',
                ylim=ylim,
                fig=fig,
                ax=ax,
                marker='o',
                **color,
            )
            x0_interp = np.linspace(x0[0], x0[-1], 201)
            self.plotter.plot(
                x0_interp,
                bloch_fun0(x0_interp, oper_type=k0[0]),
                label=f'<{k0}> fit',
                xtype='t',
                fig=fig,
                ax=ax,
                **color,
            )
            # self.plotter.flag_save = flag_fig
            self.plotter.scatter(
                x1,
                res[k1],
                label=f'<{k1}>',
                xtype='t',
                fig=fig,
                ax=ax,
                marker='o',
                **color2,
            )
            x1_interp = np.linspace(x1[0], x1[-1], 201)
            self.plotter.plot(
                x1_interp,
                bloch_fun1(x1_interp, oper_type=k0[0]),
                label=f'<{k1}> fit',
                title=f'Amp={round(QC_amp * 1e3, 3)}, '
                f'I{k0[0]}={round(v0 * 1e3, 3)}, '
                f'Z{k0[0]}={round(v1 * 1e3, 3)}MHz(QC={QC})',
                xtype='t',
                xlabel='t',
                ylabel='Expectation',
                fig=fig,
                ax=ax,
                **color2,
                flag_save=True,
            )

        if res.get('R'):
            self.plotter.plot_lines(
                self.x,
                [res['R']],
                ['|R|'],
                title=f'R Oscillation(QC={QC})',
                ylim=(-0.1, 2.1),
            )

        for k, v in zip(
            [
                'OX0',
                'OY0',
                'DZ0',
                'OX1',
                'OY1',
                'DZ1',
                'IX',
                'IY',
                'IZ',
                'ZX',
                'ZY',
                'ZZ',
            ],
            [OX0, OY0, DZ0, OX1, OY1, DZ1, IX, IY, IZ, ZX, ZY, ZZ],
        ):
            self.exp_args[k] = v

        if self.flag_data:
            save_data(
                f'C{QC}-T{QT} CR {round(QC_amp * 1e3, 3)}M Rabi exp_args-x0-x1-result',
                'qu',
                self.exp_args,
                x0,
                x1,
                self.result,
                root_path=self.root_path,
            )
            save_data(
                f'C{QC}-T{QT} CR {round(QC_amp * 1e3, 3)}M Rabi t-X0-Y0-Z0',
                'dat',
                np.c_[x0, res['X0'], res['Y0'], res['Z0']],
                root_path=self.root_path,
            )
            save_data(
                f'C{QC}-T{QT} CR {round(QC_amp * 1e3, 3)}M Rabi t-X1-Y1-Z1',
                'dat',
                np.c_[x1, res['X1'], res['Y1'], res['Z1']],
                root_path=self.root_path,
            )
            if res.get('R'):
                save_data(
                    f'C{QC}-T{QT} CR {round(QC_amp * 1e3, 3)}M Rabi t-R',
                    'dat',
                    np.c_[self.x, res['R']],
                    root_path=self.root_path,
                )
            save_data(
                f'C{QC}-T{QT} CR {round(QC_amp * 1e3, 3)}M fit OX0~DZ0-OX1~DZ1(MHz)'
                f'(rmse0={round(rmse0, 3)}, rmse1={round(rmse1, 3)})',
                'txt',
                np.c_[np.array([OX0, OY0, DZ0, OX1, OY1, DZ1]) * 1e3],
                root_path=self.root_path,
            )
            save_data(
                f'C{QC}-T{QT} CR {round(QC_amp * 1e3, 3)}M fit IX~IZ-ZX~ZZ(MHz)'
                f'(rmse0={round(rmse0, 3)}, rmse1={round(rmse1, 3)})',
                'txt',
                np.c_[np.array([IX, IY, IZ, ZX, ZY, ZZ]) * 1e3],
                root_path=self.root_path,
            )

    def analyze_HT_width(self, **kwargs):
        x0 = self.exp_args['x0']
        x1 = self.exp_args['x1']
        analy_name_list = ['X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1']
        node = len(x0)
        res = dict.fromkeys(analy_name_list)
        res['X0'] = np.asarray([res.expect[0][-1] for res in self.result[:node]])
        res['Y0'] = np.asarray([res.expect[1][-1] for res in self.result[:node]])
        res['Z0'] = np.asarray([res.expect[2][-1] for res in self.result[:node]])
        res['X1'] = np.asarray([res.expect[0][-1] for res in self.result[node:]])
        res['Y1'] = np.asarray([res.expect[1][-1] for res in self.result[node:]])
        res['Z1'] = np.asarray([res.expect[2][-1] for res in self.result[node:]])

        if len(x0) == len(x1):
            if np.allclose(x0, x1):
                res['R'] = np.sqrt(
                    (res['X0'] + res['X1']) ** 2
                    + (res['Y0'] + res['Y1']) ** 2
                    + (res['Z0'] + res['Z1']) ** 2
                )

        self._analyze_HT(res, **kwargs)

        # self.exp_args['res'] = res
        # popt0, rmse0, bloch_fun0 = self.fit_bloch_flat(QC_state=0)
        # popt1, rmse1, bloch_fun1 = self.fit_bloch_flat(QC_state=1)
        # OX0, OY0, DZ0 = popt0
        # OX1, OY1, DZ1 = popt1
        # IX = (OX0 + OX1) / 2
        # IY = (OY0 + OY1) / 2
        # IZ = (DZ0 + DZ1) / 2
        # ZX = (OX0 - OX1) / 2
        # ZY = (OY0 - OY1) / 2
        # ZZ = (DZ0 - DZ1) / 2
        #
        # ylim = kwargs.get('ylim', (-1.05, 1.05))
        # self.plotter.plot_lines_fit(self.x, [res['X0'], res['X1']], ['<X0>', '<X1>'],
        #                                 [partial(bloch_fun0, oper_type='X'), partial(bloch_fun1, oper_type='X')],
        #                                 title=f'IX={round(IX*1e3, 3)}, ZX={round(ZX*1e3, 3)}MHz(QC={QC})', ylim=ylim)
        #
        # self.plotter.plot_lines_fit(self.x, [res['Y0'], res['Y1']], ['<Y0>', '<Y1>'],
        #                                 [partial(bloch_fun0, oper_type='Y'), partial(bloch_fun1, oper_type='Y')],
        #                                 title=f'IY={round(IY*1e3, 3)}, ZY={round(ZY*1e3, 3)}MHz(QC={QC})', ylim=ylim)
        #
        # self.plotter.plot_lines_fit(self.x, [res['Z0'], res['Z1']], ['<Z0>', '<Z1>'],
        #                                 [partial(bloch_fun0, oper_type='Z'), partial(bloch_fun1, oper_type='Z')],
        #                                 title=f'IZ={round(IZ*1e3, 3)}, ZZ={round(ZZ*1e3, 3)}MHz(QC={QC})', ylim=ylim)
        #
        # self.plotter.plot_lines(self.x, [res['R']], ['|R|'],
        #                             title=f'R Oscillation(QC={QC})', ylim=(-0.1, 2.1))
        #
        # for k, v in zip(['OX0', 'OY0', 'DZ0', 'OX1', 'OY1', 'DZ1', 'IX', 'IY', 'IZ', 'ZX', 'ZY', 'ZZ'],
        #                 [OX0, OY0, DZ0, OX1, OY1, DZ1, IX, IY, IZ, ZX, ZY, ZZ]):
        #     self.exp_args[k] = v
        #
        # if self.flag_data:
        #     save_data(f'C{QC}-T{QT} CR Rabi exp_args-x-result', 'qu', self.exp_args, self.x, self.result,
        #               root_path=self.root_path)
        #     save_data(f'C{QC}-T{QT} CR Rabi t-X0,1-Y0,1-Z0,1-R', 'dat',
        #               np.c_[self.x, res['X0'], res['X1'], res['Y0'], res['Y1'], res['Z0'], res['Z1'],
        #                     res['R']], root_path=self.root_path)
        #     save_data(f'C{QC}-T{QT} CR fit OX0~DZ0-OX1~DZ1-IX~IZ-ZX~ZZ', 'txt',
        #               np.c_[OX0, OY0, DZ0, OX1, OY1, DZ1, IX, IY, IZ, ZX, ZY, ZZ], root_path=self.root_path)
        return res

    def analyze_HT_width_amp(self, **kwargs):
        QC = self.exp_args['QC']
        QT = self.exp_args['QT']
        x0 = self.exp_args['x0']
        x1 = self.exp_args['x1']

        analy_name_list = ['X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1']
        node = len(x0) * len(self.y)  # len(self.result) // 2
        res2D = dict.fromkeys(analy_name_list)
        res2D['X0'] = np.asarray(
            [res2D.expect[0][-1] for res2D in self.result[:node]]
        ).reshape(len(x0), len(self.y))
        res2D['Y0'] = np.asarray(
            [res2D.expect[1][-1] for res2D in self.result[:node]]
        ).reshape(len(x0), len(self.y))
        res2D['Z0'] = np.asarray(
            [res2D.expect[2][-1] for res2D in self.result[:node]]
        ).reshape(len(x0), len(self.y))
        res2D['X1'] = np.asarray(
            [res2D.expect[0][-1] for res2D in self.result[node:]]
        ).reshape(len(x1), len(self.y))
        res2D['Y1'] = np.asarray(
            [res2D.expect[1][-1] for res2D in self.result[node:]]
        ).reshape(len(x1), len(self.y))
        res2D['Z1'] = np.asarray(
            [res2D.expect[2][-1] for res2D in self.result[node:]]
        ).reshape(len(x1), len(self.y))
        self.exp_args['res2D'] = res2D

        if self.flag_data:
            save_data(
                f'C{QC}-T{QT} CR Spectrum exp_args-x0-x1-y-result',
                'qu',
                self.exp_args,
                x0,
                x1,
                self.y,
                self.result,
                root_path=self.root_path,
            )

        self.plotter.plot_heatmap(
            self.y,
            x0,
            res2D['X0'],
            xlabel=r'$\Omega_{CR}$',
            zlabel='<X0>',
            title=f'X0 CR Spectrum(QC={QC})',
        )
        self.plotter.plot_heatmap(
            self.y,
            x0,
            res2D['Y0'],
            xlabel=r'$\Omega_{CR}$',
            zlabel='<Y0>',
            title=f'Y0 CR Spectrum(QC={QC})',
        )
        self.plotter.plot_heatmap(
            self.y,
            x0,
            res2D['Z0'],
            xlabel=r'$\Omega_{CR}$',
            zlabel='<Z0>',
            title=f'Z0 CR Spectrum(QC={QC})',
        )
        self.plotter.plot_heatmap(
            self.y,
            x1,
            res2D['X1'],
            xlabel=r'$\Omega_{CR}$',
            zlabel='<X1>',
            title=f'X1 CR Spectrum(QC={QC})',
        )
        self.plotter.plot_heatmap(
            self.y,
            x1,
            res2D['Y1'],
            xlabel=r'$\Omega_{CR}$',
            zlabel='<Y1>',
            title=f'Y1 CR Spectrum(QC={QC})',
        )
        self.plotter.plot_heatmap(
            self.y,
            x1,
            res2D['Z1'],
            xlabel=r'$\Omega_{CR}$',
            zlabel='<Z1>',
            title=f'Z1 CR Spectrum(QC={QC})',
        )

        coef_name_list = ['IX', 'IY', 'IZ', 'ZX', 'ZY', 'ZZ']
        coef = dict.fromkeys(coef_name_list, [])
        for i, QC_amp in enumerate(self.y):
            self.exp_args['QC_amp'] = QC_amp
            res = dict.fromkeys(analy_name_list)
            for key in analy_name_list:
                res[key] = res2D[key][:, i]

            self._analyze_HT(res, **kwargs)

            for key in coef_name_list:
                coef[key].append(self.exp_args[key])

        self.plotter.plot_lines(
            self.y,
            [coef[k] for k in coef_name_list],
            coef_name_list,
            xtype='w',
            xlabel=r'$\Omega_{CR}$',
            ytype='w',
            ylabel='Coefficient Strength',
            title=f'Hamiltonian Tomography',
        )
        self.exp_args['coef'] = coef

    @staticmethod
    def bloch_eq(t, OX, OY, DZ, oper_type: str):
        OX, OY, DZ = [2 * np.pi * v for v in (OX, OY, DZ)]
        O = np.sqrt(DZ**2 + OX**2 + OY**2)
        # print(
        #     f'O={O / 2 / np.pi * 1e3:.3f}, OX={OX / 2 / np.pi * 1e3:.3f}, OY={OY / 2 / np.pi * 1e3:.3f}, '
        #     f'DZ={DZ / 2 / np.pi * 1e3:.3f}MHz')
        if oper_type == 'X':
            return (
                1
                / O**2
                * (DZ * OX - DZ * OX * np.cos(2 * O * t) + OY * O * np.sin(2 * O * t))
            )
        elif oper_type == 'Y':
            return (
                1
                / O**2
                * (DZ * OY - DZ * OY * np.cos(2 * O * t) - OX * O * np.sin(2 * O * t))
            )
        elif oper_type == 'Z':
            return 1 / O**2 * (DZ**2 + (OX**2 + OY**2) * np.cos(2 * O * t))
        else:
            raise ValueError(f'operator {oper_type} is not supported.')

    def _bloch_eq_flat(self, t, OX, OY, DZ):
        t = t[: len(t) // 3]
        bloch = partial(self.bloch_eq, t, OX, OY, DZ)
        return np.asarray([bloch('X'), bloch('Y'), bloch('Z')]).flatten()

    def _evalVars(self, Vars):
        # 为了避免拟合参数过小，拟合时将时间单位调整为us，对应频率单位为MHz
        avg = self.exp_args['res']
        t = self.x * 1e-3
        if self.exp_args['QC_state'] == 0:
            y = np.asarray([avg['X0'], avg['Y0'], avg['Z0']]).flatten()
        else:
            y = np.asarray([avg['X1'], avg['Y1'], avg['Z1']]).flatten()

        rmse = []
        for i in range(Vars.shape[0]):
            rmse.append(
                mean_squared_error(y, self._bloch_eq_flat(np.tile(t, 3), *Vars[i, :]))
            )
        return np.vstack(rmse)

    def fit_bloch_flat(self, QC_state):
        rmse = np.inf
        loop = 0
        popt = None
        while rmse > 0.06 and loop < 10:
            self.exp_args['QC_state'] = QC_state
            problem = ea.Problem(
                name='bloch equation fit',
                M=1,
                maxormins=[1],
                Dim=3,
                varTypes=[0, 0, 0],
                lb=[-10, -10, -10],
                ub=[10, 10, 10],
                evalVars=self._evalVars,
            )

            algorithm = ea.soea_DE_best_1_bin_templet(
                problem,
                ea.Population(Encoding='RI', NIND=100),
                MAXGEN=300,
                logTras=1,
            )
            date = datetime.now().strftime('%Y-%m')
            time = datetime.now().strftime('%m%d-%H.%M.%S')
            res = ea.optimize(
                algorithm,
                prophet=popt,
                verbose=False,
                drawing=1,
                outputMsg=True,
                drawLog=False,
                saveFlag=True,
                dirName=str(
                    self.root_path
                    / date
                    / f'{time}_CR {round(self.exp_args["QC_amp"] * 1e3, 3)}M_soea_DE'
                ),
            )
            print(f'soea_DE_best_1_bin optimize result: {res}')
            rmse = res['ObjV'][0][0]
            popt = res['Vars'].flatten() * 1e-3
            loop += 1
        bloch_fun = partial(self.bloch_eq, OX=popt[0], OY=popt[1], DZ=popt[2])
        return popt, rmse, bloch_fun
