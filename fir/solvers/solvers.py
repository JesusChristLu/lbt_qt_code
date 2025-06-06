# -*- coding: utf-8 -*-
# @Time     : 2022/9/20 15:09
# @Author   : WTL
# @Software : PyCharm
import copy
import numpy as np
import qutip as qp
from scipy.optimize import *
import re
from typing import Union
from chip_hamiltonian.chip_hamiltonian import ChipDynamic
from functions import *


class Solver(ChipDynamic):
    """
    系统哈密顿量求解器，需要继承chip_hamiltonian.Chip基类
    """

    # def level_sort_old(self, ei_vector0: list, ei_energy0: list, ei_energy: list):
    #     """
    #     bare_ref中的某个态bare索引为idx_bare，这个态在ei_vector0中的对应位置为idx0。ei_energy_bare/dress的顺序与bare_ref相同，
    #     和ei_energy0/energy映射关系为idx_bare<->idx_0
    #     :param ei_vector0:
    #     :param ei_energy:
    #     :param ei_energy0:
    #     :return:
    #     """
    #     ei_energy_bare = []
    #     ei_energy_dress = []
    #
    #     for state_number in qp.state_number_enumerate(self.dim):
    #         idx0 = list(ei_vector0).index(qp.state_number_qobj(self.dim, state_number))
    #         ei_energy_bare.append(ei_energy0[idx0])
    #         ei_energy_dress.append(ei_energy[idx0])
    #
    #     return ei_energy_bare, ei_energy_dress
    #
    # def vector_sort_old(self, ei_vector0: list, ei_vector: list):
    #     ei_vector_bare = []
    #     ei_vector_dress = []
    #     for state_number in qp.state_number_enumerate(self.dim):
    #         idx0 = list(ei_vector0).index(qp.state_number_qobj(self.dim, state_number))
    #         ei_vector_bare.append(np.array(ei_vector0[idx0]))
    #         ei_vector_dress.append(np.array(ei_vector[idx0]))
    #
    #     return np.hstack(ei_vector_bare), np.hstack(ei_vector_dress)
    #
    # def eigen_solve_old(self, chip_dict_exp: dict = None):
    #     """
    #
    #     :param chip_dict_exp: 芯片参数实验列表(可以改变频率、非谐、耦合强度等)
    #     :return:
    #     """
    #     if chip_dict_exp is None:
    #         chip_dict_exp = {}
    #     chip_dic = copy.deepcopy(self.chip_dic)
    #     update_nested_dic(chip_dic, chip_dict_exp)
    #     q_dic_exp = chip_dic['qubits']
    #     c_dic_exp = chip_dic['couplers']
    #     rho_map_exp = chip_dic['rho_map']
    #     H0_exp = self.H0(q_dic_exp)
    #     Hsys_exp = self.Hsys({**q_dic_exp, **c_dic_exp}, rho_map_exp)
    #
    #     ei_energy, ei_vector = Hsys_exp.eigenstates()
    #     ei_energy0, ei_vector0 = H0_exp.eigenstates()
    #     ei_energy_bare, ei_energy_dress = self.level_sort(ei_vector0, ei_energy0 / (2 * np.pi),
    #                                                       ei_energy / (2 * np.pi))
    #     return ei_energy_bare, ei_energy_dress

    def eigen_solve(self, chip_dict_exp: dict = None, method: str = 'index'):
        """
        本征值求解器
        :param chip_dict_exp:
        :param method: 标签分配方案。'index' 表示裸态、缀饰态都按照能量从低到高排序，然后将裸态标签依次分配给缀饰态；
        'overlap' 表示求缀饰态与裸态之间的overlap，将overlap最大的裸态标签分配给缀饰态
        :return:
        """
        if chip_dict_exp is None:
            chip_dict_exp = {}
        chip_dic = copy.deepcopy(self.chip_dic)
        update_nested_dic(chip_dic['qubits'], chip_dict_exp)
        # update_nested_dic(chip_dic, chip_dict_exp)

        q_dic_exp = chip_dic['qubits']
        # c_dic_exp = chip_dic['couplers']
        rho_map_exp = chip_dic['rho_map']

        # ei_energy, ei_vector = self.Hsys(
            # {**q_dic_exp, **c_dic_exp}, rho_map_exp
        # ).eigenstates()
        ei_energy, ei_vector = self.Hsys(
            {**q_dic_exp}, rho_map_exp
        ).eigenstates()
        ei_energy0, ei_vector0 = self.H0(q_dic_exp).eigenstates()
        if method == 'index':
            (
                ei_vector_bare,
                ei_energy_bare,
                ei_vector_dress,
                ei_energy_dress,
            ) = self.eigen_sort(
                ei_vector0,
                ei_energy0,
                ei_vector,
                ei_energy,
            )
        # elif method == 'overlap':
        #     ei_vector_bare, ei_energy_bare, ei_vector_dress, ei_energy_dress = self.eigen_overlap_sort(
        #         ei_vector0, ei_energy0,
        #         ei_vector, ei_energy,
        #     )
        else:
            raise ValueError(f'Method {method} is not supported.')

        return ei_energy_bare, ei_energy_dress

    def ZZ_min(self, scan_range, qubit0, qubit1, save_flag=True, update_flag=True):
        """
        在一定范围内搜索两比特间ZZ最小值
        :param save_flag: 是否更新coupler关断点
        :param qubit0:
        :param qubit1:
        :param scan_range: 'in'表示比特频率之间，'out'表示比特频率之外，也可以直接传入搜索范围
        :return:
        """
        if scan_range == 'in':
            bounds = (
                min(self.q_dic[qubit0]['w_idle'], self.q_dic[qubit1]['w_idle']),
                max(self.q_dic[qubit0]['w_idle'], self.q_dic[qubit1]['w_idle']),
            )
        elif scan_range == 'out':
            bounds = (
                max(self.q_dic[qubit0]['w_idle'], self.q_dic[qubit1]['w_idle']),
                10,
            )
        else:
            bounds = tuple(scan_range)
        result = minimize_scalar(
            self.ZZcoupling,
            args=(qubit0, qubit1, None, True),
            bounds=bounds,
            method='bounded',
        )
        coupler_freq_off = float(result.x)
        zz_min = float(result.fun)
        if save_flag:
            update_nested_dic(
                self.q_dic,
                {'C' + qubit0[1:] + '_' + qubit1[1:]: {'w_idle': coupler_freq_off}},
            )
            if update_flag:
                self.update_spectrum()
        return coupler_freq_off, zz_min

    def energy_diff(
        self,
        changed_freq,
        changed_qubit,
        energy0,
        energy1,
        q_dict_exp: dict = None,
        offset: float = 0,
    ):
        """
        改变比特频率，计算能级差
        :param changed_freq: 比特频率
        :param changed_qubit: 比特名称
        :param energy0: 能级，格式为((Qi,Qj,...),(state_i, state_j, ...))
        :param energy1: 能级，格式为((Qi,Qj,...),(state_i, state_j, ...))
        :param q_dict_exp: 外部比特参数
        :param offset: 希望能级差接近某个值，而非最小值
        :return:
        """
        q_dict_exp_new = {changed_qubit: {'w_idle': changed_freq}}
        if q_dict_exp is not None:
            q_dict_exp = {**q_dict_exp_new, **q_dict_exp}
        else:
            q_dict_exp = q_dict_exp_new

        diff = self.Ed(energy0[0], energy0[1], q_dict_exp) - self.Ed(
            energy1[0], energy1[1], q_dict_exp
        )
        return abs(abs(diff) - offset)

    def resonate_point(
        self,
        changed_qubit,
        scan_range,
        energy0,
        energy1,
        q_dict_exp: dict = None,
        offset: float = 0,
    ):
        """
        改变比特频率，搜索能级共振点
        :param changed_qubit: 比特名称
        :param scan_range: 搜索范围
        :param energy0: 能级，格式为((Qi,Qj,...),(state_i, state_j, ...))
        :param energy1: 能级，格式为((Qi,Qj,...),(state_i, state_j, ...))
        :param q_dict_exp: 外部比特参数
        :param offset: 希望能级差接近某个值，而非最小值
        :return:
        """
        result = minimize_scalar(
            self.energy_diff,
            args=(changed_qubit, energy0, energy1, q_dict_exp, offset),
            bounds=scan_range,
            method='bounded',
        )
        # resonate_freq = {changed_qubit: {'w_resonate': result.x}}
        resonate_freq = result.x
        diff_min = result.fun
        return resonate_freq, diff_min


class SolverDynamic(Solver):
    """
    含时哈密顿量求解器，需要继承chip_hamiltonian.ChipDynamic基类
    """

    @parallel_allocation
    def state_solve(
        self, pulse_dic: dict, state0, e_ops=None, c_ops=None, wR=None, **kwargs
    ):
        """
        主方程/薛定谔方程求解器
        :param wR:
        :param pulse_dic: XY和Z波形参数，key为qubit的名称，value为一个字典，键为'XY'和'Z'。
        e.g. pulse_dic = {'Q1': {'XY': xy_pulse, 'Z': z_pulse}, 'Q2': {'XY': xy_pulse, 'Z': z_pulse}, ...}
        :param state0: 系统初始时刻状态
        :param e_ops: 待求解期望值的算符列表
        :param c_ops: 退相干算符列表
        :param kwargs: 如果需要调用parallel_map()并行一个以上的参数，请在parallel_args中指定并行参数的名称
        :return:
        """
        if self.flag_trans:
            Ht = self.Ht_trans(pulse_dic)
        elif self.flag_R:
            print(f'wR passed to state_solve: {wR}')
            Ht = self.HRt(pulse_dic, wR)
        else:
            Ht = self.Ht(pulse_dic)
        return qp.mesolve(Ht, state0, self.t, c_ops=c_ops, e_ops=e_ops)

    @parallel_allocation
    def propagator_solve(
        self,
        pulse_dic: dict,
        c_ops=None,
        wR=None,
        flag_last: bool = False,
        flag_parallel: bool = False,
    ):
        """
        时间演化算符求解器
        :param flag_last: 为True时只返回最后一个Unitary，为False时返回所有时刻的Unitary
        :param wR:
        :param pulse_dic: XY和Z波形参数，key为qubit的名称，value为一个字典，键为'XY'和'Z'。
        e.g. pulse_dic = {'Q1': {'XY': xy_pulse, 'Z': z_pulse}, 'Q2': {'XY': xy_pulse, 'Z': z_pulse}, ...}
        :param c_ops: 退相干算符列表
        :param flag_parallel: 是否并行
        :return:
        """
        if c_ops is None:
            c_ops = []
        if self.flag_trans:
            Ht = self.Ht_trans(pulse_dic)
        elif self.flag_R:
            print(f'wR passed to propagator_solve: {wR}')
            Ht = self.HRt(pulse_dic, wR)
        else:
            Ht = self.Ht(pulse_dic)
        if flag_parallel:
            U = qp.propagator(
                Ht, self.t, c_op_list=c_ops, parallel=True, num_cpus=self.num_cpus
            )
        else:
            U = qp.propagator(Ht, self.t, c_op_list=c_ops, unitary_mode='single')
        if flag_last:
            return U[-1]
        else:
            return U

    # def FU_calc(
    #         self,
    #         pulse_dic: dict, U_ideal: qp.Qobj,
    #         c_ops=None, wR=None
    # ):
    #     qubit = list(pulse_dic.keys())[0]
    #     Ufull = self.propagator_solve(pulse_dic, c_ops, wR)[-1]
    #     U = np.zeros((2, 2), dtype=np.complex128)
    #     for r in range(2):
    #         for c in range(2):
    #             U[r, c] = complex(self.Kd(qubit, r).overlap(Ufull * self.Kd(qubit, c)))
    #
    #     print(f'Ureal:{qp.Qobj(U)}')
    #     SU2_param(U)
    #     # print(f'Ureal params:\n{SU2_param(U)}')
    #     phi, Uphi = cali_phi(qp.Qobj(U), U_ideal)
    #     # print(f'Uphi params:\n{SU2_param(Uphi)}')
    #     SU2_param(Uphi)
    #     return fU(Uphi, U_ideal)

    def get_c_ops(self, c_dic):
        """
        根据c_dic中的信息创建可被qutip求解器识别的c_ops
        :param c_dic: 包含退相干信息的字典，格式为{'qubit': {'T1': xx, 'Tphi': xx}, ...}
        :return:
        """
        c_ops = []
        for bit, dic in c_dic.items():
            for ty, va in dic.items():
                match1 = re.compile(r'T1').search(ty)
                match2 = re.compile(r'(T2|Tphi)(?:\((.*?)\))?').search(ty)
                if match1:
                    c_ops.append(np.sqrt(1 / va) * self.a[bit])

                elif match2:
                    qid = self.q_dic[bit].get('id')
                    assert self.dim[qid] in [2, 3], 'only support dim=2 or 3 operators'

                    prefix = match2.group(1)
                    suffix = match2.group(2)

                    identity_list = [qp.qeye(d) for d in self.dim]

                    # if ty == 'Tphi(t)':
                    #     # to do: time-dependent coe
                    #     pass
                    if suffix:
                        va = va(dic.get(suffix))

                    if prefix == 'T2':
                        t1 = dic.get('T1')
                        va = 1 / (1/va - 1/2/t1)

                    if self.dim[qid] == 3:
                        # ref:
                        cop1 = copy.copy(identity_list)
                        cop1[qid] = qp.Qobj([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
                        cop1 = 2 / 3 * np.sqrt(2 / va) * qp.tensor(cop1)

                        cop2 = copy.copy(identity_list)
                        cop2[qid] = qp.Qobj([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
                        cop2 = 1 / 3 * np.sqrt(2 / va) * qp.tensor(cop2)

                        cop3 = copy.copy(identity_list)
                        cop3[qid] = qp.Qobj([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
                        cop3 = 1 / 3 * np.sqrt(2 / va) * qp.tensor(cop3)

                        c_ops.extend([cop1, cop2, cop3])
                    if self.dim[qid] == 2:
                        # ref:
                        cop = copy.copy(identity_list)
                        cop[qid] = qp.sigmaz()
                        c_ops.append(np.sqrt(1 / 2 / va) * qp.tensor(cop))

        return c_ops

    def init_pulse(
        self,
        arg_bits: Union[tuple, list],
        arg_type: Union[tuple, list],
        pulse_shape: Union[tuple, list],
    ):
        """
        初始化波形
        :param arg_bits: 扫描参数的比特
        :param arg_type: 扫描参数的类型
        :param pulse_shape: 波形
        :return:
        """
        pulse0_set = dict.fromkeys(arg_bits)
        for bit, ty, shape in zip(arg_bits, arg_type, pulse_shape):
            if shape == 'Drag':
                pulse_name = f'X@{bit}' if ty == 'X' else f'Xo2@{bit}'
            else:
                pulse_name = f'{shape}_{ty}@{bit}'
            pulse0_set[bit] = self.gate_load[pulse_name]

        return pulse0_set


if __name__ == '__main__':
    pass
