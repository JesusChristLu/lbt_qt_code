# -*- coding: utf-8 -*-
# @Time     : 2022/9/19 15:52
# @Author   : WTL
# @Software : PyCharm
import pickle
from collections.abc import Iterable
import numpy as np
import qutip as qp
import yaml
from typing import Union
import copy
import itertools
from functools import reduce
import operator
from pathlib import Path
from pulse.pulse_base import PulseBase
from pulse.pulse_lib import *
from functions import *
from functions.containers import ChipContainer


class Chip:
    def __init__(
        self,
        chip_path: str,
        dim: Union[int, list],
        flag_Hqq_RWA: bool = False,
        flag_g_exact: bool = True,
    ):
        """
        根据芯片参数和模拟维度创建Chip类，主要用于创建算符和生成系统哈密顿量，被上层类调用
        :param chip_path: 芯片参数yaml文件路径
        :param dim: 模拟的Hilbert空间维度
        :param flag_Hqq_RWA: 哈密顿量的相互作用项是否采用旋波近似，默认为False
        """
        self.flag_Hqq_RWA = flag_Hqq_RWA
        self.flag_g_exact = flag_g_exact
        self.chip_path = chip_path
        with open(chip_path, 'r') as f:
            self.chip_dic = yaml.load(f.read(), Loader=yaml.Loader)
        self.q_dic = self.chip_dic['qubits']
        self.c_dic = self.chip_dic.get('couplers', {})
        self.rho_map = self.chip_dic.get('rho_map', {})
        self.bit_num = len(self.chip_dic['qubits'])
        if isinstance(dim, int):
            self.dim = [dim] * self.bit_num
        else:
            self.dim = dim
            assert (
                len(self.dim) == self.bit_num
            ), 'size of dimension list must match with qubit number!'
        self.a, self.ad = self.generate_op()  # 此方法中会为每个qubit分配一个唯一的id，写入self.q_dic中
        self.eigvec = None
        self.eigval = None
        self.idle_eigen()  # 此方法会计算系统哈密顿量本征态并更新self.eigvec, self.eigval
        self.update_spectrum()  # 此方法会更新self.q_dic中的参数
        self.coupler_offpoint = None

    def generate_op(self):
        """
        根据比特数目和考虑的维度生成产生/湮灭算符
        :return:
        """
        a = {}
        ad = {}
        identity_list = [qp.qeye(d) for d in self.dim]
        for idx, q in enumerate(self.q_dic.keys()):
            self.q_dic[q].update({'id': idx})
            alist = copy.copy(identity_list)
            alist[idx] = qp.destroy(self.dim[idx])
            a[q] = qp.tensor(alist)

            adlist = copy.copy(identity_list)
            adlist[idx] = qp.create(self.dim[idx])
            ad[q] = qp.tensor(adlist)

        return a, ad

    def update_spectrum(self):
        """
        更新比特的能谱参数，如果没有指定则分配默认值
        :return:
        """
        for q in self.q_dic.keys():
            self.q_dic[q]['w_idle~'] = float(self.Ed(q, 1))
            self.q_dic[q].setdefault('w', self.q_dic[q]['w_idle'])
            self.q_dic[q].setdefault('w_max', self.q_dic[q]['w_idle'])
            self.q_dic[q].setdefault('eta', -200e-3)
            self.q_dic[q].setdefault('period', 1.0)
            self.q_dic[q].setdefault('sws', 0.0)
            self.q_dic[q].setdefault('d', 0.25)

        for c in self.c_dic.keys():
            self.c_dic[c].setdefault('w', self.c_dic[c]['w_idle'])
            self.c_dic[c].setdefault('w_max', self.c_dic[c]['w_idle'])
            self.c_dic[c].setdefault('eta', -200e-3)
            self.c_dic[c].setdefault('period', 1.0)
            self.c_dic[c].setdefault('sws', 0.0)
            self.c_dic[c].setdefault('d', 0.25)

        with open(self.chip_path, 'w') as f:
            chip_dic2save = copy.deepcopy(self.chip_dic)
            pop_nested_dickey(
                chip_dic2save,
                key='w',
            )
            yaml.dump(chip_dic2save, f, sort_keys=False)

    def reset_working_point(self):
        for q in self.q_dic.keys():
            self.q_dic[q]['w'] = self.q_dic[q]['w_idle']

        for c in self.c_dic.keys():
            self.c_dic[c]['w'] = self.c_dic[c]['w_idle']

        # with open(self.chip_path, 'w') as f:
        #     yaml.dump(self.chip_dic, f)

    def idle_eigen(self, q_dict_exp: dict = None):
        """
        根据比特参数字典，计算系统哈密顿量的本征态，方便后续在缀饰态下考虑问题
        :param q_dict_exp: 如果为None，则根据json文件的参数进行计算，否则根据q_dict_exp中的参数进行计算
        :return:
        """
        q_dic_idle = copy.deepcopy({**self.q_dic, **self.c_dic})
        if q_dict_exp is not None:
            update_nested_dic(q_dic_idle, q_dict_exp)
        rho_map = self.rho_map

        for q in q_dic_idle.keys():
            q_dic_idle[q]['w'] = q_dic_idle.get(q).get('w_idle')

        ei_energy0, ei_vector0 = self.H0(q_dic_idle).eigenstates()
        ei_energy, ei_vector = self.Hsys(q_dic_idle, rho_map).eigenstates()

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
        if q_dict_exp is None:
            self.eigvec = ei_vector_dress
            self.eigval = ei_energy_dress

        return ei_energy_dress

    def eigen_sort(
        self,
        ei_vector0: list,
        ei_energy0: list,
        ei_vector: list,
        ei_energy: list,
    ):
        """
        为缀饰态分配标签
        :param ei_vector0: 裸态本征矢
        :param ei_energy0: 裸态本征值
        :param ei_vector: 缀饰态本征矢
        :param ei_energy: 缀饰态本征值
        :param method: 标签分配方案。'index' 表示裸态、缀饰态都按照能量从低到高排序，然后将裸态标签依次分配给缀饰态；
        'overlap' 表示求缀饰态与裸态之间的overlap，将overlap最大的裸态标签分配给缀饰态
        :return:
        """
        ei_vector_dress = {}
        ei_energy_dress = {}
        ei_vector_bare = {}
        ei_energy_bare = {}

        # state_idx_list = np.array([np.argmax(np.abs(vec.full())).item() for vec in ei_vector0])
        state_idx_list = [np.argmax(np.abs(vec.full())).item() for vec in ei_vector0]

        raw_idx_list = list(range(len(ei_vector0)))
        state_raw_idx_list = sorted(zip(state_idx_list, raw_idx_list))
        state_idx_list, raw_idx_list = list(zip(*state_raw_idx_list))

        for idx, state_idx in zip(raw_idx_list, state_idx_list):
            state_number = qp.state_index_number(self.dim, state_idx)
            # if method == 'overlap':
            #     state_number = qp.state_index_number(self.dim, idx)  # (state_number, idx)
            ei_vector_dress[state_number] = ei_vector[idx]
            ei_energy_dress[state_number] = ei_energy[idx] / 2 / np.pi
            ei_vector_bare[state_number] = ei_vector0[idx]
            ei_energy_bare[state_number] = ei_energy0[idx] / 2 / np.pi
        return ei_vector_bare, ei_energy_bare, ei_vector_dress, ei_energy_dress

    def eigen_overlap_sort(
        self,
        ei_vector0: list,
        ei_energy0: list,
        ei_vector: list,
        ei_energy: list,
    ):
        ei_vector_dress = {}
        ei_energy_dress = {}
        ei_vector_bare = {}
        ei_energy_bare = {}

        bare_idx_list = np.array(
            [np.argmax(np.abs(vec.full())).item() for vec in ei_vector0]
        )
        raw_idx_list = list(range(len(ei_vector0)))
        bare_raw_idx_list = sorted(zip(bare_idx_list, raw_idx_list))
        bare_idx_list, raw_idx_list = list(zip(*bare_raw_idx_list))

        num = len(ei_energy0)
        ei_mat = np.array(
            [np.real_if_close(ei_vector[i].full().flatten()) for i in range(num)]
        ).T
        ei_mat0 = np.array(
            [np.real_if_close(ei_vector0[i].full().flatten()) for i in raw_idx_list]
        )

        # overlap_mat = [[<1|1d>, <1|2d>, ..., <1|nd>],
        #                [<2|1d>, <2|2d>, ..., <2|nd>],
        #                [<n|1d>, <n|2d>, ..., <n|nd>]]
        overlap_mat = ei_mat0 @ ei_mat
        bare2dress_list = np.argmax(overlap_mat, axis=1)
        # dress_idx_list = bare_idx_list[dress2bare_map]

        # raw_idx_list = list(range(len(ei_vector0)))
        # dress_bare_raw_idx_list = sorted(zip(dress_idx_list, bare_idx_list, raw_idx_list))
        # dress_idx_list, bare_idx_list, raw_idx_list = list(zip(*dress_bare_raw_idx_list))
        #
        for raw, bare, b2d in zip(raw_idx_list, bare_idx_list, bare2dress_list):
            state_number = qp.state_index_number(self.dim, bare)
            ei_vector_bare[state_number] = ei_vector0[raw]
            ei_energy_bare[state_number] = ei_energy0[raw] / 2 / np.pi
            ei_vector_dress[state_number] = ei_vector[b2d]
            ei_energy_dress[state_number] = ei_energy[b2d] / 2 / np.pi

        return ei_vector_bare, ei_energy_bare, ei_vector_dress, ei_energy_dress

    def level_sort_old(self, ei_vector0: list, ei_energy0: list, ei_energy: list):
        """
        bare_ref中的某个态bare索引为idx_bare，这个态在ei_vector0中的对应位置为idx0。ei_energy_bare/dress的顺序与bare_ref相同，
        和ei_energy0/energy映射关系为idx_bare<->idx_0
        :param ei_vector0:
        :param ei_energy:
        :param ei_energy0:
        :return:
        """
        ei_energy_bare = []
        ei_energy_dress = []

        for state_number in qp.state_number_enumerate(self.dim):
            try:
                idx0 = list(ei_vector0).index(
                    qp.state_number_qobj(self.dim, state_number)
                )
            # 这样处理的主要原因是：当只传入单个比特且dim=2时，会出现计算的本征态为[[-1], [0]]的情况
            except Exception as e:
                ei_vector0 = [
                    qp.Qobj(np.abs(v.full()), dims=[self.dim, [1] * len(self.dim)])
                    for v in list(ei_vector0)
                ]
                idx0 = ei_vector0.index(qp.state_number_qobj(self.dim, state_number))
                print(e)
            ei_energy_bare.append(ei_energy0[idx0])
            ei_energy_dress.append(ei_energy[idx0])

        return ei_energy_bare, ei_energy_dress

    # def vector_sort_old(self, ei_vector0: list, ei_vector: list):
    #     ei_vector_bare = []
    #     ei_vector_dress = []
    #     for state_number in qp.state_number_enumerate(self.dim):
    #         idx0 = list(ei_vector0).index(qp.state_number_qobj(self.dim, state_number))
    #         ei_vector_bare.append(np.array(ei_vector0[idx0]))
    #         ei_vector_dress.append(np.array(ei_vector[idx0]))
    #
    #     return np.hstack(ei_vector_bare), np.hstack(ei_vector_dress)

    @staticmethod
    def rho2g(rho_value: Union[float, list], w_set: list):
        if not isinstance(rho_value, Iterable):
            wl, wr = w_set
            g = rho_value * np.sqrt(wl * wr)
        else:
            rho_lc, rho_rc, rho_lr = rho_value
            wl, wc, wr = w_set
            glr = rho_lr * np.sqrt(wl * wr)
            glc = rho_lc * np.sqrt(wl * wc)
            grc = rho_rc * np.sqrt(wr * wc)
            g = glr + 1 / 2 * glc * grc * (
                1 / (wl - wc) + 1 / (wr - wc) - 1 / (wl + wc) - 1 / (wr + wc)
            )
        return g

    def H0(self, q_dic_exp=None):
        """
        根据比特参数字典，返回无相互作用的哈密顿量
        :param q_dic_exp:
        :return:
        """
        if q_dic_exp is None:
            q_dic_exp = copy.deepcopy(self.q_dic)
        H0 = (
            2
            * np.pi
            * np.sum(
                np.array([
                    q_dic_exp[q]['w'] * self.ad[q] * self.a[q]
                    + q_dic_exp[q]['eta']
                    / 2
                    * self.ad[q]
                    * self.ad[q]
                    * self.a[q]
                    * self.a[q]
                    for q in self.q_dic.keys()
                ]),
                axis=0,
            )
        )
        return qp.Qobj(H0, dims=[self.dim] * 2)

    def Hqq(self, q_dic_exp: dict = None, rho_map_exp: dict = None):
        """
        根据比特参数字典以及相互作用映射，返回哈密顿量的相互作用项。
        注意按照映射字典的书写规范，当rho_map中存在3值列表时，默认为coupler结构且coupler不参与算符的构建
        :param q_dic_exp:
        :param rho_map_exp:
        :return:
        """
        if q_dic_exp is None:
            q_dic_exp = {**self.q_dic, **self.c_dic}
        if rho_map_exp is None:
            rho_map_exp = copy.deepcopy(self.rho_map)

        if not rho_map_exp:
            return qp.Qobj(
                np.zeros((np.prod(self.dim), np.prod(self.dim))), dims=[self.dim] * 2
            )

        if self.flag_Hqq_RWA:
            Hqq = (
                2
                * np.pi
                * np.sum(
                    [
                        self.gqq(pair, rho, q_dic_exp)
                        * (
                            self.ad[pair.split('-')[0]] * self.a[pair.split('-')[-1]]
                            + self.a[pair.split('-')[0]] * self.ad[pair.split('-')[-1]]
                        )
                        for pair, rho in rho_map_exp.items()
                    ],
                    axis=0,
                )
            )
        else:
            Hqq = (
                2
                * np.pi
                * np.sum(
                    [
                        -self.gqq(pair, rho, q_dic_exp)
                        * (self.ad[pair.split('-')[0]] - self.a[pair.split('-')[0]])
                        * (self.ad[pair.split('-')[-1]] - self.a[pair.split('-')[-1]])
                        for pair, rho in rho_map_exp.items()
                    ],
                    axis=0,
                )
            )
        return qp.Qobj(Hqq, dims=[self.dim] * 2)

    def gqq(
        self,
        rho_pair: str,
        rho_value: Union[float, list] = None,
        q_dic_exp: dict = None,
    ):
        if q_dic_exp is None:
            q_dic_exp = {**self.q_dic, **self.c_dic}

        if rho_value is None:
            rho_value = self.rho_map[rho_pair]

        pair = tuple(rho_pair.split('-'))
        w_set = [
            q_dic_exp[qubit]['w'] - self.flag_g_exact * q_dic_exp[qubit]['eta']
            for qubit in pair
        ]

        g = self.rho2g(rho_value, w_set)
        return g

    def Hsys(self, q_dic_exp=None, rho_map_exp=None):
        """
        根据比特参数字典以及相互作用映射，返回系统哈密顿量(H0+Hqq)
        :param q_dic_exp:
        :param rho_map_exp:
        :return:
        """
        return self.H0(q_dic_exp) + self.Hqq(q_dic_exp, rho_map_exp)

    def Hsys_trans(self, q_dic_exp=None, rho_map_exp=None):
        """
        根据比特参数字典以及相互作用映射，返回系统哈密顿量(H0+Hqq)
        :param q_dic_exp:
        :param rho_map_exp:
        :return:
        """
        return self.Hsys(q_dic_exp, rho_map_exp).transform(list(self.eigvec.values()))

    def _Kb1q(self, qubit: str, state: Union[int, str]):
        """
        传入单个比特的状态，返回该比特的裸态(主要是被self.Kb()调用，外部无需被使用)
        :param qubit:
        :param state:
        :return:
        """
        qid = self.q_dic[qubit].get('id')

        if isinstance(state, int):
            state = qp.basis(self.dim[qid], state)
        elif state == '+':
            state = (
                1
                / np.sqrt(2)
                * (qp.basis(self.dim[qid], 0) + qp.basis(self.dim[qid], 1))
            )
        elif state == '-':
            state = (
                1
                / np.sqrt(2)
                * (qp.basis(self.dim[qid], 0) - qp.basis(self.dim[qid], 1))
            )
        elif state == '+i':
            state = (
                1
                / np.sqrt(2)
                * (qp.basis(self.dim[qid], 0) + 1j * qp.basis(self.dim[qid], 1))
            )
        elif state == '-i':
            state = (
                1
                / np.sqrt(2)
                * (qp.basis(self.dim[qid], 0) - 1j * qp.basis(self.dim[qid], 1))
            )
        else:
            raise ValueError(f'state={state} is not supported.')

        return state

    def Kb(self, qubits: Union[tuple, str], states: Union[tuple, int, str]):
        """
        Abbreviation for ket_bare. 传入比特的状态，返回对应全空间下的裸态
        :param qubits:
        :param states:
        :return:
        """
        if not isinstance(qubits, tuple):
            qubits = (qubits,)
            states = (states,)
        qids = [self.q_dic[qubit].get('id') for qubit in qubits]
        state_full = [qp.basis(d, 0) for d in self.dim]

        for q, qid, state in zip(qubits, qids, states):
            state_full[qid] = self._Kb1q(q, state)

        state_full = qp.tensor(state_full)
        return state_full

    def Ob(self, qubits: Union[tuple, str], states: Union[tuple, int, str]):
        if not isinstance(qubits, tuple):
            qubits = (qubits,)
            states = (states,)
        qids = [self.q_dic[qubit].get('id') for qubit in qubits]
        Od_full = [qp.ket2dm(qp.basis(d, 0)) for d in self.dim]

        for q, qid, state in zip(qubits, qids, states):
            if state == 'I':
                Od_full[qid] = (
                    sum([self._Kb1q(q, i) * self._Kb1q(q, i).dag() for i in range(self.dim[qid])])
                )
            elif state == 'X':
                Od_full[qid] = (
                    self._Kb1q(q, 0) * self._Kb1q(q, 1).dag()
                    + self._Kb1q(q, 1) * self._Kb1q(q, 0).dag()
                )
            elif state == 'Y':
                Od_full[qid] = (
                    -1j * self._Kb1q(q, 0) * self._Kb1q(q, 1).dag()
                    + 1j * self._Kb1q(q, 1) * self._Kb1q(q, 0).dag()
                )
            elif state == 'Z':
                Od_full[qid] = (
                    self._Kb1q(q, 0) * self._Kb1q(q, 0).dag()
                    - self._Kb1q(q, 1) * self._Kb1q(q, 1).dag()
                )
            else:
                Od_full[qid] = qp.ket2dm(self._Kb1q(q, state))

        Od_full = qp.tensor(Od_full)
        return Od_full

    def _Kd1q(self, qubit: str, state: Union[int, str]):
        """
        传入单个比特的状态，返回对应全空间下的缀饰态
        :param qubit:
        :param state:
        :return:
        """
        qid = self.q_dic[qubit].get('id')
        dress0 = (0,) * self.bit_num
        dress1 = tuple(0 if i != qid else 1 for i in range(self.bit_num))

        if isinstance(state, int):
            dress = self.eigvec[
                tuple(0 if i != qid else state for i in range(self.bit_num))
            ]
        elif state == '+':
            dress = 1 / np.sqrt(2) * (self.eigvec[dress0] + self.eigvec[dress1])
        elif state == '-':
            dress = 1 / np.sqrt(2) * (self.eigvec[dress0] - self.eigvec[dress1])
        elif state == '+i':
            dress = 1 / np.sqrt(2) * (self.eigvec[dress0] + 1j * self.eigvec[dress1])
        elif state == '-i':
            dress = 1 / np.sqrt(2) * (self.eigvec[dress0] - 1j * self.eigvec[dress1])
        else:
            raise ValueError(f'state={state} is not supported.')

        return dress

    def Kd(self, qubits: Union[tuple, str], states: Union[tuple, int, str]):
        """
        Abbreviation for ket_dress. 传入比特的状态，返回对应全空间下的缀饰态
        :param qubits:
        :param states:
        :return:
        """
        if not isinstance(qubits, tuple):
            return self._Kd1q(qubits, states)

        qids = [self.q_dic[qubit].get('id') for qubit in qubits]
        state_numbers = []
        state_coefs = []
        dress = qp.Qobj(
            np.zeros(np.prod(self.dim)), dims=[self.dim, [1] * self.bit_num]
        )

        # 根据每个比特的状态，确定该比特在基矢下的展开式以及展开系数(如'+'->state_number=(0,1), state_coef=(1/sqrt(2),1/sqrt(2)))
        for state in states:
            if isinstance(state, int):
                state_numbers.append((state,))
                state_coefs.append((1,))
            elif state == '+':
                state_numbers.append((0, 1))
                state_coefs.append((1 / np.sqrt(2), 1 / np.sqrt(2)))
            elif state == '-':
                state_numbers.append((0, 1))
                state_coefs.append((1 / np.sqrt(2), -1 / np.sqrt(2)))
            elif state == '+i':
                state_numbers.append((0, 1))
                state_coefs.append((1 / np.sqrt(2), 1j / np.sqrt(2)))
            elif state == '-i':
                state_numbers.append((0, 1))
                state_coefs.append((1 / np.sqrt(2), -1j / np.sqrt(2)))
            else:
                raise ValueError(f'state={state} is not supported.')

        # 利用product函数获得所有比特的展开基矢和展开系数所有可能的组合
        for state_number, state_coef in zip(
            itertools.product(*state_numbers), itertools.product(*state_coefs)
        ):
            full_coef = reduce(operator.mul, state_coef)
            full_number = tuple(
                0 if i not in qids else state_number[qids.index(i)]
                for i in range(self.bit_num)
            )
            dress += full_coef * self.eigvec[full_number]

        return dress

    def _Od1q(self, qubit: str, state: Union[int, str]):
        """
        传入单个比特的态/算符，返回全空间下的算符(缀饰表象)
        :param qubit:
        :param state:
        :return:
        """
        qid = self.q_dic[qubit].get('id')
        dress0 = (0,) * self.bit_num
        dress1 = tuple(0 if i != qid else 1 for i, _ in enumerate(self.dim))

        if isinstance(state, int) or state in ['+', '-', '+i', '-i']:
            dress_op = qp.ket2dm(self._Kd1q(qubit, state))
        elif state == 'I':
            dress_set = []
            for i in self.dim[qid]:
                dressi = [0, ] * self.bit_num
                dressi[qid] = i
                dress_set.append(dressi)

            dress_op = (
                sum([self.eigvec[dressi] * self.eigvec[dressi].dag() for dressi in dress_set])
            )
        elif state == 'X':
            dress_op = (
                self.eigvec[dress0] * self.eigvec[dress1].dag()
                + self.eigvec[dress1] * self.eigvec[dress0].dag()
            )
        elif state == 'Y':
            dress_op = (
                -1j * self.eigvec[dress0] * self.eigvec[dress1].dag()
                + 1j * self.eigvec[dress1] * self.eigvec[dress0].dag()
            )
        elif state == 'Z':
            dress_op = (
                self.eigvec[dress0] * self.eigvec[dress0].dag()
                - self.eigvec[dress1] * self.eigvec[dress1].dag()
            )
        else:
            raise ValueError(f'state={state} is not supported.')

        return dress_op

    def Od(self, qubits: Union[tuple, str], states: Union[tuple, int, str]):
        """
        Abbreviation for operator_dress.传入比特的态/算符，返回全空间下的算符(缀饰表象)
        :param qubits:
        :param states:
        :return:
        """
        if not isinstance(qubits, tuple):
            return self._Od1q(qubits, states)

        qids = [self.q_dic[qubit].get('id') for qubit in qubits]
        ket_numbers = []
        bra_numbers = []
        state_coefs = []
        op_dress = qp.Qobj(
            np.zeros((np.prod(self.dim), np.prod(self.dim))), dims=[self.dim, self.dim]
        )

        # 根据每个比特的状态，确定该比特在基矢下的展开式以及展开系数(如'X'->ket_number=(0,1), bra_number=(1,0),
        # state_coef=(1,1))
        if not any(state in ['I', 'X', 'Y', 'Z'] for state in states):
            return qp.ket2dm(self.Kd(qubits, states))

        for qid, state in zip(qids, states):
            if isinstance(state, int):
                ket_numbers.append((state,))
                bra_numbers.append((state,))
                state_coefs.append((1,))
            elif state == '+':
                ket_numbers.append((0, 0, 1, 1))
                bra_numbers.append((0, 1, 0, 1))
                state_coefs.append((1 / 2, 1 / 2, 1 / 2, 1 / 2))
            elif state == '-':
                ket_numbers.append((0, 0, 1, 1))
                bra_numbers.append((0, 1, 0, 1))
                state_coefs.append((1 / 2, -1 / 2, -1 / 2, 1 / 2))
            elif state == '+i':
                ket_numbers.append((0, 0, 1, 1))
                bra_numbers.append((0, 1, 0, 1))
                state_coefs.append((1 / 2, -1j / 2, 1j / 2, 1 / 2))
            elif state == '-i':
                ket_numbers.append((0, 0, 1, 1))
                bra_numbers.append((0, 1, 0, 1))
                state_coefs.append((1 / 2, 1j / 2, -1j / 2, 1 / 2))
            elif state == 'I':
                ket_numbers.append(tuple(range(self.dim[qid])))
                bra_numbers.append(tuple(range(self.dim[qid])))
                state_coefs.append((1, ) * self.dim[qid])
            elif state == 'X':
                ket_numbers.append((0, 1))
                bra_numbers.append((1, 0))
                state_coefs.append((1, 1))
            elif state == 'Y':
                ket_numbers.append((0, 1))
                bra_numbers.append((1, 0))
                state_coefs.append((-1j, 1j))
            elif state == 'Z':
                ket_numbers.append((0, 1))
                bra_numbers.append((0, 1))
                state_coefs.append((1, -1))
            else:
                raise ValueError(f'state={state} is not supported.')

        # 利用product函数获得所有比特的展开基矢和展开系数所有可能的组合
        for ket_number, bra_number, state_coef in zip(
            itertools.product(*ket_numbers),
            itertools.product(*bra_numbers),
            itertools.product(*state_coefs),
        ):
            full_coef = reduce(operator.mul, state_coef)
            full_ket_number = tuple(
                0 if i not in qids else ket_number[qids.index(i)]
                for i in range(self.bit_num)
            )
            full_bra_number = tuple(
                0 if i not in qids else bra_number[qids.index(i)]
                for i in range(self.bit_num)
            )
            op_dress += (
                full_coef
                * self.eigvec[full_ket_number]
                * self.eigvec[full_bra_number].dag()
            )

        return op_dress

    def Ed(
        self,
        qubits: Union[tuple, str],
        states: Union[tuple, int],
        q_dict_exp: dict = None,
        state_flag: bool = False,
    ):
        """
        Abbreviation for eigen-energy_dress. 传入比特的状态，返回对应全空间下的缀饰本征值
        :param q_dict_exp:
        :param qubits:
        :param states:
        :return:
        """
        if not isinstance(qubits, tuple):
            qubits = (qubits,)
            states = (states,)
        qids = [self.q_dic[qubit].get('id') for qubit in qubits]
        # 遍历所有比特索引，如果该比特不在传入的qid列表中，则返回0；否则返回states中对应的比特状态
        state_full = tuple(
            0 if i not in qids else states[qids.index(i)] for i in range(self.bit_num)
        )

        if state_flag:
            return state_full
        elif q_dict_exp is None:
            return self.eigval[state_full]
        else:
            return self.idle_eigen(q_dict_exp)[state_full]

    @parallel_allocation
    def ZZcoupling(
        self,
        coupler_freq,
        qubit0,
        qubit1,
        q_dict_exp: dict = None,
        abs_flag: bool = False,
        **kwargs,
    ):
        """
        传入两个比特，计算比特间的ZZ耦合强度。
        :param q_dict_exp: 外部参数
        :param coupler_freq: coupler频率
        :param qubit0:
        :param qubit1:
        :return:
        """
        c_name = 'C' + qubit0[1:] + '_' + qubit1[1:]
        if coupler_freq > 0:
            if q_dict_exp:
                update_nested_dic(q_dict_exp, {c_name: {'w_idle': coupler_freq}})
            else:
                q_dict_exp = {c_name: {'w_idle': coupler_freq}}
        idle_eigen_exp = self.idle_eigen(q_dict_exp)

        ZZ = (
            idle_eigen_exp[
                self.Ed((qubit0, qubit1), (1, 1), q_dict_exp, state_flag=True)
            ]
            + idle_eigen_exp[
                self.Ed((qubit0, qubit1), (0, 0), q_dict_exp, state_flag=True)
            ]
            - idle_eigen_exp[
                self.Ed((qubit0, qubit1), (0, 1), q_dict_exp, state_flag=True)
            ]
            - idle_eigen_exp[
                self.Ed((qubit0, qubit1), (1, 0), q_dict_exp, state_flag=True)
            ]
        )
        if abs_flag:
            return abs(ZZ)
        else:
            return ZZ


class ChipDynamic(Chip):
    def __init__(
        self,
        time_step: float = 0.5,
        sample_rate: float = 100,
        gate_path: str = None,
        flag_init_1q_gates: tuple = None,
        flag_R: bool = False,
        flag_trans: bool = False,
        num_cpus: int = None,
        **kwargs,
    ):
        """
        在Chip类的基础上考虑了含时的情形，主要用于创建含时哈密顿量被上层类调用
        :param chip_path: 芯片参数yaml文件路径
        :param dim: 模拟的Hilbert空间维度
        :param time_step: 模拟动力学演化时的时间间隔
        :param sample_rate: 波形采样率
        :param gate_path: 门脉冲的保存和加载路径
        :param flag_init_1q_gates: 初始化单比特波形，根据传入的波形列表确定初始化哪些波形，如果为None则不进行初始化
        :param flag_R: 是否变换到旋转坐标系，默认为False
        :param flag_trans: 是否变换到缀饰态坐标系
        :param flag_Hqq_RWA: 是否进行旋波近似
        """
        container = ChipContainer(**kwargs)
        super().__init__(**container)
        self.time_step = time_step
        self.sample_rate = sample_rate
        self.flag_R = flag_R
        self.flag_trans = flag_trans
        self.width = 0  # 脉冲宽度会在self.Ht()中被配置
        self.t = []  # 时间列表会在self.Ht()中被配置
        self.pulse_dic = {}  # 波形字典会在self.Ht()中被配置

        if gate_path:
            self.gate_path = Path(gate_path) / 'Gate'
        else:
            if self.chip_dic.get('gates'):
                pulse_path = get_nested_dicvalue(self.chip_dic, 'path')[0]
                self.gate_path = Path(pulse_path).parent
            else:
                self.gate_path = (
                    Path(self.chip_path).parent / 'Gate'
                )  # 这里考虑到多个类会共用一组gates，所以没有加__class__.__name__子文件夹
        self.gate_path.mkdir(parents=True, exist_ok=True)
        self.gate_load = {}  # 存储波形原始数据

        for f in self.gate_path.rglob('*.pulse'):
            _ = self.load_gate(f)

        if flag_init_1q_gates is not None:
            self.clear_gate('all')
            for shape in flag_init_1q_gates:
                self.init_1q_gates(shape)

        self.num_cpus = (
            qp.utilities.available_cpu_count() if num_cpus is None else num_cpus
        )
        print(f'num_cpus: {self.num_cpus}')

    def _Halpha(self):
        Halpha = {
            q: 2
            * np.pi
            * self.q_dic[q]['eta']
            / 2
            * self.ad[q]
            * self.ad[q]
            * self.a[q]
            * self.a[q]
            for q in self.q_dic.keys()
        }
        return Halpha

    def _Hz_op(self):
        """
        注意可能出现含时项的算符里都没有乘2pi，需要在生成波形或者哈密顿量系数的时候把所有频率变成角频率(主要是为了避免波形包络比较复杂时出现混乱)
        :return:
        """
        Hz_op = {q: self.ad[q] * self.a[q] for q in self.q_dic.keys()}
        return Hz_op

    def _Hx_op(self):
        """
        注意可能出现含时项的算符里都没有乘2pi，需要在生成波形或者哈密顿量系数的时候把所有频率变成角频率(主要是为了避免波形包络比较复杂时出现混乱)
        :return:
        """
        Hx_op = {q: self.ad[q] + self.a[q] for q in self.q_dic.keys()}
        return Hx_op

    def _Had_op(self):
        """
        注意可能出现含时项的算符里都没有乘2pi，需要在生成波形或者哈密顿量系数的时候把所有频率变成角频率(主要是为了避免波形包络比较复杂时出现混乱)
        :return:
        """
        Had_op = {q: self.ad[q] for q in self.q_dic.keys()}
        return Had_op

    def _Ha_op(self):
        """
        注意可能出现含时项的算符里都没有乘2pi，需要在生成波形或者哈密顿量系数的时候把所有频率变成角频率(主要是为了避免波形包络比较复杂时出现混乱)
        :return:
        """
        Ha_op = {q: self.a[q] for q in self.q_dic.keys()}
        return Ha_op

    def _Hqq_op(self):
        """
        注意可能出现含时项的算符里都没有乘2pi，需要在生成波形或者哈密顿量系数的时候把所有频率变成角频率(主要是为了避免波形包络比较复杂时出现混乱)
        :return:
        """
        # rho_pairs = [tuple(key.split('-')) for key in self.rho_map.keys()]
        if self.flag_Hqq_RWA:
            Hqq_op = {
                key: self.ad[key.split('-')[0]] * self.a[key.split('-')[-1]]
                + self.a[key.split('-')[0]] * self.ad[key.split('-')[-1]]
                for key in self.rho_map.keys()
            }
        else:
            Hqq_op = {
                key: -(self.ad[key.split('-')[0]] - self.a[key.split('-')[0]])
                * (self.ad[key.split('-')[-1]] - self.a[key.split('-')[-1]])
                for key in self.rho_map.keys()
            }
        return Hqq_op

    def _Hada_op(self):
        Hada_op = {
            key: self.ad[key.split('-')[0]] * self.a[key.split('-')[-1]]
            for key in self.rho_map.keys()
        }
        return Hada_op

    def _Haad_op(self):
        Haad_op = {
            key: self.a[key.split('-')[0]] * self.ad[key.split('-')[-1]]
            for key in self.rho_map.keys()
        }
        return Haad_op

    def _Hadad_op(self):
        Hadad_op = {
            key: self.ad[key.split('-')[0]] * self.ad[key.split('-')[-1]]
            for key in self.rho_map.keys()
        }
        return Hadad_op

    def _Haa_op(self):
        Haa_op = {
            key: self.a[key.split('-')[0]] * self.a[key.split('-')[-1]]
            for key in self.rho_map.keys()
        }
        return Haa_op

    def gqq_t(
        self, rho_pair: str, pulse_dic: dict, rho_value: Union[float, list] = None
    ):
        if rho_value is None:
            rho_value = self.rho_map[rho_pair]

        qall_dic = {**self.q_dic, **self.c_dic}
        t = None

        pair = tuple(rho_pair.split('-'))
        w_set = []
        for qubit in pair:
            pulse_q = pulse_dic.get(qubit)
            if pulse_q and pulse_q.get('Z'):
                w_set.append(
                    pulse_q['Z'].data
                    - self.flag_g_exact * 2 * np.pi * qall_dic[qubit]['eta']
                )
                t = pulse_q['Z'].t
            else:
                w_set.append(
                    2
                    * np.pi
                    * (
                        qall_dic[qubit]['w']
                        - self.flag_g_exact * qall_dic[qubit]['eta']
                    )
                )
        gt = self.rho2g(rho_value, w_set)

        if isinstance(gt, np.ndarray):
            gt = qp.Cubic_Spline(t[0], t[-1], gt)
        return gt, t

    def gqq_Rt(
        self,
        t: np.ndarray,
        rho_pair: str,
        pulse_dic: dict,
        wR: dict,
        rho_value: Union[float, list] = None,
    ):
        gRt_dict = {}
        if rho_value is None:
            rho_value = self.rho_map[rho_pair]

        qall_dic = {**self.q_dic, **self.c_dic}
        # t = None

        pair = tuple(rho_pair.split('-'))
        w_set = []
        for qubit in pair:
            pulse_q = pulse_dic.get(qubit)
            if pulse_q and pulse_q.get('Z'):
                w_set.append(
                    pulse_q['Z'].data
                    - self.flag_g_exact * 2 * np.pi * qall_dic[qubit]['eta']
                )
                # t = pulse_q['Z'].t
            else:
                w_set.append(
                    2
                    * np.pi
                    * (
                        qall_dic[qubit]['w']
                        - self.flag_g_exact * qall_dic[qubit]['eta']
                    )
                )

            # if pulse_q and pulse_q.get('XY'):
            #     t = pulse_q['XY'].t
        gt = self.rho2g(rho_value, w_set)

        gRt_dict['ada'] = gt * np.exp(1j * 2 * np.pi * (wR[pair[0]] - wR[pair[-1]]) * t)
        gRt_dict['aad'] = gt * np.exp(1j * 2 * np.pi * (wR[pair[-1]] - wR[pair[0]]) * t)
        gRt_dict['adad'] = -gt * np.exp(
            1j * 2 * np.pi * (wR[pair[0]] + wR[pair[-1]]) * t
        )
        gRt_dict['aa'] = -gt * np.exp(
            -1j * 2 * np.pi * (wR[pair[0]] + wR[pair[-1]]) * t
        )

        for key, value in gRt_dict.items():
            gRt_dict[key] = qp.Cubic_Spline(t[0], t[-1], gRt_dict[key])
        return gRt_dict

    def Ht(self, pulse_dic: dict):
        """
        输入XY和Z线上的波形参数，返回含时哈密顿量
        NOTE: 注意哈密顿量系数如果是波形插值的话，所有频率都是角频率(已乘以2pi)。而哈密顿量系数如果是频率常量的话则默认是频率，需要额外乘以2pi
        :param pulse_dic: XY和Z波形参数，key为qubit的名称，value为一个字典，键为'XY'和'Z'。
        e.g. pulse_dic = {'Q1': {'XY': xy_pulse, 'Z': z_pulse}, 'Q2': {'XY': xy_pulse, 'Z': z_pulse}, ...}
        :return:
        """
        self.pulse_dic = pulse_dic
        Ht = [*self.Halpha.values()]

        if pulse_dic is None:
            pulse_dic = {}

        txy = None
        tz = None
        ixy = 0
        iz = 0
        for q in self.q_dic.keys():
            pulse_q = pulse_dic.get(q, {})
            # 如果获取到q的波形且存在Z波形，则在Hz_op上施加flux项
            if pulse_q.get('Z'):
                wq = pulse_q.get('Z')
                Ht.append([self.Hz_op[q], wq.interp])
                if iz == 0:
                    tz = wq.t
                    self.width = wq.t[-1]
                else:
                    assert np.array_equal(wq.t, tz), ValueError(
                        'Please make sure Z width between different qubits is the same!'
                    )
                    tz = wq.t
                iz += 1
            # 如果没有获取到q的Z波形, 则只对Hz_op上施加常数项
            else:
                wq = 2 * np.pi * self.q_dic[q]['w']
                Ht.append(wq * self.Hz_op[q])

            # 如果获取到q的波形且存在XY波形，则在Hx_op上施加驱动项
            if pulse_q.get('XY'):
                xpulse_q = pulse_q.get('XY')
                Ht.append([self.Hx_op[q], xpulse_q.interp])
                if ixy == 0:
                    txy = xpulse_q.t
                    self.width = xpulse_q.t[-1]
                else:
                    assert np.array_equal(xpulse_q.t, txy), ValueError(
                        'Please make sure XY width between different qubits is the same!'
                    )
                    txy = xpulse_q.t
                ixy += 1

        # 确保XY和Z线上波形宽度一致
        if np.any(txy) and np.any(tz):
            assert np.allclose(txy, tz), ValueError(
                'Please make sure pulse width between different lines is the same!'
            )

        # 根据qubit上是否有Z波形，决定Hqq_op前面的系数是常数还是含时项
        for key, rho in self.rho_map.items():
            gt, t = self.gqq_t(key, pulse_dic, rho)
            if t is None:
                Ht.append(gt * self.Hqq_op[key])
            else:
                Ht.append([self.Hqq_op[key], gt])
                self.width = t[-1]

        self.width = int(self.width / self.time_step) * self.time_step
        self.t = np.linspace(0, self.width, int(self.width / self.time_step) + 1)
        return Ht

    def HRt(self, pulse_dic: dict, wR: dict = None):
        """
        输入XY和Z线上的波形参数，返回旋转坐标系下的含时哈密顿量
        NOTE: 注意哈密顿量系数如果是波形插值的话，所有频率都是角频率(已乘以2pi)。而哈密顿量系数如果是频率常量的话则默认是频率，需要额外乘以2pi
        :param wR: 旋转坐标系频率，为float类型时表示所有比特都变换到该坐标系，为dict类型时对每个比特分别变换到相应的旋转坐标系
        (key为比特名称或者'others')
        :param pulse_dic: XY和Z波形参数，key为qubit的名称，value为一个字典，键为'XY'和'Z'。
        e.g. pulse_dic = {'Q1': {'XY': xy_pulse, 'Z': z_pulse}, 'Q2': {'XY': xy_pulse, 'Z': z_pulse}, ...}
        :return:
        """
        self.pulse_dic = pulse_dic
        Ht = [*self.Halpha.values()]

        if pulse_dic is None:
            pulse_dic = {}

        if wR is None:
            wR = {q: self.q_dic[q]['w'] for q in self.q_dic.keys()}

        txy = None
        tz = None
        ixy = 0
        iz = 0
        for q in self.q_dic.keys():
            pulse_q = pulse_dic.get(q, {})
            # 如果获取到q的波形且存在Z波形，则在Hz_op上施加flux项
            if pulse_q.get('Z'):
                wq = pulse_q.get('Z')
                print(f'{q} Z-pulse Rotation frequency: {wq.wR}GHz')
                Ht.append([self.Hz_op[q], wq.interpR['ad*a']])
                if iz == 0:
                    tz = wq.t
                    self.width = wq.t[-1]
                else:
                    assert np.array_equal(wq.t, tz), ValueError(
                        'Please make sure Z width between different qubits is the same!'
                    )
                    tz = wq.t
                iz += 1
            # 如果没有获取到q的Z波形, 则只对Hz_op上施加常数项
            else:
                wq = 2 * np.pi * (self.q_dic[q]['w'] - wR.get(q, wR.get('others')))
                print(f'{q} Rotation frequency: {wR.get(q, wR.get("others"))}GHz')
                if wq != 0:
                    Ht.append(wq * self.Hz_op[q])

            # 如果获取到q的波形且存在XY波形，则在Hx_op上施加驱动项
            if pulse_q.get('XY'):
                xpulse_q = pulse_q.get('XY')
                print(f'{q} XY-pulse Rotation frequency: {xpulse_q.wR}GHz')
                Ht.append([self.Had_op[q], xpulse_q.interpR['ad']])
                Ht.append([self.Ha_op[q], xpulse_q.interpR['a']])
                if ixy == 0:
                    txy = xpulse_q.t
                    self.width = xpulse_q.t[-1]
                else:
                    assert np.array_equal(xpulse_q.t, txy), ValueError(
                        'Please make sure XY width between different qubits is the same!'
                    )
                    txy = xpulse_q.t
                ixy += 1

        # 确保XY和Z线上波形宽度一致
        if np.any(txy) and np.any(tz):
            assert np.allclose(txy, tz), ValueError(
                'Please make sure pulse width between different lines is the same!'
            )

        # 根据qubit上是否有Z波形，决定Hqq_op前面的系数是常数还是含时项
        for key, rho in self.rho_map.items():
            t = txy if txy is not None else tz
            gRt = self.gqq_Rt(t, key, pulse_dic, wR, rho)
            Ht.append([self.Hada_op[key], gRt['ada']])
            Ht.append([self.Haad_op[key], gRt['aad']])
            if not self.flag_Hqq_RWA:
                Ht.append([self.Hadad_op[key], gRt['adad']])
                Ht.append([self.Haa_op[key], gRt['aa']])

        self.width = int(self.width / self.time_step) * self.time_step
        self.t = np.linspace(0, self.width, int(self.width / self.time_step) + 1)
        return Ht

    def Ht_trans(self, pulse_dic: dict):
        """
        输入XY和Z线上的波形参数，返回坐标基矢变换后的哈密顿量
        NOTE: 注意哈密顿量系数如果是波形插值的话，所有频率都是角频率(已乘以2pi)。而哈密顿量系数如果是频率常量的话则默认是频率，需要额外乘以2pi
        :param pulse_dic: XY和Z波形参数，key为qubit的名称，value为一个字典，键为'XY'和'Z'。
        e.g. pulse_dic = {'Q1': {'XY': xy_pulse, 'Z': z_pulse}, 'Q2': {'XY': xy_pulse, 'Z': z_pulse}, ...}
        :return:
        """
        self.pulse_dic = pulse_dic
        Ht = [sum([*self.Halpha.values()]).transform(list(self.eigvec.values()))]

        if pulse_dic is None:
            pulse_dic = {}

        txy = None
        tz = None
        ixy = 0
        iz = 0
        for q in self.q_dic.keys():
            pulse_q = pulse_dic.get(q, {})
            # 如果获取到q的波形且存在Z波形，则在Hz_op上施加flux项
            if pulse_q.get('Z'):
                wq = pulse_q.get('Z')
                Ht.append(
                    [self.Hz_op[q].transform(list(self.eigvec.values())), wq.interp]
                )
                if iz == 0:
                    tz = wq.t
                    self.width = wq.t[-1]
                else:
                    assert np.array_equal(wq.t, tz), ValueError(
                        'Please make sure Z width between different qubits is the same!'
                    )
                    tz = wq.t
                iz += 1
            # 如果没有获取到q的Z波形, 则只对Hz_op上施加常数项
            else:
                wq = 2 * np.pi * self.q_dic[q]['w']
                Ht.append(wq * self.Hz_op[q].transform(list(self.eigvec.values())))

            # 如果获取到q的波形且存在XY波形，则在Hx_op上施加驱动项
            if pulse_q.get('XY'):
                xpulse_q = pulse_q.get('XY')
                Ht.append(
                    [
                        self.Hx_op[q].transform(list(self.eigvec.values())),
                        xpulse_q.interp,
                    ]
                )
                if ixy == 0:
                    txy = xpulse_q.t
                    self.width = xpulse_q.t[-1]
                else:
                    assert np.array_equal(xpulse_q.t, txy), ValueError(
                        'Please make sure XY width between different qubits is the same!'
                    )
                    txy = xpulse_q.t
                ixy += 1

        # 确保XY和Z线上波形宽度一致
        if np.any(txy) and np.any(tz):
            assert np.allclose(txy, tz), ValueError(
                'Please make sure pulse width between different lines is the same!'
            )

        # 根据qubit上是否有Z波形，决定Hqq_op前面的系数是常数还是含时项
        for key, rho in self.rho_map.items():
            gt, t = self.gqq_t(key, pulse_dic, rho)
            if t is None:
                Ht.append(gt * self.Hqq_op[key].transform(list(self.eigvec.values())))
            else:
                Ht.append([self.Hqq_op[key].transform(list(self.eigvec.values())), gt])
                self.width = t[-1]

        self.width = int(self.width / self.time_step) * self.time_step
        self.t = np.linspace(0, self.width, int(self.width / self.time_step) + 1)
        return Ht

    def save_gate(self, names: Union[list, str], gates: Union[list, PulseBase]):
        if isinstance(names, str):
            names = [names]
        if isinstance(gates, PulseBase):
            gates = [gates]
        for name, gate in zip(names, gates):
            this_gate_path = self.gate_path / f'{name}.pulse'
            print(f'pulse path: {this_gate_path}')
            gate_dic = {
                'gates': {name: {'repr': gate.__repr__(), 'path': str(this_gate_path)}}
            }
            update_nested_dic(self.chip_dic, gate_dic)
            update_nested_dic(self.gate_load, {name: gate})

            with open(this_gate_path, 'wb') as f:
                pickle.dump(gate, f)

        with open(self.chip_path, 'w') as f:
            chip_dic2save = copy.deepcopy(self.chip_dic)
            pop_nested_dickey(chip_dic2save, key='w')
            yaml.dump(chip_dic2save, f, sort_keys=False)

    def clear_gate(self, names: Union[list, str]):
        if names == 'all':
            all_gates_path = self.gate_path.rglob('*.pulse')
        elif isinstance(names, str):
            all_gates_path = self.gate_path.rglob(f'{names}.pulse')
        elif isinstance(names, list):
            all_gates_path = []
            for name in names:
                all_gates_path.append(next(self.gate_path.rglob(f'{name}.pulse')))
        else:
            raise TypeError(f'names have wrong type: {type(names)}.')

        chip_dic2save = copy.deepcopy(self.chip_dic)
        for path in all_gates_path:
            path.unlink(missing_ok=True)

        pop_nested_dickey(chip_dic2save, 'gates')
        with open(self.chip_path, 'w') as f:
            yaml.dump(chip_dic2save, f, sort_keys=False)

    def load_gate(self, gate_path: Union[str, Path]):
        gate_name = Path(gate_path).stem
        with open(gate_path, 'rb') as f:
            self.gate_load[gate_name] = pickle.load(f)
        return self.gate_load[gate_name]

    def init_1q_gates(self, *args):
        """
        初始化单比特波形
        :param args: 需要生成的波形名称，如Drag, FlattopGaussian等
        :return:
        """
        names = []
        gates = []
        qall_dic = {**self.q_dic, **self.c_dic}
        for q in qall_dic.keys():
            for shape in args:
                q_dic_g, rho_map_g = None, None
                if shape.endswith('_g'):
                    if not q.startswith('C'):
                        break
                    new_pair = []
                    new_rho = []
                    for pair in self.rho_map.keys():
                        if q not in pair.split('-'):
                            continue

                        pair_list = pair.split('-')
                        if len(pair_list) == 3:
                            rho_map_g = {pair: self.rho_map[pair]}
                            ql, c, qr = pair.split('-')
                            q_dic_g = {bit: self.q_dic[bit] for bit in (ql, qr)}
                            break
                        elif len(pair_list) == 2:
                            new_pair.extend(list(set(pair_list) - {q}))
                            new_rho.append(self.rho_map[pair])
                        else:
                            raise ValueError(
                                f'The length of rho_pair is {len(pair_list)}, which is not supported.'
                            )

                    if rho_map_g is None:
                        if len(new_pair) != 2:
                            # new_pair长度不为2时，默认这个bit不是coupler结构，直接跳出这个比特_g波形的构建
                            break
                        ql, qr = new_pair
                        new_rho.insert(
                            1,
                            self.rho_map.get(
                                f'{ql}-{qr}', self.rho_map.get(f'{qr}-{ql}', 0)
                            ),
                        )
                        new_pair.insert(1, q)
                        new_pair = '-'.join(new_pair)
                        rho_map_g = {new_pair: new_rho}
                        q_dic_g = {bit: self.q_dic[bit] for bit in (ql, qr)}

                if shape == 'Drag':
                    pi_pulse = Drag(
                        width=20,
                        wd=qall_dic[q]['w_idle'],
                        amp=50e-3,
                        sample_rate=self.sample_rate,
                    )
                    pi_pulse.get_pulse()
                    pio2_pulse = Drag(
                        width=20,
                        wd=qall_dic[q]['w_idle'],
                        amp=25e-3,
                        sample_rate=self.sample_rate,
                    )
                    pio2_pulse.get_pulse()
                    names.extend([f'X@{q}', f'Xo2@{q}'])
                    gates.extend([pi_pulse, pio2_pulse])

                elif shape == 'FlattopGaussian_wq':
                    ftg_pulse = FlattopGaussian(
                        width=10,
                        arg=qall_dic[q]['w_idle'],
                        arg_type='wq',
                        arg_idle=qall_dic[q]['w_idle'],
                        sample_rate=self.sample_rate,
                    )
                    ftg_pulse.get_pulse()
                    names.append(f'FlattopGaussian_wq@{q}')
                    gates.append(ftg_pulse)

                elif shape == 'FlattopGaussian_flux':
                    ftg_pulse = FlattopGaussian(
                        width=10,
                        arg=0,
                        arg_type='flux',
                        arg_idle=0,
                        sample_rate=self.sample_rate,
                        q_dic=qall_dic[q],
                    )
                    ftg_pulse.get_pulse()
                    names.append(f'FlattopGaussian_flux@{q}')
                    gates.append(ftg_pulse)

                elif shape == 'FlattopGaussian_g':
                    ftg_pulse = FlattopGaussian(
                        width=10,
                        arg=0,
                        arg_type='g',
                        arg_idle=0,
                        sample_rate=self.sample_rate,
                        q_dic=q_dic_g,
                        rho_map=rho_map_g,
                    )
                    ftg_pulse.get_pulse()
                    names.append(f'FlattopGaussian_g@{q}')
                    gates.append(ftg_pulse)

                elif shape == 'Constant_wq':
                    const_pulse = Constant(
                        width=10,
                        arg=qall_dic[q]['w_idle'],
                        arg_type='wq',
                        sample_rate=self.sample_rate,
                    )
                    const_pulse.get_pulse()
                    names.append(f'Constant_wq@{q}')
                    gates.append(const_pulse)

                elif shape == 'Constant_flux':
                    const_pulse = Constant(
                        width=10,
                        arg=0,
                        arg_type='flux',
                        sample_rate=self.sample_rate,
                        q_dic=qall_dic[q],
                    )
                    const_pulse.get_pulse()
                    names.append(f'Constant_flux@{q}')
                    gates.append(const_pulse)

                elif shape == 'Constant_g':
                    const_pulse = Constant(
                        width=10,
                        arg=0,
                        arg_type='g',
                        sample_rate=self.sample_rate,
                        q_dic=q_dic_g,
                        rho_map=rho_map_g,
                    )
                    const_pulse.get_pulse()
                    names.append(f'Constant_g@{q}')
                    gates.append(const_pulse)

                elif shape == 'Trig_wq':
                    sin_pulse = Trig(
                        width=10,
                        arg=qall_dic[q]['w_idle'],
                        shape='sin',
                        arg_type='wq',
                        arg_idle=qall_dic[q]['w_idle'],
                        period=0.25,
                        sample_rate=self.sample_rate,
                    )
                    sin_pulse.get_pulse()

                    cos_pulse = Trig(
                        width=10,
                        arg=qall_dic[q]['w_idle'],
                        shape='cos',
                        arg_type='wq',
                        arg_idle=qall_dic[q]['w_idle'],
                        period=0.25,
                        sample_rate=self.sample_rate,
                    )
                    cos_pulse.get_pulse()
                    names.extend([f'Sin_wq@{q}', f'Cos_wq@{q}'])
                    gates.extend([sin_pulse, cos_pulse])

                elif shape == 'Trig_flux':
                    sin_pulse = Trig(
                        width=10,
                        arg=0,
                        shape='sin',
                        arg_type='wq',
                        arg_idle=0,
                        period=0.5,
                        sample_rate=self.sample_rate,
                        q_dic=qall_dic[q],
                    )
                    sin_pulse.get_pulse()

                    cos_pulse = Trig(
                        width=10,
                        arg=0,
                        shape='cos',
                        arg_type='wq',
                        arg_idle=0,
                        period=0.5,
                        sample_rate=self.sample_rate,
                        q_dic=qall_dic[q],
                    )
                    cos_pulse.get_pulse()
                    names.extend([f'Sin_wq@{q}', f'Cos_wq@{q}'])
                    gates.extend([sin_pulse, cos_pulse])

                elif shape == 'Trig_g':
                    sin_pulse = Trig(
                        width=10,
                        arg=0,
                        shape='sin',
                        arg_type='g',
                        arg_idle=0,
                        period=0.25,
                        sample_rate=self.sample_rate,
                        q_dic=q_dic_g,
                        rho_map=rho_map_g,
                    )
                    sin_pulse.get_pulse()

                    cos_pulse = Trig(
                        width=10,
                        arg=0,
                        shape='cos',
                        arg_type='g',
                        arg_idle=0,
                        period=0.25,
                        sample_rate=self.sample_rate,
                        q_dic=q_dic_g,
                        rho_map=rho_map_g,
                    )
                    cos_pulse.get_pulse()
                    names.extend([f'Sin_g@{q}', f'Cos_g@{q}'])
                    gates.extend([sin_pulse, cos_pulse])

        self.save_gate(names, gates)

    @property
    def Halpha(self):
        return self._Halpha()

    @property
    def Hz_op(self):
        return self._Hz_op()

    @property
    def Hx_op(self):
        return self._Hx_op()

    @property
    def Had_op(self):
        return self._Had_op()

    @property
    def Ha_op(self):
        return self._Ha_op()

    @property
    def Hqq_op(self):
        return self._Hqq_op()

    @property
    def Hada_op(self):
        return self._Hada_op()

    @property
    def Haad_op(self):
        return self._Haad_op()

    @property
    def Hadad_op(self):
        return self._Hadad_op()

    @property
    def Haa_op(self):
        return self._Haa_op()


if __name__ == '__main__':
    pass
