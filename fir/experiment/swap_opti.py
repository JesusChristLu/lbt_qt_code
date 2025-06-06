# -*- coding: utf-8 -*-
# @Time     : 2023/3/15 10:51
# @Author   : WTL
# @Software : PyCharm
from typing import Union
import numpy as np
import qutip as qp
from scipy import optimize
import copy
from pathlib import Path

from experiment.swap import SWAP
from functions import *
from functions.containers import ExpBaseDynamicContainer


class SWAPOpti(SWAP):
    def __init__(self, QL: str, QR: str, **kwargs):
        self.opti_fun = None
        self.Uswap = qp.Qobj(
            np.array([[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]]),
            dims=[[2, 2]] * 2,
        )

        kwargs.get('plot_params', {}).setdefault(
            'units', {'x-t': 'ns', 'x-w': 'GHz', 'y-t': 'ns', 'y-w': 'MHz'}
        )
        kwargs.setdefault(
            'flag_init_1q_gates',
            (
                'FlattopGaussian_wq',
                'FlattopGaussian_flux',
                'Constant_wq',
                'Constant_flux',
            ),
        )

        container = ExpBaseDynamicContainer(**kwargs)
        super().__init__(QL, QR, **container)

    def _generate_pulse(self, x, *params):
        width, arg_tup = x
        arg_bits, arg_type, shape, init_state = params
        arg_tup = to_tuple(arg_tup)

        # 初始化波形
        zpulse0_set = dict.fromkeys(arg_bits)
        for bit, t, s in zip(arg_bits, arg_type, shape):
            zpulse0_set[bit] = self.gate_load[f'{s}_{t}@{bit}']

        # 根据实验参数生成波形列表
        zpulse_dic = {}
        for bit, arg in zip(arg_bits, arg_tup):
            zpulse = copy.deepcopy(zpulse0_set[bit])
            zpulse(width=width, arg=arg)
            zpulse_dic.update({bit: {'Z': zpulse}})

        return zpulse_dic

    def opti_P(self, x, *params):
        QL = self.QL
        QR = self.QR
        *_, init_state = params

        # 放入求解器中求解
        init_qobj = self.exp_args.get('init_state', self.Kd((QL, QR), init_state))

        mea_ops = self.exp_args.get(
            'mea_ops', [self.Od((QL, QR), (0, 1)), self.Od((QL, QR), (1, 0))]
        )

        zpulse_dic = self._generate_pulse(x, *params)

        self.result = self.state_solve(zpulse_dic, state0=init_qobj, e_ops=mea_ops)

        *_, init_state = params
        resP01 = self.result.expect[0][-1]
        resP10 = self.result.expect[1][-1]
        if init_state == (0, 1):
            return np.mean([resP01, 1 - resP10])
        elif init_state == (1, 0):
            return np.mean([resP10, 1 - resP01])
        else:
            raise ValueError(f'init_state {init_state} is not supported.')

    def opti_F(self, x, *params):
        zpulse_dic = self._generate_pulse(x, *params)

        self.result = self.propagator_solve(zpulse_dic, flag_last=True)

        idx_compu = self.generate_idx()
        Ucompu = self.result[:, idx_compu][idx_compu, :]
        Ucompu = qp.Qobj(Ucompu, dims=[[2, 2]] * 2)
        error, phi_cali = errorU_cali_phi(Ucompu, self.Uswap)
        return error

    def generate_idx(self):
        QL = self.QL
        QR = self.QR
        idx_list = []
        state_list = []
        for l, r in qp.state_number_enumerate([2] * 2):
            # 将state和比特的id一一对应
            state = {
                self.q_dic[qstr]['id']: state for qstr, state in zip((QL, QR), (l, r))
            }
            # 按照id从小到大的顺序对state重新排序，这就是qutip中U矩阵的对应顺序
            state = [state.get(qid) for qid in sorted(state)]
            state_list.append(state)
            # 将state转换为index
            idx = qp.state_number_index(self.dim, state)
            idx_list.append(idx)
        # print(f'state_list: \n{state_list}')
        # print(f'idx_list: \n{idx_list}')
        return idx_list

    def opti_swap(
        self,
        opti_method: str,
        opti_obj: str,
        width0: float,
        arg_tup0: Union[tuple, float],
        lower_bound: Union[list, np.ndarray],
        upper_bound: Union[list, np.ndarray],
        arg_bits: tuple,
        arg_type: Union[str, tuple[str]] = 'wq',
        shape: Union[str, tuple[str]] = 'Constant',
        init_state: tuple = (0, 1),
        flag_plot: bool = True,
        **kwargs,
    ):
        arg_bits, arg_type, shape, arg_tup0 = to_tuple(
            arg_bits, arg_type, shape, arg_tup0
        )
        arg_type = arg_type * len(arg_bits) if len(arg_type) == 1 else arg_type
        shape = shape * len(arg_bits) if len(shape) == 1 else shape
        params = (arg_bits, arg_type, shape, init_state)

        x0 = np.array([width0, *arg_tup0])
        x_lower = x0 + np.asarray(lower_bound)
        x_upper = x0 + np.asarray(upper_bound)

        assert opti_obj in [
            'opti_P',
            'opti_F',
        ], f'opti_obj {opti_obj} is not supported.'
        self.opti_fun = eval(f'self.{opti_obj}')

        if opti_method == 'brute':
            steps = kwargs['steps']
            ranges = []
            for l, u, s in zip(x_lower, x_upper, steps):
                ranges.append(slice(l, u, s))

            res = optimize.brute(
                self.opti_fun,
                ranges=ranges,
                args=params,
                full_output=True,
                finish=optimize.fmin,
                workers=kwargs.get('workers', -1),
            )

            if self.flag_data:
                save_data('SWAP brute opti result', 'qu', res, root_path=self.root_path)

            if flag_plot:
                bits_wq = []
                bits_flux = []
                for bit, t in zip(arg_bits, arg_type):
                    if t == 'wq':
                        bits_wq.append(bit)
                    elif t == 'flux':
                        bits_flux.append(bit)
                    else:
                        raise ValueError(f'arg_type {t} is not supported.')
                xlabel_wq = ",".join(bits_wq)
                xlabel_wq += r" $\omega$(GHz)" if xlabel_wq else ""
                xlabel_flux = ",".join(bits_flux)
                xlabel_flux += " Zamp(V)" if xlabel_flux else ""
                xlabel = xlabel_wq + xlabel_flux

                if opti_obj == 'opti_P':
                    zlabel = 'Pmin'
                elif opti_obj == 'opti_F':
                    zlabel = 'error'
                else:
                    raise ValueError(f'{opti_obj} is not supported.')

                self.plotter.plot_heatmap(
                    x=res[2][1][0, :],
                    y=res[2][0][:, 0],
                    Z=res[3],
                    xlabel=xlabel,
                    zlabel=zlabel,
                    title=f'{self.QL}-{self.QR} SWAP Opti',
                    flag_unify_x=False,
                )
        else:
            raise ValueError(f'opti_method {opti_method} is not supported.')

        return res
