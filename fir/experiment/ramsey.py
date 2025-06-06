# -*- coding: utf-8 -*-
# @Time     : 2022/12/19 17:15
# @Author   : WTL
# @Software : PyCharm
from typing import Union
from pathlib import Path
import copy
import qutip as qp
from operator import itemgetter

from experiment.experiment_base import ExpBaseDynamic
from pulse.pulse_base import PulseBase
from pulse.pulse_lib import *
from functions import *
from functions.containers import ExpBaseDynamicContainer


class Ramsey(ExpBaseDynamic):
    def __init__(self, **kwargs):
        kwargs.get('plot_params', {}).setdefault(
            'units', {'x-t': 'ns', 'x-w': 'MHz', 'y-t': 'ns', 'y-w': 'MHz'}
        )
        kwargs.setdefault('flag_init_1q_gates', ('Drag',))

        container = ExpBaseDynamicContainer(**kwargs)
        super().__init__(**container)

    def scan(
        self,
        qubit: str,
        delay_list: np.ndarray,
        fringe: float = 0,
        gate: PulseBase = None,
        arg_bits: Union[tuple[str, ...], str] = None,
        arg_tup: Union[tuple[float, ...], float] = None,
        arg_type: Union[str, tuple[str, ...]] = 'wq',
        arg_shape: Union[str, tuple[str, ...]] = 'Constant',
    ):
        """

        :param qubit: 比特名称
        :param delay_list: 扫描的delay列表
        :param fringe:
        :param gate:
        :param arg_bits: 施加Z波形的比特
        :param arg_tup: Z的参数
        :param arg_type: Z的参数类型
        :param arg_shape: Z的波形
        :return:
        """
        self.exp_args.update(
            {'qubit': qubit, 'delay_list': delay_list, 'fringe': fringe}
        )

        # 初始化波形
        xpio2 = self.gate_load[f'Xo2@{qubit}'] if gate is None else gate
        # 根据实验参数生成波形列表
        pulse_dic_list = []

        zpulse0_set = None
        if arg_bits:
            arg_bits, arg_tup, arg_type, arg_shape = to_tuple(
                arg_bits, arg_tup, arg_type, arg_shape
            )
            arg_type, arg_shape = repeat_tuple(arg_type, arg_shape, repeats=len(arg_bits))
            zpulse0_set = self.init_pulse(arg_bits, arg_type, arg_shape)

        for delay in delay_list:
            xfront = copy.deepcopy(xpio2)
            xcenter = Square(width=delay, wd=self.q_dic[qubit]['w_idle'], amp=0)
            xcenter.get_pulse()
            xrear = copy.deepcopy(xpio2)
            xrear(phi=2 * np.pi * fringe * delay)

            xpulse = xfront + xcenter + xrear
            # xpulse_dic_list.append({qubit: {'XY': xpulse}})
            pulse_dic = {qubit: {'XY': xpulse}}

            if arg_bits:
                for bit, arg, ty in zip(arg_bits, arg_tup, arg_type):
                    zfront = Constant(width=xpio2.width, arg=self.q_dic[bit]['w_idle'], arg_type='wq')
                    zfront.get_pulse()
                    zrear = copy.deepcopy(zfront)

                    zcenter = copy.deepcopy(zpulse0_set[bit])
                    zcenter(width=delay, arg=arg, arg_type=ty)
                    zpulse = copy.deepcopy(zfront) + zcenter + zrear
                    update_nested_dic(pulse_dic, {bit: {'Z': zpulse}})
                    if delay == delay_list[-1]:
                        self.ploter.plot_pulse(zpulse, title=f'{bit} z pulse')

            pulse_dic_list.append(pulse_dic)

        # 放入求解器中求解
        init_state = self.exp_args.get('init_state', self.Kd(qubit, 0))

        mea_ops = self.exp_args.get('mea_ops', [self.Od(qubit, 0), self.Od(qubit, 1)])

        self.result = qp.parallel_map(
            self.state_solve,
            pulse_dic_list,
            task_kwargs={
                'state0': init_state,
                'e_ops': mea_ops,
                'wR': self.exp_args.get('wR'),
            },
        )

        save_data(
            f'{qubit} scan ramsey exp_args-result',
            'qu',
            self.exp_args,
            self.result,
            root_path=self.root_path,
        )

    def analyze(self, **kwargs):
        # qubit = self.exp_args['qubit']
        # fringe = self.exp_args['fringe']
        qubit, delay_list, fringe = itemgetter('qubit', 'delay_list', 'fringe')(
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
            popt, rmse, fun_cos = fit_cos(delay_list, res_analy[key])
            popt_list.append(popt)
            rmse_list.append(rmse)
            fun_cos_list.append(fun_cos)
        popt = popt_list[np.argmin(rmse_list)]

        freq = popt[0]
        xtype = 't'
        title = f'{qubit} scan ramsey(freq={freq*1e3:.3f}MHz, fringe={fringe*1e3}MHz)'
        self.plotter.plot_lines_fit(
            delay_list,
            list(res_analy.values()),
            analyze_names,
            fun_cos_list,
            title=title,
            xtype=xtype,
            xlabel='delay(ns)',
        )

        if extra_names:
            self.plotter.plot_lines(
                delay_list,
                list(res_extra.values()),
                extra_names,
                xtype=xtype,
                xlabel='delay(ns)',
            )

        if self.flag_data:
            save_data(
                title,
                'dat',
                np.vstack([delay_list, list(res_analy.values())]).T,
                root_path=self.root_path,
            )
