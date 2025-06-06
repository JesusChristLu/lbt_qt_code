# -*- coding: utf-8 -*-
# @Time     : 2022/9/27 20:35
# @Author   : WTL
# @Software : PyCharm
from typing import Union
import numpy as np
import copy
import pandas as pd
import qutip as qp
from operator import itemgetter
import inspect
from collections.abc import Iterable

from experiment.experiment_base import ExpBaseDynamic
from pulse import *
from functions import *
from functions.containers import ExpBaseDynamicContainer


class SWAP(ExpBaseDynamic):
    def __init__(self, QL: str, QR: str, **kwargs):
        self.QL = QL
        self.QR = QR

        plot_params = kwargs.get('plot_params', {})
        plot_params.setdefault(
            'units',
            {
                'x-t': 'ns',
                'x-w': 'GHz',
                'x-vol': 'V',
                'y-t': 'ns',
                'y-w': 'MHz',
                'y-vol': 'V',
            },
        )
        kwargs.update({'plot_params': plot_params})
        kwargs.setdefault(
            'flag_init_1q_gates',
            (
                'FlattopGaussian_wq',
                'FlattopGaussian_flux',
                'FlattopGaussian_g',
                'Constant_wq',
                'Constant_flux',
            ),
        )

        container = ExpBaseDynamicContainer(**kwargs)
        super().__init__(**container)

    def scan_width(
        self,
        width: Union[float, np.array],
        arg_bits: Union[tuple[str, ...], str],
        arg_tup: Union[tuple[float, ...], float],
        arg_type: Union[str, tuple[str, ...]] = 'wq',
        shape: Union[str, tuple[str, ...]] = 'Constant',
        g_bits: Union[tuple[str, ...], str] = None,
        g_tup: Union[tuple[float, ...], float] = None,
        g_type: Union[str, tuple[str, ...]] = 'g',
        g_shape: Union[str, tuple[str, ...]] = 'Constant',
    ):
        """
        固定波形幅值扫描波形宽度，对应实验上的单条SWAP曲线
        :param width: 波形宽度列表
        :param arg_bits: 需要扫描幅值的比特名称
        :param arg_tup: 需要扫描的参数数组，格式为(arg1_Q1, ..., arg1_QN)
        :param arg_type: Z线参数类型，可以选择'wq'/'flux'。如果输入类型为tuple则可为每个比特单独设置arg_type
        :param shape: Z线波形。如果输入类型为tuple则可为每个比特单独设置shape
        :param g_bits: 调节耦合强度的coupler
        :param g_tup: 耦合强度参数
        :param g_type: 耦合强度可由g或coupler频率wc或电压zc决定
        :param g_shape: 耦合强度波形
        :return:
        """
        QL = self.QL
        QR = self.QR
        arg_bits, arg_tup, arg_type, shape = to_tuple(
            arg_bits, arg_tup, arg_type, shape
        )
        arg_type, shape = repeat_tuple(arg_type, shape, repeats=len(arg_bits))
        width_list = width if isinstance(width, Iterable) else [width]
        fix_width = False if isinstance(width, Iterable) else True

        self.exp_args.update(
            {
                'width_list': width_list,
                'arg_bits': arg_bits,
                'arg_tup': arg_tup,
                'arg_type': arg_type,
                'shape': shape,
                'fix_width': fix_width,
            }
        )

        if g_bits:
            g_bits, g_tup, g_type, g_shape = to_tuple(g_bits, g_tup, g_type, g_shape)
            g_type, g_shape = repeat_tuple(g_type, g_shape, repeats=len(g_bits))
            self.exp_args.update(
                {'g_bits': g_bits, 'g_tup': g_tup, 'g_type': g_type, 'g_shape': g_shape}
            )

        # 初始化波形
        zpulse0_set = self.init_pulse(arg_bits, arg_type, shape)
        zpulse0_coupler_set = (
            self.init_pulse(g_bits, g_type, g_shape) if g_bits else None
        )

        # 根据实验参数生成波形列表
        pulse_dic_list = []
        for width in width_list:
            zpulse_dic = {}
            for bit, arg in zip(arg_bits, arg_tup):
                zpulse = copy.deepcopy(zpulse0_set[bit])
                zpulse(width=width, arg=arg)
                zpulse_dic.update({bit: {'Z': zpulse}})

            if g_bits:
                for bit, arg, shape, ty in zip(g_bits, g_tup, g_shape, g_type):
                    zpulse_coupler = copy.deepcopy(zpulse0_coupler_set[bit])
                    zpulse_coupler(width=width, arg=arg)
                    if ty == 'g':
                        q_dic_gate, rho_map_gate = index_qmap(
                            self, bit, arg_bits, arg_tup, arg_type
                        )
                        zpulse_coupler(q_dic=q_dic_gate, rho_map=rho_map_gate)
                    zpulse_dic.update({bit: {'Z': zpulse_coupler}})

            print(zpulse_dic)
            # for key, pulse in zpulse_dic.items():
            #     self.plotter.plot_pulse(pulse['Z'], title=f'{key} pulse')
            pulse_dic_list.append(zpulse_dic)

        # 放入求解器中求解
        init_state, mea_ops, mea_keys = self.generate_init_mea()

        if fix_width:
            self.result = self.state_solve(
                pulse_dic_list[0],
                state0=init_state,
                e_ops=mea_ops,
            )
        else:
            self.result = qp.parallel_map(
                self.state_solve,
                pulse_dic_list,
                task_kwargs={'state0': init_state, 'e_ops': mea_ops},
            )

        if self.flag_data:
            save_data(
                f'{QL}-{QR} {self.__class__.__name__} {inspect.currentframe().f_code.co_name} exp_args-result',
                'qu',
                self.exp_args,
                self.result,
                root_path=self.root_path,
            )

        self.exp_args.update(
            {
                'mea_keys': mea_keys,
                'width_list': self.result.times if fix_width else width_list,
            }
        )

    def scan_arg_width(
        self,
        width: Union[float, np.array],
        arg_bits: tuple,
        arg_tup_arr: list[tuple],
        arg_type: Union[str, tuple[str, ...]] = 'wq',
        shape: Union[str, tuple[str, ...]] = 'Constant',
        g_bits: Union[tuple[str, ...], str] = None,
        g_tup_arr: list[tuple] = None,
        g_type: Union[str, tuple[str, ...]] = 'g',
        g_shape: Union[str, tuple[str, ...]] = 'Constant',
    ):
        """
        固定宽度，扫描幅值，观察时间演化二维图。注意观察到的是固定波形宽度下每一时刻的时间演化，
        如果想要和实验对应观察末态布局数，请运行scan_arg_scan_width()[Note: 对于Constant波形两者没有差别]
        :param width: 波形宽度
        :param arg_bits: 需要扫描幅值的比特名称
        :param arg_tup_arr: 需要扫描的参数数组，格式为[(arg1_Q1, ..., arg1_QN), ... (argn_Q1, ..., argn_QN)]
        :param arg_type: Z线参数类型，可以选择'wq'/'flux'。如果输入类型为tuple则可为每个比特单独设置arg_type
        :param shape: Z线波形。如果输入类型为tuple则可为每个比特单独设置shape
        :param g_bits: 调节耦合强度的coupler
        :param g_tup_arr: 耦合强度参数
        :param g_type: 耦合强度可由g或coupler频率wc或电压zc决定
        :param g_shape: 耦合强度波形
        :return:
        """
        QL = self.QL
        QR = self.QR
        arg_tup_arr = np.asarray(arg_tup_arr).reshape(len(arg_tup_arr), -1)
        arg_bits, arg_type, shape = to_tuple(arg_bits, arg_type, shape)
        arg_type, shape = repeat_tuple(arg_type, shape, repeats=len(arg_bits))

        width_list = width if isinstance(width, Iterable) else [width]
        fix_width = False if isinstance(width, Iterable) else True

        assert len(arg_bits) == len(
            arg_tup_arr[0]
        ), f'arg_bits and arg_tuple must have the same length!'

        self.exp_args.update(
            {
                'width': width,
                'arg_bits': arg_bits,
                'arg_tup_arr': arg_tup_arr,
                'arg_type': arg_type,
                'shape': shape,
                'fix_width': fix_width,
            }
        )

        if g_bits:
            g_tup_arr = np.asarray(g_tup_arr).reshape(len(g_tup_arr), -1)
            g_bits, g_type, g_shape = to_tuple(g_bits, g_type, g_shape)
            g_type, g_shape = repeat_tuple(g_type, g_shape, repeats=len(g_bits))

            self.exp_args.update(
                {
                    'g_bits': g_bits,
                    'g_tup_arr': g_tup_arr,
                    'g_type': g_type,
                    'g_shape': g_shape,
                }
            )

        # 初始化波形
        zpulse0_set = self.init_pulse(arg_bits, arg_type, shape)
        zpulse0_coupler_set = (
            self.init_pulse(g_bits, g_type, g_shape) if g_bits else None
        )

        # 根据实验参数生成波形列表
        pulse_dic_list = []
        for width in width_list:
            all_arg_tup_arr = zip(arg_tup_arr, g_tup_arr) if g_bits else arg_tup_arr
            for arg_tups in all_arg_tup_arr:
                if g_bits:
                    arg_tup, g_tup = arg_tups
                else:
                    arg_tup, g_tup = arg_tups, None

                zpulse_dic = {}
                for bit, arg in zip(arg_bits, arg_tup):
                    zpulse = copy.deepcopy(zpulse0_set[bit])
                    zpulse(width=width, arg=arg)
                    zpulse_dic.update({bit: {'Z': zpulse}})

                if g_bits:
                    for bit, arg, shape, ty in zip(g_bits, g_tup, g_shape, g_type):
                        zpulse_coupler = copy.deepcopy(zpulse0_coupler_set[bit])
                        zpulse_coupler(width=width, arg=arg)
                        if ty == 'g':
                            q_dic_gate, rho_map_gate = index_qmap(
                                self, bit, arg_bits, arg_tup, arg_type
                            )
                            zpulse_coupler(q_dic=q_dic_gate, rho_map=rho_map_gate)
                        zpulse_dic.update({bit: {'Z': zpulse_coupler}})

                    # if width == width_list[-1] and np.all(arg_tup == arg_tup_arr[-1]):
                    #     self.plotter.plot_pulse(zpulse, title=f'{bit} swap pulse')

                pulse_dic_list.append(zpulse_dic)
                print(zpulse_dic)

        # 放入求解器中求解
        init_state, mea_ops, mea_keys = self.generate_init_mea()

        self.result = qp.parallel_map(
            self.state_solve,
            pulse_dic_list,
            task_kwargs={'state0': init_state, 'e_ops': mea_ops},
        )

        if self.flag_data:
            save_data(
                f'{QL}-{QR} {self.__class__.__name__} {inspect.currentframe().f_code.co_name} exp_args-result',
                'qu',
                self.exp_args,
                self.result,
                root_path=self.root_path,
            )

        self.exp_args.update(
            {
                'mea_keys': mea_keys,
                'width_list': self.result[0].times if fix_width else width_list,
            }
        )

    def analyze_1d(self, plot_cos: bool = True, **kwargs):
        QL = self.QL
        QR = self.QR
        (
            width_list,
            arg_bits,
            arg_tup,
            arg_type,
            shape,
            fix_width,
            mea_keys,
        ) = itemgetter(
            'width_list',
            'arg_bits',
            'arg_tup',
            'arg_type',
            'shape',
            'fix_width',
            'mea_keys',
        )(
            self.exp_args
        )
        analyze_names = kwargs.get(
            'analyze_names', ('P01', 'P10')
        )  # analy_name_list指定需要拟合的数据
        extra_names = kwargs.get('extra_names', ())  # extra_name_list指定不需要拟合的数据

        self.extract_result(
            analyze_names,
            extra_names,
            fix_width=fix_width,
            shape=(len(width_list),),
            mea_keys=mea_keys,
        )

        # res_val = list(self.res_analy.values())
        # res01 = self.res_analy.get('P01', res_val[0])
        # res10 = self.res_analy.get('P10', res_val[1])

        _ = self._analyze_1d([arg_tup], width_list, plot_cos)

        xlabel = f'pulse width(ns)'
        arg_str = [
            f'{bit} {t}={round(arg, 3)}GHz'
            if t == 'wq'
            else f'{bit} {t}={round(arg, 3)}V'
            for bit, arg, t in zip(arg_bits, arg_tup, arg_type)
        ]
        title = f'{QL}-{QR} SWAP evolution'

        if extra_names:
            self.plotter.plot_lines(
                width_list,
                list(self.res_extra.values()),
                extra_names,
                xlabel=xlabel,
                title=title,
            )

        if self.flag_data:
            save_data(
                f'{QL}-{QR} SWAP evolution{analyze_names + extra_names}({", ".join(arg_str)})',
                'dat',
                np.vstack(
                    [
                        width_list,
                        list(self.res_analy.values()) + list(self.res_extra.values()),
                    ]
                ).T,
                root_path=self.root_path,
            )

    def analyze_2d(
        self,
        plot_cos: bool = False,
        plot_bits: Union[tuple[str, ...], str] = None,
        fit_arg_bits: Union[tuple, str] = None,
        **kwargs,
    ):
        """
        qubit二维参数扫描分析类。目前仅支持扫描单个参数或多个参数绑定扫描的实验，
        不支持多个参数组合扫描的实验(itertools.product())
        :param plot_cos: 是否绘制cos拟合结果
        :param plot_bits: 选择heatmap中绘制哪些比特
        :param fit_arg_bits: 设定需要拟合分析结果的arg_bits
        :return:
        """
        QL = self.QL
        QR = self.QR

        (
            width_list,
            arg_bits,
            arg_tup_arr,
            arg_type,
            shape,
            fix_width,
            mea_keys,
        ) = itemgetter(
            'width_list',
            'arg_bits',
            'arg_tup_arr',
            'arg_type',
            'shape',
            'fix_width',
            'mea_keys',
        )(
            self.exp_args
        )  # noqa

        plot_bits = arg_bits if plot_bits is None else to_tuple(plot_bits)
        analyze_names = kwargs.get(
            'analyze_names', ('P01', 'P10')
        )  # analy_name_list指定需要拟合的数据
        extra_names = kwargs.get('extra_names', ())  # extra_name_list指定不需要拟合的数据
        suffix = f'({kwargs["suffix"]})' if kwargs.get('suffix') else ''

        self.extract_result(
            analyze_names,
            extra_names,
            fix_width=fix_width,
            shape=(len(width_list), len(arg_tup_arr)),
            mea_keys=mea_keys,
        )

        self._analyze_2d(
            width_list,
            arg_bits,
            arg_tup_arr,
            arg_type,
            plot_bits,
            analyze_names,
            extra_names,
            suffix,
        )

        swap_freq_list, popt01_list, popt10_list = self._analyze_1d(
            arg_tup_arr, width_list, plot_cos
        )

        self.ana_args.update(
            {
                'popt01_list': popt01_list,
                'popt10_list': popt10_list,
                'swap_freq_list': swap_freq_list,
            }
        )

        if fit_arg_bits is not None:
            fit_arg_bits = to_tuple(fit_arg_bits)
            self.ana_args.update({'wmin': [], 'zmin': []})
            for fit_bit in fit_arg_bits:
                bit_idx = arg_bits.index(fit_bit)
                arg_arr = np.array(arg_tup_arr)[:, bit_idx]
                t = arg_type[bit_idx]
                s = shape[bit_idx]
                xmin, g, rmse_swap, fun_swap = fit_swap(
                    arg_arr, swap_freq_list, arg_type=t
                )
                if t == 'wq':
                    wmin = xmin
                    zmin = freq2flux(wmin, **self.q_dic[fit_bit])
                    xlabel = fr'{fit_bit} $\omega$'
                    xtype = 'w'
                else:
                    zmin = xmin
                    wmin = qubit_spectrum(zmin, **self.q_dic[fit_bit])
                    xlabel = f'{fit_bit} Zamp'
                    xtype = 'vol'

                xmin_str = (
                    fr'wmin={xmin * 1e3:.2f}MHz' if t == 'wq' else fr'Zmin={xmin:.4f}V'
                )
                title = f'{QL}-{QR} SWAP_scan {fit_bit}({xmin_str}, g={g * 1e3:.2f}MHz)'
                # self.plotter.plot_swap(arg_arr, swap_freq_list, fun_swap, xlabel=xlabel, title=title)
                self.plotter.plot_lines_fit(
                    arg_arr,
                    [swap_freq_list],
                    ['swap'],
                    [fun_swap],
                    xlabel=xlabel,
                    xtype=xtype,
                    ylabel='SWAP Freq',
                    ytype='w',
                    title=title,
                )

                if self.flag_gate:
                    t_swap = 1 / (4 * g)
                    if s == 'Constant':
                        zpulse = Constant(
                            t_swap,
                            xmin,
                            arg_type=t,
                            sample_rate=self.sample_rate,
                            q_dic=self.q_dic[fit_bit],
                        )
                        zpulse.get_pulse()
                    elif s == 'FlattopGaussian':
                        arg_idle = self.q_dic[fit_bit]['w_idle'] if t == 'wq' else 0
                        zpulse = FlattopGaussian(
                            t_swap,
                            xmin,
                            arg_idle=arg_idle,
                            arg_type=t,
                            sample_rate=self.sample_rate,
                            q_dic=self.q_dic[fit_bit],
                        )
                        zpulse.get_pulse()
                    else:
                        raise ValueError(f'pulse shape {shape} is not supported!')

                    self.save_gate(f'SWAP({QL}-{QR})@{fit_bit}', zpulse)

                if self.flag_data:
                    save_data(
                        title,
                        'dat',
                        np.c_[arg_arr, swap_freq_list],
                        root_path=self.root_path,
                    )
                    save_data(
                        f'{QL}-{QR} SWAP_scan {fit_bit}_zmin-wmin-g',
                        'txt',
                        np.c_[zmin, wmin, g],
                        root_path=self.root_path,
                    )

                self.ana_args['wmin'].append(wmin)
                self.ana_args['zmin'].append(zmin)

            if len(self.ana_args['wmin']) == 1:
                self.ana_args['wmin'] = self.ana_args['wmin'][0]
                self.ana_args['zmin'] = self.ana_args['zmin'][0]

        if self.flag_data:
            index_name = [r'time\arg']
            df_list = []
            for name in analyze_names + extra_names:
                res_all = {**self.res_analy, **self.res_extra}
                df = pd.DataFrame(
                    res_all[name],
                    index=width_list,
                    columns=[
                        str(tuple(np.around(arg_tup, 3))) for arg_tup in arg_tup_arr
                    ],
                )
                df.index.names = index_name
                df_list.append(df)

            save_data(
                f'{QL}-{QR} SWAP Spectrum',
                'xlsx',
                *df_list,
                sheet_name=analyze_names + extra_names,
                root_path=self.root_path,
            )

    def analyze_c2d(
        self,
        plot_cos: bool = False,
        plot_bits: Union[tuple[str, ...], str] = None,
        fit_arg_bits: Union[tuple, str] = None,
        **kwargs,
    ):
        """
        coupler二维参数扫描分析类。目前仅支持扫描单个参数或多个参数绑定扫描的实验，
        不支持多个参数组合扫描的实验(itertools.product())
        :param plot_cos: 是否绘制cos拟合结果
        :param plot_bits: 选择heatmap中绘制哪些比特
        :param fit_arg_bits: 设定需要拟合分析结果的arg_bits
        :return:
        """
        QL = self.QL
        QR = self.QR

        (
            width_list,
            g_bits,
            g_tup_arr,
            g_type,
            g_shape,
            fix_width,
            mea_keys,
        ) = itemgetter(
            'width_list',
            'g_bits',
            'g_tup_arr',
            'g_type',
            'g_shape',
            'fix_width',
            'mea_keys',
        )(
            self.exp_args
        )
        plot_bits = g_bits if plot_bits is None else to_tuple(plot_bits)
        analyze_names = kwargs.get(
            'analyze_names', ('P01', 'P10')
        )  # analy_name_list指定需要拟合的数据
        extra_names = kwargs.get('extra_names', ())  # extra_name_list指定不需要拟合的数据
        suffix = f'({kwargs["suffix"]})' if kwargs.get('suffix') else ''

        self.extract_result(
            analyze_names,
            extra_names,
            fix_width=fix_width,
            shape=(len(width_list), len(g_tup_arr)),
            mea_keys=mea_keys,
        )

        self._analyze_2d(
            width_list,
            g_bits,
            g_tup_arr,
            g_type,
            plot_bits,
            analyze_names,
            extra_names,
            suffix,
        )

        swap_freq_list, popt01_list, popt10_list = self._analyze_1d(
            g_tup_arr, width_list, plot_cos
        )

        self.ana_args.update(
            {
                'popt01_list': popt01_list,
                'popt10_list': popt10_list,
                'swap_freq_list': swap_freq_list,
            }
        )

        if fit_arg_bits is not None:
            fit_arg_bits = to_tuple(fit_arg_bits)
            self.ana_args.update({'geff': []})
            for fit_bit in fit_arg_bits:
                bit_idx = g_bits.index(fit_bit)
                arg_arr = np.array(g_tup_arr)[:, bit_idx]
                t = g_type[bit_idx]

                popt_geff, rmse_geff, fun_geff = fit_geff(
                    arg_arr, swap_freq_list, arg_type=t
                )
                print(f'units before: {self.plotter.units}')

                if t == 'wq':
                    xlabel = fr'{fit_bit} $\omega$'
                    xtype = 'w'
                elif t == 'flux':
                    xlabel = f'{fit_bit} Zamp'
                    xtype = 'vol'
                elif t == 'g':
                    xlabel = f'{fit_bit} g'
                    xtype = 'w'
                    self.plotter.units['x-w'] = 'MHz'
                else:
                    raise ValueError(f'g_type {g_type} is not supported.')

                self.plotter.plot_lines_fit(
                    arg_arr,
                    [swap_freq_list],
                    ['geff'],
                    [fun_geff],
                    xlabel=xlabel,
                    xtype=xtype,
                    ylabel='SWAP Freq',
                    ytype='w',
                    title=f'{QL}-{QR} SWAP_scan {fit_bit}',
                )
                self.ana_args['geff'].append(fun_geff)
                self.plotter.reset()
                print(f'units after: {self.plotter.units}')

            if len(self.ana_args['geff']) == 1:
                self.ana_args['geff'] = self.ana_args['geff'][0]

    def _analyze_1d(
        self,
        arg_tup_arr,
        width_list,
        plot_cos: bool,
    ):
        res_val = list(self.res_analy.values())
        res01 = self.res_analy.get('P01', res_val[0])
        res01 = res01.reshape(len(res01), -1)
        res10 = self.res_analy.get('P10', res_val[1])
        res10 = res10.reshape(len(res10), -1)

        swap_freq_list = []
        popt01_list = []
        popt10_list = []
        for i, arg_tup in enumerate(arg_tup_arr):
            popt01, rmse01, fun_cos01 = fit_cos(width_list, res01[:, i])
            popt10, rmse10, fun_cos10 = fit_cos(width_list, res10[:, i])
            popt01_list.append(popt01)
            popt10_list.append(popt10)

            if plot_cos:
                print(f'arg: {arg_tup}\npopt01: {popt01}\npopt10: {popt10}\n')
                popt_mean = np.around(
                    np.mean(
                        [
                            [popt10[0] * 1e3, popt10[1], popt10[2]],
                            [popt01[0] * 1e3, popt01[1], popt01[2]],
                        ],
                        axis=0,
                    ),
                    3,
                )

                self.plotter.plot_lines_fit(
                    width_list,
                    [res01[:, i], res10[:, i]],
                    ['P01', 'P10'],
                    [fun_cos01, fun_cos10],
                    xtype='t',
                    xlabel='time',
                    ytype='',
                    ylabel=r'Expectation',
                    title=f'{self.QL}-{self.QR} SWAP(freq={round(popt10[0] * 1e3, 2)}MHz, amp={round(2*popt10[1], 3)})',
                    save_name=f'arg={list(arg_tup)}, popt={list(popt_mean)}',
                )

            if rmse01 < rmse10:
                popt = popt01
            else:
                popt = popt10
            swap_freq, *_ = popt
            swap_freq_list.append(swap_freq)

        return swap_freq_list, popt01_list, popt10_list

    def _analyze_2d(
        self,
        width_list,
        arg_bits,
        arg_tup_arr,
        arg_type,
        plot_bits,
        analyze_names,
        extra_names,
        suffix,
    ):
        res_all = {**self.res_analy, **self.res_extra}
        xlabel, arg_tup_arr_plot = generate_x2d(
            self, arg_bits, arg_tup_arr, arg_type, plot_bits
        )

        if self.plotter.flag_save:
            for name in analyze_names + extra_names:
                self.plotter.plot_heatmap(
                    arg_tup_arr_plot,
                    width_list,
                    res_all[name],
                    xlabel=xlabel,
                    xtype='',
                    zlabel=name,
                    title=f'{self.QL}-{self.QR} SWAP Spectrum({name}){suffix}',
                    units={'x-w': 'GHz'},
                    rotation=None
                )

    def generate_init_mea(self):
        init_state = self.exp_args.get(
            'init_state', self.Kd((self.QL, self.QR), (1, 0))
        )

        mea_ops = self.exp_args.get(
            'mea_ops',
            [self.Od((self.QL, self.QR), (0, 1)), self.Od((self.QL, self.QR), (1, 0))],
        )

        mea_keys = self.exp_args.get('mea_keys', ['P01', 'P10'])
        return init_state, mea_ops, mea_keys

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
                pulse0_set[bit] = self.gate_load[pulse_name]
            elif shape in ['Cos1', 'Cos2']:
                fun = eval(shape)
                pulse0_set[bit] = fun(
                    width=10,
                    arg=0,
                    arg_type='g',
                    arg_idle=0,
                    sample_rate=self.sample_rate,
                    q_dic={'Q1': {'w': 4.0}, 'Q2': {'w': 4.0}},
                    rho_map={'Q1-C12-Q2': [0.03, 0.03, 0.003]},
                )
            elif shape in ['Gauss1', 'Gauss2']:
                fun = eval(shape)
                pulse0_set[bit] = fun(
                    width=10,
                    arg=0,
                    arg_type='g',
                    arg_idle=0,
                    sample_rate=self.sample_rate,
                    q_dic={'Q1': {'w': 4.0}, 'Q2': {'w': 4.0}},
                    rho_map={'Q1-C12-Q2': [0.03, 0.03, 0.003]},
                )
                pass
            else:
                pulse_name = f'{shape}_{ty}@{bit}'
                pulse0_set[bit] = self.gate_load[pulse_name]

        return pulse0_set


def index_qmap(self, coupler: str, arg_bits, arg_tup, arg_type):
    # q_dic_gate = None
    rho_map_gate = None
    ql, qr = None, None
    new_pair = []
    new_rho = []
    for pair in self.rho_map.keys():
        pair_list = pair.split('-')
        if coupler not in pair_list:
            continue

        if len(pair_list) == 3:
            ql, c, qr = pair_list
            rho_map_gate = {pair: self.rho_map[pair]}
            break
        elif len(pair_list) == 2:
            new_pair.extend(list(set(pair_list) - {coupler}))
            new_rho.append(self.rho_map[pair])
        else:
            raise ValueError(
                f'The length of rho_pair is {len(pair_list)}, which is not supported.'
            )

    if rho_map_gate is None:
        assert len(new_pair) == 2, f'Length of pair = {len(new_pair)} is not supported.'
        ql, qr = new_pair
        new_rho.insert(
            2, self.rho_map.get(f'{ql}-{qr}', self.rho_map.get(f'{qr}-{ql}', 0))
        )
        new_pair.insert(1, coupler)
        new_pair = '-'.join(new_pair)
        rho_map_gate = {new_pair: new_rho}

    try:
        idxl = arg_bits.index(ql)
        wl = (
            arg_tup[idxl]
            if arg_type[idxl] == 'wq'
            else qubit_spectrum(arg_tup[idxl], **self.q_dic[ql])
        )
    except Exception as e:
        # print(f'{e}. Retrieve index failed.')
        wl = self.q_dic[ql]['w']

    try:
        idxr = arg_bits.index(qr)
        wr = (
            arg_tup[idxr]
            if arg_type[idxr] == 'wq'
            else qubit_spectrum(arg_tup[idxr], **self.q_dic[qr])
        )
    except Exception as e:
        # print(f'{e}. Retrieve index failed.')
        wr = self.q_dic[qr]['w']

    q_dic_gate = {ql: {'w': wl}, qr: {'w': wr}}

    return q_dic_gate, rho_map_gate


def generate_x2d(
    self,
    arg_bits: Union[tuple, list],
    arg_tup_arr: np.ndarray,
    arg_type: Union[tuple, list],
    plot_bits: Union[tuple[str, ...], str] = None,
):
    plot_bits = arg_bits if plot_bits is None else to_tuple(plot_bits)
    bits_wq = []
    bits_flux = []
    bits_g = []
    arg_tup_arr_plot = []
    for bit, t, arg_tup in zip(arg_bits, arg_type, arg_tup_arr.T):
        if bit not in plot_bits:
            continue
        if t == 'wq':
            bits_wq.append(bit)
            _, arg_tup_new = self.plotter.unify_units('x', 'w', label='', values=arg_tup)
            arg_tup_arr_plot.append(arg_tup_new)
        elif t == 'flux':
            bits_flux.append(bit)
            _, arg_tup_new = self.plotter.unify_units(
                'x', 'vol', label='', values=arg_tup
            )
            arg_tup_arr_plot.append(arg_tup_new)
        elif t == 'g':
            bits_g.append(bit)
            arg_tup_new = arg_tup * 1e3
            arg_tup_arr_plot.append(arg_tup_new)
        else:
            raise ValueError(f'arg_type {t} is not supported.')
    unit_wq, unit_flux = itemgetter('x-w', 'x-vol')(self.plotter.units)
    xlabel = fr'{",".join(bits_wq)} $\omega$({unit_wq})' if bits_wq else ''
    xlabel += f'{",".join(bits_flux)} Z({unit_flux})' if bits_flux else ''
    xlabel += f'{",".join(bits_g)} g(MHz)' if bits_g else ''
    return xlabel, np.asarray(arg_tup_arr_plot).T


if __name__ == '__main__':
    pass
