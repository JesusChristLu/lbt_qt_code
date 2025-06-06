# -*- coding: utf-8 -*-
# @Time     : 2022/9/27 22:33
# @Author   : WTL
# @Software : PyCharm
import numpy as np
from typing import Union
from pathlib import Path
from solvers.solvers import SolverDynamic
from functions.plot_tools import PlotTool
from functions.containers import ChipDynamicContainer, ExpBaseContainer


class ExpBase(SolverDynamic):
    def __init__(
        self,
        flag_data: bool = False,
        flag_ana_data: bool = True,
        flag_fig: bool = True,
        flag_close: bool = False,
        root_path: Union[str, Path] = None,
        plot_params: dict = None,
        **kwargs
    ):
        """
        为所有实验封装了一个实验基类，继承自``solvers.SolverDynamic``。主要是为了方便数据保存，并且规范每个实验类需要包含的基本方法: run(), analyze()
        :param chip_path: 芯片参数yaml文件路径
        :param dim: 模拟的Hilbert空间维度
        :param flag_data: 是否保存数据
        :param flag_fig: 是否保存图片
        :param flag_close: 绘图后是否删除图片(一般可以在非交互模式下保存图片以后选择将图片删除，防止占用过多内存)
        :param root_path: 保存根路径，默认在yaml文件的父目录创建一个名称与实验类相同的子文件夹
        :param plot_params: 绘图参数，详见``functions.plot_tools.PlotTool``
        """
        container = ChipDynamicContainer(**kwargs)
        super().__init__(**container)
        self.flag_data = flag_data
        self.flag_ana_data = flag_ana_data
        self.flag_fig = flag_fig
        self.flag_close = flag_close
        self.root_path = Path(root_path) if root_path else Path(self.chip_path).parent / self.__class__.__name__
        plot_params = {} if plot_params is None else plot_params
        self.plotter = PlotTool(
            flag_save=self.flag_fig,
            flag_close=self.flag_close,
            root_path=self.root_path,
            **plot_params,
        )
        self.exp_args = {}
        self.result = None
        self.res_dress = {}
        self.res_bare = {}
        self.res_analy = {}
        self.res_extra = {}
        self.ana_args = {}

    def run(self, *args, **kwargs):
        pass

    def analyze(self, *args, **kwargs):
        """
        为了提高代码可读性，约定自定义的实验类中分析模拟结果的方法名称为analyze_XX
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def extract_result_energy(self):
        bare_keys = list(self.result[0][0].keys())
        dress_keys = list(self.result[0][1].keys())
        ei_bare_arr_dic = {state: [] for state in bare_keys}
        ei_dress_arr_dic = {state: [] for state in dress_keys}

        for res in self.result:
            for bk, dk in zip(bare_keys, dress_keys):
                ei_bare_arr_dic[bk] = np.append(ei_bare_arr_dic[bk], res[0][bk])
                ei_dress_arr_dic[dk] = np.append(ei_dress_arr_dic[dk], res[1][dk])

        self.res_bare = ei_bare_arr_dic
        self.res_dress = ei_dress_arr_dic


class ExpBaseDynamic(ExpBase):
    def __init__(
            self,
            flag_gate: bool = False,
            **kwargs
    ):
        """
        为所有实验封装了一个实验基类，继承自``solvers.SolverDynamic``。主要是为了方便数据保存，并且规范每个实验类需要包含的基本方法: run(), analyze()
        :param chip_path: 芯片参数yaml文件路径
        :param dim: 模拟的Hilbert空间维度
        :param time_step: 模拟动力学演化时的时间间隔
        :param sample_rate: 波形采样率
        :param flag_R: 是否变换到旋转表象
        :param flag_trans: 是否变换到缀饰态表象
        :param flag_data: 是否保存数据
        :param flag_gate: 如果是扫描门实验，是否保存最优波形
        :param flag_fig: 是否保存图片
        :param flag_close: 绘图后是否删除图片(一般可以在非交互模式下保存图片以后选择将图片删除，防止占用过多内存)
        :param root_path: 保存根路径，默认在yaml文件的父目录创建一个名称与实验类相同的子文件夹
        :param plot_params: 绘图参数，详见``functions.plot_tools.PlotTool``
        """
        # container = ChipDynamicContainer(**kwargs)
        container = ExpBaseContainer(**kwargs)
        super().__init__(**container)

        self.flag_gate = flag_gate
        self.pulse_dic_list = []

    def run(self, *args, **kwargs):
        """
        为了提高代码可读性，约定自定义的实验类中运行动力学模拟的方法名称为run_XX, scan_XX, scan2d_XX等
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def scan(self, *args, **kwargs):
        """
        为了提高代码可读性，约定自定义的实验类中运行动力学模拟的方法名称为run_XX, scan_XX, scan2d_XX等
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def scan2d(self, *args, **kwargs):
        """
        为了提高代码可读性，约定自定义的实验类中运行动力学模拟的方法名称为run_XX, scan_XX, scan2d_XX等
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def analyze(self, *args, **kwargs):
        """
        为了提高代码可读性，约定自定义的实验类中分析模拟结果的方法名称为analyze_XX
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)
        self.analyze()

    def extract_result(
        self,
        analyze_names,
        extra_names,
        shape,
        fix_width=False,
        mea_keys=None,
        auto_t=True,
        save_name=None
    ):
        """
        将self.result中的原始结果分别保存到self.res_analy和self.res_extra中
        :param analyze_names: 保存到self.res_analy的名称
        :param extra_names: 保存到self.res_extra的名称
        :param shape: 数组形状
        :param fix_width: 是否为固定脉宽的实验(注意只有Constant波形时固定脉宽才和扫描脉宽的实验等价)
        :param mea_keys: 测量算符的名称
        :param auto_t: 对于大多数扫描波形脉宽和幅值的实验，通常外层扫width_list，内层扫arg_tup_arr。
                       而且绘制heatmap时shape也遵循len(width_list)*len(arg_tup_arr)的格式。
                       但是对于fix_width的情形，width_list为self.result.times，外层为arg_tup_arr。
                       此时如果设置auto_t=True，将自动对self.res_analy和self.res_extra进行转置。
                       如果不符合这一使用场景请将auto_t设为False
        :param save_name: 如果self.flag_ana_data为True时，保存文件名称为save_name
        :return:
        """
        results = self.result if isinstance(self.result, list) else [self.result]

        self.res_analy = dict.fromkeys(analyze_names)
        self.res_extra = dict.fromkeys(extra_names)

        for idx, key in enumerate(analyze_names + extra_names):
            if mea_keys:
                idx = mea_keys.index(key)

            if key in analyze_names:
                self.res_analy[key] = np.asarray(
                    [
                        res.expect[idx] if fix_width else res.expect[idx][-1]
                        for res in results
                    ]
                )

                if fix_width and len(shape) > 1 and auto_t:
                    self.res_analy[key] = self.res_analy[key].T

                self.res_analy[key] = self.res_analy[key].reshape(shape)

            if key in extra_names:
                self.res_extra[key] = np.asarray(
                    [
                        res.expect[idx] if fix_width else res.expect[idx][-1]
                        for res in results
                    ]
                )

                if fix_width and len(shape) > 1 and auto_t:
                    self.res_extra[key] = self.res_extra[key].T

                self.res_extra[key] = self.res_extra[key].reshape(shape)
