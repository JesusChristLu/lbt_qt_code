# -*- coding: utf-8 -*-
# @Time     : 2022/9/29 17:10
# @Author   : WTL
# @Software : PyCharm
import numpy as np
import qutip as qp

from experiment.experiment_base import ExpBase
from functions.containers import ExpBaseContainer
from functions import *


class EnergyLevel(ExpBase):
    def __init__(self, **kwargs):
        plot_params = kwargs.get('plot_params', {})
        plot_params.setdefault(
            'units', {'x-t': 'ns', 'x-w': 'GHz', 'y-t': 'ns', 'y-w': 'GHz'}
        )
        kwargs.update({'plot_params': plot_params})
        container = ExpBaseContainer(**kwargs)
        super().__init__(**container)

    def run(
        self,
        qubit: str,
        arg_list: np.ndarray,
        arg_type: str = 'wq',
        sort_method: str = 'index',
    ):
        self.exp_args.update(
            {
                'arg_list': arg_list,
            }
        )

        wq_list = [
            arg if arg_type == 'wq' else qubit_spectrum(arg, **self.q_dic[qubit])
            for arg in arg_list
        ]
        exp_dic_list = [{'qubits': {qubit: {'w': wq}}} for wq in wq_list]

        result = qp.parallel_map(
            self.eigen_solve, exp_dic_list, task_kwargs={'method': sort_method}
        )

        bare_keys = list(result[0][0].keys())
        dress_keys = list(result[0][1].keys())
        ei_bare_arr_dic = {state: [] for state in bare_keys}
        ei_dress_arr_dic = {state: [] for state in dress_keys}

        for res in result:
            for bk, dk in zip(bare_keys, dress_keys):
                ei_bare_arr_dic[bk] = np.append(ei_bare_arr_dic[bk], res[0][bk])
                ei_dress_arr_dic[dk] = np.append(ei_dress_arr_dic[dk], res[1][dk])

        self.result = {'bare': ei_bare_arr_dic, 'dress': ei_dress_arr_dic}
        # self.analyze()

    def analyze(self, excit_total=None, excit_1q=None):
        arg_list = self.exp_args['arg_list']
        ei_bare_dic_list = self.result['bare']
        ei_dress_dic_list = self.result['dress']

        self.plotter.plot_levels(
            x=arg_list,
            energy_dress=ei_dress_dic_list,
            energy_bare=ei_bare_dic_list,
            excit_total=excit_total,
            excit_1q=excit_1q,
        )
