# -*- coding: utf-8 -*-
# @Time     : 2022/9/21 16:34
# @Author   : WTL
# @Software : PyCharm
from collections.abc import Mapping
import numpy as np
import qutip as qp
from typing import Union
from datetime import datetime
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from functools import wraps
from functions import *


def update_nested_dic(dic: dict, dic1: Mapping):
    for key, value in dic1.items():
        if isinstance(value, Mapping):
            update_nested_dic(dic.get(key, dic.setdefault(key, {})), value)
        else:
            dic[key] = value


def pop_nested_dickey(dic: dict, key: str):
    try:
        for k, v in dic.items():
            if isinstance(v, dict):
                if dic.get(key):
                    dic.pop(key)
                else:
                    pop_nested_dickey(v, key)
            else:
                if dic.get(key):
                    dic.pop(key)
                    # break
    except Exception as e:
        pop_nested_dickey(dic, key)


def get_nested_dicvalue(dic: dict, key: str):
    value = []

    def _get_nested_dicvalue(dic: dict, key: str):
        for k, v in dic.items():
            if isinstance(v, dict):
                _get_nested_dicvalue(v, key)
            else:
                if dic.get(key):
                    value.append(dic[key])

    _get_nested_dicvalue(dic, key)

    return list(set(value))


def to_tuple(*args):
    args_new = []
    for arg in args:
        if isinstance(arg, (float, int, complex, str)):
            arg_new = (arg,)
        elif isinstance(arg, (list, np.ndarray)):
            if isinstance(arg[0], tuple):
                arg_new = arg
            else:
                arg_new = [(v,) for v in arg]
            # arg_new = tuple(arg)
        elif isinstance(arg, tuple):
            arg_new = arg
        else:
            raise ValueError(f'arg type {type(arg)} is not supported now.')

        args_new.append(arg_new)
    if len(args_new) == 1:
        return args_new[0]
    else:
        return args_new


def repeat_tuple(*args, repeats: Union[int, tuple, list] = 1, flag_check: bool = True):
    args_new = []
    if isinstance(repeats, int):
        repeats = (repeats,) * len(args)

    for arg, repeat in zip(args, repeats):
        if flag_check and len(arg) != 1:
            args_new.append(arg)
        else:
            arg_new = arg * repeat
            args_new.append(arg_new)

    if len(args_new) == 1:
        return args_new[0]
    else:
        return args_new


def save_data(save_name: str, data_type: str, *data, root_path: Path = None, **kwargs):
    date = datetime.now().strftime('%Y-%m')
    time = datetime.now().strftime('%m%d-%H.%M.%S')
    root_path = root_path if root_path else Path.cwd()
    current_path = root_path / date
    current_path.mkdir(parents=True, exist_ok=True)
    data_path = current_path / f'{time}_{save_name}.{data_type}'

    if data_type == 'xlsx':
        excel_spec = pd.ExcelWriter(data_path, engine='xlsxwriter')
        sheet_names = kwargs.get('sheet_name')
        for i, d in enumerate(tuple(data)):
            if sheet_names is None:
                sheet_name = i
            else:
                sheet_name = sheet_names[i]

            d.to_excel(excel_spec, sheet_name=sheet_name)
        # excel_spec.save()
        excel_spec.close()

    elif data_type == 'csv':
        header = kwargs.get('header', '')
        np.savetxt(data_path, *data, delimiter=',', comments='', header=header)

    elif data_type in ['dat', 'txt']:
        np.savetxt(data_path, *data, delimiter=' ' * 4)

    elif data_type == 'qu':
        data_path = current_path / f'{time}_{save_name}'
        qp.qsave(data, data_path)

    else:
        raise ValueError(f'data type {data_type} is not supported now!')
    print(f'data saved: {data_path}')


def load_data(data_type: str, **kwargs):
    data_file_type_map = {
        'xlsx': ('Excel工作表', '.xlsx'),
        'dat': ('DAT文件', '.dat'),
        'qu': ('qu文件', '.qu'),
        'txt': ('文本文档', '.txt'),
    }

    window = tk.Tk()
    window.withdraw()

    default_load_dir = Path.home() / '.transmon_qubit'
    default_load_dir.mkdir(parents=True, exist_ok=True)
    try:
        default_load_file = next(default_load_dir.glob('load_path.txt'))
        with open(default_load_file, mode='r') as f:
            initialdir = f.readline()
    except StopIteration as si:
        print(si)
        initialdir = Path.cwd()

    if data_file_type_map.get(data_type):
        filetypes = [data_file_type_map[data_type]]
    else:
        filetypes = list(data_file_type_map.values())

    file = filedialog.askopenfile(
        title='选择文件', initialdir=initialdir, filetypes=filetypes
    )

    print(f'load file: {file.name}')
    with open(default_load_dir / 'load_path.txt', mode='w') as f:
        f.write(str(Path(file.name).parent))
    window.destroy()

    if data_type == 'xlsx':
        data = []
        sheet_names = kwargs.get('sheet_name', (0,))
        for sheet_name in sheet_names:
            df = pd.read_excel(file.name, sheet_name=sheet_name, index_col=0)
            data.append(df)
    elif data_type in ['dat', 'txt']:
        data = np.loadtxt(file.name)
    elif data_type == 'qu':
        data = qp.qload(file.name)
    else:
        raise ValueError(f'data type {data_type} is not supported now!')
    return data


def generate_idx(chip, q_list):
    id_list = [chip.q_dic[q]['id'] for q in q_list]
    id_list.sort()
    substate_list = list(qp.state_number_enumerate([2] * len(q_list)))
    state_list = [
        tuple(
            [
                substate[id_list.index(id)] if id in id_list else 0
                for id in range(len(chip.q_dic))
            ]
        )
        for substate in substate_list
    ]
    idx_list = [qp.state_number_index(chip.dim, state) for state in state_list]
    return idx_list


def parallel_allocation(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 多变量并行实验传入的格式为((arg1, arg2, ...))，需要进行一定的处理
        if isinstance(args[0], tuple) and len(args) == 1:
            args = args[0]

        if kwargs.get('parallel_args'):
            for arg_idx, arg_name in enumerate(kwargs['parallel_args']):
                kwargs.update({arg_name: args[arg_idx]})
            return func(self, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return wrapper


def parallel_allocation_def(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 多变量并行实验传入的格式为((arg1, arg2, ...))，需要进行一定的处理
        if isinstance(args[0], tuple) and len(args) == 1:
            args = args[0]

        if kwargs.get('parallel_args'):
            for arg_idx, arg_name in enumerate(kwargs['parallel_args']):
                kwargs.update({arg_name: args[arg_idx]})
            return func(**kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def generate_chip_dic_list(
    self
):
    arg_bits, arg_type, arg_tup_arr, g_bits, g_tup_arr, g_type, rho_pair, rho_arr = [
        self.exp_args.get(key) for key in
        ('arg_bits', 'arg_type', 'arg_tup_arr', 'g_bits', 'g_tup_arr', 'g_type',
         'rho_pair', 'rho_arr')
    ]

    len_arg = 0 if arg_tup_arr is None else len(arg_tup_arr)
    len_g = 0 if g_tup_arr is None else len(g_tup_arr)
    len_rho = 0 if rho_arr is None else len(rho_arr)
    len_set = list({len_arg, len_g, len_rho} - {0})
    assert len(len_set) == 1,\
        'length of all variables must be equal when they are assigned.'
    len_list = len_set[0]

    if arg_bits:
        arg_tup_arr = np.asarray(arg_tup_arr).reshape(len(arg_tup_arr), -1)
        arg_bits, arg_type = to_tuple(arg_bits, arg_type)
        arg_type = repeat_tuple(arg_type, repeats=len(arg_bits))
        self.exp_args.update({
            'arg_bits': arg_bits,
            'arg_type': arg_type,
            'arg_tup_arr': arg_tup_arr,
        })

    if g_bits:
        g_tup_arr = np.asarray(g_tup_arr).reshape(len(g_tup_arr), -1)
        g_bits, g_type = to_tuple(g_bits, g_type)
        g_type = repeat_tuple(g_type, repeats=len(g_bits))
        self.exp_args.update({
            'g_bits': g_bits,
            'g_tup_arr': g_tup_arr,
            'g_type': g_type,
        })

    if rho_pair:
        self.exp_args.update({
            'rho_pair': rho_pair,
            'rho_arr': rho_arr
        })

    exp_dic_list = []
    # 对参数列表进行循环
    for idx in range(len_list):
        # q_dic, c_dic = {}, {}
        exp_dic = copy.deepcopy(self.chip_dic)

        if len_arg:
            arg_tup = arg_tup_arr[idx]
            for bit, arg, ty in zip(arg_bits, arg_tup, arg_type):
                wq = arg if ty == 'wq' else qubit_spectrum(arg, **self.q_dic[bit])
                # q_dic.update({bit: {'w': wq}})
                update_nested_dic(exp_dic, {bit: {'w': wq}})

        if len_g:
            g_tup = g_tup_arr[idx]
            for bit, arg, ty in zip(g_bits, g_tup, g_type):
                if ty == 'wq':
                    wq = arg
                elif ty == 'flux':
                    wq = qubit_spectrum(arg, **self.c_dic[bit])
                elif ty == 'g':
                    arg_tup = arg_tup_arr[idx]
                    q_dic_exp, rho_map_exp = index_qmap(self, bit, arg_bits, arg_tup, arg_type)
                    rho_name, *_ = rho_map_exp.keys()
                    rho_value, *_ = rho_map_exp.values()
                    ql, c, qr = rho_name.split('-')
                    wl, wr = [q_dic_exp[bit]['w'] for bit in (ql, qr)]
                    wq = geff2wc(arg, wl, wr, rho_value)
                else:
                    raise ValueError(f'g_type {ty} is not supported.')
                update_nested_dic(exp_dic, {bit: {'w': wq}})

        if len_rho:
            rho = rho_arr[idx]
            update_nested_dic(exp_dic, {rho_pair: rho})

        exp_dic_list.append(exp_dic)
    return exp_dic_list


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


if __name__ == '__main__':
    pass
