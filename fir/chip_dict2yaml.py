# -*- coding: utf-8 -*-
# @Time     : 2022/9/19 15:27
# @Author   : WTL
# @Software : PyCharm
import yaml
import numpy as np

CHIP_DICT = {
        'qubits': {
            'Q1': {
                'w_idle': 4.35,
                'eta': -200e-3
            },
            'Q2': {
                'w_idle': 5.16,
                'eta': -200e-3
            },
            'Q3': {
                'w_idle': 4.56,
                'eta': -200e-3
            },
            'Q4': {
                'w_idle': 5.20,
                'eta': -200e-3
            },
            'Q5': {
                'w_idle': 4.38,
                'eta': -200e-3
            },
        },

        'rho_map': {
            'Q1-Q2': 0.005,
            'Q2-Q3': float(0.005 * np.sqrt(6) / 2),
            'Q3-Q4': float(0.005 * np.sqrt(6) / 2),
            'Q4-Q5': 0.005
        }
    }

if __name__ == '__main__':
    with open('E:\simulation_data\simu_gpst\chip_gpst_5q.yaml', 'w') as f:
        yaml.dump(CHIP_DICT, f, sort_keys=False)

    # with open(r'E:\simulation_data\simu_swap\swap.yaml', 'r') as f:
    #     y = yaml.load(f.read(), Loader=yaml.Loader)
    #     print(y)

