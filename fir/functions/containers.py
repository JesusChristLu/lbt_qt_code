# -*- coding: utf-8 -*-
# @Time     : 2023/9/5 22:51
# @Author   : WTL
# @Software : PyCharm


class Container(dict):
    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def get(self, key, default=None):
        return self.get(key, default)

    def setdefault(self, key, default=None):
        return self.setdefault(key, default)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)


class ChipContainer(Container):
    default_CC = {
        'chip_path': None,
        'dim': 3,
        'flag_Hqq_RWA': False,
        'flag_g_exact': False,
    }

    def __init__(self, **kwargs):
        for key, value in self.default_CC.items():
            self.__dict__[key] = kwargs.get(key, value)

        super().__init__(**kwargs)


class ChipDynamicContainer(ChipContainer):
    default_CDC = {
        'time_step': 0.5,
        'sample_rate': 100.0,
        'gate_path': None,
        'flag_init_1q_gates': None,
        'flag_R': False,
        'flag_trans': False,
        'flag_Hqq_RWA': False,
        'num_cpus': None,
    }

    def __init__(self, **kwargs):
        for key, value in self.default_CDC.items():
            self.__dict__[key] = kwargs.get(key, value)

        super().__init__(**kwargs)


class ExpBaseContainer(ChipDynamicContainer):
    default_EBC = {
        'flag_data': False,
        'flag_ana_data': True,
        'flag_fig': True,
        'flag_close': False,
        'root_path': None,
        'plot_params': {},
    }

    def __init__(self, **kwargs):
        for key, value in self.default_EBC.items():
            self.__dict__[key] = kwargs.get(key, value)

        super().__init__(**kwargs)


class ExpBaseDynamicContainer(ExpBaseContainer):
    default_EBDC = {
        'flag_gate': False,
    }

    def __init__(self, **kwargs):
        for key, value in self.default_EBDC.items():
            self.__dict__[key] = kwargs.get(key, value)

        super().__init__(**kwargs)
