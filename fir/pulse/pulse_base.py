# -*- coding: utf-8 -*-
# @Time     : 2022/9/22 16:41
# @Author   : WTL
# @Software : PyCharm
import copy
from abc import ABC, abstractmethod
import qutip as qp
import numpy as np
from typing import Union
from functions import *


class PulseBase(ABC):
    @abstractmethod
    def __init__(self, width: float, sample_rate: float = 100.0, wR: float = None):
        """
        波形基类，所有自定义的波形都要继承自此基类。定义了get_pulse()和__repr__()两个必须在自定义波形中配置的抽象方法。
        重定义了加法操作__add__()实现波形拼接，乘法操作__mul__()实现两段波形的相乘或波形与常数的相乘，__call__实现实例类似函数一样被调用，
        sqrt()实现波形数据求平方根(主要是Z波形在计算哈密顿量耦合项系数时会用到)
        :param width: 波形宽度
        :param sample_rate: 采样率(指创建插值函数的采样率，并非动力学模拟的时间间隔)，默认为100.0GHz
        """
        self._width = int(width * sample_rate) / sample_rate
        self.sample_rate = sample_rate
        self._t = np.linspace(0, self.width, int(self.width * self.sample_rate) + 1)
        self._wR = wR
        self._data = None
        self._interp = None
        self._dataR = None
        self._interpR = None
        self._envelope = None

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, width):
        self._width = int(width * self.sample_rate) / self.sample_rate
        self._t = np.linspace(0, self._width, int(self._width * self.sample_rate) + 1)

    @property
    def wR(self) -> float:
        return self._wR

    @wR.setter
    def wR(self, wR):
        self._wR = wR

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def interp(self) -> qp.Cubic_Spline:
        return self._interp

    @property
    def dataR(self) -> dict:
        return self._dataR

    @property
    def interpR(self) -> dict:
        return self._interpR

    @property
    def envelope(self) -> dict:
        return self._envelope

    @abstractmethod
    def get_pulse(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        return self.__repr__()

    def __call__(self, **kwargs):
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise AttributeError(
                        f'{self.__class__.__name__} does not have attribute {key}'
                    )
        self.get_pulse()

    @abstractmethod
    def __add__(self, other: Union['XYPulseBase', 'ZPulseBase']):
        new_self = copy.deepcopy(self)
        new_self._t = np.hstack((self._t, self._t[-1] + other._t[1:]))
        new_self._width = new_self._t[-1]
        return new_self

    @abstractmethod
    def __mul__(self, repeat: int):
        new_self = copy.deepcopy(self)
        t_repeat = new_self._t
        for _ in range(repeat - 1):
            t_repeat = np.hstack((t_repeat, t_repeat[-1] + new_self._t[1:]))
        new_self._t = t_repeat
        new_self._width = new_self._t[-1]
        return new_self

    @abstractmethod
    def __rmul__(self, repeat: int):
        return self.__mul__(repeat)

    @abstractmethod
    def __imul__(self, repeat: int):
        return self.__mul__(repeat)

    @abstractmethod
    def __truediv__(self, other: Union['XYPulseBase', 'ZPulseBase', float]):
        new_self = copy.deepcopy(self)
        if isinstance(other, PulseBase):
            assert len(self._data) == len(other._data), AssertionError(
                f'Pulse length must match when perform / operation!'
            )
            new_self._data = self._data / other._data
            new_self._dataR = {
                key: self._dataR[key] / other._dataR[key] for key in self._dataR.keys()
            }
        else:
            new_self._data = self._data / other
            new_self._dataR = {
                key: self._dataR[key] / other for key in self._dataR.keys()
            }

        new_self._interp = qp.Cubic_Spline(
            new_self._t[0], new_self._t[-1], new_self._data
        )
        new_self._interpR = {
            key: qp.Cubic_Spline(new_self._t[0], new_self._t[-1], new_self._dataR[key])
            for key in self._dataR.keys()
        }
        return new_self

    @abstractmethod
    def __rtruediv__(self, other: Union['XYPulseBase', 'ZPulseBase', float]):
        new_self = copy.deepcopy(self)
        if isinstance(other, PulseBase):
            assert len(self._data) == len(other._data), AssertionError(
                f'Pulse length must match when perform / operation!'
            )
            new_self._data = other._data / self._data
            new_self._dataR = {
                key: other._dataR[key] / self._dataR[key] for key in self._dataR.keys()
            }
        else:
            new_self._data = other / self._data
            new_self._dataR = {
                key: other / self._dataR[key] for key in self._dataR.keys()
            }

        new_self._interp = qp.Cubic_Spline(
            new_self._t[0], new_self._t[-1], new_self._data
        )
        new_self._interpR = {
            key: qp.Cubic_Spline(new_self._t[0], new_self._t[-1], new_self._dataR[key])
            for key in self._dataR.keys()
        }
        return new_self


class XYPulseBase(PulseBase):
    def __init__(
        self,
        width: float,
        wd: float,
        phi: float,
        sample_rate: float = 100.0,
        wR: float = None,
    ):
        """
        XY波形基类，所有自定义的XY波形都要继承自此基类。
        :param width: 波形宽度
        :param sample_rate: sample_rate: 采样率(指创建插值函数的采样率，并非动力学模拟的时间间隔)，默认为100.0GHz
        """
        wR = wR if wR else wd
        super().__init__(width, sample_rate, wR)
        self._dataR = dict.fromkeys(('ad', 'a'))
        self._interpR = dict.fromkeys(('ad', 'a'))
        self._envelope = {}  # dict.fromkeys(('X', 'Y'))
        self.wd = wd
        self.phi = phi

    @abstractmethod
    def get_pulse(self):
        """
        Hd = (Xquad * cos(wd * t) + Yquad * sin(wd * t)) * (ad + a)
        HRd = R * Hd * Rd
            = (Xquad * cos(wd * t) + Yquad * sin(wd * t)) * (exp(1j*wR*t)*ad + exp(-1j*wR*t)*a)
            (RWA) =>  [Xquad/2 * exp(-1j*(wd-wR)*t) - Yquad/2j * exp(-1j*(wd-wR)*t)] * ad +
                      [Xquad/2 * exp(1j*(wd-wR)*t) + Yquad/2j * exp(1j*(wd-wR)*t)] * a
        :return:
        """
        self._data = 0
        self._dataR['ad'] = 0
        self._dataR['a'] = 0
        for wd, wR in self.envelope.keys():
            Xquad = self.envelope[(wd, wR)]['X']
            Yquad = self.envelope[(wd, wR)]['Y']

            self._data += Xquad * np.cos(2*np.pi * wd * self.t) + Yquad * np.sin(2*np.pi * wd * self.t)

            # # RWA
            # Dt = 2*np.pi * (self.wd - self.wR) * self.t
            # self._dataR['ad'] = Xquad / 2 * np.exp(-1j * Dt) - Yquad / 2j * np.exp(-1j * Dt)
            # self._dataR['a'] = Xquad / 2 * np.exp(1j * Dt) + Yquad / 2j * np.exp(1j * Dt)

            # no RWA
            self._dataR['ad'] += self._data * np.exp(1j * 2*np.pi * wR * self.t)
            self._dataR['a'] += self._data * np.exp(-1j * 2*np.pi * wR * self.t)

        self._interp = qp.Cubic_Spline(self.t[0], self.t[-1], self.data)
        self._interpR = {key: qp.Cubic_Spline(self.t[0], self.t[-1], self._dataR[key]) for key in ('ad', 'a')}

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def __add__(self, other: 'XYPulseBase'):
        new_self = super().__add__(other)
        wd_wR_set = list({**self.envelope, **other.envelope}.keys())
        for wd_wR in wd_wR_set:
            self_env = self.envelope.get(wd_wR, {'X': np.zeros_like(self.t), 'Y': np.zeros_like(self.t)})
            other_env = other.envelope.get(wd_wR, {'X': np.zeros_like(other.t), 'Y': np.zeros_like(other.t)})

            new_self._envelope[wd_wR] = {key: np.hstack((self_env[key], other_env[key][1:])) for key in ('X', 'Y')}

        super(type(new_self), new_self).get_pulse()
        return new_self

    def __mul__(self, repeat: int):
        new_self = super().__mul__(repeat)
        envelope_repeat = copy.deepcopy(new_self._envelope)
        for _ in range(repeat - 1):
            envelope_repeat = {
                key: np.hstack((envelope_repeat[key], new_self._envelope[key][1:]))
                for key in ('X', 'Y')
            }
        new_self._envelope = envelope_repeat
        super(type(new_self), new_self).get_pulse()
        return new_self

    def __rmul__(self, repeat: int):
        return self.__mul__(repeat)

    def __imul__(self, repeat: int):
        return self.__mul__(repeat)

    def __truediv__(self, other: 'XYPulseBase'):
        new_self = super().__truediv__(other)
        if isinstance(other, XYPulseBase):
            assert len(self._data) == len(other._data), AssertionError(
                f'Pulse length must match when perform * operation!'
            )
            new_self._envelope = {
                key: self.envelope[key] / other.envelope[key] for key in ('X', 'Y')
            }
        else:
            new_self._envelope = {key: self.envelope[key] / other for key in ('X', 'Y')}
        return new_self

    def __rtruediv__(self, other: 'XYPulseBase'):
        new_self = super().__rtruediv__(other)
        if isinstance(other, XYPulseBase):
            assert len(self._data) == len(other._data), AssertionError(
                f'Pulse length must match when perform * operation!'
            )
            new_self._envelope = {
                key: other.envelope[key] / self.envelope[key] for key in ('X', 'Y')
            }
        else:
            new_self._envelope = {key: other / self.envelope[key] for key in ('X', 'Y')}
        return new_self


class ZPulseBase(PulseBase):
    def __init__(self, width: float, sample_rate: float = 100.0, wR: float = None):
        """
        XY波形基类，所有自定义的XY波形都要继承自此基类。
        :param width: 波形宽度
        :param sample_rate: sample_rate: 采样率(指创建插值函数的采样率，并非动力学模拟的时间间隔)，默认为100.0GHz
        """
        super().__init__(width, sample_rate, wR)
        self._arg_data = None
        self._dataR = dict.fromkeys(('ad*a',))
        self._interpR = dict.fromkeys(('ad*a',))
        self._wq = None
        self._wq_idle = None
        self.arg = None
        self.arg_idle = None
        self.arg_type = None
        self.q_dic = None
        self.rho_map = None

    @property
    def wq(self):
        return self._wq

    @property
    def wq_idle(self):
        return self._wq_idle

    @property
    def arg_data(self):
        return self._arg_data

    @abstractmethod
    def get_pulse(self):
        self._interp = qp.Cubic_Spline(self.t[0], self.t[-1], self._data)

        # pulse in Rotation Frame
        if self.wR is None:
            self._wR = self._wq_idle
        wR = 2 * np.pi * self.wR
        self._dataR['ad*a'] = self._data - wR
        self._interpR['ad*a'] = qp.Cubic_Spline(
            self.t[0], self.t[-1], self._dataR['ad*a']
        )

    def arg2wq(self):
        """
        将Z线上的任意参数类型转换为频率类型，并给self.data赋值
        :return:
        """
        if self.arg_type == 'g':
            rho_pair, *_ = self.rho_map.keys()
            rho_value, *_ = self.rho_map.values()
            ql, c, qr = rho_pair.split('-')
            wl, wr = [self.q_dic[bit]['w'] for bit in (ql, qr)]
            self._wq = geff2wc(self.arg, wl, wr, rho_value)
            self._wq_idle = geff2wc(self.arg_idle, wl, wr, rho_value)
            self._data = 2 * np.pi * geff2wc(self.arg_data, wl, wr, rho_value)

        elif self.arg_type == 'flux':
            self._wq = qubit_spectrum(self.arg, **self.q_dic)
            self._wq_idle = qubit_spectrum(self.arg_idle, **self.q_dic)
            self._data = 2 * np.pi * qubit_spectrum(self.arg_data, **self.q_dic)

        elif self.arg_type == 'wq':
            self._wq = self.arg
            self._wq_idle = self.arg_idle
            self._data = 2 * np.pi * self.arg_data

        else:
            raise ValueError(f'arg_type {self.arg_type} is not supported.')

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def __add__(self, other: 'ZPulseBase'):
        new_self = super().__add__(other)
        new_self._data = np.hstack((self._data, other._data[1:]))
        super(type(new_self), new_self).get_pulse()
        return new_self

    def __mul__(self, repeat: int):
        new_self = super().__mul__(repeat)
        data_repeat = new_self.data
        for _ in range(repeat - 1):
            data_repeat = np.hstack((data_repeat, new_self.data[1:]))
        new_self._data = data_repeat
        super(type(new_self), new_self).get_pulse()
        return new_self

    def __rmul__(self, repeat: int):
        return self.__mul__(repeat)

    def __imul__(self, repeat: int):
        return self.__mul__(repeat)

    def __truediv__(self, other: Union['ZPulseBase', float]):
        return super().__truediv__(other)

    def __rtruediv__(self, other: Union['ZPulseBase', float]):
        return super().__rtruediv__(other)
