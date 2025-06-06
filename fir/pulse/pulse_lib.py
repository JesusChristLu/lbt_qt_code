# -*- coding: utf-8 -*-
# @Time     : 2022/9/22 20:35
# @Author   : WTL
# @Software : PyCharm
import numpy as np
from scipy.special import erf
from collections.abc import Iterable
from pulse.pulse_base import XYPulseBase, ZPulseBase


class Square(XYPulseBase):
    def __init__(
        self,
        width,
        wd,
        amp,
        phi=0,
        detu=0,
        sample_rate: float = 100.0,
        wR: float = None,
    ):
        # 基类中会将width调整为1/sample_rate的整数倍self.width，并根据self.width计算出波形时间列表self.t
        super().__init__(width, wd, phi, sample_rate, wR)

        self.amp = amp
        self.detu = detu

    def get_pulse(self):
        """
        pulse = real[Xquad*exp(1j*(detu*t-phi)) * exp(1j*wd*t)]
              = real{[Xquad*cos(detu*t-phi) -1j*(-Xquad*sin(detu*t-phi))] * exp(1j*wd*t)}
              = Xquad*cos(detu*t-phi) * cos(wd*t) - Xquad*sin(detu*t-phi) * sin(wd*t)
              = Xquad'*cos(wd*t) + Yquad'*sin(wd*t)
        Xquad' = Xquad*cos(detu*t-phi)
        Yquad' = -Xquad*sin(detu*t-phi)
        :return:
        """
        wd, amp, detu = [2 * np.pi * arg for arg in (self.wd, self.amp, self.detu)]

        Xquad0 = amp * np.ones_like(self.t)

        self._envelope[(self.wd, self.wR)] = {
            'X': np.real(Xquad0 * np.exp(1j * (detu * self.t - self.phi))),
            'Y': -np.imag(Xquad0 * np.exp(1j * (detu * self.t - self.phi)))
        }
        super().get_pulse()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}: width={self.width:.2f}ns, wd={self.wd * 1e3:.2f}MHz, '
            f'amp={self.amp * 1e3:.2f}MHz, detu={self.detu * 1e3:.2f}MHz, '
            f'phi={self.phi / np.pi:.2f}*pi. wR={self.wR * 1e3:.2f}MHz'
        )


class Drag(XYPulseBase):
    def __init__(
        self,
        width,
        wd,
        amp,
        phi=0,
        detu=0,
        lam=0.5,
        eta=-200e-3,
        sample_rate: float = 100.0,
        wR: float = None,
    ):
        # 基类中会将width调整为1/sample_rate的整数倍self.width，并根据self.width计算出波形时间列表self.t
        super().__init__(width, wd, phi, sample_rate, wR)

        self.amp = amp
        self.detu = detu
        self.lam = lam
        self.eta = eta

    def get_pulse(self):
        """
        pulse = real[(Xquad - 1j*Yquad)*exp(1j*(detu*t-phi)) * exp(1j*wd*t)]
              = real{[Xquad*cos(detu*t-phi) + Yquad*sin(detu*t-phi)
                     -1j*(-Xquad*sin(detu*t-phi) + Yquad*cos(detu*t-phi))] * exp(1j*wd*t)}
              = [Xquad*cos(detu*t-phi) + Yquad*sin(detu*t-phi)] * cos(wd*t) +
                [-Xquad*sin(detu*t-phi) + Yquad*cos(detu*t-phi)] * sin(wd*t)
              = Xquad'*cos(wd*t) + Yquad'*sin(wd*t)
        Xquad' = Xquad*cos(detu*t-phi) + Yquad*sin(detu*t-phi)
        Yquad' = -Xquad*sin(detu*t-phi) + Yquad*cos(detu*t-phi)
        :return:
        """
        wd, amp, detu, eta = [
            2 * np.pi * arg for arg in (self.wd, self.amp, self.detu, self.eta)
        ]

        Xquad0 = amp / 2 * (1 - np.cos(2 * np.pi * self.t / self.width))
        Yquad0 = - self.lam / eta * amp / 2 * 2 * np.pi / self.width * np.sin(2 * np.pi * self.t / self.width)

        self._envelope[(self.wd, self.wR)] = {
            'X': np.real((Xquad0 - 1j * Yquad0) * np.exp(1j * (detu * self.t - self.phi))),
            'Y': -np.imag((Xquad0 - 1j * Yquad0) * np.exp(1j * (detu * self.t - self.phi)))
        }
        super().get_pulse()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}: width={self.width:.2f}ns, wd={self.wd * 1e3:.2f}MHz, '
            f'amp={self.amp * 1e3:.2f}MHz, detu={self.detu * 1e3:.2f}MHz, '
            f'eta={self.eta * 1e3:.2f}MHz, lam={self.lam:.2f}, phi={self.phi / np.pi:.3f}pi. '
            f'wR={self.wR * 1e3:.2f}MHz'
        )


class DragGaussian(XYPulseBase):
    def __init__(
        self,
        width,
        wd,
        amp,
        phi=0,
        detu=0,
        lam=0.5,
        eta=-200e-3,
        sigma=None,
        sample_rate: float = 100.0,
        wR: float = None,
    ):
        # 基类中会将width调整为1/sample_rate的整数倍self.width，并根据self.width计算出波形时间列表self.t
        super().__init__(width, wd, phi, sample_rate, wR)

        self.amp = amp
        self.detu = detu
        self.lam = lam
        self.eta = eta
        self.sigma = sigma if sigma else 0.5 * width

    def get_pulse(self):
        wd, amp, detu, eta = [
            2 * np.pi * arg for arg in (self.wd, self.amp, self.detu, self.eta)
        ]
        sigma = self.sigma
        t = self.t
        width = self.width
        lam = self.lam

        Xquad0 = amp * np.exp(-(t - width/2)**2 / (2*sigma**2)) - amp * np.exp(-(width/2)**2 / (2*sigma**2))
        Yquad0 = lam / eta * amp * (t - width/2)/sigma**2 * np.exp(-(t - width/2)**2 / (2*sigma**2))

        self._envelope[(self.wd, self.wR)] = {
            'X': np.real((Xquad0 - 1j * Yquad0) * np.exp(1j * (detu * self.t - self.phi))),
            'Y': -np.imag((Xquad0 - 1j * Yquad0) * np.exp(1j * (detu * self.t - self.phi)))
        }
        super().get_pulse()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}: width(total)={self.width:.2f}ns, wd={self.wd * 1e3:.2f}MHz, '
            f'amp={self.amp * 1e3:.2f}MHz, detu={self.detu * 1e3:.2f}MHz, '
            f'eta={self.eta * 1e3:.2f}MHz, lam={self.lam:.2f}, phi={self.phi / np.pi:.3f}pi, '
            f'sigma={self.sigma}ns. wR={self.wR * 1e3:.2f}MHz'
        )


class DragTanh(XYPulseBase):
    def __init__(
        self,
        width,
        wd,
        amp,
        detu=0,
        phi=0,
        lam=0.5,
        eta=-200e-3,
        sigma=None,
        sample_rate: float = 100.0,
        wR: float = None,
    ):
        # 基类中会将width调整为1/sample_rate的整数倍self.width，并根据self.width计算出波形时间列表self.t
        super().__init__(width, wd, phi, sample_rate, wR)

        self.amp = amp
        self.detu = detu
        self.lam = lam
        self.eta = eta
        self.sigma = sigma if sigma else 0.6 * width - 1.4

    def get_pulse(self):
        wd, amp, detu, eta = [
            2 * np.pi * arg for arg in (self.wd, self.amp, self.detu, self.eta)
        ]
        sigma = self.sigma
        t = self.t
        width = self.width
        lam = self.lam

        Xquad0 = amp * (np.tanh(t/sigma) + np.tanh((width-t)/sigma)) - amp * np.tanh(width/sigma)
        Yquad0 = - lam / eta * amp / sigma * (-np.tanh(t/sigma)**2 + np.tanh((width-t)/sigma)**2)

        self._envelope[(self.wd, self.wR)] = {
            'X': np.real((Xquad0 - 1j * Yquad0) * np.exp(1j * (detu * self.t - self.phi))),
            'Y': -np.imag((Xquad0 - 1j * Yquad0) * np.exp(1j * (detu * self.t - self.phi)))
        }
        super().get_pulse()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}: width(total)={self.width:.2f}ns, wd={self.wd * 1e3:.2f}MHz, '
            f'amp={self.amp * 1e3:.2f}MHz, detu={self.detu * 1e3:.2f}MHz, '
            f'eta={self.eta * 1e3:.2f}MHz, lam={self.lam:.2f}, phi={self.phi / np.pi:.3f}pi, '
            f'sigma={self.sigma}ns. wR={self.wR * 1e3:.2f}MHz'
        )


class FlattopGaussianEnv(XYPulseBase):
    def __init__(
        self,
        width,
        wd,
        amp,
        phi=0,
        detu=0,
        sigma=1.25,
        buffer=5,
        sample_rate: float = 100.0,
        wR: float = None,
    ):
        # 基类中会将width调整为1/sample_rate的整数倍self.width，并根据self.width计算出波形时间列表self.t
        super().__init__(width, wd, phi, sample_rate, wR)

        self.amp = amp
        self.detu = detu
        self.sigma = sigma
        self.buffer = buffer

    def get_pulse(self):
        wd, amp, detu = [2 * np.pi * arg for arg in (self.wd, self.amp, self.detu)]

        Xquad0 = amp / 2 * \
                 (erf((self.t - self.buffer) / (np.sqrt(2) * self.sigma)) -
                  erf((self.t - self.t[-1] + self.buffer) / (np.sqrt(2) * self.sigma)))  # noqa

        self._envelope[(self.wd, self.wR)] = {
            'X': np.real(Xquad0 * np.exp(1j * (detu * self.t - self.phi))),
            'Y': -np.imag(Xquad0 * np.exp(1j * (detu * self.t - self.phi)))
        }
        super().get_pulse()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}: width={self.width:.2f}ns, wd={self.wd * 1e3:.2f}MHz, '
            f'amp={self.amp * 1e3:.2f}MHz, detu={self.detu * 1e3:.2f}MHz, phi={self.phi / np.pi:.2f}*pi,'
            f'sigma={self.sigma}ns, buffer={self.buffer}ns. wR={self.wR * 1e3:.2f}MHz'
        )


class Constant(ZPulseBase):
    def __init__(
        self,
        width,
        arg,
        arg_type: str = 'wq',
        buffer=0,
        sample_rate: float = 100.0,
        q_dic: dict = None,
        rho_map: dict = None,
        wR: float = None,
    ):
        # 基类中会将width调整为1/sample_rate的整数倍self.width，并根据self.width计算出波形时间列表self.t
        super().__init__(width, sample_rate, wR)

        self.arg = arg
        self.arg_idle = arg
        self.arg_type = arg_type
        self.q_dic = q_dic
        self.rho_map = rho_map
        self.buffer = buffer

    def get_pulse(self):
        self._arg_data = self.arg * np.ones_like(self.t)
        if self.buffer:
            mask = (self.t <= self.buffer) | (self.t >= (self.width - self.buffer))
            self._arg_data[mask] = 0

        self.arg2wq()
        super().get_pulse()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}: width={self.width:.2f}ns, wq={self.wq * 1e3:.2f}MHz. '
            f'wR={self.wR * 1e3:.2f}MHz'
        )


class FlattopGaussian(ZPulseBase):
    def __init__(
        self,
        width,
        arg,
        arg_type: str = 'wq',
        sigma=1.25,
        buffer=5,
        arg_idle=0,
        sample_rate: float = 100.0,
        q_dic: dict = None,
        rho_map: dict = None,
        wR: float = None,
    ):
        # 基类中会将width调整为1/sample_rate的整数倍self.width，并根据self.width计算出波形时间列表self.t
        super().__init__(width, sample_rate, wR)

        self.arg = arg
        self.arg_type = arg_type
        self.arg_idle = arg_idle
        self.q_dic = q_dic
        self.rho_map = rho_map
        self.sigma = sigma
        self.buffer = buffer

    def get_pulse(self):
        self._arg_data = (self.arg - self.arg_idle) / 2 * (
            erf((self.t - self.buffer) / (np.sqrt(2) * self.sigma))
            - erf((self.t - self.t[-1] + self.buffer) / (np.sqrt(2) * self.sigma))
        ) + self.arg_idle

        self.arg2wq()
        super().get_pulse()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}: width={self.width:.2f}ns, wq={self._wq * 1e3:.2f}MHz, '
            f'sigma={self.sigma:.2f}, buffer={self.buffer:.2f}ns. wR={self.wR * 1e3:.2f}MHz'
        )


class Trig(ZPulseBase):
    def __init__(
        self,
        width,
        arg,
        shape,
        arg_type: str = 'g',
        arg_idle: float = 0,
        period: float = 0.25,
        sample_rate: float = 100.0,
        q_dic: dict = None,
        rho_map: dict = None,
        wR: float = None,
    ):
        # 基类中会将width调整为1/sample_rate的整数倍self.width，并根据self.width计算出波形时间列表self.t
        super().__init__(width, sample_rate, wR)

        self.shape = shape
        self.arg = arg
        self.arg_type = arg_type
        self.arg_idle = arg_idle
        self.period = period
        self.q_dic = q_dic
        self.rho_map = rho_map

    def get_pulse(self):
        if self.shape in ['sin', 'cos']:
            func = eval(f'np.{self.shape}')
        else:
            raise ValueError(f'Trig shape {self.shape} is not supported.')

        self._arg_data = (self.arg - self.arg_idle) * func(
            2 * np.pi * self.period * self.t / self.width
        ) + self.arg_idle

        self.arg2wq()
        super().get_pulse()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.shape}): width={self.width:.2f}ns, period={self.period}ns, '
            f'wq={self._wq * 1e3:.2f}MHz. wR={self.wR * 1e3:.2f}MHz'
        )


class PiecewiseConstant(ZPulseBase):
    def __init__(
        self, width, data: Iterable, sample_rate: float = 1.0, wR: float = None
    ):
        # 基类中会将width调整为1/sample_rate的整数倍self.width，并根据self.width计算出波形时间列表self.t
        super().__init__(width, sample_rate, wR)
        self._data = data

    def get_pulse(self):

        super().get_pulse()

    def __repr__(self):
        return f'{self.__class__.__name__}: width={self.width:.2f}ns'


if __name__ == '__main__':
    pass
