# -*- coding: utf-8 -*-
# @Time     : 2022/9/28 11:00
# @Author   : WTL
# @Software : PyCharm
from typing import Union
from functools import partial
from scipy.optimize import curve_fit, least_squares
from scipy.fftpack import fft, fftfreq
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy.polynomial import Polynomial
from functions.formulas import qubit_spectrum, cosine, swap, geff


def fit_spectrum(flux_list, fq_list):
    #    [fq_max, eta,           sws,      period, d]
    lb = [3, -np.inf, np.min(flux_list), 0, 0]
    ub = [10, 0, np.max(flux_list), 5, 1]

    def error(paras, *args):
        fq_max, eta, sws, period, d = paras
        flux, fq = args
        return fq - qubit_spectrum(
            flux, w_max=fq_max, eta=eta, sws=sws, period=period, d=d
        )

    popt0 = [np.max(fq_list), -200e-3, np.median(flux_list), 1, 0.25]
    res = least_squares(error, x0=popt0, bounds=(lb, ub), args=(flux_list, fq_list))
    popt = res.x
    fun_spectrum = partial(
        qubit_spectrum,
        w_max=popt[0],
        eta=popt[1],
        sws=popt[2],
        period=popt[3],
        d=popt[4],
    )
    rmse = mean_squared_error(fq_list, fun_spectrum(flux_list), squared=False)
    return popt, rmse, fun_spectrum


def fit_fft(x, y, freq_max=np.inf):
    # 对y进行傅里叶变换，并根据列表长度和间隔计算采样频率
    y_fft = fft(y)
    dx = x[1] - x[0]
    f = fftfreq(len(x), dx)

    # 处理fft数据，保留频率为正的部分并归一化
    mask = (f > 0) & (f < freq_max)
    x1 = f[mask]
    y1 = np.abs(y_fft[mask])
    y1 = y1 / len(x) * 2  # / np.max(y1)

    # peaks, _ = find_peaks(y1)
    # if len(peaks) == 0:
    peaks = [np.argmax(y1)]

    basic_freq = x1[np.min(peaks)]
    basic_amp = y1[np.min(peaks)]
    basic_phase = np.angle(y_fft[mask])[np.min(peaks)]
    basic_offset = abs(y_fft[0]) / len(x)
    return basic_freq, basic_amp, basic_phase, basic_offset, peaks, x1, y1


def fit_cos(x, y):
    f0, amp0, phi0, offset0, *_ = fit_fft(x, y)
    # amp0 = (max(y) - min(y)) / 2
    # phi0 = 0
    # offset0 = np.mean(y)
    lb = [0, 0, -2 * np.pi, -np.inf]
    ub = [np.inf, np.inf, 2 * np.pi, np.inf]
    popt0_list = [[f0, amp0, phi0, offset0], [f0, (max(y) - min(y)) / 2, 0, np.mean(y)]]
    popt_list = []
    rmse_list = []
    for popt0 in popt0_list:
        try:
            popt, *_ = curve_fit(cosine, x, y, p0=popt0, bounds=(lb, ub))
        except Exception as e:
            print(e)
            popt = popt0
        rmse = mean_squared_error(y, cosine(x, *popt), squared=False)
        popt_list.append(popt)
        rmse_list.append(rmse)
    rmse = np.min(rmse_list)
    popt = popt_list[np.argmin(rmse_list)]
    return (
        popt,
        rmse,
        partial(cosine, **dict(zip(('freq', 'amp', 'phi', 'offset'), popt))),
    )  # lambda arg: cosine(arg, *popt)


def fit_swap(x, y, arg_type: str = 'wq'):
    g0 = np.min(y) / 2
    if arg_type == 'wq':
        xmin0 = x[np.argmin(y)]
        a0 = 1
        popt, *_ = curve_fit(partial(swap, arg_type=arg_type), x, y, p0=[g0, xmin0, a0])
        g, xmin, *_ = popt
    elif arg_type == 'flux':
        a0 = 10
        b0 = 10
        c0 = x[np.argmin(y)]

        lb = [0, -np.inf, -np.inf, -np.inf]
        ub = np.inf
        popt, *_ = curve_fit(
            partial(swap, arg_type=arg_type), x, y, p0=[g0, b0, a0, c0], bounds=(lb, ub)
        )
        g, b, a, c = popt
        xmin_l = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        xmin_r = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        print(f'popt: {popt}, xmin_l: {xmin_l}, xmin_r: {xmin_r}')
        if abs(xmin_l - np.mean(x)) < abs(xmin_r - np.mean(x)):
            xmin = xmin_l
        else:
            xmin = xmin_r
    else:
        raise ValueError(f'arg_type type {arg_type} is not supported.')

    rmse = mean_squared_error(y, swap(x, *popt, arg_type=arg_type), squared=False)
    return xmin, g, rmse, lambda arg: swap(arg, *popt, arg_type=arg_type)


def fit_geff(x, y, arg_type: str = 'wq'):
    g120 = 10e-3
    coe0 = 1e-3
    wq0 = 4.0
    lb = [0, 0, 0]
    ub = [np.inf, np.inf, np.inf]
    if arg_type in ['wq', 'g']:
        try:
            popt, *_ = curve_fit(
                partial(geff, arg_type=arg_type),
                x,
                y,
                p0=[g120, coe0, wq0],
                bounds=(lb, ub),
            )
        except Exception as e:
            print(e)
    elif arg_type == 'flux':
        a0, b0, c0 = -32.0, 0.0, 8.0
        lb += [-np.inf, -np.inf, 0]
        ub += [0, np.inf, np.inf]
        popt, *_ = curve_fit(
            partial(geff, arg_type=arg_type),
            x,
            y,
            p0=[g120, coe0, wq0, a0, b0, c0],
            bounds=(lb, ub),
        )
    else:
        raise ValueError(f'arg_type type {arg_type} is not supported.')

    rmse = mean_squared_error(y, geff(x, *popt, arg_type=arg_type), squared=False)
    return popt, rmse, lambda arg: geff(arg, *popt, arg_type=arg_type)


def fit_poly(x, y, deg: Union[int, list]):
    popt_list = []
    fun_poly_list = []
    rmse_list = []
    for d in np.atleast_1d(deg):
        poly, stats = Polynomial.fit(x, y, d, full=True)
        popt_list.append(poly.convert().coef)
        p = Polynomial(poly.convert().coef)
        print(f'root: {p.deriv().roots()}')
        fun_poly_list.append(lambda arg: p(np.array(arg)))
        rmse_list.append(stats[0])

    idx = np.argmin(rmse_list)
    popt = popt_list[idx]
    fun_poly = fun_poly_list[idx]
    rmse = rmse_list[idx]
    return popt, rmse, fun_poly
