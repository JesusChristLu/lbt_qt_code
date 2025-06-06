# -*- coding: utf-8 -*-
# @Time     : 2022/9/23 1:29
# @Author   : WTL
# @Software : PyCharm
from typing import Union
import numpy as np
import qutip as qp
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_widths, welch
from qutip.qip.operations.gates import rx, ry, rz
from itertools import product
import cmath
import geatpy as ea


def qubit_spectrum(flux, **q_dic):
    if q_dic is None:
        q_dic = {}

    fq_max = q_dic.get('w_max', 5)
    eta = q_dic.get('eta', -200e-3)
    sws = q_dic.get('sws', 0)
    period = q_dic.get('period', 1.0)
    d = q_dic.get('d', 0.25)

    phi = (flux - sws) / period
    k_phi = np.sqrt(np.cos(np.pi * phi) ** 2 + d**2 * np.sin(np.pi * phi) ** 2)
    return (fq_max + eta) * np.sqrt(k_phi) - eta


def freq2flux(fq, branch='right', **q_dic):
    if q_dic is None:
        q_dic = {}

    fq_max = q_dic.get('w_max', 5)
    eta = q_dic.get('eta', -200e-3)
    sws = q_dic.get('sws', 0)
    period = q_dic.get('period', 1.0)
    d = q_dic.get('d', 0.25)

    k_phi = ((fq + eta) / (fq_max + eta)) ** 2
    phi0 = 1 / np.pi * np.arccos(np.sqrt((k_phi**2 - d**2) / (1 - d**2)))
    if branch == 'right':
        phi = phi0
    elif branch == 'left':
        phi = -phi0
    else:
        raise Exception("branch must be 'right' or 'left'.")

    flux = period * phi + sws
    return flux


def calc_sws(fq_idle, branch='right', **q_dic):
    if q_dic is None:
        q_dic = {}

    fq_max = q_dic.get('w_max', 5)
    eta = q_dic.get('eta', -200e-3)
    period = q_dic.get('period', 1.0)
    d = q_dic.get('d', 0.25)

    k_phi = ((fq_idle + eta) / (fq_max + eta)) ** 2
    phi0 = 1 / np.pi * np.arccos(np.sqrt((k_phi**2 - d**2) / (1 - d**2)))
    if branch == 'right':
        phi = phi0
    elif branch == 'left':
        phi = -phi0
    else:
        raise Exception("branch must be 'right' or 'left'.")
    sws = -period * phi
    return sws


def cosine(x, freq, amp, phi, offset):
    return amp * np.cos(2 * np.pi * freq * x + phi) + offset


def swap(x, *popt, arg_type: str = 'wq'):
    """
    swap频率随电压或者比特频率的变化关系
    swap_freq = np.sqrt(4 * g**2 + D**2), where D = wq1 - wq2.
    :param popt:
    :param x:
    :param arg_type:
    :return:
    """
    if arg_type == 'wq':
        g, xmin, a = popt
        freq = np.sqrt(4 * g**2 + a * (x - xmin) ** 2)
    elif arg_type == 'flux':
        g, b, a, c = popt
        freq = np.sqrt(
            4 * g**2 + (a * x**2 + b * x + c) ** 2
        )  # np.sqrt(4 * g**2 + (a * (x - xmin)**2 + c)**2)
    else:
        raise ValueError(f'arg_type type {arg_type} is not supported!')

    return freq


def geff(x, *popt, arg_type: str = 'wq'):
    """
    有效耦合强度随电压或者比特频率的变化关系
    geff = g12 + 1/2*g1c*g2c * (1/D1c + 1/D2c - 1/S1c - 1/S2c)
    = g12 + 1/2*rho1c*rho2c * np.sqrt(w1*w2)*wc * (1/(w1-wc) + 1/(w2-wc) - 1/(w1+wc) - 1/(w2+wc))
    当两比特共振时，w1=w2=wq
    geff = g12 + rho1c*rho2c * wq * wc * (1/(wq-wc) - 1/(wq+wc))
    :param popt:
    :param x:
    :param arg_type:
    :return:
    """
    if arg_type in ['wq', 'g']:
        g12, coe, wq = popt
        wc = x
    elif arg_type == 'flux':
        g12, coe, wq, a, b, c = popt
        wc = a * x**2 + b * x + c
    else:
        raise ValueError(f'arg_type type {arg_type} is not supported!')

    return g12 + coe * wq * wc * (1 / (wq - wc) - 1 / (wq + wc))


def wc2geff(wc, wl, wr, rho_value):
    """
    根据coupler频率计算有效耦合强度
    :param wc: coupler频率
    :param wl: ql频率
    :param wr: qr频率
    :param rho_value: 归一化耦合强度
    :return:
    """
    rho_lc, rho_rc, rho_lr = rho_value

    glr = rho_lr * np.sqrt(wl * wr)
    glc = rho_lc * np.sqrt(wl * wc)
    grc = rho_rc * np.sqrt(wr * wc)
    geff = glr + 1 / 2 * glc * grc * (
        1 / (wl - wc) + 1 / (wr - wc) - 1 / (wl + wc) - 1 / (wr + wc)
    )
    return geff


def geff2wc(g, wl, wr, rho_value):
    """
    根据目标有效耦合强度计算所需的coupler频率，依据的公式为
    geff = glr + 1 / 2 * glc * grc * (1 / (wl - wc) + 1 / (wr - wc) - 1 / (wl + wc) - 1 / (wr + wc))
    其中
    glr = rho_lr * np.sqrt(wl * wr), glc = rho_lc * np.sqrt(wl * wc), grc = rho_rc * np.sqrt(wr * wc)
    :param g: 目标有效耦合强度
    :param wl: 目标有效耦合强度时的wl
    :param wr: 目标有效耦合强度时的wr
    :param rho_value: 归一化耦合强度
    :return:
    """
    rho_lc, rho_rc, rho_lr = rho_value

    k = rho_lc * rho_rc * np.sqrt(wl * wr) / (g + rho_lr * np.sqrt(wl * wr))
    a, b = wl**2, wr**2
    wc_1 = np.sqrt(
        (
            (1 - k) * (a + b)
            + np.sqrt((1 - k) ** 2 * (a + b) ** 2 - 4 * (1 - 2 * k) * a * b)
        )
        / (2 * (1 - 2 * k))
    )
    wc_2 = np.sqrt(
        (
            (1 - k) * (a + b)
            - np.sqrt((1 - k) ** 2 * (a + b) ** 2 - 4 * (1 - 2 * k) * a * b)
        )
        / (2 * (1 - 2 * k))
    )

    if np.all(wc_1 > wl) and np.all(wc_1 > wr):
        wc = wc_1
    elif np.all(wc_2 > wl) and np.all(wc_2 > wr):
        wc = wc_2
    else:
        raise ValueError(f'Solution {[wc_1, wc_2]} is not physical.')

    return wc


def U_XY(gate_type: str):
    if gate_type == 'X':
        U = rx(np.pi)
    elif gate_type == 'X/2':
        U = rx(np.pi / 2)
    elif gate_type == 'Y':
        U = ry(np.pi)
    elif gate_type == 'Y/2':
        U = ry(np.pi / 2)
    else:
        raise ValueError(f'{gate_type} is not supported gate type.')
    return U


def U_rz(U: qp.Qobj, phase: Union[float, list]):
    """
    在U矩阵上施加单比特相位(如果是多比特U矩阵则可以在每个比特上分别施加单比特相位)
    :param U: U矩阵
    :param phase: 单比特相位
    :return:
    """
    if isinstance(phase, float):
        phase = [
            phase,
        ]
    phase_gates = [rz(p) for p in phase]
    Phase = qp.tensor(phase_gates)
    return Phase * U


def U_rphi(U, phi: Union[float, list]):
    """
    对U矩阵施加转轴旋转(如果是多比特U矩阵则可以在每个比特上分别施加旋转)
    :param U: U矩阵
    :param phi: 旋转角度
    :return:
    """
    if isinstance(phi, float):
        phi = [
            phi,
        ]
    Z_phi_gates = [rz(p) for p in phi]
    Z_phi = qp.tensor(Z_phi_gates)
    Z_pphi_gates = [rz(-p) for p in phi]
    Z_pphi = qp.tensor(Z_pphi_gates)
    return Z_phi * U * Z_pphi


def fU(Ureal: qp.Qobj, Uideal: qp.Qobj):
    """
    计算实际U矩阵相对于理想U矩阵的保真度。
    ref: Pedersen, L. H., et al. (2007). "Fidelity of quantum operations." Physics Letters A 367(1-2): 47-51.
    :param Ureal: 实际U矩阵
    :param Uideal: 理想U矩阵
    :return:
    """
    d = Ureal.shape[0]
    return ((Ureal.dag() * Ureal).tr() + abs((Uideal.dag() * Ureal).tr()) ** 2) / (
        d * (d + 1)
    )


def errorU(Ureal: qp.Qobj, Uideal: qp.Qobj):
    """
    计算实际U矩阵相对于理想U矩阵的误差。
    :param Ureal: 实际U矩阵
    :param Uideal: 理想U矩阵
    :return:
    """
    error = 1 - fU(Ureal, Uideal)
    return error


def cali_phi(Ureal: qp.Qobj, Uideal: qp.Qobj):
    dim = int(np.log2(Ureal.shape[0]))

    def error(phi):
        # Uphi = U_rphi(Ureal, phi)
        Uphi = U_rz(Ureal, phi)
        return errorU(Uphi, Uideal)

    def error_multi(phi_list):
        error_list = []
        for phi in phi_list:
            Uphi = U_rz(Ureal, phi)
            error_list.append(errorU(Uphi, Uideal))
        return np.vstack(error_list)
    # NM
    # res = minimize(error, x0=np.zeros(int(np.log2(d))), method='Nelder-Mead')
    # res = minimize(
    #     error, x0=np.random.randint(0, 2 * np.pi, dim), method='Nelder-Mead'
    # )
    # DE
    problem = ea.Problem(
        name='soea err model',
        M=1,  # 初始化M（目标维数）
        maxormins=[1],  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim=dim,  # 决策变量维数
        varTypes=[0] * dim,  # 决策变量的类型列表，0：实数；1：整数
        lb=[0] * dim,  # 决策变量下界
        ub=[2 * np.pi] * dim,  # 决策变量上界
        evalVars=error_multi
    )

    algorithm = ea.soea_DE_best_1_bin_templet(
        problem,
        ea.Population(Encoding='RI', NIND=15),
        MAXGEN=15,
        logTras=5,
        # trappedValue=1e-10,
        # maxTrappedCount=20
    )
    algorithm.mutOper.F = 0.7
    algorithm.recOper.XOVR = 0.7

    res = ea.optimize(
        algorithm,
        # prophet=np.array(self.experiment_options.FIR0),
        verbose=False, drawing=0, outputMsg=False,
        drawLog=False, saveFlag=False
    )
    phi_opti = res['Vars'][0]
    # phi_opti = res.x
    # print(res['ObjV'])
    # U_phi_opti = U_rphi(Ureal, phi_opti)
    U_phi_opti = U_rz(Ureal, phi_opti)
    return phi_opti, U_phi_opti


def fU_cali_phi(Ureal: qp.Qobj, Uideal: qp.Qobj):
    phi_cali, Ucali = cali_phi(Ureal, Uideal)
    F = fU(Ucali, Uideal)
    return F


def errorU_cali_phi(Ureal: qp.Qobj, Uideal: qp.Qobj):
    phi_cali, Ucali = cali_phi(Ureal, Uideal)
    error = errorU(Ucali, Uideal)
    return error, phi_cali


def SU2_param(U2x2: Union[np.ndarray, qp.Qobj]):
    """
    给定一个2x2的U矩阵，从矩阵元中提取theta, phi, lam参数
    U(theta, phi, lam)
    = Z(phi) * X(theta) * Z(lam)
    = [exp(-1j*(phi+lam)/2)*cos(theta/2),      -j*exp(1j*(lam-phi)/2)*sin(theta/2)]
      [-j*exp(1j*(phi-lam)/2)*sin(theta/2),    exp(1j*(phi+lam)/2)*cos(theta/2)]
    = [exp(-1j*(phi+lam)/2)*cos(theta/2),      exp(1j*(lam-phi-pi)/2)*sin(theta/2)]
      [exp(1j*(phi-lam-pi)/2)*sin(theta/2),    exp(1j*(phi+lam)/2)*cos(theta/2)]
    :param U2x2:
    :return:
    """
    if isinstance(U2x2, qp.Qobj):
        U2x2 = U2x2.full()

    amp00, angle00 = cmath.polar(U2x2[0, 0])
    amp01, angle01 = cmath.polar(U2x2[0, 1])
    amp10, angle10 = cmath.polar(U2x2[1, 0])
    amp11, angle11 = cmath.polar(U2x2[1, 1])

    theta00 = 2 * np.arccos(amp00)
    theta11 = 2 * np.arccos(amp11)
    theta01 = 2 * np.arcsin(amp01)
    theta10 = 2 * np.arcsin(amp10)

    phase01, phase10, phase11 = [
        angle - angle00 for angle in (angle01, angle10, angle11)
    ]
    lam = np.pi / 2 + phase01
    phi = np.pi / 2 + phase10
    phiplam = phase11

    # print(f'angle(00,11,01,10) = {np.array([angle00, angle11, angle01, angle10]) / np.pi}pi\n'
    #       f'theta(00,11,01,10) = {np.array([theta00, theta01, theta10, theta11]) / np.pi}pi\n'
    #       f'lam = {lam / np.pi}pi\n'
    #       f'phi = {phi / np.pi}pi\n'
    #       f'phi+lam = {phiplam / np.pi}pi\n')

    return np.mean([theta00, theta01, theta10, theta11]), lam, phi, phiplam


def qpt_rho_in(U_ideal: qp.Qobj):
    rho_basis_1 = [0, 1, '+', '+i']
    rho_basis = [rho_basis_1] * len(U_ideal.dims)
    rho_in_list = list(product(*rho_basis))
    return rho_in_list


def qpt(rho_in_list: list, rho_out_list: list, U_ideal: qp.Qobj):
    op_basis = [[qp.qeye(2), qp.sigmax(), qp.sigmay(), qp.sigmaz()]] * len(U_ideal.dims)
    E_op_list = [qp.tensor(*E_op) for E_op in product(*op_basis)]
    EE_op_list = [qp.sprepost(E1, E2.dag()) for E1 in E_op_list for E2 in E_op_list]
    Beta = np.hstack([qp.mat2vec(EE.full()) for EE in EE_op_list])

    rho_in_mat = np.hstack([qp.mat2vec(rho_in) for rho_in in rho_in_list])
    rho_out_mat = np.hstack([qp.mat2vec(rho_out) for rho_out in rho_out_list])
    Gamma = np.linalg.solve(rho_in_mat.T, rho_out_mat.T)
    Gamma = Gamma.T

    vecGamma = qp.mat2vec(Gamma)
    vecChi = np.linalg.solve(Beta, vecGamma)
    Chi = qp.vec2mat(vecChi)

    vecGamma_ideal = qp.mat2vec(U_ideal.full())
    vecChi_ideal = np.linalg.solve(Beta, vecGamma_ideal)
    Chi_ideal = qp.vec2mat(vecChi_ideal)

    error = 1 - abs(np.trace(np.dot(Chi_ideal, Chi)))
    return error


def pulse_psd(pulse_list, sample_rate):
    #######################################################
    # FFT using Welch method
    # windows = np.ones(nfft) - no windowing
    # if windows = 'hamming', etc.. this function will
    # normalize to an equivalent noise bandwidth (ENBW)
    #######################################################
    nfft = len(pulse_list)
    f, psd = welch(
        pulse_list,
        fs=sample_rate,
        window=np.ones_like(pulse_list),
        nperseg=nfft,
        scaling='density',
    )
    df = f[1] - f[0]

    peaks, properties = find_peaks(psd, height=np.max(psd) / 5, distance=5)
    peaks_fwhm = peak_widths(psd, peaks)
    #     peaks = [np.argmax(y1)]
    basic_freq = f[peaks]
    basic_amp = psd[peaks]
    basic_phase = np.angle(psd)[peaks]
    basic_offset = abs(psd[0]) / len(f)
    return (
        basic_freq,
        basic_amp,
        basic_phase,
        basic_offset,
        peaks,
        peaks_fwhm,
        df,
        f,
        psd,
    )
