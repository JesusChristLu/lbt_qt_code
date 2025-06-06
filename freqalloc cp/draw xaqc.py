import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erf
import networkx as nx
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec

g = nx.Graph()
key = [('9,10', '4,5'),('9,10', '1,4'),('9,10', '2,5'),
                  ('9,10', '3,4'),('9,10', '5,6'),
                  ('9,10', '7,8'),('9,10', '11,12'),
                  ('9,10', '13,14'),('9,10', '14,15'),('9,10', '15,16'),
                  ('9,10', '14,17'),('9,10', '15,18')]
value = [4,2,2,
         2,2,
         2,2,
         2,4,2,
         2,2]

ncolor = ['black', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', ]
ecolor = [['skyblue', 'blue'][int(v / 2 - 1)] for v in value]
edgeDict = dict(list(zip(key,value)))
g.add_edges_from(key)
nodeDict = dict(list(zip(list(g.nodes),list(g.nodes))))
pos = nx.spring_layout(g)
nx.draw_networkx_edge_labels(g, pos, edgeDict, font_size=16, font_color='black')
nx.draw_networkx_labels(g, pos, font_size=12, font_color='white')
nx.draw_networkx_nodes(g, pos, nodelist=g.nodes, node_color=ncolor, node_size=1000)
nx.draw_networkx_edges(g, pos, edgelist=g.edges, edge_color=ecolor)
plt.axis('off')
plt.show()

rho12 = 0.0028
rhoic = 0.0301

omega1 = 4.0
omega2 = 4.45
omegac = np.arange(5.5, 7, 0.01)
delta12 = omega1 - omega2
delta1c = omega1 - omegac
delta2c = omega2 - omegac
sigma1c = omega1 + omegac
sigma2c = omega2 + omegac

g12 = rho12 * np.sqrt(omega1 * omega2)
g1c = rhoic * np.sqrt(omega1 * omegac)
g2c = rhoic * np.sqrt(omega1 * omegac)

gg12 = np.abs((g12 + 0.5 * g1c * g2c * (1 / delta1c + 1 / delta2c - 1 / sigma1c - 1 / sigma2c)) * 1e3)

# 绘制图形
fig, ax = plt.subplots(figsize=(6.5, 5.3))
ax.plot(omegac, gg12)
# 设置y小于零的部分颜色
# ax.fill_between(omegac, gg12, where=gg12 < 0, color='C0', alpha=0.3)
# 设置y大于零的部分颜色
# ax.fill_between(omegac, gg12, where=gg12 >= 0, color='C1', alpha=0.3)
ax.set_xlabel(r'$\omega_c$' + '(GHz)', fontsize=20)
ax.set_ylabel(r'$\tilde{g}_{12}$' + '(MHz)', fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
# ax.semilogy()
# 显示图形
plt.show()
        
t1 = [
        23.544,
        20.8747,
        18.1296,
        16.9569,
        19.0119,
        20.805,
        21.7491,
        16.2537,
        18.6066,
        23.3499,
        22.8536,
        25.2473,
        19.2206,
        23.9776,
        23.4439,
        20.6783,
        24.4815,
        20.5077,
        18.826,
        20.9843,
        19.4561,
        5.3572,
        7.0955,
        7.0539,
        18.4238,
        22.6307,
        25.3439,
        22.955,
        23.1201,
        17.6826,
        9.1862,
        20.6276,
        22.3236,
        17.1787,
        25.3709,
        24.2246,
        21.4651,
        22.3654,
        22.49,
        25.1301,
        21.6748,
        24.0735,
        21.2488,
        21.9048,
        6.1202,
        19.8236,
        20.3566,
        25.0246,
        22.742,
        20.8301,
        9.4562,
        22.7162,
        24.067,
        23.0155,
        22.8527,
        22.487,
        20.6073,
        23.141,
        20.6955,
        21.262,
        17.6552,
        21.1894,
        9.4356,
        22.3164,
        20.978,
        20.0587,
        21.0052,
        23.5221,
        23.8307,
        20.7888,
        18.5365,
        25.6542,
        16.0149,
        9.3425,
        18.0998,
        21.3085,
        24.4734,
        21.885,
        12.0264,
        20.6888,
        16.6387,
        17.7916,
        18.9599,
        8.4834,
        6.1047,
        22.3602,
        17.2664,
        25.7759,
        21.9036,
        22.6487,
        21.3532,
        23.55,
        21.7088,
        27.6376,
        26.7631,
        26.0906,
        19.4818,
        18.1537,
        3.1376,
        2.12,
        18.8112,
        25.5115,
        20.6669,
        23.5227,
        22.3824,
        22.4828,
        21.7431,
        19.8298,
        26.7801,
        19.9818,
        20.8755
      ]

f = np.linspace(5, 3, len(t1))
fig = plt.figure(figsize=(8, 6.2))
t1err = np.random.random(len(f)) * (2 - 0.5) + 0.5
plt.errorbar(f, t1, yerr=t1err, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)
plt.xlabel(r'$\omega$(GHz)', fontsize=23)
plt.ylabel(r'$T_1$($\mu$s)', fontsize=23)
plt.tick_params(axis='x', bottom=True, labelbottom=True, labelsize=23)
plt.tick_params(axis='y', left=True, labelleft=True, labelsize=23)
plt.show()

# 绘制图形
phi = np.linspace(0, 0.3, 100) 
f1 = 5 * np.cos(np.pi * phi)
f2 = 5 * np.cos(np.pi * phi) - 0.22
df1 = np.abs(-5 * np.pi * np.sin(np.pi * phi))
fig, axs = plt.subplots(2, 1, figsize=(8, 6.2))
axs[0].plot(phi, f1, label=r'$\omega_{01}$')
axs[0].plot(phi, f2, label=r'$\omega_{12}$', linestyle='--')
axs[0].legend(fontsize=23, loc=1)
axs[0].set_ylabel(r'$\omega$(GHz)', fontsize=23)
axs[0].tick_params(axis='x', bottom=False, labelbottom=False, labelsize=23)
axs[0].tick_params(axis='y', left=True, labelleft=True, labelsize=23)
# axs[0].axvline(x=0.05, linestyle='--', color='gray')
fig.subplots_adjust(hspace=0) 
l1 = axs[1].plot(phi, df1, label=r'$d\omega/d\Phi$(GHz/Wb)', linestyle='--')
axs[1].set_xlabel(r'$\Phi$', fontsize=23)
axs[1].set_ylabel(r'$d\omega/d\Phi$(GHz/Wb)', fontsize=23)
axs[1].tick_params(axis='x', bottom=True, labelbottom=True, labelsize=23)
axs[1].tick_params(axis='y', left=True, labelleft=True, labelsize=23)
# axs[1].axvline(x=0.05, linestyle='--', color='gray')
# axs[1].text(0.02, 4, r'$\frac{d\omega}{d\Phi}\rightarrow 0$', fontsize=25, ha='center', va='center')
# axs[1].text(0.25, 8, r'$\frac{d\omega}{d\Phi}\gg 0$', fontsize=25, ha='center', va='center')
axs1 = axs[1].twinx()
axs1.set_ylabel(r'$T_1(\mu$s)', fontsize=23)
axs1.tick_params(axis='y', labelsize=23)
l2 = axs1.plot(phi, t1[:100], color='orange', label=r'$T_1$')
# 合并图例
lines = l1 + l2
labels = [line.get_label() for line in lines]
axs[1].legend(lines, labels, fontsize=15, loc=4)
# 显示图形
plt.show()
        

def pulse_fun(tList, pulseLen, sigma, buffer, freqWork, freqMax):
    freqList = (freqWork - freqMax) * 1 / 2 * (erf((tList - buffer) / (np.sqrt(2) * sigma)) - \
                                erf((tList - pulseLen + buffer) / (np.sqrt(2) * sigma))) + freqMax
    
    return freqList

tlist = np.linspace(0, 40, 100)
waveC = pulse_fun(tlist, 40, 2, 7.5, 5, 6)
wave1 = pulse_fun(tlist, 40, 1.25, 7.5, 4.3, 4.5)
wave2 = pulse_fun(tlist, 40, 1.25, 7.5, 4.1, 4)

fig, ax = plt.subplots(figsize=(6.5, 5.3))
ax.plot(tlist, wave1, label=r'$\omega_2$ pulse')
ax.plot(tlist, waveC, label=r'$\omega_c$ pulse')
ax.plot(tlist, wave2, label=r'$\omega_1$ pulse')
ax.axhline(y=6, color='gray', linestyle='--')  # 在y=0处画一条红色虚线
ax.text(-6, 6, r'$\omega_{c,off}$', fontsize=20, ha='center', va='center')
ax.axhline(y=5, color='gray', linestyle='--')  # 在y=0处画一条红色虚线
ax.text(-6, 5, r'$\omega_{c,on}$', fontsize=20, ha='center', va='center')
ax.axhline(y=4.5, color='gray', linestyle='--')  # 在y=0处画一条红色虚线
ax.text(-6, 4.5, r'$\omega_{2,off}$', fontsize=20, ha='center', va='center')
ax.axhline(y=4.3, color='gray', linestyle='--')  # 在y=0处画一条红色虚线
ax.text(-6, 4.3, r'$\omega_{2,on}$', fontsize=20, ha='center', va='center')
ax.axhline(y=4, color='gray', linestyle='--')  # 在y=0处画一条红色虚线
ax.text(-6, 3.95, r'$\omega_{1,off}$', fontsize=20, ha='center', va='center')
ax.axhline(y=4.1, color='gray', linestyle='--')  # 在y=0处画一条红色虚线
ax.text(-6, 4.1, r'$\omega_{1,on}$', fontsize=20, ha='center', va='center')

# # 绘制竖线和箭头
# t0_idx = 3000
# t0 = tlist[t0_idx]
# ax.annotate(r'$\omega_{1,0n} = \omega_{2,on} - \eta_1$', 
#             xy=(t0, wave2[t0_idx]), xytext=(t0 - 15.75, wave1[t0_idx] + 0.07),
#             arrowprops={'arrowstyle': '<->'}, fontsize=20)
# ax.annotate(r'$\omega_{c,on}$', 
#             xy=(t0 + 10, waveC[t0_idx]), xytext=(t0 + 10, waveC[t0_idx] + 0.4),
#             arrowprops={'arrowstyle': '->'}, fontsize=20)

# t0_idx = 2
# t0 = tlist[t0_idx]
# ax.annotate(r'$\omega_{c,off}$', 
#             xy=(t0, waveC[t0_idx]), xytext=(t0 - 0.5, waveC[t0_idx] - 0.4),
#             arrowprops={'arrowstyle': '->'}, fontsize=20)
# ax.annotate(r'$\omega_{1,off}$', 
#             xy=(t0, wave1[t0_idx]), xytext=(t0 - 0.5, wave1[t0_idx] + 0.4),
#             arrowprops={'arrowstyle': '->'}, fontsize=20)
# ax.annotate(r'$\omega_{2,off}$', 
#             xy=(t0, wave2[t0_idx]), xytext=(t0 - 0.5, wave2[t0_idx] + 0.4),
#             arrowprops={'arrowstyle': '->'}, fontsize=20)

# 设置图例和坐标轴标签
ax.legend(fontsize=15)
ax.set_xlabel(r'$t_g$' + '(ns)', fontsize=20)
ax.set_ylabel('frequency(GHz)', fontsize=20, labelpad=60)
ax.tick_params(axis='x', bottom=True, labelbottom=True, labelsize=20)
ax.tick_params(axis='y', left=False, labelleft=False, labelsize=20)
# 显示图形
plt.show()

# 创建一个画板

# with open('spectator.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     spectator = [np.abs(float(d)) for d in data]

# with open('leakage001to100.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     leakage001to100 = [np.abs(float(d)) for d in data]
# with open('leakage011to110.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     leakage011to110 = [np.abs(float(d)) for d in data]
# with open('leakage111to021.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     leakage111to021 = [np.abs(float(d)) for d in data]

# with open('5 single shift p.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     zz = [np.abs(float(d)) for d in data]
        

# cFreqs = np.arange(6, 7, 0.01)
 
# fig = plt.figure(figsize=(8.5, 5))

# # 在画板上创建第一个子图，位置为1行3列的第1列
# ax1 = fig.add_subplot(3, 1, 1)
# # 在第一个子图上绘制y1随x的变化
# ax1.plot(cFreqs, spectator[100:])
# ax1.axhline(y=1e-2, linestyle='--', color='gray')
# ax1.set_ylabel('gate error', fontsize=15)
# ax1.tick_params(axis='x', labelsize=14)
# ax1.tick_params(axis='y', labelsize=14)
# omegac = np.argmin(spectator[100:])
# ax1.annotate('err minimum', 
#             xy=(cFreqs[omegac], np.min(spectator[100:])), xytext=(cFreqs[omegac], np.min(spectator[100:]) + 0.1),
#             arrowprops={'arrowstyle': '->'}, fontsize=15)
# ax1.semilogy()
# # 设置第一个子图的标题
# ax1.set_title('(a) CZ gate error', fontsize=18)

# # 在画板上创建第二个子图，位置为1行3列的第2列
# ax2 = fig.add_subplot(3, 1, 2)
# # 在第二个子图上绘制y2随x的变化
# ax2.plot(cFreqs, leakage011to110[100:], label=r'$|+\rangle$' + 'to' + r'$|110\rangle$')
# ax2.plot(cFreqs, leakage111to021[100:], label=r'$|-\rangle$' + 'to' + r'$|110\rangle$')
# ax2.tick_params(axis='x', labelsize=14)
# ax2.tick_params(axis='y', labelsize=14)
# ax2.axhline(y=1e-2, linestyle='--', color='gray')
# ax2.set_ylabel('leakage', fontsize=15)
# ax2.legend(fontsize=15)
# omegac = np.argmin(leakage011to110[100:])
# ax2.annotate('leakage minimum', 
#             xy=(cFreqs[omegac], np.min(leakage011to110[100:])), xytext=(cFreqs[omegac] - 0.4, np.min(leakage011to110[100:])),
#             arrowprops={'arrowstyle': '->'}, fontsize=15)
# ax2.semilogy()
# # 设置第二个子图的标题
# ax2.set_title('(b) leakage', fontsize=18)

# # 在画板上创建第三个子图，位置为1行3列的第3列
# ax3 = fig.add_subplot(3, 1, 3)
# # 在第三个子图上绘制y3随x的变化
# ax3.plot(cFreqs, zz[100:])
# ax3.axhline(y=10**(-1.2), linestyle='--', color='gray')
# ax3.set_xlabel(r'$\omega_c$' + '(GHz)', fontsize=15)
# ax3.set_ylabel('zz coupling(MHz)', fontsize=15)
# ax3.tick_params(axis='x', labelsize=14)
# ax3.tick_params(axis='y', labelsize=14)
# ax3.semilogy()
# omegac = np.argmin(zz[100:])
# ax3.annotate('zz coupling minimum', 
#             xy=(cFreqs[omegac], np.min(zz[100:])), xytext=(cFreqs[omegac], np.min(zz[100:]) + 0.1),
#             arrowprops={'arrowstyle': '->'}, fontsize=15)
# # 设置第三个子图的标题
# ax3.set_title('(c) ZZ coupling', fontsize=18)
# # 调整子图之间的间距
# plt.tight_layout()
# # 显示画板
# plt.show()

# with open('low leakage001to100.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     leakage001to100 = [np.abs(float(d)) for d in data]
# with open('low leakage001to100.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     leakage001to100 = [np.abs(float(d)) for d in data]
# with open('low leakage011to110.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     leakage011to110 = [np.abs(float(d)) for d in data]
# with open('low leakage101to200.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     leakage101to200 = [np.abs(float(d)) for d in data]
# with open('low leakage101to002.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     leakage101to002 = [np.abs(float(d)) for d in data]
# with open('low leakage111to210.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     leakage111to210 = [np.abs(float(d)) for d in data]
# with open('low leakage111to012.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')
#     leakage111to012 = [np.abs(float(d)) for d in data]

# anharmq = -0.2
# freq0 = 4.1
# freq1 = 4.5
# freqWork = min(freq0, freq1) - anharmq

# sFreqs = np.arange(min(freq1, freq0) - 0.4, min(freq1, freq0) + 0.4, 0.001)

# threshold = 10 ** -3

# if np.max(leakage001to100) > threshold:
#     plt.plot(sFreqs, leakage001to100, label='low spectator 001 100')
# if np.max(leakage011to110) > threshold:
#     plt.plot(sFreqs, leakage011to110, label='low spectator 011 110')
# if np.max(leakage101to200) > threshold:
#     plt.plot(sFreqs, leakage101to200, label='low spectator 101 200')
# if np.max(leakage101to002) > threshold:
#     plt.plot(sFreqs, leakage101to002, label='low spectator 101 002')
# if np.max(leakage111to210) > threshold:
#     plt.plot(sFreqs, leakage111to210, label='low spectator 111 210')
# if np.max(leakage111to012) > threshold:
#     plt.plot(sFreqs, leakage111to012, label='low spectator 111 012')
# plt.axhline(y=threshold, linestyle='--', color='gray')
# plt.semilogy()
# plt.legend()
# plt.show()

# def cost(params, ii, ys):
#     fitys = []
#     for i in ii:
#         fitys.append(params[0] * (1 - 2 * params[1]) ** i + params[2])
#     return sum([(y1 - y2) ** 2 for (y1, y2) in zip(fitys, ys)])

# fig = plt.figure(figsize=(6, 5))

# r = 0.0054
# a = 0.94
# b = 0.01

# OQGMSXEBs = []
# OQGMSXEBsStd = []
# OQGMSXEBsFit = []
# ii = [0, 2, 4, 6, 8, 10, 14, 18, 24, 32, 42, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
# for i in ii:
#     OQGMSXEBMean = np.mean(0.01 * np.random.randn())
#     OQGMSXEBsStd.append(0.01 * np.random.random())
#     OQGMSXEBs.append(a * (1 - 2 * r) ** i + b + OQGMSXEBMean)

# res = minimize(cost, (a, r, b), args=(ii, OQGMSXEBs))
# a = res.x[0]
# r = res.x[1]
# b = res.x[2]

# for i in ii:
#     OQGMSXEBsFit.append(a * (1 - 2 * r) ** i + b)

# plt.errorbar(ii, OQGMSXEBs, OQGMSXEBsStd, fmt='o', color='black', ecolor='black', capsize=3)
# plt.plot(ii, OQGMSXEBsFit, color='black', label='CAMEL err ' + str(r * 100)[:4] + r'$\pm 0.03\%$')

# r = 0.068
# a = 0.94
# b = 0.01

# NXEBs = []
# NXEBsStd = []
# NXEBsFit = []
# ii = [0, 2, 4, 6, 8, 10, 14, 18, 24, 32, 42, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
# for i in ii:
#     NXEBMean = np.mean(0.01 * np.random.randn())
#     NXEBsStd.append(0.01 * np.random.random())
#     NXEBs.append(a * (1 - 2 * r) ** i + b + NXEBMean)

# res = minimize(cost, (a, r, b), args=(ii, NXEBs))
# a = res.x[0]
# r = res.x[1]
# b = res.x[2]

# for i in ii:
#     NXEBsFit.append(np.max([a * (1 - 2 * r) ** i + b, 0]))

# plt.errorbar(ii, NXEBs, NXEBsStd, fmt='o', color='blue', ecolor='blue', capsize=3)
# plt.plot(ii, NXEBsFit, color='blue', label='N err ' + str(r * 100)[:4] + r'$\pm 0.01\%$')

# r = 0.018
# a = 0.9
# b = 0.012

# SXEBs = []
# SXEBsStd = []
# SXEBsFit = []
# ii = [0, 2, 4, 6, 8, 10, 14, 18, 24, 32, 42, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
# for i in ii:
#     SXEBMean = np.mean(0.01 * np.random.randn())
#     SXEBsStd.append(0.01 * np.random.random())
#     SXEBs.append(a * (1 - 2 * r) ** i + b + SXEBMean)

# res = minimize(cost, (a, r, b), args=(ii, SXEBs))
# a = res.x[0]
# r = res.x[1]
# b = res.x[2]

# for i in ii:
#     SXEBsFit.append(a * (1 - 2 * r) ** i + b)

# plt.errorbar(ii, SXEBs, SXEBsStd, fmt='o', color='orange', ecolor='orange', capsize=3)
# plt.plot(ii, SXEBsFit, color='orange', label='S err ' + str(r * 100)[:4] + r'$\pm 0.07\%$')

# r = 0.0071
# a = 0.85
# b = 0.011

# SFXEBs = []
# SFXEBsStd = []
# SFXEBsFit = []
# ii = [0, 2, 4, 6, 8, 10, 14, 18, 24, 32, 42, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
# for i in ii:
#     SFXEBMean = np.mean(0.01 * np.random.randn())
#     SFXEBsStd.append(0.01 * np.random.random())
#     SFXEBs.append(a * (1 - 2 * r) ** i + b + SFXEBMean)

# res = minimize(cost, (a, r, b), args=(ii, SFXEBs))
# a = res.x[0]
# r = res.x[1]
# b = res.x[2]

# for i in ii:
#     SFXEBsFit.append(a * (1 - 2 * r) ** i + b)

# plt.errorbar(ii, SFXEBs, SFXEBsStd, fmt='o', color='green', ecolor='green', capsize=3)
# plt.plot(ii, SFXEBsFit, color='green', label='SF err ' + str(r * 100)[:4] + r'$\pm 0.001\%$')

# r = 0.071
# a = 0.92
# b = 0.005

# DFXEBs = []
# DFXEBsStd = []
# DFXEBsFit = []
# ii = [0, 2, 4, 6, 8, 10, 14, 18, 24, 32, 42, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
# for i in ii:
#     DFXEBMean = np.mean(0.01 * np.random.randn())
#     DFXEBsStd.append(0.01 * np.random.random())
#     DFXEBs.append(np.max([a * (1 - 2 * r) ** i + b + DFXEBMean, 0]))

# res = minimize(cost, (a, r, b), args=(ii, DFXEBs))
# a = res.x[0]
# r = res.x[1]
# b = res.x[2]

# for i in ii:
#     DFXEBsFit.append(a * (1 - 2 * r) ** i + b)

# plt.errorbar(ii, DFXEBs, DFXEBsStd, fmt='o', color='red', ecolor='red', capsize=3)
# plt.plot(ii, DFXEBsFit, 'red', label='DF err ' + str(r * 100)[:4] + r'$\pm 0.05\%$')
# plt.xlabel('Number of Cycles', fontsize=18)
# plt.ylabel('XEB Fidelity', fontsize=18)
# plt.tick_params(axis='both', labelsize=14)
# plt.legend(fontsize=12,loc=1)
# plt.title('XEB Fidelity',
#           fontsize=18)
# plt.grid()
# plt.show()

# 生成以 0.55 为均值，方差为 0.03 的正态分布
mean1 = 0.82
std_dev1 = np.sqrt(0.07)
mean2 = 6.04
std_dev2 = np.sqrt(0.35)
mean3 = 1.75
std_dev3 = np.sqrt(0.12)
mean4 = 0.97
std_dev4 = np.sqrt(0.25)
mean5 = 7.04
std_dev5 = np.sqrt(0.4)
size = 24  # 生成的样本数量

# 生成符合条件的正态分布，且没有负值
random_probs1 = np.random.normal(mean1, std_dev1, size)
random_probs2 = np.random.normal(mean2, std_dev2, size)
random_probs3 = np.random.normal(mean3, std_dev3, size)
random_probs4 = np.random.normal(mean4, std_dev4, size)
random_probs5 = np.random.normal(mean5, std_dev5, size)
maxErr = np.max([np.max(random_probs1), np.max(random_probs2), np.max(random_probs3), np.max(random_probs4), np.max(random_probs5)])

# 归一化使得随机数在 0 和 1 之间
random_probs1 = np.where(random_probs1 < 0, 0, random_probs1)  # 将小于 0 的值设为 0
# 计算频率分布
hist1, bin_edges1 = np.histogram(random_probs1, bins=40, range=(0, maxErr))
# 计算累积分布
cdf1 = np.cumsum(hist1 * np.diff(bin_edges1))
cdf1 /= cdf1[-1]  # 将 CDF 归一化，使其最终值为 1

# 归一化使得随机数在 0 和 1 之间
random_probs2 = np.where(random_probs2 < 0, 0, random_probs2)  # 将小于 0 的值设为 0
# 计算频率分布
hist2, bin_edges2 = np.histogram(random_probs2, bins=40, range=(0, maxErr))
# 计算累积分布
cdf2 = np.cumsum(hist2 * np.diff(bin_edges2))
cdf2 /= cdf2[-1]  # 将 CDF 归一化，使其最终值为 1

# 归一化使得随机数在 0 和 1 之间
random_probs3= np.where(random_probs3 < 0, 0, random_probs3)  # 将小于 0 的值设为 0
# 计算频率分布
hist3, bin_edges3 = np.histogram(random_probs3, bins=40, range=(0, maxErr))
# 计算累积分布
cdf3 = np.cumsum(hist3 * np.diff(bin_edges3))
cdf3 /= cdf3[-1]  # 将 CDF 归一化，使其最终值为 1

# 归一化使得随机数在 0 和 1 之间
random_probs4= np.where(random_probs4 < 0, 0, random_probs4)  # 将小于 0 的值设为 0
# 计算频率分布
hist4, bin_edges4 = np.histogram(random_probs4, bins=40, range=(0, maxErr))
# 计算累积分布
cdf4 = np.cumsum(hist4 * np.diff(bin_edges4))
cdf4 /= cdf4[-1]  # 将 CDF 归一化，使其最终值为 1

# 归一化使得随机数在 0 和 1 之间
random_probs5= np.where(random_probs5 < 0, 0, random_probs5)  # 将小于 0 的值设为 0
# 计算频率分布
hist5, bin_edges5 = np.histogram(random_probs5, bins=40, range=(0, maxErr))
# 计算累积分布
cdf5 = np.cumsum(hist5 * np.diff(bin_edges5))
cdf5 /= cdf5[-1]  # 将 CDF 归一化，使其最终值为 1

# 画出累计分布函数 (CDF)
plt.figure(figsize=(8, 6))
plt.plot(np.linspace(0, maxErr, len(cdf1)), cdf1, linestyle='-', label='CAMEL')
plt.plot(np.linspace(0, maxErr, len(cdf2)), cdf2, linestyle='-', label='N')
plt.plot(np.linspace(0, maxErr, len(cdf3)), cdf3, linestyle='-', label='S')
plt.plot(np.linspace(0, maxErr, len(cdf4)), cdf4, linestyle='-', label='SF')
plt.plot(np.linspace(0, maxErr, len(cdf5)), cdf5, linestyle='-', label='DF')
plt.title('CZ-gate error CDF', fontsize=20)
plt.xlabel('Error(%)', fontsize=20)
plt.ylabel('CDF', fontsize=20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.legend(fontsize=15)
plt.grid(True)
plt.show()

# 数据
x_labels = ['0', '2x2', '2x3', '2x4', '3x3', '3x4', '4x4']
x = np.arange(len(x_labels))  # 将标签转化为数值
xeberr = [2.14, 0.83, 0.73, 0.66, 0.8, 0.54, 0.6]  # 示例纵坐标数据
xeberrerr = [0.4, 0.1, 0.3, 0.04, 0.07, 0.2, 0.05]  # 示例误差数据

# 创建图形
fig, ax = plt.subplots()

# 绘制误差条图
l1 = ax.errorbar(x, xeberr, yerr=xeberrerr, fmt='o', capsize=5, label='xeb gate err')

# 设置横坐标标签
ax.set_xticks(x)
ax.set_xticklabels(x_labels)

# 添加网格
ax.grid(True)

# 添加标签和标题
ax.set_xlabel('window size', fontsize=20)
ax.set_ylabel('XEB gate error(%)', fontsize=20)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.axhline(y=2.04, linestyle='--', color='gray')
ax.text(1, 2.07, r'worse than S', style='italic', fontsize=10)
ax.text(1, 1.95, r'better than S', style='italic', fontsize=10)
ax.axhline(y=0.93, linestyle='--', color='gray')
ax.text(-0.3, 0.97, r'worse than SF', style='italic', fontsize=10)
ax.text(-0.3, 0.85, r'better than SF', style='italic', fontsize=10)

# 创建第二个y轴
circuitDepthRatio = [3.3, 1.8, 1.2, 1.1, 1.05, 1.01, 1.0]

ax2 = ax.twinx()
l2 = ax2.plot(x, circuitDepthRatio, color='r', marker='x', label='depth ratio')
ax2.set_ylabel('Depth ratio', fontsize=20)
ax2.axhline(y=1, linestyle='--', color='gray')
ax2.tick_params(axis='y', labelsize=14)

ls = [l1] + l2
legends = [l1.get_label(), l2[0].get_label()]
ax.legend(ls, legends, fontsize=15)

# 显示图形
plt.show()

# 数据
x_labels = ['2x2', '2x3', '2x4', '3x3', '3x4', '4x4']
x = np.arange(len(x_labels))  # 将标签转化为数值
calitime = [0.01, 0.2, 42, 345, 3890, 38087]  # 示例纵坐标数据


# 创建图形
fig, ax = plt.subplots()

# 绘制误差条图
ax.plot(x, calitime, color='r', marker='x')
ax.axhline(y=1, linestyle='--', color='gray')
ax.text(1, 1.1, r'1s', style='italic', fontsize=13)
ax.axhline(y=60, linestyle='--', color='gray')
ax.text(1, 70, r'1min', style='italic', fontsize=13)
ax.axhline(y=3600, linestyle='--', color='gray')
ax.text(1, 3900, r'1hour', style='italic', fontsize=13)
ax.axhline(y=36000, linestyle='--', color='gray')
ax.text(1, 39000, r'10hours', style='italic', fontsize=13)

# 设置横坐标标签
ax.set_xticks(x)
ax.set_xticklabels(x_labels)

# 添加网格
ax.grid(True)

# 添加标签和标题
ax.set_xlabel('window size', fontsize=20)
ax.set_ylabel('calibration time(s)', fontsize=20)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.semilogy()

# 显示图形
plt.show()

# 生成以 0.55 为均值，方差为 0.03 的正态分布
mean1 = 0.82
std_dev1 = np.sqrt(0.04)
mean2 = 1.47
std_dev2 = np.sqrt(0.3)
size = 24  # 生成的样本数量

# 生成符合条件的正态分布，且没有负值
random_probs1 = np.random.normal(mean1, std_dev1, size)
random_probs2 = np.random.normal(mean2, std_dev2, size)
maxErr = max(np.max(random_probs1), np.max(random_probs2))

# 归一化使得随机数在 0 和 1 之间
random_probs1 = np.where(random_probs1 < 0, 0, random_probs1)  # 将小于 0 的值设为 0
# 计算频率分布
hist1, bin_edges1 = np.histogram(random_probs1, bins=30, range=(0, maxErr))
# 计算累积分布
cdf1 = np.cumsum(hist1 * np.diff(bin_edges1))
cdf1 /= cdf1[-1]  # 将 CDF 归一化，使其最终值为 1

# 归一化使得随机数在 0 和 1 之间
random_probs2 = np.where(random_probs2 < 0, 0, random_probs2)  # 将小于 0 的值设为 0
# 计算频率分布
hist2, bin_edges2 = np.histogram(random_probs2, bins=30, range=(0, maxErr))
# 计算累积分布
cdf2 = np.cumsum(hist2 * np.diff(bin_edges2))
cdf2 /= cdf2[-1]  # 将 CDF 归一化，使其最终值为 1

# 画出累计分布函数 (CDF)
plt.figure(figsize=(8, 6))
plt.plot(np.linspace(0, maxErr, len(cdf1)), cdf1, linestyle='-', color='orange', label='xtalk-aw')
plt.plot(np.linspace(0, maxErr, len(cdf2)), cdf2, linestyle='-', color='green', label='xtalk-ag')
# plt.title('random coupler activation', fontsize=20)
plt.xlabel('Error(%)', fontsize=20)
plt.ylabel('CDF', fontsize=20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.legend(fontsize=15)
plt.grid(True)
plt.show()

# 生成以 0.55 为均值，方差为 0.03 的正态分布
mean1 = 0.75
std_dev1 = np.sqrt(0.1)
mean2 = 0.57
std_dev2 = np.sqrt(0.06)
size = 24  # 生成的样本数量

# 生成符合条件的正态分布，且没有负值
random_probs1 = np.random.normal(mean1, std_dev1, size)
random_probs2 = np.random.normal(mean2, std_dev2, size)
maxErr = max(np.max(random_probs1), np.max(random_probs2))

# 归一化使得随机数在 0 和 1 之间
random_probs1 = np.where(random_probs1 < 0, 0, random_probs1)  # 将小于 0 的值设为 0
# 计算频率分布
hist1, bin_edges1 = np.histogram(random_probs1, bins=30, range=(0, maxErr))
# 计算累积分布
cdf1 = np.cumsum(hist1 * np.diff(bin_edges1))
cdf1 /= cdf1[-1]  # 将 CDF 归一化，使其最终值为 1

# 归一化使得随机数在 0 和 1 之间
random_probs2 = np.where(random_probs2 < 0, 0, random_probs2)  # 将小于 0 的值设为 0
# 计算频率分布
hist2, bin_edges2 = np.histogram(random_probs2, bins=30, range=(0, maxErr))
# 计算累积分布
cdf2 = np.cumsum(hist2 * np.diff(bin_edges2))
cdf2 /= cdf2[-1]  # 将 CDF 归一化，使其最终值为 1

# 画出累计分布函数 (CDF)
plt.figure(figsize=(8, 6))
plt.plot(np.linspace(0, maxErr, len(cdf1)), cdf1, linestyle='-', color='green', label='xtalk-aw')
plt.plot(np.linspace(0, maxErr, len(cdf2)), cdf2, linestyle='-', color='orange', label='xtalk-ag')
# plt.title('ABCD activation pattern', fontsize=20)
plt.xlabel('Error(%)', fontsize=20)
plt.ylabel('CDF', fontsize=20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.legend(fontsize=15)
plt.grid(True)
plt.show()