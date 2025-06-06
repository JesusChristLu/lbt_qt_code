import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erf
import networkx as nx
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
import numpy as np

# 调度方案
scheduling_algorithms = ['FCFS', 'SJF', 'QSJF', 'SRTF', 'RR', 'MFQ', 'HRRF', 'QHRRF']

# 5组泊松过程的速率参数
lambda_values = [5, 20, 50, 100, 500]

# 假设每个速率参数下的吞吐量数据（示例数据）
throughputs = [
    [5.1, 5.2, 6.2, 6, 4, 4.3, 5.1, 5.9],   # 对应 lambda = 5
    [17.3, 22, 24, 28, 16, 20, 21, 23],   # 对应 lambda = 20
    [42, 55, 65, 68, 34, 36, 53, 55],   # 对应 lambda = 50
    [80, 92, 115, 128, 74, 86, 103, 105],   # 对应 lambda = 100
    [85, 82, 133, 128, 84, 76, 113, 125]   # 对应 lambda = 500
]

# 假设每个速率参数下的误差数据
errors = [
    [1, 1.2, 1.1, 1.3, 1.2, 1.5, 1.1, 1.3],
    [2.5, 3.7, 3.6, 1.8, 2.9, 2.0, 1.8, 1.9],
    [5, 8.1, 3.2, 9.3, 8.1, 2.5, 7.0, 5.2],
    [15, 26, 37, 18, 57, 10, 28, 29],
    [18, 20, 17, 28, 77, 30, 18, 39]
]

# 设置图形
# plt.figure(figsize=(10, 6))

# 对每个泊松速率参数绘制一条带误差棒的曲线
for i, lambda_val in enumerate(lambda_values):
    plt.errorbar(scheduling_algorithms, throughputs[i], yerr=errors[i], fmt='o-', capsize=5, markersize=8, label=rf'$λ$ = {lambda_val}')

# 添加图形标题和坐标轴标签
plt.title('Throughput', fontsize=15)
plt.xlabel('Scheduling Algorithms', fontsize=15)
plt.ylabel(r'Throughput $(\frac{1}{s})$', fontsize=15)

plt.xticks(fontsize=15) # set the x-axis ticks to the group labels
plt.yticks(fontsize=15) # set the x-axis ticks to the group labels

# 显示图例
plt.legend(fontsize=10)

# 添加网格
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()

# 调度方案
scheduling_algorithms = ['FCFS', 'SJF', 'QSJF', 'SRTF', 'RR', 'MFQ', 'HRRF', 'QHRRF']

# 5组泊松过程的速率参数
lambda_values = [5, 20, 50, 100, 500]

# 假设每个速率参数下的吞吐量数据（示例数据）
utilization = np.array([
    [0.3, 0.29, 0.30, 0.33, 0.35, 0.33, 0.34, 0.38],   # 对应 lambda = 5
    [0.6, 0.76, 0.90, 0.77, 0.61, 0.56, 0.71, 0.83],   # 对应 lambda = 20
    [0.7, 0.8, 0.92, 0.70, 0.54, 0.66, 0.53, 0.85],   # 对应 lambda = 50
    [0.67, 0.72, 0.86, 0.75, 0.41, 0.63, 0.63, 0.82],   # 对应 lambda = 100
    [0.60, 0.70, 0.90, 0.78, 0.52, 0.73, 0.53, 0.92]   # 对应 lambda = 500
]) * 100

# 假设每个速率参数下的误差数据
errors = np.array([
    [0.01, 0.01, 0.01, 0.013, 0.012, 0.05, 0.07, 0.013],
    [0.07, 0.03, 0.06, 0.018, 0.029, 0.02, 0.018, 0.03],
    [0.09, 0.08, 0.02, 0.013, 0.011, 0.024, 0.08, 0.01],
    [0.05, 0.016, 0.07, 0.018, 0.017, 0.01, 0.048, 0.04],
    [0.03, 0.05, 0.043, 0.08, 0.07, 0.08, 0.04, 0.08]
]) * 100

# 设置图形
# plt.figure(figsize=(10, 6))

# 对每个泊松速率参数绘制一条带误差棒的曲线
for i, lambda_val in enumerate(lambda_values):
    plt.errorbar(scheduling_algorithms, utilization[i], yerr=errors[i], fmt='o-', capsize=5, markersize=8, label=rf'$λ$ = {lambda_val}')

# 添加图形标题和坐标轴标签
plt.title('utilization', fontsize=15)
plt.xlabel('Scheduling Algorithms', fontsize=15)
plt.ylabel('utilization(%)', fontsize=15)

plt.xticks(fontsize=15) # set the x-axis ticks to the group labels
plt.yticks(fontsize=15) # set the x-axis ticks to the group labels

# 显示图例
plt.legend(fontsize=10)

# 添加网格
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()

# 调度方案
scheduling_algorithms = ['FCFS', 'SJF', 'QSJF', 'SRTF', 'RR', 'MFQ', 'HRRF', 'QHRRF']

# 5组泊松过程的速率参数
lambda_values = [5, 20, 50, 100, 500]

# 假设每个速率参数下的吞吐量数据（示例数据）
turnaround = [
    [1.2, 1.01, 1.98, 1.63, 1.5, 1.43, 1.51, 1.09],   # 对应 lambda = 5
    [1.6, 4.06, 4.90, 4.77, 2.61, 2.56, 2.71, 1.83],   # 对应 lambda = 20
    [2.7, 5.8, 7.92, 5.70, 3.34, 3.66, 3.53, 2.85],   # 对应 lambda = 50
    [4.7, 6.2, 8.86, 9.75, 4.41, 4.4, 4.63, 3.82],   # 对应 lambda = 100
    [8.7, 9.2, 15.86, 14.75, 8.41, 7.63, 8.63, 7.82]   # 对应 lambda = 500
]

# 假设每个速率参数下的误差数据
errors = [
    [0.1, 0.1, 0.1, 0.13, 0.12, 0.1, 0.1, 0.13],
    [1.3, 0.3, 0.6, 0.18, 0.29, 0.2, 0.18, 0.13],
    [1, 0.8, 0.2, 0.13, 0.11, 0.24, 0.8, 0.1],
    [1, 1.6, 0.7, 1.8, 1.7, 0.1, 0.48, 0.4],
    [2, 2.6, 1.7, 0.8, 0.7, 2.1, 1.48, 1.4]
]

# 设置图形
# plt.figure(figsize=(10, 6))

# 对每个泊松速率参数绘制一条带误差棒的曲线
for i, lambda_val in enumerate(lambda_values):
    plt.errorbar(scheduling_algorithms, turnaround[i], yerr=errors[i], fmt='o-', capsize=5, markersize=8, label=rf'$λ$ = {lambda_val}')

# 添加图形标题和坐标轴标签
plt.title('weighted turnaround time', fontsize=15)
plt.xlabel('Scheduling Algorithms', fontsize=15)
plt.ylabel('weighted turnaround time', fontsize=15)

plt.xticks(fontsize=15) # set the x-axis ticks to the group labels
plt.yticks(fontsize=15) # set the x-axis ticks to the group labels

# 显示图例
plt.legend(fontsize=10)

# 添加网格
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()


# 调度方案
scheduling_algorithms = ['QSRA', 'codar', 'qhsp', 'redlent', 'xtalkaw']

# 5组泊松过程的速率参数
lambda_values = [5, 20, 50, 100, 500]

# 假设每个速率参数下的吞吐量数据（示例数据）
throughputs = [
    [6.1, 5.01, 4.98, 6.63, 1.5],   # 对应 lambda = 5
    [19, 6.06, 10.90, 8.77, 12.61],   # 对应 lambda = 20
    [46, 5.8, 17.92, 40.70, 33.34],   # 对应 lambda = 50
    [87, 6.2, 18.86, 79.75, 64.41],   # 对应 lambda = 100
    [127, 4.2, 15.86, 104.75, 88.41]   # 对应 lambda = 500
]

# 假设每个速率参数下的误差数据
errors = [
    [1, 2, 2, 1.3, 1.2],
    [1.3, 1.3, 6, 1.8, 2.9],
    [5, 1.8, 2, 1.3, 1.1],
    [16, 1.6, 7, 18, 1.7],
    [20, 2.6, 1.7, 8, 8.7]
]

# 设置图形
# plt.figure()
# plt.figure(figsize=(10, 6))

# 对每个泊松速率参数绘制一条带误差棒的曲线
for i, lambda_val in enumerate(lambda_values):
    plt.errorbar(scheduling_algorithms, throughputs[i], yerr=errors[i], fmt='o-', capsize=5, markersize=8, label=rf'$λ$ = {lambda_val}')

# 添加图形标题和坐标轴标签
plt.title('throughput', fontsize=15)
plt.xlabel('Scheduling Algorithms', fontsize=15)
plt.ylabel(r'Throughput $(\frac{1}{s})$', fontsize=15)

plt.xticks(fontsize=15) # set the x-axis ticks to the group labels
plt.yticks(fontsize=15) # set the x-axis ticks to the group labels

# 显示图例
plt.legend(fontsize=10)

# 添加网格
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()


# 调度方案
scheduling_algorithms = ['QSRA', 'codar', 'qhsp', 'redlent', 'xtalkaw']

# 5组泊松过程的速率参数
lambda_values = [5, 20, 50, 100, 500]

# 假设每个速率参数下的吞吐量数据（示例数据）
utilization = [
    [0.61, 0.1, 0.498, 0.663, 0.5],   # 对应 lambda = 5
    [0.70, 0.06, 0.49, 0.77, 0.61],   # 对应 lambda = 20
    [0.83, 0.08, 0.592, 0.88, 0.34],   # 对应 lambda = 50
    [0.82, 0.02, 0.686, 0.85, 0.41],   # 对应 lambda = 100
    [0.85, 0.042, 0.586, 0.83, 0.41]   # 对应 lambda = 500
]

# 假设每个速率参数下的误差数据
errors = [
    [0.01, 0.02, 0.02, 0.013, 0.012],
    [0.013, 0.013, 0.06, 0.018, 0.029],
    [0.05, 0.018, 0.02, 0.013, 0.011],
    [0.016, 0.016, 0.07, 0.018, 0.017],
    [0.020, 0.026, 0.017, 0.08, 0.087]
]

# 设置图形
# plt.figure()
# plt.figure(figsize=(10, 6))

# 对每个泊松速率参数绘制一条带误差棒的曲线
for i, lambda_val in enumerate(lambda_values):
    plt.errorbar(scheduling_algorithms, utilization[i], yerr=errors[i], fmt='o-', capsize=5, markersize=8, label=rf'$λ$ = {lambda_val}')

# 添加图形标题和坐标轴标签
plt.title('utilization', fontsize=15)
plt.xlabel('Scheduling Algorithms', fontsize=15)
plt.ylabel('utilization(%)', fontsize=15)

plt.xticks(fontsize=15) # set the x-axis ticks to the group labels
plt.yticks(fontsize=15) # set the x-axis ticks to the group labels

# 显示图例
plt.legend(fontsize=10)

# 添加网格
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()

# 调度方案
scheduling_algorithms = ['QSRA', 'codar', 'qhsp', 'redlent', 'xtalkaw']

# 5组泊松过程的速率参数
lambda_values = [5, 20, 50, 100, 500]

# 假设每个速率参数下的吞吐量数据（示例数据）
turnaround = [
    [1.61, 4.1, 4.98, 1.663, 5],   # 对应 lambda = 5
    [1.70, 26.06, 49, 2.67, 10.61],   # 对应 lambda = 20
    [2.86, 70.8, 59, 6.60, 34],   # 对应 lambda = 50
    [3.87, 120, 86, 12.65, 41],   # 对应 lambda = 100
    [7.90, 801, 586, 20.70, 441]   # 对应 lambda = 500
]

# 假设每个速率参数下的误差数据
errors = [
    [0.1, 0.02, 0.2, 0.13, 1.2],
    [0.13, 13, 6, 0.18, 2.9],
    [0.5, 18, 12, 0.13, 11],
    [1.6, 16, 7, 1.8, 17],
    [2, 56, 17, 4, 87]
]

# 设置图形
# plt.figure()
# plt.figure(figsize=(10, 6))

# 对每个泊松速率参数绘制一条带误差棒的曲线
for i, lambda_val in enumerate(lambda_values):
    plt.errorbar(scheduling_algorithms, turnaround[i], yerr=errors[i], fmt='o-', capsize=5, markersize=8, label=rf'$λ$ = {lambda_val}')

# 添加图形标题和坐标轴标签
plt.title('weighted turnaround time', fontsize=15)
plt.xlabel('Scheduling Algorithms', fontsize=15)
plt.ylabel('weighted turnaround time', fontsize=15)

plt.xticks(fontsize=15) # set the x-axis ticks to the group labels
plt.yticks(fontsize=15) # set the x-axis ticks to the group labels

# 显示图例
plt.legend(fontsize=10)

# 添加网格
plt.grid(True)

plt.semilogy()

# 显示图形
plt.tight_layout()
plt.show()


# 调度方案
scheduling_algorithms = ['QSRA', 'codar', 'qhsp', 'redlent', 'xtalkaw']

# 假设每个速率参数下的吞吐量数据（示例数据）
pst = [0.72, 0.75, 0.68, 0.33, 0.78]

# 假设每个速率参数下的误差数据
errors = [0.1, 0.24, 0.2, 0.13, 0.12]

# 设置图形
# plt.figure()
# plt.figure(figsize=(10, 6))

# 绘制带误差棒的柱形图
# 设置柱的宽度
bar_width = 0.6
bars = plt.bar(scheduling_algorithms, pst, yerr=errors, capsize=5, color='skyblue', width=bar_width)

# 添加图形标题和坐标轴标签
plt.title('pst', fontsize=15)
plt.xlabel('Scheduling Algorithms', fontsize=15)
plt.ylabel('pst(%)', fontsize=15)

plt.xticks(fontsize=15) # set the x-axis ticks to the group labels
plt.yticks(fontsize=15) # set the x-axis ticks to the group labels

# 显示图例
# plt.legend(fontsize=10)

# 添加网格
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()


# 调度方案
scheduling_algorithms = ['QSRA', 'QHSP', 'XtalkAw', 'RedLent']

# 假设每个速率参数下的吞吐量数据（示例数据）
ratio = [0.27, 0.14, 0.12, 0.09]

# 假设每个速率参数下的误差数据
errors = [0.17, 0.03, 0.04, 0.02]

# 设置图形
# plt.figure()
# plt.figure(figsize=(10, 6))

# 绘制带误差棒的柱形图
# 设置柱的宽度
bar_width = 0.6
bars = plt.bar(scheduling_algorithms, ratio, yerr=errors, capsize=5, color='grey', width=bar_width)

# 添加图形标题和坐标轴标签
plt.title('ratio between intra-connection and total connection', fontsize=15)
plt.xlabel('Scheduling Algorithms', fontsize=15)
plt.ylabel(r'$\frac{r_i}{r_a}$', fontsize=25)

plt.xticks(fontsize=15) # set the x-axis ticks to the group labels
plt.yticks(fontsize=15) # set the x-axis ticks to the group labels

# 显示图例
# plt.legend(fontsize=10)

# 添加网格
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()

