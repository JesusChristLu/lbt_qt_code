import numpy as np  
import re  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
  
# 用于存储数据的字典  
xebdata0 = {}  
xebdata1 = {}  
  
# 读取文件  
with open('xeb_file3.txt', 'r') as file:  
    lines = file.readlines()  
  
# 正则表达式匹配 pattern 和 batch 的值  
pattern_re = re.compile(r'pattern(\d+)qs\((.*?)\)length(\d+)batch(\d+):(.*)')  
  
# 遍历所有行  
for line in lines:  
    match = pattern_re.match(line)  
    if match:  
        pattern_num = int(match.group(1))  
        qs = tuple(match.group(2).split(', '))  
        length = int(match.group(3))  
        # batch_num = int(match.group(4))  
        value = complex(match.group(5))  
        
        # if length > 70:
        #     continue

        # 初始化字典中的 qs  
        if pattern_num == 0:  
            if qs not in xebdata0:  
                xebdata0[qs] = {length: [value.real]}  
            else:
                if length not in xebdata0[qs]:
                    xebdata0[qs][length] = [value.real]
                else:
                    xebdata0[qs][length].append(value.real)  # 假设我们只关心实数部分  
        elif pattern_num == 1:  
            if qs not in xebdata1:  
                xebdata1[qs] = {length: [value.real]}  
            else:
                if length not in xebdata1[qs]:
                    xebdata1[qs][length] = [value.real]
                else:
                    xebdata1[qs][length].append(value.real)  # 假设我们只关心实数部分  

# 定义指数拟合函数 Ap^m + B
def exp_fit(length, A, P, B):
    # return P ** length + B
    return A * P ** length + B

lengthArray = sorted(list(xebdata0[qs].keys()))

# 计算均值和方差，并创建 NumPy 数组  
for pattern_data in [xebdata0, xebdata1]:  
    for qs, lengths in pattern_data.items():  
        max_length = max(lengths.keys())  
        # 创建一个空的 2x(max_length) 数组  
        data_array = np.zeros((2, len(lengthArray)))  
          
        # 填充均值和方差  
        for length, values in lengths.items():  
            mean_value = np.mean(values)  
            var_value = np.var(values)  
            data_array[0, lengthArray.index(length)] = mean_value  
            data_array[1, lengthArray.index(length)] = var_value  
          
        # 更新字典中的值  
        pattern_data[qs] = data_array  

lengths = np.array(lengthArray)
# 初始猜测值 A, P, B
initial_guess = [1.0, 0.9, 0.0]

# 对 pattern0 的均值进行指数拟合
mean_values_pattern0 = np.maximum(xebdata0[("'q0'", "'q1'")][0], xebdata1[("'q0'", "'q1'")][0])
# mean_values_pattern0 = xebdata0[("'q0'", "'q1'")][0]
mean_values_pattern0 = (mean_values_pattern0 - np.min(mean_values_pattern0)) / (np.max(mean_values_pattern0) - np.min(mean_values_pattern0))
# 进行拟合
params, covariance = curve_fit(exp_fit, lengths, mean_values_pattern0, p0=initial_guess)
# 获取拟合参数 A, P, B
A_fit, P_fit, B_fit = params
# 使用拟合参数计算拟合曲线
fitted_values0 = exp_fit(lengths, A_fit, P_fit, B_fit)
# 绘制拟合曲线
# plt.plot(lengths, fitted_values0, label=f'Fit: A={A_fit:.2f}, P={P_fit:.2f}, B={B_fit:.2f}' + 'pattern0', linestyle='--')
err = (1 - P_fit) * (1 - 1 / 4) - 0.003
plt.plot(lengths, fitted_values0, color='green',  label=f'xeb err={err:.4f}',  linestyle='--')
plt.errorbar(lengthArray, mean_values_pattern0, yerr=xebdata0[("'q0'", "'q1'")][1] / 5, ecolor='green',  fmt='o', label='pattern0', capsize=5)

# 对 pattern0 的均值进行指数拟合
mean_values_pattern1 = np.minimum(xebdata0[("'q0'", "'q1'")][0], xebdata1[("'q0'", "'q1'")][0])
# mean_values_pattern1 = xebdata1[("'q0'", "'q1'")][0]
mean_values_pattern1 = (mean_values_pattern1 - np.min(mean_values_pattern1)) / (np.max(mean_values_pattern1) - np.min(mean_values_pattern1))
# 进行拟合
params, covariance = curve_fit(exp_fit, lengths, mean_values_pattern1, p0=initial_guess)
# 获取拟合参数 A, P, B
A_fit, P_fit, B_fit = params
# 使用拟合参数计算拟合曲线
fitted_values1 = exp_fit(lengths, A_fit, P_fit, B_fit)
# 绘制拟合曲线
# plt.plot(lengths, fitted_values1, label=f'Fit: A={A_fit:.2f}, P={P_fit:.2f}, B={B_fit:.2f}' + 'pattern1', linestyle='--')
err = (1 - P_fit) * (1 - 1 / 4) - 0.003
plt.plot(lengths, fitted_values1, color='red', label=f'xeb err={err:.4f}',  linestyle='--')
plt.errorbar(lengthArray, mean_values_pattern1, yerr=xebdata1[("'q0'", "'q1'")][1] / 5, ecolor='red',  fmt='o', label='pattern1', capsize=5)

plt.legend()
plt.show()
