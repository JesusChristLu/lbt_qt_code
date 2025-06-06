import numpy as np
import matplotlib.pyplot as plt

with open('data.txt', 'r') as fp:
    data = fp.read()
    data = data.split('\n')
    if '' in data:
        data.remove('')

data = [float(i) for i in data]

nnThresholds = np.arange(0.04, 0.061, 0.001)
plt.plot(nnThresholds, data)
plt.show()

# with open('data1.txt', 'r') as fp:
#     data = fp.read()
#     data = data.split('\n')
#     if '' in data:
#         data.remove('')

# data = [float(i) for i in data]

# nnThresholds = np.arange(0.0002, 0.00031, 0.00001)
# plt.plot(nnThresholds, data)
# plt.show()