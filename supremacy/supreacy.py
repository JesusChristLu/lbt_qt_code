import matplotlib.pyplot as plt
import numpy as np

def exp_dist(x, a):
    return a * np.exp(-a * x)


x = np.linspace(0, 1, 100)
for a in [2, 4, 8, 16]:
    plt.plot(x, exp_dist(x, a), label=str(a))
    
plt.legend()
plt.show()