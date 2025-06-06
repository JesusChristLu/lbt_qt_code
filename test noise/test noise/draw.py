from pyqpanda import *
from math import pi
import cupy as np
import numpy as nmp
import matplotlib.pyplot as plt
import time
import scipy.stats as sta
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
#for beta_num in [0, 1, 0.5, 0.25]:

for beta_num in [0]:
    step = 4
    loss = []
    with open('step' + str(step) + 'precisionpi' + str(int(32 * beta_num)) + '.txt', "r") as f:  
        while True:
            data_txt = f.readline()
            if data_txt == '':
                break
            loss.append(-float(data_txt[0 : 9]))
    mu_Prec = np.mean(np.array(loss))
    sigma_Prec = np.std(np.array(loss))
    n, bins, patches = plt.hist(loss, bins = 100, density = 1, facecolor = "red", alpha = 1) 
    y_norm = sta.norm.pdf(np.asnumpy(bins), np.asnumpy(mu_Prec), np.asnumpy(sigma_Prec)) 

    plt.title('Loss distribution', font2)
    #plt.plot(bins, y_norm, color='blue', label='norm')  
    plt.xlim([9.5, 10.1])
    #plt.ylim([0, 25])
    plt.yticks(fontproperties = 'Times New Roman', size = 15)
    plt.xticks(fontproperties = 'Times New Roman', size = 15)
    plt.xlabel('loss', font2) 
    plt.ylabel('Probability', font2)    
    plt.show()
    #plt.savefig('step' + str(step) + 'precisionpi' + str(int(32 * beta_num)) + '.png')
    #plt.close()