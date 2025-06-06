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

colors = ['black', 'red', 'yellowgreen', 'brown', 
          'teal', 'darkblue', 'purple', 'orange', 
          'grey', 'crimson']

precision_range = [pi / 32, pi / 16, pi / 8]

def get_pdf(p_v, bin):
    p_v = np.array(p_v)
    pv_range = np.arange(2, 10.5, (10.5 - 2.0) / bin)
    frequency = []
    probability = []
    for i in pv_range:
        probability.append(len(np.where(p_v < i)[0]))
    probability = np.array(probability)
    frequency = probability[1:] - probability[:len(probability) - 1]
    frequency = (np.array(frequency) / sum(frequency))
    return frequency

def relative_entropy(pdf1, pdf2):
    return -np.sum(pdf1 * np.log(pdf2 + 1e-20))
    
standard = ['epsilon\\step0precisionpi0.txt',
            'epsilon\\step1precisionpi0.txt',
            'epsilon\\step2precisionpi0.txt',
            'epsilon\\step3precisionpi0.txt',
            'epsilon\\step4precisionpi0.txt',]
file_name = [['epsilon\\step0precisionpi32.txt',
              'epsilon\\step0precisionpi16.txt',
              'epsilon\\step0precisionpi8.txt'],
             ['epsilon\\step1precisionpi32.txt',
              'epsilon\\step1precisionpi16.txt',
              'epsilon\\step1precisionpi8.txt'],
             ['epsilon\\step2precisionpi32.txt',
              'epsilon\\step2precisionpi16.txt',
              'epsilon\\step2precisionpi8.txt'],
             ['epsilon\\step3precisionpi32.txt',
              'epsilon\\step3precisionpi16.txt',
              'epsilon\\step3precisionpi8.txt'],
             ['epsilon\\step4precisionpi32.txt',
              'epsilon\\step4precisionpi16.txt',
              'epsilon\\step4precisionpi8.txt']]
bin = 1000

for step in range(5):
    variance = []
    maxmin = []
    std = []
    with open(standard[step], "r") as f:  
        loss = []
        while True:
            data_txt = f.readline()
            if data_txt == '':
                break
            loss.append(-float(data_txt[0 : 9]))
    #variance.append(np.std(np.array(loss)))
    maxmin.append(max(loss) - min(loss))
    for precision in [1, 2, 3]:
        with open(file_name[step][precision - 1], "r") as f:  
            loss = []
            while True:
                data_txt = f.readline()
                if data_txt == '':
                    break
                loss.append(-float(data_txt[0 : 9]))
        #variance.append(np.std(np.array(loss)))
        maxmin.append(max(loss) - min(loss))
    plt.title('')
    #plt.plot(variance, colors[step], label='p=' + str(step + 1))  
    plt.plot(maxmin, colors[step], label='p=' + str(step + 1))
    plt.yticks(fontproperties = 'Times New Roman', size = 15)
    plt.xticks(fontproperties = 'Times New Roman', size = 15)
    plt.ylabel('variance', font2) 
    plt.xlabel('epsilon', font2) 
    plt.legend()
    #plt.savefig('smaller' + str(step) + '.png', dpi=800)
plt.show()

