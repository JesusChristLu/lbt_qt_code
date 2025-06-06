from pyqpanda import *
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats as sta
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

'''
fname = ['F:\\vs experiment\\qaoa6\\qaoa6\\20200513 153535.txt',]
for name in fname:
    with open(name) as f: 
        data_txt = f.read()
        reading = False
        is_number = False
        number_text = ''
        loss = []
        for i in data_txt:
            if i == '[':
                reading = True
            elif reading and (i.isdigit() or i == '-' or i == '.'):
                is_number = True
                number_text += i
            elif i == ']' and is_number:
                is_number = False
                loss.append(float(number_text))
                number_text = ''
            elif is_number:
                number_text += i
            elif i == 'p' and reading:
                reading = False
                break
    losses.append(loss)
    '''
p = 1
colors = ['black', 'red', 'yellowgreen', 'brown', 
          'teal', 'darkblue', 'purple', 'orange', 
          'grey', 'crimson']
for loss in losses:
    plt.title('loss function')
    plt.plot(np.array(loss) / 10, color=colors[p - 1], label='p=' + str(p))  
    #plt.xlim([3.5, 6.5])
    plt.yticks(fontproperties = 'Times New Roman', size = 15)
    plt.xticks(fontproperties = 'Times New Roman', size = 15)
    plt.xlabel('epoch', font2) 
    plt.ylabel('loss', font2) 
    plt.legend()
    #plt.savefig('smaller' + str(step) + '.png', dpi=800)
    p += 1
plt.show()
