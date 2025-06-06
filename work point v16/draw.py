import os
from tkinter.tix import DirList
import numpy as np
import matplotlib.pyplot as plt

# 读数据

def read_data(f):
    with open(f, 'r') as fp:
        data = fp.read()
        data = data.split('\n')
        if '' in data:
            data.remove('')
    f = []
    for d in data:
        if '(' in d:
            d = d[1:-2]
        f.append(float(d))
    return f

# 画热力图

def draw_heat_map(xx, yy, mat, title, picname, xlabel, ylabel, drawtype=None, threshold=None):
    mat = np.array([mat]).reshape(len(xx), len(yy))

    if drawtype == None:
        mat = mat.T
    elif drawtype == 'log':
        mat = np.log10(mat.T)
    elif drawtype == 'abs':
        mat = np.abs(mat.T)
    elif 'log' in drawtype and 'abs' in drawtype:
        mat = np.log10(np.abs(mat.T) + 1e-6)

    xx,yy = np.meshgrid(xx, yy)
    font2 = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 20}
    fig, ax = plt.subplots()

    if threshold == None:
        cs = plt.contour(xx, yy, mat, colors="r", linewidths=0.5) 
    else:
        cs = plt.contour(xx, yy, mat, [threshold], colors="r", linewidths=0.5) 

    plt.clabel(cs, fontsize=12, inline=True) 
    plt.contourf(xx, yy, mat, 200, cmap=plt.cm.jet)

    plt.colorbar()

    plt.xlabel(xlabel, font2)
    plt.ylabel(ylabel, font2)
    plt.title(title)
    plt.savefig(picname + '.pdf', dpi=300)

if __name__ == '__main__':

    path = os.getcwd()
    datanames = os.listdir(path)
    fileList = []
    for i in datanames:
        if '.txt' in i:
            fileList.append(i)

    anharm = {'q0' : -240 * 1e-3, 'c0' : -200 * 1e-3, 'q1' : -200 * 1e-3, 'c1' : -200 * 1e-3, 'q2' : -300 * 1e-3}

    omegal, omegahi = 3.6, 4.5
    omegah = omegal - anharm['q1']

    omegas = [omegahi, omegal - anharm['q1'], omegal, omegah - anharm['q2'], omegah + anharm['q1'] - anharm['q2'], omegah + 2 * anharm['q1'], omegahi + anharm['q1']]

    omega0 = 3.6
    omega1 = omega0 - anharm['q1']
    omega2Low, omega2High = 3.2, 4.6
    omega2step = 120
    omega2List = np.arange(omega2Low, omega2High, (omega2High - omega2Low) / omega2step)[:omega2step]
    delta12s =  omega1 - omega2List

    i = 0

    data_lists = []
    for name in fileList:
        if '0.txt' in name:
            data_list = read_data(name)
            Picname = name[:-4]
            order = int(name[-5])
            for name1 in fileList:
                if name1[:-5] == name[:-5] and int(name1[-5]) == order + 1:
                    data_list += read_data(name1)
                    order += 1

            data_list = np.array(data_list).reshape(omega2step, 1)
            data_lists.append(data_list)
            i += 1

    plt.figure(figsize=(7, 8))
    ax = plt.subplot(211)

    for i in omegas[:3]:
        ax.axvline(x=i + 0.01, ymin=0.05, ymax=0.9, color='red')
        ax.axvline(x=i + 0.01 + 0.06, ymin=0.05, ymax=0.9, color='green', linestyle='--')
        ax.axvline(x=i + 0.01 - 0.06, ymin=0.05, ymax=0.9, color='green', linestyle='--')

    ax.plot(omega2List, data_lists[0])
    ax.set_ylabel('0 error')
    # plt.legend()
    ax = plt.subplot(212)

    for i in omegas:
        ax.axvline(x=i + 0.01, ymin=0.05, ymax=0.9, color='red')
        ax.axvline(x=i + 0.01 + 0.055, ymin=0.05, ymax=0.9, color='green', linestyle='--')
        ax.axvline(x=i + 0.01 - 0.055, ymin=0.05, ymax=0.9, color='green', linestyle='--')

    ax.plot(omega2List, data_lists[1])
    ax.set_ylabel('1 error')
    # plt.legend()
    plt.show()