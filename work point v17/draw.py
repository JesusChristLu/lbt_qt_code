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

    anharm = {'q0' : -220 * 1e-3, 'c0' : -200 * 1e-3, 'q1' : -220 * 1e-3, 'c1' : -200 * 1e-3, 'q2' : -220 * 1e-3}




    # omega0 = 4.5

    # delta12Low, delta12High = -0.04, 0.04
    # delta12step = 50
    # delta12s = np.arange(delta12Low, delta12High, (delta12High - delta12Low) / delta12step)
    # omega1List = omega0 - delta12s

    # clow, chigh = 5, 7
    # cstep = 50
    # omegacList = np.arange(clow, chigh, (chigh - clow) / cstep)
    # omegacList = omegacList


    # for name in fileList:
    #     if '0.txt' in name:
    #         data_list = read_data(name)
    #         Picname = name[:-4]
    #         if 'cphase' in name:
    #             data_list = list(np.abs(np.abs(np.array(data_list)) - np.pi))
    #         order = int(name[-5])
    #         for name1 in fileList:
    #             if name1[:-5] == name[:-5] and int(name1[-5]) == order + 1:
    #                 if 'cphase' in name1:
    #                     data_list += list(np.abs(np.abs(np.array(read_data(name1))) - np.pi))
    #                 else:
    #                     data_list += read_data(name1)
    #                 order += 1

    #         data_list = np.array(data_list).reshape(delta12step, cstep)
    #         data_list = data_list[::-1]
    #         if 'cphase' in name:
    #             threshold = -2
    #         else:
    #             threshold = -3
    #         draw_heat_map(delta12s, omegacList, data_list, Picname, Picname, 'detune12(MHz)', 'omegac(GHz)', 'logabs', threshold=threshold)



    step = 50

    rho0cLow, rho0cHigh = 1e-4, 0.08
    rhoqcList = np.arange(rho0cLow, rho0cHigh, (rho0cHigh - rho0cLow) / step)[:step]

    rho01Low, rho01High = 1e-4, 0.015
    rhoqqList = np.arange(rho01Low, rho01High, (rho01High - rho01Low) / step)[:step]
    

    for name in fileList:
        if '0.txt' in name:
            data_list = read_data(name)
            Picname = name[:-4]
            order = int(name[-5])
            for name1 in fileList:
                if name1[:-5] == name[:-5] and int(name1[-5]) == order + 1:
                    data_list += read_data(name1)
                    order += 1

            if 'amp' in name:
                typ = 'abs'
                threshold = 0.8
                for i in range(len(data_list)):
                    if data_list[i] == 1:
                        data_list[i] = 1.4
                dt = np.array(data_list).reshape((step, step))
                print('fuck')
            elif 'err' in name:
                typ = 'logabs'
                threshold = -3
                dt = np.array(data_list).reshape((step, step))
                print('fuck')
            elif 'sta' in name:
                typ = 'abs'
                threshold = 1
                data_list = np.array(data_list) * 100
                for i in range(len(data_list)):
                    if data_list[i] > 2:
                        data_list[i] = 2
                dt = np.array(data_list).reshape((step, step))
                print('fuck')
            elif 'zz rho' in name:
                typ = 'logabs'
                threshold = -1.3
                dt = np.array(data_list).reshape(step, step)
                print('fuck')
            draw_heat_map(rhoqcList, rhoqqList, data_list, Picname, Picname, 'rho1c', 'rho12', typ, threshold)