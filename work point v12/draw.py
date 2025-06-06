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

    path = 'F:\\vs experiment\\work point v12'
    datanames = os.listdir(path)
    fileList = []
    for i in datanames:
        if '.txt' in i:
            fileList.append(i)

    anharm = {'q0' : -200 * 1e-3, 'c0' : -200 * 1e-3, 'q1' : -200 * 1e-3, 'c1' : 200 * 1e-3, 'q2' : -200 * 1e-3}

    omega0 = 4.5

    delta12Low, delta12High = -0.6, omega0 - max(omega0 - anharm['q1'], omega0 + anharm['q0'])
    delta12step = 25
    delta12s = np.arange(delta12Low, delta12High, (delta12High - delta12Low) / delta12step)
    omega1List = omega0 - delta12s

    clow, chigh = 5, 7
    cstep = 25
    omegacList = np.arange(clow, chigh, (chigh - clow) / cstep)
    omegacList = omegacList

    alphc = 0
    
    for name in fileList:
        if 'wp' in name:
            omega1Work = []
            omegacWork = []
            sigma1 = []
            sigma2 = []
            
            with open(name, 'r') as fp:
                data = fp.read()
                data = data.split('\n')
                while '' in data:
                    data.remove('')
                for i in data:
                    i = i.split(' ')
                    while '' in i:
                        i.remove('')
                    lenI = len(i)
                    for strNum in range(lenI):
                        if strNum > 0 and i[strNum - 1] == 'omega1':
                            omega1Work.append([[float(i[strNum + 1][:-1])]])
                            omegacWork.append([[float(i[strNum + 1][:-1])]])
                            sigma1.append([[float(i[strNum + 1][:-1])]])
                            sigma2.append([[float(i[strNum + 1][:-1])]])
                        elif i[strNum - 1] == 'omegac':
                            omega1Work[-1][0].append(float(i[strNum][1 : -1]))
                            omegacWork[-1][0].append(float(i[strNum][1 : -1]))
                            sigma1[-1][0].append(float(i[strNum][1 : -1]))
                            sigma2[-1][0].append(float(i[strNum][1 : -1]))
                        elif i[strNum - 1] == 'omega1Work':
                            omega1Work[-1].append(float(i[strNum]))
                            omega1Work[-1] = omega1Work[-1][::-1]
                        elif i[strNum - 1] == 'omegacWork':
                            omegacWork[-1].append(float(i[strNum]))
                            omegacWork[-1] = omegacWork[-1][::-1]
                        elif i[strNum - 1] == 'sigma1':
                            sigma1[-1].append(float(i[strNum]))
                            sigma1[-1] = sigma1[-1][::-1]
                        elif i[strNum - 1] == 'sigma2':
                            sigma2[-1].append(float(i[strNum]))
                            sigma2[-1] = sigma2[-1][::-1]

    omega1Work = list(dict(sorted(omega1Work, key=lambda x : x[1])).keys())
    omega1Work = np.array(omega1Work).reshape(delta12step, cstep)
    omega1Work = omega1Work[::-1]
    draw_heat_map(delta12s[:-5], omegacList[2:], omega1Work[:-5, 2:] - max(omega0 - anharm['q1'], omega0 + anharm['q0']), 
    'omega1 work', 'omega1 work', 'detune12(MHz)', 'omegac(GHz)', threshold=0.14)
    omegacWork = list(dict(sorted(omegacWork, key=lambda x : x[1])).keys())
    omegacWork = np.array(omegacWork).reshape(delta12step, cstep)
    omegacWork = omegacWork[::-1]
    draw_heat_map(delta12s[:-5], omegacList[3:], omegacWork[:-5, 3:], 'omegac work', 'omegac work', 'detune12(MHz)', 'omegac(GHz)', threshold=5.05)
    sigma1 = list(dict(sorted(sigma1, key=lambda x : x[1])).keys())
    sigma1 = np.array(sigma1).reshape(delta12step, cstep)
    sigma1 = sigma1[::-1]
    draw_heat_map(delta12s, omegacList, sigma1, 'sigma1', 'sigma1', 'detune12(MHz)', 'omegac(GHz)')
    sigma2 = list(dict(sorted(sigma2, key=lambda x : x[1])).keys())
    sigma2 = np.array(sigma2).reshape(delta12step, cstep)
    sigma2 = sigma2[::-1]
    draw_heat_map(delta12s, omegacList, sigma2, 'sigma2', 'sigma2', 'detune12(MHz)', 'omegac(GHz)')


    # for name in fileList:
    #     if '0.' in name:
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
    #             threshold = -4
    #         draw_heat_map(delta12s, omegacList, data_list, Picname, Picname, 'detune12(MHz)', 'omegac(GHz)', 'logabs', threshold=threshold)

                
    