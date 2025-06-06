import os
import numpy as np
import matplotlib.pyplot as plt

# 读数据

def read_data(f):
    with open(f, 'r') as fp:
        data = fp.read()
        data = data.split('\n')
        if '' in data:
            data.remove('')
    argsNum = len(data) - 1
    args = []
    for _ in range(argsNum):
        args.append([])

    f = []
    d = data[0].split(' ')
    for v in d:
        if v == '':
            break
        f.append(float(v))
    for i in range(argsNum):
        strXyz = data[i + 1].split('\n')
        for strX in strXyz:
            strX = strX.split(' ')
            for v in strX:
                if v == '':
                    break
                args[i].append(float(v))
    return f, args

# 画热力图

def draw_heat_map(xx, yy, mat, title, picname, xlabel, ylabel, drawtype=None, threshold=None):
    mat = np.array([mat]).reshape(len(xx), len(yy))
    if drawtype == 'log':
        mat = np.log10(mat.T)
    elif drawtype == 'abs':
        mat = np.abs(mat.T)
    elif 'log' in drawtype and 'abs' in drawtype:
        mat = np.log10(np.abs(mat.T))
    else:
        mat = mat.T

    xx,yy = np.meshgrid(xx, yy)
    font2 = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 20}
    fig, ax = plt.subplots()

    if threshold == None:
        cs = plt.contour(xx, yy, mat, colors="r", linewidths=0.5) 
    else:
        cs = plt.contour(xx, yy, mat, [threshold], colors="r", linewidths=0.5) 

    plt.clabel(cs, fontsize=12, inline=True) 
    plt.contourf(xx, yy, mat, 200)

    plt.colorbar()

    plt.xlabel(xlabel, font2)
    plt.ylabel(ylabel, font2)
    plt.title(title)
    plt.savefig(picname + '.pdf', dpi=300)

if __name__ == '__main__':

    zeta_list, omegaqcList = read_data('zz coupling.txt')
    omega2List = omegaqcList[0]
    omegacList = omegaqcList[1]
    zeta_list = zeta_list

    # 画图
    draw_heat_map(omega2List, omegacList, zeta_list, 'xi', 'zz coupling', 'omega_1', 'omega_c', 'logabs')

    err0_list, omegaqcList = read_data('error0.txt')
    omega2List = omegaqcList[0]
    omegacList = omegaqcList[1]
    err0_list = err0_list

    # 画图
    draw_heat_map(omega2List, omegacList, err0_list, 'lg(err0)', 'error0', 'omega_1', 'omega_c', 'logabs', -5)

    err1_list, omegaqcList = read_data('error1.txt')
    omega2List = omegaqcList[0]
    omegacList = omegaqcList[1]
    err1_list = err1_list

    draw_heat_map(omega2List, omegacList, err1_list, 'lg(err1)', 'error1', 'omega_1', 'omega_c', 'logabs' , -5)