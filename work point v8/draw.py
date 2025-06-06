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
    f = []
    for d in data:
        f.append(float(d))
    return f

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

    path = 'F:\\vs experiment\\work point v8'
    datanames = os.listdir(path)
    fileList = []
    for i in datanames:
        if '.txt' in i:
            fileList.append(i)

    omega0 = 5
    qlow = 4
    qhigh = 6
    clow = 4
    chigh = 8
    step = 50
    omega1List = np.arange(qlow, qhigh, (qhigh - qlow) / step)
    omegacList = np.arange(clow, chigh, (chigh - clow) / step)

    # detunelow = -0.1
    # detunelow = -0.15
    detunelow = -0.2
    detunehigh = 0
    amplow = 0.1
    amphigh = 0.35
    step = 50
    detuneList = np.arange(detunelow, detunehigh, (detunehigh - detunelow) / step)
    ampList = np.arange(amplow, amphigh, (amphigh - amplow) / step)


    err0_dict = dict()
    err1_dict = dict()
    errx_dict = dict()
    err0_list = []
    err1_list = []
    errx_list = []
    errampdetune_list = []

    for name in fileList:
        if 'zz' in name:
            zeta_list = read_data(name)
            draw_heat_map(omega1List, omegacList, zeta_list, 'xi', 'zz coupling', 'omega_1', 'omega_c', 'logabs')
        elif 'err' in name:
            if 'i0' in name:
                err0_dict[int(name[7])] = read_data(name)
            elif 'i1' in name:
                err1_dict[int(name[7])] = read_data(name)
            elif 'xx' in name:
                errx_dict[int(name[7])] = read_data(name)
            elif 'ampdetune' in name:
                errampdetune_list = read_data(name)
                draw_heat_map(detuneList, ampList, errampdetune_list, 'lg(err)', name[:-4], 'detune', 'amp', 'logabs', -4)

    k = sorted(err0_dict.keys())
    for kk in k:
        err0_list += err0_dict[kk]
        err1_list += err1_dict[kk]
        errx_list += errx_dict[kk]

    draw_heat_map(omega1List, omegacList, err0_list, 'lg(err0)', 'error0', 'omega_1', 'omega_c', 'logabs', -5)
    draw_heat_map(omega1List, omegacList, err1_list, 'lg(err1)', 'error1', 'omega_1', 'omega_c', 'logabs' , -5)
    draw_heat_map(omega1List, omegacList, errx_list, 'lg(errx)', 'errorx', 'omega_1', 'omega_c', 'logabs', -5)