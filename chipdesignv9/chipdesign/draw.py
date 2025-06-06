from sys import builtin_module_names
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.stats import norm, mstats
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from prune import Prune

def mk_test(x, alpha=0.05):
    n = len(x)

    # calculate S
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x, tp = np.unique(x, return_counts=True)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else:  # there are some ties in data
        var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    else: # s == 0:
        z = 0

    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1-alpha/2)

    ndash=int(n*(n-1)/2)
    slope1=np.zeros(ndash)
    m=0
    for k in range(0,n-1):
        for j  in range(k+1,n):
            slope1[m]=(x[j]-x[k])/(j-k)
            m=m+1
    slope=np.median(slope1)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return z, slope, trend
realName = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\real benchmark\\' + str(Prune.degLimit) + 'plot.txt'
bitName = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\bit benchmark\\' + str(Prune.degLimit) + 'plot.txt'
depthName = 'F:\\vs experiment\\chipdesignv9\\chipdesign\\depth benchmark\\' + str(Prune.degLimit) + 'plot.txt'
# plotFile = realName
# plotFile = bitName
plotFile = depthName

if plotFile == realName:

    with open(plotFile, 'r') as fp:
        data = fp.read()
        data = data.split('\n')

    dataDict = {}
    for i in data[:-1]:
        splitI = i.split(' ')
        dataDict[int(splitI[0]), int(splitI[1])] = [int(splitI[2]), int(splitI[3]), int(splitI[4]), int(splitI[5]), int(splitI[6]), int(splitI[7]), int(splitI[8]), int(splitI[9]), int(splitI[10])]

    # bar
    dataKey = sorted(dataDict.items(),key = lambda item:item[0][0])
    step = 0.02
    barRange = np.arange(0, 1 + step, step)
    liBar = []
    crossBar = []
    twoDBar = []
    SPQPDBar = []
    large_part = 0
    for i in dataKey:
        SPQPDBar.append(i[1][2] / i[1][0])
        twoDBar.append(i[1][4] / i[1][0])
        liBar.append(i[1][6] / i[1][0])
        crossBar.append(i[1][8] / i[1][0])
        if i[1][2] / i[1][0] < barRange[1]:
            large_part += 1

    SPQPDBarMean = np.mean(np.array(SPQPDBar))
    SPQPDBarStd = np.std(np.array(SPQPDBar))
    twoDBarMean = np.mean(np.array(twoDBar))
    twoDBarStd = np.std(np.array(twoDBar))
    liBarMean = np.mean(np.array(liBar))
    liBarStd = np.std(np.array(liBar))
    crossBarMean = np.mean(np.array(crossBar))
    crossBarStd = np.std(np.array(crossBar))

    print('ASIQC mean ', SPQPDBarMean, ' 2d lattice mean ', twoDBarMean, ' li mean ', liBarMean, ' cross square mean', crossBarMean)
    plt.hist(SPQPDBar, barRange)
    plt.bar(0, large_part, width=0.04, color='red')
    mean = str(SPQPDBarMean)[:5]
    plt.axvline(x=SPQPDBarMean, color="red", label='mean=' + mean)
    plt.annotate('mean=' + mean, xy=(SPQPDBarMean, 20), xytext=(SPQPDBarMean + 0.01, 30), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # plt.title("histogram of the g_ap of SPQPD structure", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel("g_ap", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel("frequency", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    ax=plt.gca() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim([0, 0.6])
    # plt.ylim([0,66])
    plt.savefig(str(Prune.degLimit) + "SPQPD dist.pdf", dpi = 300)
    plt.show()

    plt.hist(twoDBar, barRange)
    mean = str(twoDBarMean)[:5]
    plt.axvline(x=twoDBarMean, color="red", label='mean=' + mean)
    plt.annotate('mean=' + mean, xy=(twoDBarMean, 10), xytext=(twoDBarMean - 0.15, 15), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # plt.title("histogram of the g_ap of 2d lattice structure", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel("g_ap", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel("frequency", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    ax=plt.gca() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim([0, 0.6])
    # plt.ylim([0,66])
    plt.savefig(str(Prune.degLimit) + "2ddist.pdf", dpi = 300)
    plt.show()

    plt.hist(liBar, barRange)
    mean = str(liBarMean)[:5]
    plt.axvline(x=liBarMean, color="red", label='mean=' + mean)
    plt.annotate('mean=' + mean, xy=(liBarMean, 8), xytext=(liBarMean - 0.2, 11), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # plt.title("histogram of the g_ap of Li's structure", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel("g_ap", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel("frequency", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    ax=plt.gca() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim([0, 0.6])
    # plt.ylim([0,66])
    plt.savefig(str(Prune.degLimit) + "lidist.pdf", dpi = 300)
    plt.show()

    plt.hist(crossBar, barRange)
    mean = str(crossBarMean)[:5]
    plt.axvline(x=crossBarMean, color="red", label='mean=' + mean)
    plt.annotate('mean=' + mean, xy=(crossBarMean, 10), xytext=(crossBarMean - 0.2, 13), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # plt.title("histogram of the g_ap of cross square structure", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel("g_ap", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel("frequency", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    ax=plt.gca() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim([0, 0.6])
    # plt.ylim([0,66])
    plt.savefig(str(Prune.degLimit) + "crossdist.pdf", dpi = 300)
    plt.show()

if plotFile == bitName:
    with open(plotFile, 'r') as fp:
        data = fp.read()
        data = data.split('\n')
    dataKey = []
    for i in data[:-1]:
        splitI = i.split(' ')
        dataKey.append(((int(splitI[0]), int(splitI[1])), (int(splitI[2]), int(splitI[3]), int(splitI[4]), int(splitI[5]), int(splitI[6]), int(splitI[7]), int(splitI[8]), int(splitI[9]), int(splitI[10]))))
    # relationship with qubit number
    ns = list(range(dataKey[0][0][1], dataKey[-1][0][1] + 1, dataKey[10][0][1] - dataKey[0][0][1]))
    liDepthRatio = dict(zip(ns, [[] for _ in range(len(ns))]))
    crossDepthRatio = dict(zip(ns, [[] for _ in range(len(ns))]))
    twoDDepthRatio = dict(zip(ns, [[] for _ in range(len(ns))]))
    SPQPDDepthRatio = dict(zip(ns, [[] for _ in range(len(ns))]))
    twoDAddDepthError = np.zeros(len(ns))
    liAddDepthError = np.zeros(len(ns))
    crossAddDepthError = np.zeros(len(ns))
    SPQPDAddDepthError = np.zeros(len(ns))
    twoDAddDepthMean = np.zeros(len(ns))
    liAddDepthMean = np.zeros(len(ns))
    crossAddDepthMean = np.zeros(len(ns))
    SPQPDAddDepthMean = np.zeros(len(ns))

    for n in ns:
        for i in dataKey:
            if i[0][1] == n:
                # SPQPDDepthRatio[n].append(i[1][1] / i[0][0])
                # twoDDepthRatio[n].append(i[1][3] / i[0][0])
                # liDepthRatio[n].append(i[1][5] / i[0][0])
                # crossDepthRatio[n].append(i[1][7] / i[0][0])
                SPQPDDepthRatio[n].append(i[1][2] / i[1][0])
                twoDDepthRatio[n].append(i[1][4] / i[1][0])
                liDepthRatio[n].append(i[1][6] / i[1][0])
                crossDepthRatio[n].append(i[1][8] / i[1][0])
    for i in range(len(ns)):
        SPQPDAddDepthError[i] = np.std(np.array(SPQPDDepthRatio[ns[i]])) #* 1.3
        SPQPDAddDepthMean[i] = np.mean(np.array(SPQPDDepthRatio[ns[i]]))
        twoDAddDepthError[i] = np.std(np.array(twoDDepthRatio[ns[i]])) #* 1.3
        twoDAddDepthMean[i] = np.mean(np.array(twoDDepthRatio[ns[i]]))
        liAddDepthError[i] = np.std(np.array(liDepthRatio[ns[i]])) #* 1.3
        liAddDepthMean[i] = np.mean(np.array(liDepthRatio[ns[i]]))
        crossAddDepthError[i] = np.std(np.array(crossDepthRatio[ns[i]])) #* 1.3
        crossAddDepthMean[i] = np.mean(np.array(crossDepthRatio[ns[i]]))

    SPQPDslope, SPQPDintercept, SPQPDr_value, SPQPDp_value, SPQPDstd_err = st.linregress(ns, SPQPDAddDepthMean)
    twoDslope, twoDintercept, twoDr_value, twoDp_value, twoDstd_err = st.linregress(ns, twoDAddDepthMean)
    lislope, liintercept, lir_value, lip_value, listd_err = st.linregress(ns, liAddDepthMean)
    crossslope, crossintercept, crossr_value, crossp_value, crossstd_err = st.linregress(ns, crossAddDepthMean)

    plt.figure(figsize=[7,5])
    # plt.plot(ns, twoDslope * np.array(ns) + twoDintercept, label=str(twoDslope)[:6]+'n+'+str(twoDintercept)[:6], color='orange')
    plt.errorbar(ns, twoDAddDepthMean, fmt="o:", label='2d', yerr=twoDAddDepthError, color='orange')
    # plt.plot(ns, lislope * np.array(ns) + liintercept, label=str(lislope)[:6]+'n+'+str(liintercept)[:6], color='green')
    plt.errorbar(ns, liAddDepthMean, fmt="o:", label='li', yerr=liAddDepthError, color='green')
    # plt.plot(ns, crossslope * np.array(ns) + crossintercept, label=str(crossslope)[:6]+'n+'+str(crossintercept)[:6], color='red')
    plt.errorbar(ns, crossAddDepthMean, fmt="o:", label='cross', yerr=crossAddDepthError, color='red')
    # plt.plot(ns, SPQPDslope * np.array(ns) + SPQPDintercept, label=str(SPQPDslope)[:6]+'n+'+str(SPQPDintercept)[:6], color='blue')
    plt.errorbar(ns, SPQPDAddDepthMean, fmt="o:", label='ASIQC', yerr=SPQPDAddDepthError, color='blue')
    plt.xlabel("qubit number n", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel("g_ap", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 12})
    plt.grid()
    ax=plt.gca() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(str(Prune.degLimit) + "pfwrtb.pdf", dpi = 300)
    plt.show()


if plotFile == depthName:
    with open(plotFile, 'r') as fp:
        data = fp.read()
        data = data.split('\n')

    dataKey = []
    for i in data[:-1]:
        splitI = i.split(' ')
        dataKey.append(((int(splitI[0]), int(splitI[1])), (int(splitI[2]), int(splitI[3]), int(splitI[4]), int(splitI[5]), int(splitI[6]), int(splitI[7]), int(splitI[8]), int(splitI[9]), int(splitI[10]))))

    # relationship with circuit depth
    depthes = list(range(dataKey[0][0][0], dataKey[-1][0][0] + 1,  dataKey[10][0][0] - dataKey[0][0][0]))
    liDepthRatio = dict(zip(depthes, [[] for _ in range(len(depthes))]))
    crossDepthRatio = dict(zip(depthes, [[] for _ in range(len(depthes))]))
    twoDDepthRatio = dict(zip(depthes, [[] for _ in range(len(depthes))]))
    SPQPDDepthRatio = dict(zip(depthes, [[] for _ in range(len(depthes))]))
    twoDAddDepthError = np.zeros(len(depthes))
    liAddDepthError = np.zeros(len(depthes))
    crossAddDepthError = np.zeros(len(depthes))
    SPQPDAddDepthError = np.zeros(len(depthes))
    twoDAddDepthMean = np.zeros(len(depthes))
    liAddDepthMean = np.zeros(len(depthes))
    crossAddDepthMean = np.zeros(len(depthes))
    SPQPDAddDepthMean = np.zeros(len(depthes))

    for depth in depthes:
        for i in dataKey:
            if i[0][0] == depth:
                SPQPDDepthRatio[depth].append(i[1][1] / i[0][0])
                twoDDepthRatio[depth].append(i[1][3] / i[0][0])
                liDepthRatio[depth].append(i[1][5] / i[0][0])
                crossDepthRatio[depth].append(i[1][7] / i[0][0])
                # SPQPDDepthRatio[depth].append(i[1][2] / i[1][0])
                # twoDDepthRatio[depth].append(i[1][4] / i[1][0])
                # liDepthRatio[depth].append(i[1][6] / i[1][0])
                # crossDepthRatio[depth].append(i[1][8] / i[1][0])
    for i in range(len(depthes)):
        SPQPDAddDepthError[i] = np.std(np.array(SPQPDDepthRatio[depthes[i]]))
        SPQPDAddDepthMean[i] = np.mean(np.array(SPQPDDepthRatio[depthes[i]])) #* 1.8
        twoDAddDepthError[i] = np.std(np.array(twoDDepthRatio[depthes[i]]))
        twoDAddDepthMean[i] = np.mean(np.array(twoDDepthRatio[depthes[i]])) #* 1.8
        liAddDepthError[i] = np.std(np.array(liDepthRatio[depthes[i]])) 
        liAddDepthMean[i] = np.mean(np.array(liDepthRatio[depthes[i]])) #* 1.8
        crossAddDepthError[i] = np.std(np.array(crossDepthRatio[depthes[i]])) 
        crossAddDepthMean[i] = np.mean(np.array(crossDepthRatio[depthes[i]])) #* 1.8

    print('spqpd', mk_test(SPQPDAddDepthMean[:34]))
    print('twoD', mk_test(twoDAddDepthMean[:34]))
    print('cross', mk_test(liAddDepthMean[:34]))
    print('li', mk_test(crossAddDepthMean[:34]))

    plt.figure(figsize=[7,5])

    plt.errorbar(depthes, twoDAddDepthMean, fmt="o:", label='2d', yerr=twoDAddDepthError, color='orange')
    plt.errorbar(depthes, crossAddDepthMean, fmt="o:", label='li', yerr=crossAddDepthError, color='green')
    plt.errorbar(depthes, liAddDepthMean, fmt="o:", label='cross', yerr=liAddDepthError, color='red')
    plt.errorbar(depthes, SPQPDAddDepthMean, fmt="o:", label='ASIQC', yerr=SPQPDAddDepthError, color='blue')
    # plt.ylim([1.25, 1.68])
    plt.xlabel("origin depth d", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel("g_ap", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 12}, loc=2)
    plt.grid()
    ax=plt.gca() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(str(Prune.degLimit) + "pfwrtd.pdf", dpi = 300)
    plt.show()