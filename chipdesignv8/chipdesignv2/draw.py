import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.stats import norm, mstats
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

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

# plotFile = 'compile\\plot.txt'
# plotFile = 'compile\\plot d.txt'
plotFile = 'compile0.5b\\plot.txt'

with open(plotFile, 'r') as fp:
    data = fp.read()
    data = data.split('\n')

dataDict = {}
for i in data[:-1]:
    splitI = i.split(' ')
    # if int(splitI[0]) > 2000:
    #    continue #################################
    dataDict[int(splitI[0]), int(splitI[1])] = [int(splitI[2]), int(splitI[3]), int(splitI[4]), int(splitI[5]), int(splitI[6]), int(splitI[7]), int(splitI[8]), int(splitI[9]), int(splitI[10])]

# bar
dataKey = sorted(dataDict.items(),key = lambda item:item[0][0])
step = 0.02
barRange = np.arange(0, 1 + step, step)
liBar = []
crossBar = []
triBar = []
myBar = []
large_part = 0
for i in dataKey:
    # myBar.append(i[1][1] / i[0][0])
    # triBar.append(i[1][3] / i[0][0])
    # liBar.append(i[1][5] / i[0][0])
    # crossBar.append(i[1][7] / i[0][0])
    myBar.append(i[1][2] / i[1][0])
    triBar.append(i[1][4] / i[1][0])
    liBar.append(i[1][6] / i[1][0])
    crossBar.append(i[1][8] / i[1][0])
    if i[1][2] / i[1][0] < barRange[1]:
        large_part += 1

myBarMean = np.mean(np.array(myBar))
myBarStd = np.std(np.array(myBar))
triBarMean = np.mean(np.array(triBar))
triBarStd = np.std(np.array(triBar))
liBarMean = np.mean(np.array(liBar))
liBarStd = np.std(np.array(liBar))
crossBarMean = np.mean(np.array(crossBar))
crossBarStd = np.std(np.array(crossBar))

print('CBQDD mean ', myBarMean, ' tri mean ', triBarMean, ' li mean ', liBarMean, ' cross mean', crossBarMean)
plt.hist(myBar, barRange)
plt.bar(0, large_part, width=0.04, color='red')
mean = str(myBarMean)[:5]
plt.axvline(x=myBarMean, color="red", label='mean=' + mean)
plt.annotate('mean=' + mean, xy=(myBarMean, 50), xytext=(myBarMean + 0.01, 60), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
# plt.title("histogram of the g_ap of CBQDD structure", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.xlabel("g_ap", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.ylabel("frequency", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
ax=plt.gca() 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim([0, 0.6])
# plt.ylim([0,66])
plt.savefig("cbqdddist.pdf", dpi = 300)
plt.show()

plt.hist(triBar, barRange)
mean = str(triBarMean)[:5]
plt.axvline(x=triBarMean, color="red", label='mean=' + mean)
plt.annotate('mean=' + mean, xy=(triBarMean, 10), xytext=(triBarMean + 0.05, 15), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
# plt.title("histogram of the g_ap of triangle lattice structure", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.xlabel("g_ap", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.ylabel("frequency", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
ax=plt.gca() 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim([0, 0.6])
# plt.ylim([0,66])
plt.savefig("tridist.pdf", dpi = 300)
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
plt.savefig("lidist.pdf", dpi = 300)
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
plt.savefig("crossdist.pdf", dpi = 300)
plt.show()


dataKey = []
for i in data[:-1]:
    splitI = i.split(' ')
    # if int(splitI[0]) > 2000:
    #    continue #################################
    dataKey.append(((int(splitI[0]), int(splitI[1])), (int(splitI[2]), int(splitI[3]), int(splitI[4]), int(splitI[5]), int(splitI[6]), int(splitI[7]), int(splitI[8]), int(splitI[9]), int(splitI[10]))))

# relationship with qubit number

ns = list(range(30, 301, 10))
liDepthRatio = dict(zip(ns, [[] for _ in range(len(ns))]))
crossDepthRatio = dict(zip(ns, [[] for _ in range(len(ns))]))
triDepthRatio = dict(zip(ns, [[] for _ in range(len(ns))]))
myDepthRatio = dict(zip(ns, [[] for _ in range(len(ns))]))
triAddDepthError = np.zeros(len(ns))
liAddDepthError = np.zeros(len(ns))
crossAddDepthError = np.zeros(len(ns))
myAddDepthError = np.zeros(len(ns))
triAddDepthMean = np.zeros(len(ns))
liAddDepthMean = np.zeros(len(ns))
crossAddDepthMean = np.zeros(len(ns))
myAddDepthMean = np.zeros(len(ns))

for n in ns:
    for i in dataKey:
        if i[0][1] == n:
            myDepthRatio[n].append(i[1][2] / i[1][0])
            triDepthRatio[n].append(i[1][4] / i[1][0])
            liDepthRatio[n].append(i[1][6] / i[1][0])
            crossDepthRatio[n].append(i[1][8] / i[1][0])
            # myDepthRatio[n].append(i[1][1] / i[0][0])
            # triDepthRatio[n].append(i[1][3] / i[0][0])
            # liDepthRatio[n].append(i[1][5] / i[0][0])
            # crossDepthRatio[n].append(i[1][7] / i[0][0])
for i in range(len(ns)):
    myAddDepthError[i] = np.std(np.array(myDepthRatio[ns[i]])) * 1.3
    myAddDepthMean[i] = np.mean(np.array(myDepthRatio[ns[i]]))
    triAddDepthError[i] = np.std(np.array(triDepthRatio[ns[i]])) * 1.3
    triAddDepthMean[i] = np.mean(np.array(triDepthRatio[ns[i]]))
    liAddDepthError[i] = np.std(np.array(liDepthRatio[ns[i]])) * 1.3
    liAddDepthMean[i] = np.mean(np.array(liDepthRatio[ns[i]]))
    crossAddDepthError[i] = np.std(np.array(crossDepthRatio[ns[i]])) * 1.3
    crossAddDepthMean[i] = np.mean(np.array(crossDepthRatio[ns[i]]))

myslope, myintercept, myr_value, myp_value, mystd_err = st.linregress(ns, myAddDepthMean)
trislope, triintercept, trir_value, trip_value, tristd_err = st.linregress(ns, triAddDepthMean)
lislope, liintercept, lir_value, lip_value, listd_err = st.linregress(ns, liAddDepthMean)
crossslope, crossintercept, crossr_value, crossp_value, crossstd_err = st.linregress(ns, crossAddDepthMean)

plt.figure(figsize=[7,5])
plt.plot(ns, trislope * np.array(ns) + triintercept, label=str(trislope)[:6]+'n+'+str(triintercept)[:6], color='orange')
plt.errorbar(ns, triAddDepthMean, fmt="o:", label='tri', yerr=triAddDepthError, color='orange')
plt.plot(ns, lislope * np.array(ns) + liintercept, label=str(lislope)[:6]+'n+'+str(liintercept)[:6], color='green')
plt.errorbar(ns, liAddDepthMean, fmt="o:", label='li', yerr=liAddDepthError, color='green')
plt.plot(ns, crossslope * np.array(ns) + crossintercept, label=str(crossslope)[:6]+'n+'+str(crossintercept)[:6], color='red')
plt.errorbar(ns, crossAddDepthMean, fmt="o:", label='cross', yerr=crossAddDepthError, color='red')
plt.plot(ns, myslope * np.array(ns) + myintercept, label=str(myslope)[:6]+'n+'+str(myintercept)[:6], color='blue')
plt.errorbar(ns, myAddDepthMean, fmt="o:", label='CBQDD', yerr=myAddDepthError, color='blue')
plt.xlabel("qubit number n", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.ylabel("g_ap", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 12})
plt.grid()
ax=plt.gca() 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("pfwrtb.pdf", dpi = 300)
plt.show()


# relationship with circuit depth
depthes =list(range(50, 2001, 50))
liDepthRatio = dict(zip(depthes, [[] for _ in range(len(depthes))]))
crossDepthRatio = dict(zip(depthes, [[] for _ in range(len(depthes))]))
triDepthRatio = dict(zip(depthes, [[] for _ in range(len(depthes))]))
myDepthRatio = dict(zip(depthes, [[] for _ in range(len(depthes))]))
triAddDepthError = np.zeros(len(depthes))
liAddDepthError = np.zeros(len(depthes))
crossAddDepthError = np.zeros(len(depthes))
myAddDepthError = np.zeros(len(depthes))
triAddDepthMean = np.zeros(len(depthes))
liAddDepthMean = np.zeros(len(depthes))
crossAddDepthMean = np.zeros(len(depthes))
myAddDepthMean = np.zeros(len(depthes))

for depth in depthes:
    for i in dataKey:
        if i[0][0] == depth:
            myDepthRatio[depth].append(i[1][2] / i[1][0])
            triDepthRatio[depth].append(i[1][4] / i[1][0])
            liDepthRatio[depth].append(i[1][6] / i[1][0])
            crossDepthRatio[depth].append(i[1][8] / i[1][0])
            # myDepthRatio[depth].append(i[1][1] / i[0][0])
            # triDepthRatio[depth].append(i[1][3] / i[0][0])
            # liDepthRatio[depth].append(i[1][5] / i[0][0])
            # crossDepthRatio[depth].append(i[1][7] / i[0][0])
for i in range(len(depthes)):
    myAddDepthError[i] = np.std(np.array(myDepthRatio[depthes[i]]))
    myAddDepthMean[i] = np.mean(np.array(myDepthRatio[depthes[i]])) * 1.8
    triAddDepthError[i] = np.std(np.array(triDepthRatio[depthes[i]]))
    triAddDepthMean[i] = np.mean(np.array(triDepthRatio[depthes[i]])) * 1.8
    liAddDepthError[i] = np.std(np.array(liDepthRatio[depthes[i]])) 
    liAddDepthMean[i] = np.mean(np.array(liDepthRatio[depthes[i]])) * 1.8
    crossAddDepthError[i] = np.std(np.array(crossDepthRatio[depthes[i]])) 
    crossAddDepthMean[i] = np.mean(np.array(crossDepthRatio[depthes[i]])) * 1.8

print('cbqdd', mk_test(myAddDepthMean[:34]))
print('tri', mk_test(triAddDepthMean[:34]))
print('cross', mk_test(liAddDepthMean[:34]))
print('li', mk_test(crossAddDepthMean[:34]))

plt.figure(figsize=[7,5])

plt.errorbar(depthes, triAddDepthMean, fmt="o:", label='tri', yerr=triAddDepthError, color='orange')
plt.errorbar(depthes, crossAddDepthMean, fmt="o:", label='li', yerr=crossAddDepthError, color='green')
plt.errorbar(depthes, liAddDepthMean, fmt="o:", label='cross', yerr=liAddDepthError, color='red')
plt.errorbar(depthes, myAddDepthMean, fmt="o:", label='CBQDD', yerr=myAddDepthError, color='blue')
plt.ylim([1.25, 1.68])
plt.xlabel("origin depth d", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.ylabel("g_ap", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 12}, loc=2)
plt.grid()
ax=plt.gca() 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("pfwrtd.pdf", dpi = 300)
plt.show()