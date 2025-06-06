import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from scipy.stats import norm, mstats

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

plotFile = 'compile\\plot.txt'
# plotFile = 'compile0.5b\\plot.txt'

with open(plotFile, 'r') as fp:
    data = fp.read()
    data = data.split('\n')

dataKey = []
for i in data[:-1]:
    splitI = i.split(' ')
    dataKey.append(((int(splitI[0]), int(splitI[1])), (int(splitI[2]), int(splitI[3]), int(splitI[4]), int(splitI[5]), int(splitI[6]), int(splitI[7]), int(splitI[8]), int(splitI[9]), int(splitI[10]))))

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