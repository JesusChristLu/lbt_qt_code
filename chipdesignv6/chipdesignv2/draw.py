import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plotFile = 'compile0.5\\plot.txt'

with open(plotFile, 'r') as fp:
    data = fp.read()
    data = data.split('\n')

depthes = []
liAddDepth = []
crossAddDepth = []
triAddDepth = []
myAddDepth = []
cxs = []
liAddSwap = []
crossAddSwap = []
triAddSwap = []
myAddSwap = []
dataDict ={}
for i in data[:-1]:
    splitI = i.split(' ')
    dataDict[int(splitI[0])] = [int(splitI[1]), int(splitI[2]), int(splitI[3]), int(splitI[4]), int(splitI[5]), int(splitI[6]), int(splitI[7]), int(splitI[8]), int(splitI[9])]

dataKey =  sorted(dataDict.items(),key = lambda item:item[0])

depthes = []
liAddDepth = []
crossAddDepth = []
triAddDepth = []
myAddDepth = []
cxs = []
liAddSwap = []
crossAddSwap = []
triAddSwap = []
myAddSwap = []

for i in dataKey:
    depthes.append(i[0])
    cxs.append(i[1][0])
    myAddDepth.append(i[1][1])
    myAddSwap.append(i[1][2])
    triAddDepth.append(i[1][3])
    triAddSwap.append(i[1][4])
    liAddDepth.append(i[1][5])
    liAddSwap.append(i[1][6])
    crossAddDepth.append(i[1][7])
    crossAddSwap.append(i[1][8])

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

#ax.plot(depthes, cxs, marker='*', label="cx")
#ax.plot(depthes, myAddDepth, marker='*', label="my add depth")
ax.plot(depthes, myAddSwap, marker='*', label="my add swap")
#ax.plot(depthes, triAddDepth, marker='*', label="tri add depth")
ax.plot(depthes, triAddSwap, marker='*', label="tri add swap")
#ax.plot(depthes, liAddDepth, marker='*', label="li add depth")
ax.plot(depthes, liAddSwap, marker='*', label="li add swap")
#ax.plot(depthes, crossAddDepth, marker='*', label="cross add depth")
ax.plot(depthes, crossAddSwap, marker='*', label="cross add swap")
axins = inset_axes(ax, width="35%", height="25%", loc='lower left',
                   bbox_to_anchor=(0.4, 0.75, 1, 1),
                   bbox_transform=ax.transAxes)

#ax.plot(depthes, cxs, marker='*', label="cx")
#ax.plot(depthes, myAddDepth, marker='*', label="my add depth")
axins.plot(depthes[0:20], myAddSwap[0:20], marker='*', label="my add swap")
#ax.plot(depthes, triAddDepth, marker='*', label="tri add depth")
axins.plot(depthes[0:20], triAddSwap[0:20], marker='*', label="tri add swap")
#ax.plot(depthes, liAddDepth, marker='*', label="li add depth")
axins.plot(depthes[0:20], liAddSwap[0:20], marker='*', label="li add swap")
#ax.plot(depthes, crossAddDepth, marker='*', label="cross add depth")
axins.plot(depthes[0:20], crossAddSwap[0:20], marker='*', label="cross add swap")
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='k', lw=1)
ax.grid()
axins.grid()
ax.legend(loc = "upper left")

#plt.xlabel("时间")
#plt.ylabel("温度")
#plt.title("标题")

plt.show()