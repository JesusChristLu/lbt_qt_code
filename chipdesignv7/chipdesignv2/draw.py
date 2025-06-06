import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plotFile = 'compile0.5b\\plot.txt'

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
    #if int(splitI[0]) > 800:
    #    continue #################################
    dataDict[int(splitI[0]), int(splitI[1])] = [int(splitI[2]), int(splitI[3]), int(splitI[4]), int(splitI[5]), int(splitI[6]), int(splitI[7]), int(splitI[8]), int(splitI[9]), int(splitI[10])]

dataKey = sorted(dataDict.items(),key = lambda item:item[0][0])

depthes = []
liAddDepth = []
crossAddDepth = []
triAddDepth = []
myAddDepth = []
liAddSwap = []
crossAddSwap = []
triAddSwap = []
myAddSwap = []

for i in dataKey:
    depthes.append(i[0][0])
    myAddDepth.append(i[1][1])
    myAddSwap.append(i[1][2])
    triAddDepth.append(i[1][3])
    triAddSwap.append(i[1][4])
    liAddDepth.append(i[1][5])
    liAddSwap.append(i[1][6])
    crossAddDepth.append(i[1][7])
    crossAddSwap.append(i[1][8])

#xlim = 100 #1750
xlim = 95 #700
#xlim = 50 #100
#xlim = 30 #60

plt.plot(depthes[:xlim], triAddSwap[:xlim], marker='*', label='tri add swap')
plt.plot(depthes[:xlim], liAddSwap[:xlim], marker='*', label='li add swap')
plt.plot(depthes[:xlim], crossAddSwap[:xlim], marker='*', label='cross add swap')
plt.plot(depthes[:xlim], myAddSwap[:xlim], marker='*', label='CBQDD add swap')
plt.xlabel("origin depth")
plt.grid()
plt.legend()
plt.show()

plt.plot(depthes, triAddSwap, marker='*', label='tri add swap')
plt.plot(depthes, liAddSwap, marker='*', label='li add swap')
plt.plot(depthes, crossAddSwap, marker='*', label='cross add swap')
plt.plot(depthes, myAddSwap, marker='*', label='CBQDD add swap')
plt.xlabel("origin depth")
plt.grid()
plt.legend()
plt.show()

# plt.plot(depthes[:xlim], myAddDepth[:xlim], marker='*', label='CBQDD add depth')
# plt.plot(depthes[:xlim], triAddDepth[:xlim], marker='*', label='tri add depth')
# plt.plot(depthes[:xlim], liAddDepth[:xlim], marker='*', label='li add depth')
# plt.plot(depthes[:xlim], crossAddDepth[:xlim], marker='*', label='cross add depth')
# plt.xlabel("origin depth")
# plt.grid()
# plt.legend()
# plt.show()

# plt.plot(depthes, myAddDepth, marker='*', label='CBQDD add depth')
# plt.plot(depthes, triAddDepth, marker='*', label='tri add depth')
# plt.plot(depthes, liAddDepth, marker='*', label='li add depth')
# plt.plot(depthes, crossAddDepth, marker='*', label='cross add depth')
# plt.xlabel("origin depth")
# plt.grid()
# plt.legend()
# plt.show()


dataKey = sorted(dataDict.items(),key = lambda item:item[0][1])

ns = []
liAddDepth = []
crossAddDepth = []
triAddDepth = []
myAddDepth = []
liAddSwap = []
crossAddSwap = []
triAddSwap = []
myAddSwap = []

liAddDepthError = []
crossAddDepthError = []
triAddDepthError = []
myAddDepthError = []
liAddSwapError = []
crossAddSwapError = []
triAddSwapError = []
myAddSwapError = []

templiAddDepth = []
tempcrossAddDepth = []
temptriAddDepth = []
tempmyAddDepth = []
templiAddSwap = []
tempcrossAddSwap = []
temptriAddSwap = []
tempmyAddSwap = []

for i in dataKey:
    if ns == []:
        ns.append(i[0][1])
        templiAddDepth = [i[1][1]]
        tempcrossAddDepth = [i[1][2]]
        temptriAddDepth = [i[1][3]]
        tempmyAddDepth = [i[1][4]]
        templiAddSwap = [i[1][5]]
        tempcrossAddSwap = [i[1][6]]
        temptriAddSwap = [i[1][7]]
        tempmyAddSwap = [i[1][8]]
    elif not (i[0][1] in ns):
        ns.append(i[0][1])
        myAddDepth.append(np.mean(np.array(tempmyAddDepth)))
        myAddSwap.append(np.mean(np.array(tempmyAddSwap)))
        triAddDepth.append(np.mean(np.array(temptriAddDepth)))
        triAddSwap.append(np.mean(np.array(temptriAddSwap)))
        liAddDepth.append(np.mean(np.array(templiAddDepth)))
        liAddSwap.append(np.mean(np.array(templiAddSwap)))
        crossAddDepth.append(np.mean(np.array(tempcrossAddDepth)))
        crossAddSwap.append(np.mean(np.array(tempcrossAddSwap)))

        liAddDepthError.append(np.std(np.array(tempmyAddDepth)))
        crossAddDepthError.append(np.std(np.array(tempmyAddDepth)))
        triAddDepthError.append(np.std(np.array(tempmyAddDepth)))
        myAddDepthError.append(np.std(np.array(tempmyAddDepth)))
        liAddSwapError.append(np.std(np.array(tempmyAddDepth)))
        crossAddSwapError.append(np.std(np.array(tempmyAddDepth)))
        triAddSwapError.append(np.std(np.array(tempmyAddDepth)))
        myAddSwapError.append(np.std(np.array(tempmyAddDepth)))

        templiAddDepth = [i[1][1]]
        tempcrossAddDepth = [i[1][2]]
        temptriAddDepth = [i[1][3]]
        tempmyAddDepth = [i[1][4]]
        templiAddSwap = [i[1][5]]
        tempcrossAddSwap = [i[1][6]]
        temptriAddSwap = [i[1][7]]
        tempmyAddSwap = [i[1][8]]
    else:
        tempmyAddDepth.append(i[1][1])
        tempmyAddSwap.append(i[1][2])
        temptriAddDepth.append(i[1][3])
        temptriAddSwap.append(i[1][4])
        templiAddDepth.append(i[1][5])
        templiAddSwap.append(i[1][6])
        tempcrossAddDepth.append(i[1][7])
        tempcrossAddSwap.append(i[1][8])

myAddDepth.append(np.mean(np.array(tempmyAddDepth)))
myAddSwap.append(np.mean(np.array(tempmyAddSwap)))
triAddDepth.append(np.mean(np.array(temptriAddDepth)))
triAddSwap.append(np.mean(np.array(temptriAddSwap)))
liAddDepth.append(np.mean(np.array(templiAddDepth)))
liAddSwap.append(np.mean(np.array(templiAddSwap)))
crossAddDepth.append(np.mean(np.array(tempcrossAddDepth)))
crossAddSwap.append(np.mean(np.array(tempcrossAddSwap)))

liAddDepthError.append(np.std(np.array(tempmyAddDepth)))
crossAddDepthError.append(np.std(np.array(tempmyAddDepth)))
triAddDepthError.append(np.std(np.array(tempmyAddDepth)))
myAddDepthError.append(np.std(np.array(tempmyAddDepth)))
liAddSwapError.append(np.std(np.array(tempmyAddDepth)))
crossAddSwapError.append(np.std(np.array(tempmyAddDepth)))
triAddSwapError.append(np.std(np.array(tempmyAddDepth)))
myAddSwapError.append(np.std(np.array(tempmyAddDepth)))

xlim = 6

plt.errorbar(ns[:xlim], triAddSwap[:xlim], fmt="o:", label='tri add swap', yerr=triAddSwapError[:xlim])
plt.errorbar(ns[:xlim], liAddSwap[:xlim], fmt="o:", label='li add swap', yerr=liAddSwapError[:xlim])
plt.errorbar(ns[:xlim], crossAddSwap[:xlim], fmt="o:", label='cross add swap', yerr=crossAddSwapError[:xlim])
plt.errorbar(ns[:xlim], myAddSwap[:xlim], fmt="o:", label='CBQDD add swap', yerr=myAddSwapError[:xlim])
plt.xlabel("origin qubit number")
plt.grid()
plt.legend()
plt.show()


plt.errorbar(ns, triAddSwap, fmt="o:", label='tri add swap', yerr=triAddSwapError)
plt.errorbar(ns, liAddSwap, fmt="o:", label='li add swap', yerr=liAddSwapError)
plt.errorbar(ns, crossAddSwap, fmt="o:", label='cross add swap', yerr=crossAddSwapError)
plt.errorbar(ns, myAddSwap, fmt="o:", label='CBQDD add swap', yerr=myAddSwapError)
plt.xlabel("origin qubit number")
plt.grid()
plt.legend()
plt.show()

# xlim = 6
# plt.errorbar(ns[:xlim], myAddDepth[:xlim], fmt="o:", label='CBDD add depth', yerr=myAddDepthError[:xlim])
# plt.errorbar(ns[:xlim], triAddDepth[:xlim], fmt="o:", label='tri add depth', yerr=triAddDepth[:xlim])
# plt.errorbar(ns[:xlim], liAddDepth[:xlim], fmt="o:", label='li add depth', yerr=liAddDepthError[:xlim])
# plt.errorbar(ns[:xlim], crossAddDepth[:xlim], fmt="o:", label='cross add depth', yerr=crossAddDepthError[:xlim])
# plt.xlabel("origin qubit number")
# plt.grid()
# plt.legend()
# plt.show()

# plt.errorbar(ns, myAddDepth, fmt="o:", label='CBDD add depth', yerr=myAddDepthError)
# plt.errorbar(ns, triAddDepth, fmt="o:", label='tri add depth', yerr=triAddDepthError)
# plt.errorbar(ns, liAddDepth, fmt="o:", label='li add depth', yerr=liAddDepthError)
# plt.errorbar(ns, crossAddDepth, fmt="o:", label='cross add depth', yerr=crossAddDepthError)
# plt.xlabel("origin qubit number")
# plt.grid()
# plt.legend()
# plt.show()

occupyFile = 'compile0.5b\\' + 'qft_16' + 'occupy.txt' #######################################################
CBDD = []
origin = []
with open(occupyFile, 'r') as fp:
    data = fp.read()
    data = data.split('\n')
    for i in data[:-1]:
        i = i.split(' ')
        CBDD.append(float(i[0]))
        origin.append(float(i[1]))


bit_list = range(len(CBDD))

x = list(range(len(CBDD)))
total_width, n = 0.8, 2
width = total_width / n
plt.bar(x, CBDD, width=width, label='CBQDD', fc='b')
for i in range(len(x)):
    x[i] += width
plt.bar(x, origin, width=width, label='origin', tick_label=bit_list, fc='g')
plt.xlabel('qubit')
plt.ylabel('gate load')
plt.legend()
plt.show()