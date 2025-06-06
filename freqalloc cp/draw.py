import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# 读数据

def read_data(f):
    # with open(f, 'r', encoding='utf-8') as fp:
    with open(f, 'r') as fp:
        data = fp.read()
        data = data.split('\n')
        if '' in data:
            data.remove('')
    f1 = []

    for d in data:
        d = d.split(' ')
        # data1 = d[-1][:-1]
        data1 = d[0]
        f1.append(float(data1))
    return f1

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
        cs = plt.contour(xx, yy, mat, threshold, colors="g", linewidths=0.5) 

    plt.clabel(cs, fontsize=12, inline=True) 
    plt.contourf(xx, yy, mat, 200, cmap=plt.cm.jet)

    plt.colorbar()

    plt.xlabel(xlabel, font2)
    plt.ylabel(ylabel, font2)
    plt.title(title)
    plt.savefig(picname + '.pdf', dpi=300)

def pulse_fun(tList, pulseLen, sigma, freqWork, freqMax):
    freqList = freqMax + (freqWork - freqMax) * (1 - np.sum(sigma)) * (1 - np.cos(tList / pulseLen * (len(sigma) + 1) * 2 * np.pi)) / 2
    w = 1
    for i in sigma:
        freqList += (freqWork - freqMax) * i * (1 - np.cos(tList / pulseLen * w * 2 * np.pi)) / 2
        w += 1
    
    return freqList


if __name__ == '__main__':
    
    # pulseLen = 60
    # tList = np.arange(0, pulseLen, 0.01)
    # sigma = 0.9
    # freqWork = 4.9
    # freqMax = 7
    # plt.plot(tList, gaus_flat(tList, pulseLen, sigma, freqWork, freqMax))
    # plt.show()

    # path = os.getcwd()
    # datanames = os.listdir(path)
    # fileList = []
    # for i in datanames:
    #     if '.txt' in i:
    #         fileList.append(i)

    # anharm = {'q0' : -220 * 1e-3, 'c0' : -180 * 1e-3, 'q1' : -220 * 1e-3, 'c1' : -180 * 1e-3, 'q2' : -220 * 1e-3}


    # sFreqLow = 4.5 - 0.05 
    # sFreqHigh = 4.5 + 0.05
    # cFreqLow = 6.7
    # cFreqHigh = 7.9
    # # cFreqLow = 1.9
    # # cFreqHigh = 2.2
    # sFreqs = np.arange(sFreqLow, sFreqHigh, 0.001)
    # cFreqs = np.arange(cFreqLow, cFreqHigh, 0.02)


    # freq0 = 4.5
    # freq1 = 4.4
    # # freq1 = 4.6
    # coupler = ((0, 0), (0, 1))
    # step = 0.01
    # # omegaOns = np.arange(freq1 + 0.1, freq1 + 0.5, step)
    # # omegaOns = np.arange(freq1, freq1 + 0.3, step)
    # omegaOns = np.arange(3, 4.4, step)
    # # omegaOns = np.arange(freq0 - 0.2, freq0 - 0.1, step)
    # sigmas = np.arange(0.01, 0.7, step)
    # # sigmas = np.arange(0.4, 0.9, step)


    # 两比特门串扰

    # freq0 = 4.6
    # freq1 = 5
    # freqs = np.arange(freq0 - 0.1, freq0 + 0.1, 0.005)
    # name = 'flat gaussian two Q para noncq.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # plt.plot(freqs, np.abs(data_list), label=name)
    # name = 'flat gaussian two Q para cq.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # plt.plot(freqs, np.abs(data_list), label=name)
    # name = 'flat gaussian two Q para cq2.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # plt.plot(freqs, np.abs(data_list), label='dif rho')
    # plt.axhline(y=1e-3, xmin=0.05, xmax=0.99, color='red', linestyle='--')
    # plt.semilogy()
    # plt.xlabel('spectator freq(Ghz)', fontsize='16')
    # plt.ylabel('error', fontsize='16')
    # plt.legend()
    # plt.show()

    # 提取横向耦合

    # anharmq = -0.22

    # freq0 = 4.14
    # freqWork1 = freq0 - anharmq + 0.001
    # freqOff1 = 4.53

    # # name = 'nnshift c.txt'
    # # name = str(tqFreq)[:3] + ' single shift p.txt'
    # name = 'leakage001to100.txt'
    # # name = 'leakage011to110.txt'
    # # name = 'leakage111to021.txt'

    # sFreqs = np.arange(freq0 - 0.1, freq0 + 0.1, 0.005)
    # cFreqs = np.arange(5.5, 5.9, 0.01)

    # data_list = read_data(name)
    # Picname = name[:-4]
    # zzthreshold = [-2]
    # draw_heat_map(sFreqs, cFreqs, np.array(data_list), Picname, Picname, 'spectator freq(GHz)', 'coupler freq(GHz)', 'logabs', threshold=zzthreshold)


    # 实验数据
    freq0 = 4.1
    freq1 = 4.5

    # sFreqs = np.linspace(freq0 - 0.05, freq0 + 0.05, 60)
    # sFreqs = [4.080508474576271]
    sFreqs = [4.104237288135593]
    cFreqs = np.linspace(6.2, 7.2, 50)
    # cFreqs = [7.1]

    # sFreqs = np.linspace(freq0 - 0.05, freq0 + 0.05, 40)
    # cFreqs = np.linspace(6.2, 7.2, 40)

    # name = 'spectator.txt'
    # name = 'spectator n=1.txt'
    # name = 'spectator n=10.txt'
    # name = 'spectator long swap.txt'
    name1 = 'coupler1 n=10.txt'
    name2 = 'coupler2 n=10.txt'
    dataDict1 = dict()
    dataDict2 = dict()

    with open(name1, 'r') as fp:
        data = fp.read()
        for d in data.split('\n'):
            if d == '':
                continue
            d = d.replace('array', '')
            d = d.replace('([', '')
            d = d.replace('])', '')
            d = eval(d)
            for key in d:
                app = d[key]
                if not(key in dataDict1):
                    dataDict1[key] = [app]
                else:
                    dataDict1[key].append(app)
    with open(name2, 'r') as fp:
        data = fp.read()
        for d in data.split('\n'):
            if d == '':
                continue
            d = d.replace('array', '')
            d = d.replace('([', '')
            d = d.replace('])', '')
            d = eval(d)
            for key in d:
                app = d[key]
                if not(key in dataDict2):
                    dataDict2[key] = [app]
                else:
                    dataDict2[key].append(app)

    rho_qc = 0.029
    rho_qq = 0.0026
    anharmc = -0.1
    freq0 = (4.075 + 4.10) / 2

    geff1 = []
    geff2 = []
    for cfreq in cFreqs:
        g2c = rho_qc * np.sqrt((freq0 + 0.0125) * cfreq)
        gsc = rho_qc * np.sqrt((freq0 + 0.0125) * cfreq)
        g2s = rho_qq * np.sqrt((freq0 + 0.0125) * (freq0 + 0.0125))
        g2s1 = g2s + 0.5 * g2c * gsc * (2 / (freq0 + 0.0125 - cfreq) - 2 / (freq0 + 0.0125 + cfreq))
        geff1.append(np.abs(g2s1))
        g2c = rho_qc * np.sqrt((freq0 - 0.0125) * cfreq)
        gsc = rho_qc * np.sqrt((freq0 - 0.0125) * cfreq)
        g2s = rho_qq * np.sqrt((freq0 - 0.0125) * (freq0 - 0.0125))
        g2s2 = g2s + 0.5 * g2c * gsc * (2 / (freq0 - 0.0125 - cfreq) - 2 / (freq0 - 0.0125 + cfreq))
        geff2.append(np.abs(g2s2))
    
    
    for excitedNum in ['00111t', '00011t']:
        for key in dataDict1:
            dat1 = np.array(dataDict1[key]).reshape(len(sFreqs), len(cFreqs))
            dat2 = np.array(dataDict2[key]).reshape(len(sFreqs), len(cFreqs))
            if excitedNum in key:
                if np.abs(np.max(dat1) - np.min(dat1)) > 1e-3:
                    # plt.plot(cFreqs, geff, label='geff')
                    plt.plot(cFreqs, geff1, label='geff1')
                    plt.plot(cFreqs, geff2, label='geff2')
                    plt.plot(cFreqs, dat1[0], label='res1')
                    plt.plot(cFreqs, dat2[0], label='res2')
                    plt.semilogy()
                    plt.legend()
                    # plt.savefig(key + 'n=1' + '.pdf', dpi=300)
                    plt.savefig('cp ' + key + 'n=10' + '.pdf', dpi=300)
                    # plt.savefig(key + 'long' + '.pdf', dpi=300)
                    plt.close()

                    # draw_heat_map(sFreqs, cFreqs, dat, key + 'n=1', key + 'n=1', 'spectator freq(GHz)', 'coupler freq(GHz)', 'abs', threshold=[0.001])
                    # draw_heat_map(sFreqs, cFreqs, dat, key, key, 'spectator freq(GHz)', 'coupler freq(GHz)', 'abs', threshold=[0.001])
                    # draw_heat_map(sFreqs, repeatT, np.array(dataDict[key]), key, key, 'spectator freq(GHz)', 'cz repeat', 'abs', threshold=[0.001])
                    # draw_heat_map(tg, sFreqs, np.array(dataDict[key]), 'long swap' + key, 'long swap' + key, 't(ns)', 'spectator freq(GHz)', 'abs')
                    # plt.plot(sFreqs, dataDict[key], label=key)
        # plt.legend()
        # plt.show()

    # freq0 = 4.0
    # freq1 = 4.5

    # sFreqs = np.linspace(freq0 - 0.05, freq0 + 0.05, 20)
    # # sFreqs = [4.1]
    # cFreqs = np.linspace(6, 7, 50)
    # # cFreqs = [7.0]
    # name = 'spectator n=10.txt'

    # data = []
    # with open(name, 'r') as fp:
    #     strData = fp.read()
    #     for d in strData.split('\n'):
    #         if d == '':
    #             continue
    #         data.append(float(d))
    
    # draw_heat_map(sFreqs, cFreqs, np.array(data), picname='spectator.pdf', title='spectator', xlabel='spectator freq(GHz)', ylabel='coupler freq(GHz)', drawtype='logabs')


    # 耦合常数
    # anharmq = -0.2
    # offDetune = 0.4
    # onDetune = -anharmq
    # nearOnDetune = -anharmq - 0.01
    # freq2 = 5.1

    # freq1On = freq2 + onDetune
    # freq1Off1 = freq2 + offDetune
    # freq1Off2 = freq2 + nearOnDetune

    # # cFreqOff1Low = max([freq1Off1, freq2]) + 0.01
    # # cFreqOff2Low = max([freq1Off2, freq2]) + 0.01
    # # cFreqOnLow = max([freq1On, freq2]) + 0.01
    # # cFreqHigh = 7
    # cFreqLow = 3
    # cFreqOff1High = min([freq1Off1, freq2]) - 0.01
    # cFreqOff2High = min([freq1Off1, freq2]) - 0.01
    # cFreqOnHigh = min([freq1Off1, freq2]) - 0.01
    # # cOff1Freqs = np.arange(cFreqOff1Low, cFreqHigh, 0.02)
    # # cOff2Freqs = np.arange(cFreqOff2Low, cFreqHigh, 0.02)
    # # cOnFreqs = np.arange(cFreqOnLow, cFreqHigh, 0.02)
    # cOff1Freqs = np.arange(cFreqLow, cFreqOff1High, 0.02)
    # cOff2Freqs = np.arange(cFreqLow, cFreqOff2High, 0.02)
    # cOnFreqs = np.arange(cFreqLow, cFreqOnHigh, 0.02)
    # # rho_qcs = np.arange(0.01, 0.03, 0.001)
    # # rho_qqs = np.arange(0.001, 0.0025, 0.0001)
    # rho_qcs = np.arange(0.01, 0.04, 0.001)
    # rho_qqs = np.arange(-0.001, -0.0025, -0.0001)
    # anharmcs = np.arange(-0.1, -0.3, -0.005)
    # nameOff1 = str(freq1Off1)[:3] + ' ' + str(freq2)[:3] + ' shift11 n.txt'
    # offData1 = read_data(nameOff1)
    # offData1 = np.array(np.abs(offData1)).reshape(len(rho_qcs), len(rho_qqs), len(anharmcs), len(cOff1Freqs))
    # nameOff2 = str(freq1Off2)[:3] + ' ' + str(freq2)[:3] + ' shift11 n.txt'
    # offData2 = read_data(nameOff2)
    # offData2 = np.array(np.abs(offData2)).reshape(len(rho_qcs), len(rho_qqs), len(anharmcs), len(cOff2Freqs))
    # offWidth = np.zeros((len(rho_qcs), len(rho_qqs), len(anharmcs)))
    # nameOn = str(freq1On)[:3] + ' ' + str(freq2)[:3] + ' shift11 n.txt'
    # onData = read_data(nameOn)
    # onData = np.array(np.abs(onData)).reshape(len(rho_qcs), len(rho_qqs), len(anharmcs), len(cOnFreqs))
    # onWidth = np.zeros((len(rho_qcs), len(rho_qqs), len(anharmcs)))

    # minMaxc = np.zeros((len(rho_qcs), len(rho_qqs), len(anharmcs)))

    # for i in range(len(rho_qcs)):
    #     for j in range(len(rho_qqs)):
    #         for k in range(len(anharmcs)):
    #             od1 = []
    #             od2 = []
    #             for l in range(len(cOff1Freqs)):
    #                 if np.log10(offData1[i, j, k, l]) < -2.4:
    #                     od1.append(cOff1Freqs[l])
    #             # if len(od1) > 0:
    #                 # minMaxc[i, j, k] = max(od1)

    #             for l in range(len(cOff2Freqs)):
    #                 if np.log10(offData2[i, j, k, l]) < -2.4:
    #                     od2.append(cOff2Freqs[l])
    #             for m in od1:
    #                 for n in od2:
    #                     if np.abs(m - n) < 0.02 and m > 3 and n > 3:
    #                         offWidth[i, j, k] += 0.02
    #                         break
    #             for l in range(len(cOnFreqs)):
    #                 if np.log10(onData[i, j, k, l]) > 1.9 and np.log10(onData[i, j, k, l]) < 2.5:
    #                     onWidth[i, j, k] += 0.02

    # # m = 0
    # # draw_heat_map(rho_qcs, rho_qqs, minMaxc[:, :, m], nameOff1 + str(anharmcs[m])[:4] + 'mac c', 
    # #               nameOff1 + str(anharmcs[m])[:4] + 'max c', 'rho qc', 'rho qq', threshold=[3])

    # # for m in range(40):
    # #     for i in range(len(rho_qcs)):
    # #         for j in range(len(rho_qqs)):
    # #             if offWidth[i, j, m] > 0.2 and onWidth[i, j, m] > 0.2:
    # #                 print(m, anharmcs[m], i, rho_qcs[i], j, rho_qqs[j])

    # # plt.plot(anharmcs, offWidth[16, 7, :], label='off width')
    # # plt.plot(anharmcs, onWidth[16, 7, :], label='on width')
    # plt.plot(anharmcs, offWidth[17, 1, :], label='off width')
    # plt.plot(anharmcs, onWidth[17, 1, :], label='on width')
    # plt.legend()
    # plt.show()

    # m = 0
    # # m = 39

    # draw_heat_map(rho_qcs, rho_qqs, offWidth[:, :, m], nameOff1 + str(anharmcs[m])[:4] + 'off width', 
    #               nameOff1 + str(anharmcs[m])[:4] + 'off width', 'rho qc', 'rho qq')
    # draw_heat_map(rho_qcs, rho_qqs, onWidth[:, :, m], nameOff1 + str(anharmcs[m])[:4] + 'on width', 
    #               nameOff1 + str(anharmcs[m])[:4] + 'on width', 'rho qc', 'rho qq')
    


    # 可调耦合

    # 改变omegac和omegaq

    # freq0 = 4.1
    # freq1 = 4.5
    # sFreqs = np.linspace(freq0 - 0.5, freq0 + 0.5, 50)
    # cFreqs = np.linspace(6, 8, 50)
    # tqFreq = freq1

    # name = str(tqFreq)[:3] + ' single shift p.txt'
    # # name = str(tqFreq)[:3] + ' single shift n.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]

    # zzthreshold = [-2, -1.2, 1.5]

    # draw_heat_map(sFreqs, cFreqs, data_list, Picname, Picname, 'omega1(GHz)', 'omegac(GHZ)', 'logabs', threshold=zzthreshold)

    # 改变omegac和rho12
    # rho_qc = 0.031
    # tqFreq = 5
    # spFreq = tqFreq - 0.4
    # # cFreqLow = tqFreq + 0.01
    # # cFreqHigh = 7
    # cFreqLow = 3
    # cFreqHigh = spFreq - 0.01
    # cFreqs = np.arange(cFreqLow, cFreqHigh, 0.02)
    # rho12Low = -0.003
    # rho12High = -0.001
    # rho12s = np.arange(rho12Low, rho12High, 0.0001)

    # name = 'rho omegac' + str(np.abs(tqFreq - spFreq))[:5] + \
    #  str(rho_qc)[:5] + '.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # zzthreshold = [-2, 1.5]
    # draw_heat_map(rho12s, cFreqs, data_list, Picname, Picname, 'rho_qq', 'omegac(GHz)', 'logabs', threshold=zzthreshold)

    # 改变freq1和detune
    # freq1s = np.arange(4, 6, 0.01)
    # detunes = np.arange(0.04, 0.6, 0.01)
    # name = 'freq shift11 n.txt'
    # data_list = read_data(name)
    # coffMin = np.zeros((len(freq1s), len(detunes)))
    # coffMax = np.zeros((len(freq1s), len(detunes)))
    # coffWidth = np.zeros((len(freq1s), len(detunes)))
    # conWidth = np.zeros((len(freq1s), len(detunes)))
    # id = 0
    # for freq1 in freq1s:
    #     for detune in detunes:
    #         # freqcs = np.arange(max(freq1, freq1 - detune), 7.1, 0.02)
    #         freqcs = np.arange(2.9, min(freq1, freq1 - detune), 0.02)
    #         offC = []
    #         if np.abs(detune - 0.2) < 0.01:
    #             Onc = []
    #         for cFreq in freqcs:
    #             if np.log10(data_list[id]) < -2:
    #                 offC.append(cFreq)
    #             if np.abs(detune - 0.2) < 0.01 and np.log10(data_list[id]) > 1.5 and np.log10(data_list[id]) < 2:
    #                 Onc.append(cFreq)
    #             id += 1
    #         if len(offC) > 0:
    #             # coffMin[list(freq1s).index(freq1), list(detunes).index(detune)] = min(offC)
    #             coffMax[list(freq1s).index(freq1), list(detunes).index(detune)] = max(offC)
    #             coffWidth[list(freq1s).index(freq1), list(detunes).index(detune)] = len(offC) * 0.02
    #         if np.abs(detune - 0.2) < 0.01 and len(offC) > 0:
    #             conWidth[list(freq1s).index(freq1), list(detunes).index(detune)] = len(Onc) * 0.02
    
    # # draw_heat_map(freq1s, detunes * 1e3, coffMin, 'min c p', 'min c p', 'freq1(GHz)', 'detune(MHz)')
    # # draw_heat_map(freq1s, detunes * 1e3, coffWidth, 'off width c p', 'off width c p', 'freq1(GHz)', 'detune(MHz)')
    # # conWidth = np.max(conWidth, axis=1)
    # # plt.figure()
    # # plt.plot(freq1s, conWidth)
    # # plt.title('on width c p')
    # # plt.xlabel('freq1(GHz)')
    # # plt.ylabel('on width(GHz)')
    # # plt.show()
    # # draw_heat_map(freq1s, detunes * 1e3, coffMax, 'min c n', 'min c n', 'freq1(GHz)', 'detune(MHz)')
    # # draw_heat_map(freq1s, detunes * 1e3, coffWidth, 'off width c n', 'off width c n', 'freq1(GHz)', 'detune(MHz)')
    # conWidth = np.max(conWidth, axis=1)
    # plt.figure()
    # plt.plot(freq1s, conWidth)
    # plt.title('on width c n')
    # plt.xlabel('freq1(GHz)')
    # plt.ylabel('on width(GHz)')
    # plt.show()


    # g线
    # omega_1 = 5
    # omega_2 = 5 - 0.1
    # rho_qc = 0.02
    # rho_qq = 0.001
    # alpha_q = -0.2
    # omega_c = np.arange(6, 10, 0.02)
    # alpha_c = -0.2

    # g_12 = rho_qq * np.sqrt(omega_1 * omega_2)
    # g_1c = rho_qc * np.sqrt(omega_1 * omega_c)
    # g_2c = rho_qc * np.sqrt(omega_2 * omega_c)
    # delta_1c = omega_1 - omega_c
    # delta_2c = omega_2 - omega_c
    # delta_12 = omega_1 - omega_2
    # g = g_12 + 0.5 * g_1c * g_2c * (1 / delta_1c + 1 / delta_2c)
    # mu = g_1c * g_2c / (2 * delta_1c * delta_2c)

    # # plt.plot(omega_c, np.abs(g), label='g')
    # # plt.plot(omega_c, np.abs(mu), label='mu')
    # # plt.plot(omega_c, np.abs(g - alpha_q * mu), label='mu+g')
    # # plt.plot(omega_c, 
    # #          np.abs(4 * alpha_q / (delta_12 ** 2 - alpha_q ** 2) * (g - alpha_q * mu) ** 2 + (8 * alpha_c + 4 * alpha_q) * mu ** 2), 
    # #          label='xi')
    # plt.plot(omega_c, 
    #          4 * alpha_q / (delta_12 ** 2 - alpha_q ** 2) * (g - alpha_q * mu) ** 2, 
    #          label='xi1')
    # plt.plot(omega_c, 
    #          (8 * alpha_c + 4 * alpha_q) * mu ** 2, 
    #          label='xi2')
    # plt.axhline(y=0, xmin=0.05, xmax=0.99, color='red', linestyle='--')
    # # plt.semilogy()
    # plt.legend()
    # plt.show()

    # 线图
    # name = 'g p.txt'
    # gpList = np.log10(np.abs(read_data(name)))
    # name = 'g n.txt'
    # gnList = np.log10(np.abs(read_data(name)))
    # Picname = 'g.pdf'
    # fig, ax = plt.subplots()

    # ax.axvline(x=4.4, linestyle='--', color='gray')

    # index = np.abs(np.arange(4.33, 6.5, 0.02) - 4.4).argmin()
    # cpIntersect = np.arange(4.33, 6.5, 0.02)[index]
    # gpintersect = gpList[index]

    # ax.annotate(f"({cpIntersect:.2f}, {gpintersect:.2f})", 
    #          xy=(cpIntersect, gpintersect), xytext=(cpIntersect+25, gpintersect+40), 
    #          textcoords='offset points', ha='center', va='bottom',
    #          arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
    
    # line1 = ax.plot(np.arange(4.33, 6.5, 0.02), gpList, label='gp', color='blue')
    # ax.set_xlabel('gp c freq(GHz)')
    # ax.set_ylabel('g(MHz)')
    # ax2 = ax.twiny()

    # ax2.axvline(x=4.05, linestyle='--', color='gray')

    # index = np.abs(np.arange(2.7, 4.09, 0.02) - 4.05).argmin()
    # cnIntersect = np.arange(2.7, 4.09, 0.02)[index]
    # gnintersect = gnList[index]

    # ax2.annotate(f"({cnIntersect:.2f}, {gnintersect:.2f})", 
    #          xy=(cnIntersect, gnintersect), xytext=(cnIntersect-50, gnintersect+10), 
    #          textcoords='offset points', ha='center', va='bottom',
    #          arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
    

    # ax2.set_label('gn c freq(GHz)')
    # line2 = ax2.plot(np.arange(2.7, 4.09, 0.02), gnList, label='gn', color='red')
    # lines = line1 + line2
    # labels = [l.get_label() for l in lines]
    # ax.legend(lines, labels)
    # plt.show()

    # sFreqLow = 4.0
    # sFreqHigh = 4.2

    # cFreqLowP = 4.51
    # cFreqHighP = 6.5
    # sFreqs = np.arange(sFreqLow, sFreqHigh, 0.002)
    # cFreqsP = np.arange(cFreqLowP, cFreqHighP, 0.02)

    # name1 = 'single shift p1.txt'
    # name2 = 'single shift p2.txt'

    # dataP1 = read_data(name1)
    # dataP2 = read_data(name2)

    # dataP1 = np.array(dataP1).reshape(len(sFreqs), len(cFreqsP))
    # dataP2 = np.array(dataP2).reshape(len(sFreqs), len(cFreqsP))

    # intersectp = np.zeros(len(sFreqs))
    # for sfreq in range(len(sFreqs)):
    #     for cfreq in range(len(cFreqsP)):
    #         if np.log10(np.abs(dataP1[sfreq, cfreq])) < -2 and np.log10(np.abs(dataP2[sfreq, cfreq])) < -2:
    #             intersectp[sfreq] += 0.02

    # cFreqLowN = 2.7
    # cFreqHighN = 4.09
    # cFreqsN = np.arange(cFreqLowN, cFreqHighN, 0.02)

    # name3 = 'single shift n1.txt'
    # name4 = 'single shift n2.txt'

    # dataN1 = read_data(name3)
    # dataN2 = read_data(name4)

    # dataN1 = np.array(dataN1).reshape(len(sFreqs), len(cFreqsN))
    # dataN2 = np.array(dataN2).reshape(len(sFreqs), len(cFreqsN))

    # intersectn = np.zeros(len(sFreqs))
    # for sfreq in range(len(sFreqs)):
    #     for cfreq in range(len(cFreqsN)):
    #         if np.log10(np.abs(dataN1[sfreq, cfreq])) < -2 and np.log10(np.abs(dataN2[sfreq, cfreq])) < -2:
    #             intersectn[sfreq] += 0.02

    # plt.plot(sFreqs, intersectp, label='p off width')
    # plt.plot(sFreqs, intersectn, label='n off width')

    # plt.axvline(x=4.09, linestyle='--', color='gray')

    # index = np.abs(sFreqs - 4.09).argmin()
    # sIntersect = sFreqs[index]
    # cpIntersect = intersectp[index]
    # cnIntersect = intersectn[index]

    # plt.annotate(f"({sIntersect:.2f}, {cpIntersect:.2f})", 
    #          xy=(sIntersect, cpIntersect), xytext=(sIntersect+10, cpIntersect+40), 
    #          textcoords='offset points', ha='center', va='bottom',
    #          arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
    
    # plt.annotate(f"({sIntersect:.2f}, {cnIntersect:.2f})", 
    #          xy=(sIntersect, cnIntersect), xytext=(sIntersect-50, cnIntersect+45), 
    #          textcoords='offset points', ha='center', va='bottom',
    #          arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
    

    # plt.axvline(x=4.11, linestyle='--', color='gray')

    # index = np.abs(sFreqs - 4.11).argmin()
    # sIntersect = sFreqs[index]
    # cpIntersect = intersectp[index]
    # cnIntersect = intersectn[index]

    # plt.annotate(f"({sIntersect:.2f}, {cpIntersect:.2f})", 
    #          xy=(sIntersect, cpIntersect), xytext=(sIntersect+30, cpIntersect+60), 
    #          textcoords='offset points', ha='center', va='bottom',
    #          arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
    
    # plt.annotate(f"({sIntersect:.2f}, {cnIntersect:.2f})", 
    #          xy=(sIntersect, cnIntersect), xytext=(sIntersect+40, cnIntersect+5), 
    #          textcoords='offset points', ha='center', va='bottom',
    #          arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

    # plt.xlabel('s freq(Ghz)')
    # plt.ylabel('off freq width(Ghz)')
    # plt.title('the intersection freq width between on and off')
    # plt.legend()
    # plt.show()



    # 可调耦合线图
    # data_list = np.array([data_list]).reshape(len(sFreqs), len(cFreqs))
    # errDet = np.zeros(len(sFreqs))
    # for i in range(len(sFreqs)):
    #     for j in range(len(cFreqs)):
    #         if data_list[i, j] < 1e-3:
    #             errDet[i] += 0.02
    # plt.plot(sFreqs, errDet)
    # plt.grid()
    # plt.show()

    # 可调耦合线图
    # data_list = np.reshape(np.array(data_list), (len(sFreqs), len(cFreqs)))
    # for c in [0, 8, 20, 40, 52, 60]:
    #     d = np.abs(np.array(data_list[c]))
    #     plt.plot(cFreqs, d, label=str(round(sFreqs[c], 5))[:4])
    # plt.axhline(y=1e2, xmin=0.05, xmax=0.9, color='green', linestyle='--')
    # plt.axhline(y=1e-2, xmin=0.05, xmax=0.9, color='red', linestyle='--')
    # plt.semilogy()
    # plt.legend()
    # plt.show()

    # 参数数目与时间
    # a = np.arange(4.2, 4.5, 0.005)
    # name = 'two Q para detune 60.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # zzthreshold = -2
    # fthreshold = -2.5
    # data_list = np.array([data_list]).reshape(5, 28)
    # data0_list = np.abs(data_list[0])
    # data0std = np.var(data0_list)
    # data0mean = np.mean(data0_list)
    # data1_list = np.abs(data_list[1])
    # data1std = np.var(data1_list)
    # data1mean = np.mean(data1_list)
    # data2_list = np.abs(data_list[2])
    # data2std = np.var(data2_list)
    # data2mean = np.mean(data2_list)
    # data3_list = np.abs(data_list[3])
    # data3std = np.var(data3_list)
    # data3mean = np.mean(data3_list)
    # data4_list = np.abs(data_list[4])
    # data4std = np.var(data4_list)
    # data4mean = np.mean(data4_list)
    # plt.errorbar([1,2,3,4,5], [data0mean, data1mean, data2mean, data3mean, data4mean], 
    # [data0std, data1std,  data2std, data3std, data4std], capsize=3, capthick=2, label='60ns')
    # name = 'two Q para detune 65.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # zzthreshold = -2
    # fthreshold = -2.5
    # data_list = np.array([data_list]).reshape(5, 28)
    # data0_list = np.abs(data_list[0])
    # data0std = np.var(data0_list)
    # data0mean = np.mean(data0_list)
    # data1_list = np.abs(data_list[1])
    # data1std = np.var(data1_list)
    # data1mean = np.mean(data1_list)
    # data2_list = np.abs(data_list[2])
    # data2std = np.var(data2_list)
    # data2mean = np.mean(data2_list)
    # data3_list = np.abs(data_list[3])
    # data3std = np.var(data3_list)
    # data3mean = np.mean(data3_list)
    # data4_list = np.abs(data_list[4])
    # data4std = np.var(data4_list)
    # data4mean = np.mean(data4_list)
    # plt.errorbar([1,2,3,4,5], [data0mean, data1mean, data2mean, data3mean, data4mean], 
    # [data0std, data1std,  data2std, data3std, data4std], capsize=3, capthick=2, label='65ns')
    # name = 'two Q para detune 70.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # zzthreshold = -2
    # fthreshold = -2.5
    # data_list = np.array([data_list]).reshape(5, 28)
    # data0_list = np.abs(data_list[0])
    # data0std = np.var(data0_list)
    # data0mean = np.mean(data0_list)
    # data1_list = np.abs(data_list[1])
    # data1std = np.var(data1_list)
    # data1mean = np.mean(data1_list)
    # data2_list = np.abs(data_list[2])
    # data2std = np.var(data2_list)
    # data2mean = np.mean(data2_list)
    # data3_list = np.abs(data_list[3])
    # data3std = np.var(data3_list)
    # data3mean = np.mean(data3_list)
    # data4_list = np.abs(data_list[4])
    # data4std = np.var(data4_list)
    # data4mean = np.mean(data4_list)
    # plt.errorbar([1,2,3,4,5], [data0mean, data1mean, data2mean, data3mean, data4mean], 
    # [data0std, data1std,  data2std, data3std, data4std], capsize=3, capthick=2, label='70ns')
    # name = 'two Q para detune 75.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # zzthreshold = -2
    # fthreshold = -2.5
    # data_list = np.array([data_list]).reshape(5, 28)
    # data0_list = np.abs(data_list[0])
    # data0std = np.var(data0_list)
    # data0mean = np.mean(data0_list)
    # data1_list = np.abs(data_list[1])
    # data1std = np.var(data1_list)
    # data1mean = np.mean(data1_list)
    # data2_list = np.abs(data_list[2])
    # data2std = np.var(data2_list)
    # data2mean = np.mean(data2_list)
    # data3_list = np.abs(data_list[3])
    # data3std = np.var(data3_list)
    # data3mean = np.mean(data3_list)
    # data4_list = np.abs(data_list[4])
    # data4std = np.var(data4_list)
    # data4mean = np.mean(data4_list)
    # plt.errorbar([1,2,3,4,5], [data0mean, data1mean, data2mean, data3mean, data4mean], 
    # [data0std, data1std,  data2std, data3std, data4std], capsize=3, capthick=2, label='75ns')
    # name = 'two Q para detune 80.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # zzthreshold = -2
    # fthreshold = -2.5
    # data_list = np.array([data_list]).reshape(5, 28)
    # data0_list = np.abs(data_list[0])
    # data0std = np.var(data0_list)
    # data0mean = np.mean(data0_list)
    # data1_list = np.abs(data_list[1])
    # data1std = np.var(data1_list)
    # data1mean = np.mean(data1_list)
    # data2_list = np.abs(data_list[2])
    # data2std = np.var(data2_list)
    # data2mean = np.mean(data2_list)
    # data3_list = np.abs(data_list[3])
    # data3std = np.var(data3_list)
    # data3mean = np.mean(data3_list)
    # data4_list = np.abs(data_list[4])
    # data4std = np.var(data4_list)
    # data4mean = np.mean(data4_list)
    # plt.errorbar([1,2,3,4,5], [data0mean, data1mean, data2mean, data3mean, data4mean], 
    # [data0std, data1std,  data2std, data3std, data4std], capsize=3, capthick=2, label='80ns')
    # name = 'two Q para detune 85.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # zzthreshold = -2
    # fthreshold = -2.5
    # data_list = np.array([data_list]).reshape(5, 28)
    # data0_list = np.abs(data_list[0])
    # data0std = np.var(data0_list)
    # data0mean = np.mean(data0_list)
    # data1_list = np.abs(data_list[1])
    # data1std = np.var(data1_list)
    # data1mean = np.mean(data1_list)
    # data2_list = np.abs(data_list[2])
    # data2std = np.var(data2_list)
    # data2mean = np.mean(data2_list)
    # data3_list = np.abs(data_list[3])
    # data3std = np.var(data3_list)
    # data3mean = np.mean(data3_list)
    # data4_list = np.abs(data_list[4])
    # data4std = np.var(data4_list)
    # data4mean = np.mean(data4_list)
    # plt.errorbar([1,2,3,4,5], [data0mean, data1mean, data2mean, data3mean, data4mean], 
    # [data0std, data1std,  data2std, data3std, data4std], capsize=3, capthick=2, label='85ns')
    # name = 'two Q para detune 90.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # zzthreshold = -2
    # fthreshold = -2.5
    # data_list = np.array([data_list]).reshape(5, 28)
    # data0_list = np.abs(data_list[0])
    # data0std = np.var(data0_list)
    # data0mean = np.mean(data0_list)
    # data1_list = np.abs(data_list[1])
    # data1std = np.var(data1_list)
    # data1mean = np.mean(data1_list)
    # data2_list = np.abs(data_list[2])
    # data2std = np.var(data2_list)
    # data2mean = np.mean(data2_list)
    # data3_list = np.abs(data_list[3])
    # data3std = np.var(data3_list)
    # data3mean = np.mean(data3_list)
    # data4_list = np.abs(data_list[4])
    # data4std = np.var(data4_list)
    # data4mean = np.mean(data4_list)
    # plt.errorbar([1,2,3,4,5], [data0mean, data1mean, data2mean, data3mean, data4mean], 
    # [data0std, data1std,  data2std, data3std, data4std], capsize=3, capthick=2, label='90ns')
   

    # name = 'two Q para detune 60.txt'
    # data_list = read_data(name)
    # Picname = name[:-4]
    # zzthreshold = -2
    # fthreshold = -2.5
    # data_list = np.array([data_list]).reshape(10, 28)
    # data0_list = np.abs(data_list[0])
    # data0std = np.var(data0_list)
    # data0mean = np.mean(data0_list)
    # data1_list = np.abs(data_list[1])
    # data1std = np.var(data1_list)
    # data1mean = np.mean(data1_list)
    # data2_list = np.abs(data_list[2])
    # data2std = np.var(data2_list)
    # data2mean = np.mean(data2_list)
    # data3_list = np.abs(data_list[3])
    # data3std = np.var(data3_list)
    # data3mean = np.mean(data3_list)
    # data4_list = np.abs(data_list[4])
    # data4std = np.var(data4_list)
    # data4mean = np.mean(data4_list)
    # data5_list = np.abs(data_list[5])
    # data5std = np.var(data5_list)
    # data5mean = np.mean(data5_list)
    # data6_list = np.abs(data_list[6])
    # data6std = np.var(data6_list)
    # data6mean = np.mean(data6_list)
    # data7_list = np.abs(data_list[7])
    # data7std = np.var(data7_list)
    # data7mean = np.mean(data7_list)
    # data8_list = np.abs(data_list[8])
    # data8std = np.var(data8_list)
    # data8mean = np.mean(data8_list)
    # data9_list = np.abs(data_list[9])
    # data9std = np.var(data9_list)
    # data9mean = np.mean(data9_list)
    # plt.errorbar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
    # [data0mean, data1mean, data2mean, data3mean, data4mean, data5mean, data6mean, data7mean, data8mean, data9mean], 
    # [data0std, data1std,  data2std, data3std, data4std, data5std, data6std, data7std, data8std, data9std], 
    # capsize=3, capthick=2, label='60ns')
    # plt.semilogy()

    # data0_list = np.abs(data_list[0])
    
    # data1_list = np.abs(data_list[1])
    
    # data2_list = np.abs(data_list[2])
    
    # data3_list = np.abs(data_list[3])
    
    # data4_list = np.abs(data_list[4])

    # plt.plot(np.arange(4.497, 4.503, 0.00001), data_list[0], label='1')
    # plt.plot(np.arange(4.497, 4.503, 0.00001), data_list[1], label='2')
    # plt.plot(np.arange(4.497, 4.503, 0.00001), data_list[2], label='3')
    # plt.plot(np.arange(4.497, 4.503, 0.00001), data_list[3], label='4')
    # plt.plot(np.arange(4.2, 4.5, 0.005), data_list[4], label='5')

    # plt.legend()
    # plt.semilogy()
    # plt.savefig(name[:-4] + '.pdf', dpi=300)
    # plt.close()

    # tlist = np.arange(0, 60, 0.01)
    # plt.plot(tlist, pulse_fun(tlist, 60, 
    # [1.0053, -0.02142854782253074, -0.012, 0.052], 
    # 3.6763631033422612, 2.1))
    # plt.show()