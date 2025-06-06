from pathlib import Path
import numpy as np
import json
from freq_allocator.model.data_gen import MyDataset
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch
import matplotlib.pyplot as plt
import networkx as nx
import freq_allocator
from freq_allocator.model.err_model_nn import QuantumGNN
from freq_allocator.dataloader.load_chip import max_Algsubgraph
from freq_allocator.model.err_model import train_a_model, test_a_model
import re
from copy import deepcopy

if __name__ == '__main__':
    H = 5
    W = 5

    # H = 2
    # W = 4

    # czList = ['q32q33', 'q32q38', 'q33q34', 'q33q39', 'q34q35', 'q34q40', 'q35q41', 'q38q39', 'q39q40', 'q40q41']
    # qList = ['q32', 'q33', 'q34', 'q35', 'q38', 'q39', 'q40', 'q41']
    # qDict = {(0, 0) : 'q32', (0, 1) : 'q33', (0, 2) : 'q34', (0, 3) : 'q35', (1, 0) : 'q38', (1, 1) : 'q39', (1, 2) : 'q40', (1, 3) : 'q41'}

    qList = ['q0', 'q1', 'q2', 'q3', 'q4',
             'q6', 'q7', 'q8', 'q9', 'q10',
             'q12', 'q13', 'q14', 'q15', 'q16',
             'q18', 'q19', 'q20', 'q21', 'q22',
             'q24', 'q25', 'q26', 'q27', 'q28']
    qDict = dict(zip(nx.grid_2d_graph(5, 5).nodes, qList))

    xy_crosstalk_sim_path = Path.cwd() / 'chipdata' / r"xy_crosstalk_sim.json"
    freq_data = Path.cwd() / 'chipdata' / r"qubit_freq_real.json"
    qubit_data = Path.cwd() / 'chipdata' / r"qubit_data.json"

    chip = freq_allocator.load_chip_data_from_file(H, W, qubit_data, freq_data, xy_crosstalk_sim_path, loc=qDict)

    # faka data

    dataset = MyDataset(H, W, chip, ifFake=True)           
    czList = [node for node in MyDataset.gen_xtalk_graph(chip).nodes if not(node in qList)]

    errcwd = Path.cwd() / 'chipdata' / r"err_list.json"
    freqcwd = Path.cwd() / 'chipdata' / r"freq_list.json"
    with open(freqcwd, 'r') as f:
        # 读取文件内容，返回一个Python对象
        freqs = json.load(f)
        
    with open(errcwd, 'r') as f:                                                                    
        # 读取文件内容，返回一个Python对象
        errs = json.load(f)

    gList = qList + czList
    data = []
    labels = []
    for qubit in gList:
        for configuration in freqs:
            data.append((np.array(configuration)).tolist())
            data[-1].append(int(gList.index(qubit)))
            labels.append(errs[freqs.index(configuration)][gList.index(qubit)])

    # # real data
    # freq_err_cwd = Path.cwd() / 'chipdata' / r'sq_train_data.json'
    # with open(freq_err_cwd, 'r') as f:
    #     data_label = json.load(f)

    # cz_err_cwd = Path.cwd() / 'chipdata' / r'collect_train.json'
    # with open(cz_err_cwd, 'r') as f:
    #     cz_data_label = json.load(f)

    # train_cz_err_cwd1 = Path.cwd() / 'twoq_train_data' / r'collect_train1.json'
    # with open(train_cz_err_cwd1, 'r') as f:
    #     train_cz_err_label1 = json.load(f)

    # train_cz_err_cwd2 = Path.cwd() / 'twoq_train_data' / r'collect_train2.json'
    # with open(train_cz_err_cwd2, 'r') as f:
    #     train_cz_err_label2 = json.load(f)

    # data = []
    # labels = []
    # for qubit in qList:
    #     for configuration in data_label:
    #         if data_label[configuration] == {}:
    #             continue
    #         data.append([data_label[configuration][q]['freq'] for q in qList] + [0 for _ in czList])
    #         data[-1].append(int(qList.index(qubit)))
    #         labels.append(1 - data_label[configuration][qubit]['fidelity'])

    # # 两比特门数据以及单比特门数据
    # czData = []
    # czLabel = []
    # # czLabel1 = []
    # for configuration in cz_data_label:
    #     if cz_data_label[configuration] == {}:
    #         continue
    #     # 匹配正则表达式
    #     singQFreq = [cz_data_label[configuration]['q_freq_dict'][q] for q in qList]
    #     singQPattern = re.compile(r'^(\d+) ParallelRBSingle')
    #     twoQIterationPattern = re.compile(r'^Two Q iteration (\d+)$')
    #     for qc in qList + czList:
    #         for iter1Key in cz_data_label[configuration]:
    #             if qc in czList:
    #                 singQLabelMatch = None
    #                 twoQInterationMatch = twoQIterationPattern.match(iter1Key)
    #             else:
    #                 singQLabelMatch = singQPattern.match(iter1Key)
    #                 twoQInterationMatch = None
    #             if singQLabelMatch == None and twoQInterationMatch == None:
    #                 continue
    #             if not(twoQInterationMatch == None):
    #                 twoQInterationMatch = twoQInterationMatch.string
    #                 parallelXEBMultiplePattern = re.compile(r'^(\d+) ParallelXEBMultiple')
    #                 for iter2Key in cz_data_label[configuration][twoQInterationMatch]:
    #                     parallelxebMatch = parallelXEBMultiplePattern.match(iter2Key)
    #                     if not(parallelxebMatch == None):
    #                         parallelxebMatch = parallelxebMatch.string
    #                         if qc in cz_data_label[configuration][twoQInterationMatch][parallelxebMatch]:
    #                             czLabel.append(max(0, 1 - cz_data_label[configuration][twoQInterationMatch][parallelxebMatch][qc]['result']['f_xeb']))
    #                             czData.append(singQFreq + [cz_data_label[configuration][twoQInterationMatch]['pair_freq_dict'][cz] 
    #                                                         if cz in cz_data_label[configuration][twoQInterationMatch][parallelxebMatch]['parallel_units'] else 0.0
    #                                                         for cz in czList])
    #                             czData[-1].append(int((qList + czList).index(qc)))
    #             else:
    #                 singQLabelMatch = singQLabelMatch.string
    #                 for iter2Key in cz_data_label[configuration][singQLabelMatch]:
    #                     if iter2Key == qc:
    #                         czLabel.append(1 - cz_data_label[configuration][singQLabelMatch][qc]['result']['fidelity'])
    #                         # czLabel1.append(1 - cz_data_label[configuration][singQLabelMatch][qc]['result']['fidelity'])
    #                         czData.append(singQFreq + [0 for _ in czList])
    #                         czData[-1].append(int((qList + czList).index(qc)))
    #                         break

    # print(np.mean(labels) * 100, np.var(labels) * 100, np.std(labels) * 100)
    # print(np.mean(czLabel) * 100, np.var(czLabel) * 100, np.std(czLabel) * 100)
    # # print(np.mean(czLabel1) * 100, np.var(czLabel1) * 100, np.std(czLabel1) * 100)

    # data = data + czData
    # labels = labels + czLabel

    # 创建一个数据集对象，传入数据和标签

    dataset = MyDataset(H, W, chip, data=data, labels=labels)
    print('data set min max', (dataset.minErr, dataset.maxErr))
    train_size = int(0.9 * len(dataset.dataset))  # 80% 的数据用于训练
    test_size = len(dataset.dataset) - train_size  # 剩下的 20% 用于测试
    print('train size and test size', train_size, test_size)

    minMedian = 1
    bestmodel = None
    minMedians = []
    for _ in range(1):
    # for _ in range(5):
        quantumGNN = QuantumGNN(q_num=dataset.dataset[0][0].size()[0] - 1, xtalk_graph=dataset.xtalk_graph)
        # 使用 random_split 函数进行拆分
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 接下来，你可以使用 train_dataset 和 test_dataset 来创建 DataLoader 进行训练和测试
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        # train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)
        
        train_a_ii, train_b_ii = quantumGNN.train_model(quantumGNN, train_loader, (dataset.minErr, dataset.maxErr))
        test_a_ii, test_b_ii = quantumGNN.test_model(quantumGNN, test_loader, (dataset.minErr, dataset.maxErr))
        # 计算 c_i
        c_i = np.abs(test_a_ii - test_b_ii) / np.maximum(test_a_ii, test_b_ii)

        # 对 c_i 进行排序
        c_i_sorted = np.sort(c_i)
        c_i_median = np.median(c_i_sorted)

        minMedians.append(c_i_median)
        print('median err', c_i_median)
        if minMedian > c_i_median:
            train_a_i = train_a_ii
            train_b_i = train_b_ii
            test_a_i = test_a_ii
            test_b_i = test_b_ii
            minMedian = c_i_median
            bestmodel = quantumGNN

    print('minMedian', minMedian)
    print('minMedians', minMedians)
    path = Path.cwd() / 'results' / 'model.pth'

    # 保存模型的状态字典
    torch.save(quantumGNN.state_dict(), path)
    
    print('min train pred', np.min(train_a_i), 'max train pred', np.max(train_a_i))
    print('min train meas', np.min(train_b_i), 'max train meas', np.max(train_b_i))

    # plt.figure(figsize=(6, 4.5))
    plt.scatter(train_a_i, train_b_i, s=1)
    # sns.kdeplot(x=train_a_i, y=train_b_i, cmap="Reds", fill=True, bw_adjust=0.5)
    # 生成一条45度的线
    # 获取x轴和y轴的最大值和最小值，以确定45度线的范围
    x = np.linspace(dataset.minErr, dataset.maxErr, 100)
    y = x
    plt.plot(x, y, color='red', linestyle='--')
    # 添加标题和标签
    # plt.title('二维散点图和45度线')
    plt.xlabel('prediction', fontsize=20)
    plt.ylabel('measurement', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.title('train', fontsize=20)

    plt.semilogx()
    plt.semilogy()
    plt.tight_layout() 

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'train.pdf', dpi=300)
    plt.close()
    
    # 计算 c_i
    c_i = np.abs(train_a_i - train_b_i) / np.maximum(train_a_i, train_b_i)

    # 对 c_i 进行排序
    c_i_sorted = np.sort(c_i)
    c_i_median = np.median(c_i_sorted)

    # 计算累积频率
    cum_freq = np.arange(1, len(c_i_sorted) + 1) / len(c_i_sorted)

    # 创建累积频率分布图
    # plt.figure(figsize=(6, 4.5))
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median, color='r', linestyle='--', label='median=' + str(c_i_median * 100)[:4] + '%')

    # 添加标题和标签
    plt.title('train relav', fontsize=20)
    plt.semilogx()
    plt.xlabel('relav inacc', fontsize=20)
    plt.ylabel('cdf', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout() 

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'train relav.pdf', dpi=300)
    plt.close()

    # 计算 c_i
    c_i = np.abs(train_a_i - train_b_i)

    # 对 c_i 进行排序
    c_i_sorted = np.sort(c_i)
    c_i_median = np.median(c_i_sorted)

    # 计算累积频率
    cum_freq = np.arange(1, len(c_i_sorted) + 1) / len(c_i_sorted)

    # 创建累积频率分布图
    # plt.figure(figsize=(6, 4.5))
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median, color='r', linestyle='--', label='median=' + str(c_i_median)[:6])

    # 添加标题和标签
    plt.title('train abs', fontsize=20)
    plt.semilogx()
    plt.xlabel('inacc', fontsize=20)
    plt.ylabel('cdf', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout() 

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'train abs.pdf', dpi=300)
    plt.close()

    print('min test pred', np.min(test_a_i), 'max test pred', np.max(test_a_i))
    print('min test meas', np.min(test_b_i), 'max test meas', np.max(test_b_i))

    # 创建一个散点图

    # 生成一条45度的线
    x = np.linspace(dataset.minErr, dataset.maxErr, 100)
    y = x
    plt.plot(x, y, color='red', linestyle='--')

    # plt.figure(figsize=(6, 4.5))
    plt.scatter(test_a_i, test_b_i, s=1)
    # sns.kdeplot(x=test_a_i, y=test_b_i, cmap="Reds", fill=True, bw_adjust=0.5)
    plt.plot(x, y, color='red', linestyle='--')
    # 添加标题和标签
    # plt.title('二维散点图和45度线')
    plt.xlabel('prediction', fontsize=20)
    plt.ylabel('measurement', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('test', fontsize=20)

    plt.semilogx()
    plt.semilogy()
    plt.tight_layout() 

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'test.pdf', dpi=300)
    plt.close()
    
    # 计算 c_i
    c_i = np.abs(test_a_i - test_b_i) / np.maximum(test_a_i, test_b_i)

    # 对 c_i 进行排序
    c_i_sorted = np.sort(c_i)
    c_i_median_relav = np.median(c_i_sorted)

    # 计算累积频率
    cum_freq = np.arange(1, len(c_i_sorted) + 1) / len(c_i_sorted)

    # 创建累积频率分布图
    # plt.figure(figsize=(6, 4.5))
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median_relav, color='r', linestyle='--', label='median=' + str(c_i_median_relav * 100)[:4] + '%')

    # 添加标题和标签
    plt.title('test relav', fontsize=20)
    plt.semilogx()
    plt.xlabel('relev inacc', fontsize=20)
    plt.ylabel('cdf', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout() 

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'test relav.pdf', dpi=300)
    plt.close()
    
    # 计算 c_i
    c_i = np.abs(test_a_i - test_b_i)

    # 对 c_i 进行排序
    c_i_sorted = np.sort(c_i)
    c_i_median = np.median(c_i_sorted)

    # 计算累积频率
    cum_freq = np.arange(1, len(c_i_sorted) + 1) / len(c_i_sorted)

    # 创建累积频率分布图
    # plt.figure(figsize=(6, 4.5))
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median, color='r', linestyle='--', label='median=' + str(c_i_median)[:6])

    # 添加标题和标签
    plt.title('test abs', fontsize=20)
    plt.semilogx()
    plt.xlabel('inacc', fontsize=20)
    plt.ylabel('cdf', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout() 

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'test abs.pdf', dpi=300)
    plt.close()

    # real data
    # data = []
    # labels = []
    # for configuration in data_label:
    #     if data_label[configuration] == {}:
    #             continue
    #     data.append([data_label[configuration][q]['freq'] for q in qList] + [0 for _ in czList])
    #     labels.append([1 - data_label[configuration][qubit]['fidelity'] for qubit in qList] + [0 for _ in czList])

    # czData = []
    # czLabel = []
    # for configuration in cz_data_label:
    #     if cz_data_label[configuration] == {}:
    #         continue
    #     # 匹配正则表达式
    #     singQFreq = [cz_data_label[configuration]['q_freq_dict'][q] for q in qList]
    #     singQPattern = re.compile(r'^(\d+) ParallelRBSingle')
    #     twoQIterationPattern = re.compile(r'^Two Q iteration (\d+)$')
    #     for iter1Key in cz_data_label[configuration]:
    #         twoQInterationMatch = twoQIterationPattern.match(iter1Key)
    #         singQLabelMatch = singQPattern.match(iter1Key)
    #         if not(twoQInterationMatch == None):
    #             twoQInterationMatch = twoQInterationMatch.string
    #             parallelXEBMultiplePattern = re.compile(r'^(\d+) ParallelXEBMultiple')
    #             # parallelXEBMultiplePattern = re.compile(r'^(\d+) ParallelXEBMultiplee')
    #             for iter2Key in cz_data_label[configuration][twoQInterationMatch]:
    #                 parallelxebMatch = parallelXEBMultiplePattern.match(iter2Key)
    #                 if not(parallelxebMatch == None):
    #                     parallelxebMatch = parallelxebMatch.string
    #                     czLabel.append([0 for _ in qList])
    #                     for qc in czList:
    #                         if qc in cz_data_label[configuration][twoQInterationMatch][parallelxebMatch]:
    #                             czLabel[-1].append(max(0, 1 - cz_data_label[configuration][twoQInterationMatch][parallelxebMatch][qc]['result']['f_xeb']))
    #                         else:
    #                             czLabel[-1].append(0)
    #                     czData.append(singQFreq + [cz_data_label[configuration][twoQInterationMatch]['pair_freq_dict'][cz] 
    #                                                 if cz in cz_data_label[configuration][twoQInterationMatch][parallelxebMatch]['parallel_units'] else 0.0
    #                                                 for cz in czList])
    #         elif not(singQLabelMatch == None):
    #             singQLabelMatch = singQLabelMatch.string
    #             czLabel.append([1 - cz_data_label[configuration][singQLabelMatch][qc]['result']['fidelity'] for qc in qList] + [0 for _ in czList])
    #             czData.append(singQFreq + [0 for _ in czList])

    # data = data + czData
    # labels = labels + czLabel

    # # fake data
    # data = []
    # labels = []
    # for configuration in freqs:
    #     data.append((np.array(configuration)).tolist())
    #     labels.append(errs[freqs.index(configuration)])

    # dataset = MyDataset(H, W, chip, data=data, labels=labels, is_a=True)
    # xtalk_graph = dataset.xtalk_graph
    # dataset = TensorDataset(torch.tensor(data), torch.tensor(labels))
    # train_size = int(0.8 * len(dataset))  # 80% 的数据用于训练
    # test_size = len(dataset) - train_size  # 剩下的 20% 用于测试
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
    # train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)
    
    # a = train_a_model(train_loader, xtalk_graph)
    # print(a)
    
    # test_a_model(a, test_loader, xtalk_graph)