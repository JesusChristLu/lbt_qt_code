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

if __name__ == '__main__':
    # H = 12
    # W = 6

    # H = 5
    # W = 5

    H = 2
    W = 4

    xy_crosstalk_sim_path = Path.cwd() / 'chipdata' / r"xy_crosstalk_sim.json"
    freq_data = Path.cwd() / 'chipdata' / r"qubit_freq_real.json"
    qubit_data = Path.cwd() / 'chipdata' / r"qubit_data.json"

    chip = freq_allocator.load_chip_data_from_file(H, W, qubit_data, freq_data, xy_crosstalk_sim_path)

    # MyDataset(H, W, chip, ifFake=True) 

    # errcwd = Path.cwd() / 'chipdata' / r"err_list.json"
    # freqcwd = Path.cwd() / 'chipdata' / r"freq_list.json"
    # with open(freqcwd, 'r') as f:
    #     # 读取文件内容，返回一个Python对象
    #     data = json.load(f)
        
    # with open(errcwd, 'r') as f:                                                                    
    #     # 读取文件内容，返回一个Python对象
    #     labels = json.load(f)

    freq_err_cwd = Path.cwd() / 'chipdata' / r'sq_train_data.json'
    with open(freq_err_cwd, 'r') as f:
        data_label = json.load(f)

    # q40Err = []'

    # q40Freq = []
    # q38Freq = []
    # q33Freq = []
    # q34Freq = []
    # q32Freq = []
    # q35Freq = []
    # q41Freq = []
    # q39Freq = []

    # for configuration in data_label:
    #     if data_label[configuration].get('q40', False):
    #         q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
    #         q40Freq.append(data_label[configuration]['q40']['freq'])
    #     if data_label[configuration].get('q40', False) and \
    #         data_label[configuration].get('q38', False):
    #         q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
    #         q38Freq.append(data_label[configuration]['q40']['freq'] - 
    #                     data_label[configuration]['q38']['freq'])
    #     if data_label[configuration].get('q40', False) and \
    #         data_label[configuration].get('q33', False):
    #         q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
    #         q33Freq.append(data_label[configuration]['q40']['freq'] - 
    #                     data_label[configuration]['q33']['freq'])
    #     if data_label[configuration].get('q40', False) and \
    #         data_label[configuration].get('q34', False):
    #         q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
    #         q34Freq.append(data_label[configuration]['q40']['freq'] - 
    #                     data_label[configuration]['q34']['freq'])
    #     if data_label[configuration].get('q40', False) and \
    #         data_label[configuration].get('q32', False):
    #         q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
    #         q32Freq.append(data_label[configuration]['q40']['freq'] - 
    #                     data_label[configuration]['q32']['freq'])
    #     if data_label[configuration].get('q40', False) and \
    #         data_label[configuration].get('q35', False):
    #         q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
    #         q35Freq.append(data_label[configuration]['q40']['freq'] - 
    #                     data_label[configuration]['q35']['freq'])
    #     if data_label[configuration].get('q40', False) and \
    #         data_label[configuration].get('q41', False):
    #         q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
    #         q41Freq.append(data_label[configuration]['q40']['freq'] - 
    #                     data_label[configuration]['q41']['freq'])
    #     if data_label[configuration].get('q40', False) and \
    #         data_label[configuration].get('q39', False):
    #         q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
    #         q39Freq.append(data_label[configuration]['q40']['freq'] - 
    #                     data_label[configuration]['q39']['freq'])

    # sorted_40f = sorted(zip(q40Freq, q40Err))
    # q40Freq, q40Err = zip(*sorted_40f)
    # plt.plot(q40Freq, q40Err)
    # plt.show()
    # sorted_38f = sorted(zip(q38Freq, q40Err))
    # q38Freq, q38Err = zip(*sorted_38f)
    # plt.plot(q38Freq, q38Err)
    # plt.show()
    # sorted_33f = sorted(zip(q33Freq, q40Err))
    # q33Freq, q33Err = zip(*sorted_33f)
    # plt.plot(q33Freq, q33Err)
    # plt.show()
    # sorted_34f = sorted(zip(q34Freq, q40Err))
    # q34Freq, q34Err = zip(*sorted_34f)
    # plt.plot(q34Freq, q34Err)
    # plt.show()
    # sorted_32f = sorted(zip(q32Freq, q40Err))
    # q32Freq, q32Err = zip(*sorted_32f)
    # plt.plot(q32Freq, q32Err)
    # plt.show()
    # sorted_35f = sorted(zip(q35Freq, q40Err))
    # q35Freq, q35Err = zip(*sorted_35f)
    # plt.plot(q35Freq, q35Err)
    # plt.show()
    # sorted_41f = sorted(zip(q41Freq, q40Err))
    # q41Freq, q41Err = zip(*sorted_41f)
    # plt.plot(q41Freq, q41Err)
    # plt.show()
    # sorted_39f = sorted(zip(q39Freq, q40Err))
    # q39Freq, q39Err = zip(*sorted_39f)
    # plt.plot(q39Freq, q39Err)
    # plt.show()

    qList = ['q32', 'q33', 'q34', 'q35', 'q38', 'q39', 'q40', 'q41']
    data = []
    labels = []
    for qubit in qList:
        for configuration in data_label:
            if data_label[configuration] == {}:
                continue
            data.append([])
            labels.append(1 - data_label[configuration][qubit]['fidelity'])
            for q in qList:
                data[-1].append(data_label[configuration][q]['freq'])
            data[-1].append(int(qList.index(qubit)))

    # 创建一个数据集对象，传入数据和标签

    dataset = MyDataset(H, W, chip, data=data, labels=labels)
    print('data set min max', (dataset.minErr, dataset.maxErr))
    train_size = int(0.7 * len(dataset.dataset))  # 80% 的数据用于训练
    test_size = len(dataset.dataset) - train_size  # 剩下的 20% 用于测试
    print('train size and test size', train_size, test_size)
    # quantumGNN = QuantumGNN(q_num=dataset.dataset[0][0].size()[0] - 1, xtalk_graph=dataset.xtalk_graph)

    minMedian = 1
    bestmodel = None
    minMedians = []
    for _ in range(5):
    # for _ in range(1):
        quantumGNN = QuantumGNN(q_num=dataset.dataset[0][0].size()[0] - 1, xtalk_graph=dataset.xtalk_graph)
        # 使用 random_split 函数进行拆分
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 接下来，你可以使用 train_dataset 和 test_dataset 来创建 DataLoader 进行训练和测试
        train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True)
        
        train_a_ii, train_b_ii = quantumGNN.train_model(quantumGNN, train_loader, (dataset.minErr, dataset.maxErr))
        test_a_ii, test_b_ii = quantumGNN.test_model(quantumGNN, test_loader, (dataset.minErr, dataset.maxErr))
        # 计算 c_i
        c_i = np.abs(test_a_ii - test_b_ii) / np.maximum(test_a_ii, test_b_ii)

        # 对 c_i 进行排序
        c_i_sorted = np.sort(c_i)
        c_i_median = np.median(c_i_sorted)

        minMedians.append(c_i_median)
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

    plt.figure(figsize=(6, 4))
    plt.scatter(train_a_i, train_b_i, s=1)
    # sns.kdeplot(x=train_a_i, y=train_b_i, cmap="Reds", fill=True, bw_adjust=0.5)
    # 生成一条45度的线
    # 获取x轴和y轴的最大值和最小值，以确定45度线的范围
    x = np.linspace(dataset.minErr, dataset.maxErr, 100)
    y = x
    plt.plot(x, y, color='red', linestyle='--')
    # 添加标题和标签
    # plt.title('二维散点图和45度线')
    plt.xlabel('prediction')
    plt.ylabel('measurement')

    plt.title('train')

    plt.semilogx()
    plt.semilogy()

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
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median, color='r', linestyle='--', label='median=' + str(c_i_median * 100)[:4] + '%')

    # 添加标题和标签
    plt.title('train relav')
    plt.semilogx()
    plt.xlabel('relav inacc')
    plt.ylabel('cdf')
    plt.legend()

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
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median, color='r', linestyle='--', label='median=' + str(c_i_median)[:6])

    # 添加标题和标签
    plt.title('train abs')
    plt.semilogx()
    plt.xlabel('inacc')
    plt.ylabel('cdf')
    plt.legend()

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

    plt.figure(figsize=(6, 4))
    plt.scatter(test_a_i, test_b_i, s=1)
    # sns.kdeplot(x=test_a_i, y=test_b_i, cmap="Reds", fill=True, bw_adjust=0.5)
    plt.plot(x, y, color='red', linestyle='--')
    # 添加标题和标签
    # plt.title('二维散点图和45度线')
    plt.xlabel('prediction')
    plt.ylabel('measurement')
    plt.title('test')

    plt.semilogx()
    plt.semilogy()

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
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median_relav, color='r', linestyle='--', label='median=' + str(c_i_median_relav * 100)[:4] + '%')

    # 添加标题和标签
    plt.title('test')
    plt.semilogx()
    plt.xlabel('relev inacc')
    plt.ylabel('cdf')
    plt.legend()

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
    plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
    plt.axvline(x=c_i_median, color='r', linestyle='--', label='median=' + str(c_i_median)[:6])

    # 添加标题和标签
    plt.title('test abs')
    plt.semilogx()
    plt.xlabel('inacc')
    plt.ylabel('cdf')
    plt.legend()

    # 显示图形
    plt.savefig(Path.cwd() / 'results' / 'test abs.pdf', dpi=300)
    plt.close()

    # dataset = MyDataset(H, W, chip, data=data, labels=labels)
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