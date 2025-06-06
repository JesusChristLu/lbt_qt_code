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

    # q40Err = []

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
        # if data_label[configuration].get('q40', False) and \
        #     data_label[configuration].get('q38', False):
        #     q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
        #     q38Freq.append(data_label[configuration]['q40']['freq'] - 
        #                 data_label[configuration]['q38']['freq'])
        # if data_label[configuration].get('q40', False) and \
        #     data_label[configuration].get('q33', False):
        #     q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
        #     q33Freq.append(data_label[configuration]['q40']['freq'] - 
        #                 data_label[configuration]['q33']['freq'])
        # if data_label[configuration].get('q40', False) and \
        #     data_label[configuration].get('q34', False):
        #     q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
        #     q34Freq.append(data_label[configuration]['q40']['freq'] - 
        #                 data_label[configuration]['q34']['freq'])
        # if data_label[configuration].get('q40', False) and \
        #     data_label[configuration].get('q32', False):
        #     q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
        #     q32Freq.append(data_label[configuration]['q40']['freq'] - 
        #                 data_label[configuration]['q32']['freq'])
        # if data_label[configuration].get('q40', False) and \
        #     data_label[configuration].get('q35', False):
        #     q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
        #     q35Freq.append(data_label[configuration]['q40']['freq'] - 
        #                 data_label[configuration]['q35']['freq'])
        # if data_label[configuration].get('q40', False) and \
        #     data_label[configuration].get('q41', False):
        #     q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
        #     q41Freq.append(data_label[configuration]['q40']['freq'] - 
        #                 data_label[configuration]['q41']['freq'])
        # if data_label[configuration].get('q40', False) and \
        #     data_label[configuration].get('q39', False):
        #     q40Err.append(1 - data_label[configuration]['q40']['fidelity'])
        #     q39Freq.append(data_label[configuration]['q40']['freq'] - 
        #                 data_label[configuration]['q39']['freq'])

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

    qList = ['q40', 'q38', 'q33', 'q34', 'q32', 'q35', 'q41', 'q39']
    data = []
    labels = []
    for qubit in qList:
        for configuration in data_label:
            if qubit in data_label[configuration]:
                data.append([])
                labels.append(1 - data_label[configuration][qubit]['fidelity'])
                for q in qList:
                    if q in data_label[configuration]:
                        data[-1].append(data_label[configuration][q]['freq'])
                    else:
                        data[-1].append(0)
                data[-1].append(int(qList.index(qubit)))

    # 创建一个数据集对象，传入数据和标签

    dataset = MyDataset(H, W, chip, data=data, labels=labels)
    train_size = int(0.8 * len(dataset.dataset))  # 80% 的数据用于训练
    test_size = len(dataset.dataset) - train_size  # 剩下的 20% 用于测试
    # 使用 random_split 函数进行拆分
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 接下来，你可以使用 train_dataset 和 test_dataset 来创建 DataLoader 进行训练和测试
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    quantumGNN = QuantumGNN(q_num=dataset.dataset[0][0].size()[0] - 1, xtalk_graph=dataset.xtalk_graph)

    quantumGNN.train_model(quantumGNN, train_loader, (dataset.minErr, dataset.maxErr))
    quantumGNN.test_model(quantumGNN, test_loader, (dataset.minErr, dataset.maxErr))

    path = Path.cwd() / 'results' / 'model.pth'

    # 保存模型的状态字典
    torch.save(quantumGNN.state_dict(), path)



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