from pathlib import Path
import numpy as np
import json
from freq_allocator.model.data_gen import MyDataset
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import matplotlib.pyplot as plt
import networkx as nx
import freq_allocator
from freq_allocator.model.err_model_nn import QuantumGNN
from freq_allocator.dataloader.load_chip import max_Algsubgraph

if __name__ == '__main__':
    H = 3
    W = 3
    # H = 3
    # W = 1
        
    varType = 'int'

    xy_crosstalk_sim_path = Path.cwd() / 'chipdata' / r"xy_crosstalk_sim.json"
    freq_data = Path.cwd() / 'chipdata' / r"qubit_freq_real.json"
    qubit_data = Path.cwd() / 'chipdata' / r"qubit_data.json"

    chip = freq_allocator.load_chip_data_from_file(H, W, qubit_data, freq_data, xy_crosstalk_sim_path)

    MyDataset(H, W, chip, ifFake=True) 

    errcwd = Path.cwd() / 'chipdata' / r"err_list.json"
    freqcwd = Path.cwd() / 'chipdata' / r"freq_list.json"
    with open(freqcwd, 'r') as f:
        # 读取文件内容，返回一个Python对象
        data = json.load(f)
        
    with open(errcwd, 'r') as f:
        # 读取文件内容，返回一个Python对象
        labels = json.load(f)

    # # 创建一个数据集对象，传入数据和标签

    # # 假设你已经有了一个包含所有数据的数据集对象 dataset

    for qid in range(17):
        dataset = MyDataset(H, W, chip, data=data, labels=labels, qid=qid)
        train_size = int(0.8 * len(dataset.dataset))  # 80% 的数据用于训练
        test_size = len(dataset.dataset) - train_size  # 剩下的 20% 用于测试
        # 使用 random_split 函数进行拆分
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 接下来，你可以使用 train_dataset 和 test_dataset 来创建 DataLoader 进行训练和测试
        train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

        # quantumGNN = QuantumGNN(q_num=dataset.dataset[0][0].size()[0] - 1)
        quantumGNN = QuantumGNN(q_num=dataset.dataset[0][0].size()[0], xtalk_graph=dataset.xtalk_graph)

        quantumGNN.train_model(quantumGNN, train_loader, qid=qid)

        quantumGNN.test_model(quantumGNN, test_loader, qid=qid)

        path = rf'F:\OneDrive\vs experiment\FreqAllocator-2.2\results\model' + str(qid) + '.pth'

        # 保存模型的状态字典
        torch.save(quantumGNN.state_dict(), path)