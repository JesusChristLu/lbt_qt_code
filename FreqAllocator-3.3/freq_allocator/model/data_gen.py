import os
from copy import deepcopy
import freq_allocator
from torch.utils.data import Dataset
import torch
from pathlib import Path
import numpy as np
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from freq_allocator.dataloader.load_chip import max_Algsubgraph
import networkx as nx
from multiprocessing import Pool, cpu_count
from freq_allocator.model.err_model import (
    edge_distance
)

class MyDataset(Dataset):
    # 初始化方法，传入原始数据和标签
    def __init__(self, H, W, chip, data=None, labels=None, ifFake=False, qid=0):
        self.H = H
        self.W = W
        self.chip = chip
        self.xtalk_graph = self.gen_xtalk_graph()
        if ifFake:
            self.__generate_fake_data__()
        else:
            
            # 计算每个通道的均值和标准差
            
            data = torch.tensor(data)
            data = data.squeeze()
            pos = data[:, -1]
            data = data[:, :-1]

            # 创建一个与 A 同样形状的零张量 B
            mask = torch.zeros_like(data)
            # 将 A 中非零元素对应位置设置为 1
            mask[data != 0] = 1
            
            maxFreq = np.max([np.max(self.xtalk_graph.nodes[qcq]['allow freq']) for qcq in self.xtalk_graph.nodes])
            minFreq = np.min([np.min(self.xtalk_graph.nodes[qcq]['allow freq']) for qcq in self.xtalk_graph.nodes])
            x = (data - minFreq) / (maxFreq - minFreq) * mask
            # x = data * mask

            x = torch.cat([x, pos.unsqueeze(-1)], dim=1)

            print(x)

            y = torch.tensor(labels)
            self.maxErr = torch.max(y)
            self.minErr = torch.min(y)
            y = (y - self.minErr) / (self.maxErr - self.minErr)
            
            self.dataset = TensorDataset(x, y)

    # 根据索引返回一个数据样本和对应的标签
    def __getitem__(self, index):
        xy = self.dataset[index]
        return xy
    
    def __len__(self):
        # 返回数据集的大小
        return len(self.dataset)

    def __generate_fake_data__(self):
            
        # a = [1, 1, 
        #     0.2, 10, 0.5, 10,
        #     4e-4, 1e-7, 1e-2, 1e-2, 1e-5,
        #     1, 10, 0.7, 10, 
        #     1, 
        #     1, 10, 0.7, 10]    
        
        # a = [0.1, 0.1, 
        #     0.1, 20, 0.1, 20,
        #     4e-4, 1e-7, 1e-2, 1e-2, 1e-5,  
        #     0.1, 20, 0.1, 20, 
        #     0.05, 
        #     0.1, 20, 0.1, 20]      
        
        a = [0.1, 0.1, 
            0.1, 20, 0.1, 20,
            4e-4, 1e-7, 1e-2, 1e-2, 1e-5,  
            0.1, 20, 0.1, 20, 
            0.05, 
            0.1, 20, 0.1, 20]      
        
        datLen = 300
            
        frequencysid = np.zeros((datLen, len(self.xtalk_graph.nodes)), dtype=np.int32)
        
        for node in self.xtalk_graph.nodes:
            frequencysid[:, list(self.xtalk_graph.nodes).index(node)] = np.random.choice(list(range(len(self.xtalk_graph.nodes[node]['allow freq']))), size=(datLen))

        # 将 allow freq 转换为 NumPy 数组
        allow_freq_list = [(list(self.xtalk_graph.nodes[qcq]['allow freq'])) for qcq in self.xtalk_graph.nodes]

        # 找到最长的列表长度
        max_length = max(len(lst) for lst in allow_freq_list)

        # 使用列表推导来将所有列表扩展到相同的长度，并用-1填充短列表
        lists_padded = [lst + [-1] * (max_length - len(lst)) for lst in allow_freq_list]

        # 将列表转换为NumPy数组并垂直堆叠
        allow_freq_array = np.vstack(lists_padded)

        # 初始化 x 数组
        x = np.zeros((len(frequencysid), len(self.xtalk_graph.nodes)))

        # 计算 x 数组的值
        x[:, :] = allow_freq_array[np.arange(len(self.xtalk_graph.nodes)), frequencysid] * 1e-3

        xtalk_graphs = [self.xtalk_graph] * len(frequencysid)
        targets = [list(self.xtalk_graph.nodes)] * len(frequencysid)
        aes = [a] * len(frequencysid)
        isTrain = [True] * len(frequencysid) 
        p = Pool(cpu_count())
        err = p.starmap(freq_allocator.model.err_model, zip(frequencysid, xtalk_graphs, aes, targets, isTrain))
        p.close()
        p.join()

        # err_list = err.tolist()
        err_list = np.vstack(err).tolist()
        freq_list = x.tolist()
        errcwd = Path.cwd() / 'chipdata' / r"err_list.json"
        freqcwd = Path.cwd() / 'chipdata' / r"freq_list.json"
        with open(errcwd, "w") as f:
            # 把列表写入json文件
            json.dump(err_list, f)

        with open(freqcwd, "w") as f:
            # 把列表写入json文件
            json.dump(freq_list, f)
        return x, err
    
    def gen_xtalk_graph(self):
        single_qubit_graph = deepcopy(self.chip)

        two_qubit_graph = nx.Graph()
        edges_to_remove = []
        two_qubit_graph.add_nodes_from(self.chip.edges)
        for qcq in self.chip.edges():
            if self.chip.nodes[qcq[0]]['freq_max'] > self.chip.nodes[qcq[1]]['freq_max']:
                qh, ql = qcq[0], qcq[1]
            else:
                qh, ql = qcq[1], qcq[0]
            if self.chip.nodes[qh]['freq_min'] + self.chip.nodes[qh]['anharm'] > self.chip.nodes[ql]['freq_max'] or \
                self.chip.nodes[qh]['freq_max'] + self.chip.nodes[qh]['anharm'] < self.chip.nodes[ql]['freq_min']:
                edges_to_remove.append(qcq)
            else:
                two_qubit_graph.nodes[qcq]['two tq'] = 40
                two_qubit_graph.nodes[qcq]['ql'] = ql
                two_qubit_graph.nodes[qcq]['qh'] = qh
                lb = max(self.chip.nodes[ql]['freq_min'], self.chip.nodes[qh]['freq_min'] + self.chip.nodes[qh]['anharm'])
                ub = min(self.chip.nodes[ql]['freq_max'], self.chip.nodes[qh]['freq_max'] + self.chip.nodes[qh]['anharm'])
                two_qubit_graph.nodes[qcq]['allow freq'] = np.linspace(lb, ub, np.int_(ub - lb) + 1)
        two_qubit_graph.remove_nodes_from(edges_to_remove)
        maxParallelCZs = max_Algsubgraph(self.chip)
        for maxParallelCZ in maxParallelCZs:
            qcqHaveSeen = []
            for qcq1 in maxParallelCZ:
                if qcq1 in edges_to_remove:
                    continue
                for qcq2 in maxParallelCZ:
                    if qcq2 in edges_to_remove:
                        continue
                    if qcq1 == qcq2:
                        continue
                    qcqHaveSeen.append((qcq1, qcq2))
                    if edge_distance(self.chip, qcq1, qcq2) == 1:
                        two_qubit_graph.add_edge(qcq1, qcq2)

        xtalk_graph = nx.union(single_qubit_graph, two_qubit_graph)

        for qcq in two_qubit_graph:
            for qubit in single_qubit_graph:
                if (nx.has_path(self.chip, qubit, qcq[0]) and nx.has_path(self.chip, qubit, qcq[1])) and \
                    (nx.shortest_path_length(self.chip, qubit, qcq[0]) == 1 or nx.shortest_path_length(self.chip, qubit, qcq[1]) == 1) and \
                    not(qubit in qcq):
                    xtalk_graph.add_edge(qcq, qubit)

        fixQ = []
        for node in xtalk_graph.nodes:
            if len(xtalk_graph.nodes[node]['allow freq']) <= 2:
                fixQ.append(node)
                xtalk_graph.nodes[node]['frequency'] = xtalk_graph.nodes[node]['allow freq'][0]

        return xtalk_graph