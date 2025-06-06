from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import networkx as nx
from .formula import lorentzain
from pathlib import Path

class QuantumGNN(nn.Module):
    def __init__(self, q_num=None, xtalk_graph=None):
        super(QuantumGNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_num = q_num
        self.xtalk_graph = xtalk_graph

        self.posEmbed = nn.Linear(self.q_num, self.q_num)

        self.fc1 = nn.Linear(self.q_num, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 1)

    def forward(self, x):
        xPos = x[:, -1].int()
        x = x[:, :-1]

        pos = torch.zeros_like(x)
        # 使用高级索引来设置 pos[i, xPos[i]] = 1
        pos[torch.arange(pos.size(0)), xPos] = 1

        x = x + self.posEmbed(pos)

        x = torch.dropout(F.relu(self.fc1(x)), train=True, p=0.05)
        x = torch.dropout(F.relu(self.fc2(x)), train=True, p=0.05)
        x = torch.dropout(F.relu(self.fc3(x)), train=True, p=0.05)
        x = torch.dropout(F.relu(self.fc4(x)), train=True, p=0.05)
        x = torch.dropout(F.relu(self.fc5(x)), train=True, p=0.05)

        x = F.relu(self.fc6(x))
        # x = self.fc6(x)
        return x

    def train_model(self, model, data_loader):
        print('train')
        
        mseloss = nn.MSELoss()
        epoches = 300

        losses = []
        # optimizer = optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.001)
        optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=0.001)
        # scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=500, gamma=0.1)
        # optimizer = optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.001)
        # scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=500, gamma=0.9)
        
        for epoch in range(epoches):

            lossNum = 0
            a_i = None

            for batch_data in data_loader:
                optimizer.zero_grad()
                x = batch_data[0].to(self.device)
                model = model.train().to(self.device)

                output = model(x)           
                loss = mseloss(output.squeeze(), batch_data[1].to(self.device))
                
                if a_i is None:
                    a_i = output.squeeze().cpu().detach()
                    b_i = batch_data[1].to(self.device).cpu()
                else:
                    a_i = torch.cat([a_i, output.squeeze().cpu().detach()])
                    b_i = torch.cat([b_i, batch_data[1].to(self.device).cpu()])

                lossNum += 1
                loss.backward()
                optimizer.step()
                # scheduler.step()
                
            # 计算每个epoch的平均损失
            
            if not(epoch % 10) or loss < 1e-4:
            # if not(epoch % 10) or loss < 1e-3:
                print(a_i.size()[0], a_i.min() * (0.2 - 1e-3) + 1e-3, b_i.size()[0], b_i.min() * (0.2 - 1e-3) + 1e-3)
                print(a_i.size()[0], a_i.max() * (0.2 - 1e-3) + 1e-3, b_i.size()[0], b_i.max() * (0.2 - 1e-3) + 1e-3)
                losses.append(loss.cpu().detach().numpy())
                print(f'Epoch {epoch + 1}, Loss: {loss.cpu().detach().numpy()}')
                if loss < 1e-4:
                # if loss < 1e-3:
                    break
            
        plt.plot(losses)
        plt.savefig(Path.cwd() / 'results' / 'loss.pdf', dpi=300)
        plt.close()

        a_i = a_i.numpy() * (0.2 - 1e-3) + 1e-3
        b_i = b_i.numpy() * (0.2 - 1e-3) + 1e-3
        
        # 创建一个散点图
        plt.scatter(a_i, b_i)

        # 获取x轴和y轴的最大值和最小值，以确定45度线的范围
        min_val = min(min(a_i), min(b_i))
        max_val = max(max(a_i), max(b_i))

        # 生成一条45度的线
        x = np.linspace(min_val, max_val, 100)
        y = x
        plt.plot(x, y, color='red', linestyle='--')
        plt.title('train')

        # 添加标题和标签
        # plt.title('二维散点图和45度线')
        plt.xlabel('prediction')
        plt.ylabel('measurement')

        # plt.semilogx()
        # plt.semilogy()

        # 显示图形
        plt.savefig(Path.cwd() / 'results' / 'train.pdf', dpi=300)
        plt.close()
        
        # 计算 c_i
        c_i = np.abs(a_i - b_i) / b_i

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
        c_i = np.abs(a_i - b_i)

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
            
    def test_model(self, model, data_loader):
        print('test')
        
        mseloss = nn.MSELoss()
        total_loss = 0.0
        a_i = None

        for batch_data in data_loader:
            
            x = batch_data[0].to(self.device)
            model = model.eval().to(self.device)
            
            output = model(x)
            # 计算损失
            loss = mseloss(output.squeeze(), batch_data[1].to(self.device))

            if a_i is None:
                a_i = output.squeeze().cpu().detach()
                b_i = batch_data[1].to(self.device).cpu()
            else:
                a_i = torch.cat([a_i, output.squeeze().cpu().detach()])
                b_i = torch.cat([b_i, batch_data[1].to(self.device).cpu()])

            total_loss += loss.item()
            print(loss.cpu().detach().numpy())
            
        average_loss = total_loss / len(data_loader)
        print(f'test, Average Loss: {average_loss}')

        print(a_i.size()[0], a_i.min() * (0.2 - 1e-3) + 1e-3, b_i.size()[0], b_i.min() * (0.2 - 1e-3) + 1e-3)
        print(a_i.size()[0], a_i.max() * (0.2 - 1e-3) + 1e-3, b_i.size()[0], b_i.max() * (0.2 - 1e-3) + 1e-3)

        a_i = a_i.numpy() * (0.2 - 1e-3) + 1e-3
        b_i = b_i.numpy() * (0.2 - 1e-3) + 1e-3

        # 创建一个散点图
        plt.scatter(a_i, b_i)

        # 获取x轴和y轴的最大值和最小值，以确定45度线的范围
        min_val = min(min(a_i), min(b_i))
        max_val = max(max(a_i), max(b_i))

        # 生成一条45度的线
        x = np.linspace(min_val, max_val, 100)
        y = x
        plt.plot(x, y, color='red', linestyle='--')

        # 添加标题和标签
        # plt.title('二维散点图和45度线')
        plt.xlabel('prediction')
        plt.ylabel('measurement')
        plt.title('test')

        # plt.semilogx()
        # plt.semilogy()

        # 显示图形
        plt.savefig(Path.cwd() / 'results' / 'test.pdf', dpi=300)
        plt.close()
        
        # 计算 c_i
        c_i = np.abs(a_i - b_i) / b_i

        # 对 c_i 进行排序
        c_i_sorted = np.sort(c_i)
        c_i_median = np.median(c_i_sorted)

        # 计算累积频率
        cum_freq = np.arange(1, len(c_i_sorted) + 1) / len(c_i_sorted)

        # 创建累积频率分布图
        plt.plot(c_i_sorted, cum_freq, marker='o', linestyle='-', color='blue')
        plt.axvline(x=c_i_median, color='r', linestyle='--', label='median=' + str(c_i_median * 100)[:4] + '%')

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
        c_i = np.abs(a_i - b_i)

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