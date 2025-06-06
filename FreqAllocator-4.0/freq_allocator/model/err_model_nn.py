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

class QuantumGNN(nn.Module):
    def __init__(self, q_num=None, xtalk_graph=None):
        super(QuantumGNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_num = q_num
        self.xtalk_graph = xtalk_graph
        self.fca1 = nn.Linear(self.q_num, 512)
        self.fca2 = nn.Linear(512, 512)
        self.fca3 = nn.Linear(512, self.q_num)
        self.fca4 = nn.Linear(self.q_num, 4)
        self.fca5 = nn.Linear(4, 4)

        self.fcb1 = nn.Linear(self.q_num, 512)
        self.fcb2 = nn.Linear(512, 512)
        self.fcb3 = nn.Linear(512, self.q_num)
        self.fcb4 = nn.Linear(self.q_num, 4)
        self.fcb5 = nn.Linear(4, 4)
        
        self.fc6 = nn.Linear(4, 1)

    def forward(self, x1):
        # x2 = deepcopy(x1)

        x1 = torch.dropout(F.leaky_relu(self.fca1(x1)), train=True, p=0.01)
        x1 = torch.dropout(F.leaky_relu(self.fca2(x1)), train=True, p=0.01)
        x1 = torch.dropout(F.leaky_relu(self.fca3(x1)), train=True, p=0.01)
        x1 = torch.dropout(F.leaky_relu(self.fca4(x1)), train=True, p=0.01)
        x1 = torch.dropout(F.leaky_relu(self.fca5(x1)), train=True, p=0.01)

        # x2 = torch.dropout(F.leaky_relu(self.fcb1(x2)), train=True, p=0.01)
        # x2 = torch.dropout(F.leaky_relu(self.fcb2(x2)), train=True, p=0.01)
        # x2 = torch.dropout(F.leaky_relu(self.fcb3(x2)), train=True, p=0.01)
        # x2 = torch.dropout(F.leaky_relu(self.fcb4(x2)), train=True, p=0.01)
        # x2 = torch.dropout(F.leaky_relu(self.fca5(x2)), train=True, p=0.01)

        # x = self.fc6(x1 + x2)

        x = self.fc6(x1)
        return x

    def train_model(self, model, data_loader, qid):
        print('train')
        
        mseloss = nn.MSELoss()
        epoches = 3000

        losses = []
        optimizer = optim.Adam(model.parameters(), weight_decay=1e-6, lr=0.01)
        scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.9)
        for epoch in range(epoches):

            lossNum = 0
            for batch_data in data_loader:
                optimizer.zero_grad()
                x = batch_data[0].to(self.device)
                model = model.train().to(self.device)

                output = model(x)           
                loss = mseloss(output.squeeze(), batch_data[1].to(self.device))
                
                # print(loss)

                lossNum += 1
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            # 计算每个epoch的平均损失
            
            if not(epoch % 10) or loss < 5e-4:
                losses.append(loss.cpu().detach().numpy())
                print(f'Epoch {epoch + 1}, Loss: {loss.cpu().detach().numpy()}')
                if loss < 5e-4:
                    break
            
        plt.plot(losses)
        plt.savefig(rf'F:\OneDrive\vs experiment\FreqAllocator-2.2\results\loss' + str(qid) + '.pdf', dpi=300)
        plt.close()

        a_i = output.squeeze().cpu().detach().numpy() * (0.0722 - 0.007) + 0.007
        b_i = batch_data[1].to(self.device).cpu().numpy() * (0.0722 - 0.007) + 0.007
        
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

        plt.semilogx()
        plt.semilogy()

        # 显示图形
        plt.savefig(rf'F:\OneDrive\vs experiment\FreqAllocator-2.2\results\train' + str(qid) + '.pdf', dpi=300)
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
        plt.savefig(rf'F:\OneDrive\vs experiment\FreqAllocator-2.2\results\train relav' + str(qid) + '.pdf', dpi=300)
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
        plt.savefig(rf'F:\OneDrive\vs experiment\FreqAllocator-2.2\results\train abs' + str(qid) + '.pdf', dpi=300)
        plt.close()
            
    def test_model(self, model, data_loader, qid):
        print('test')
        
        mseloss = nn.MSELoss()
        total_loss = 0.0
        for batch_data in data_loader:
            
            x = batch_data[0].to(self.device)
            model = model.eval().to(self.device)
            
            output = model(x)
            # 计算损失
            loss = mseloss(output.squeeze(), batch_data[1].to(self.device))

            total_loss += loss.item()
            print(loss.cpu().detach().numpy())
            
        average_loss = total_loss / len(data_loader)
        print(f'test, Average Loss: {average_loss}')

        a_i = output.squeeze().cpu().detach().numpy() * (0.0722 - 0.007) + 0.007
        b_i = batch_data[1].to(self.device).cpu().numpy() * (0.0722 - 0.007) + 0.007
        
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

        plt.semilogx()
        plt.semilogy()

        # 显示图形
        plt.savefig(rf'F:\OneDrive\vs experiment\FreqAllocator-2.2\results\test' + str(qid) + '.pdf', dpi=300)
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
        plt.savefig(rf'F:\OneDrive\vs experiment\FreqAllocator-2.2\results\test relav' + str(qid) + '.pdf', dpi=300)
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
        plt.savefig(rf'F:\OneDrive\vs experiment\FreqAllocator-2.2\results\test abs' + str(qid) + '.pdf', dpi=300)
        plt.close()