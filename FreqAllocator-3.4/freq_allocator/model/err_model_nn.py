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
import seaborn as sns

# 定义每个头的多层感知机
class HeadMLP(nn.Module):
    def __init__(self, q_num):
        super(HeadMLP, self).__init__()
        self.fc1 = nn.Linear(q_num, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 4)
        
        self.fcout = nn.Linear(4, 4)
        
    def forward(self, x):
        x = torch.dropout(F.relu(self.fc1(x)), train=True, p=0.1)
        x = torch.dropout(F.relu(self.fc2(x)), train=True, p=0.005)
        x = torch.dropout(F.relu(self.fc3(x)), train=True, p=0.001)
        x = torch.dropout(F.relu(self.fc4(x)), train=True, p=0.001)
        x = torch.dropout(F.relu(self.fc5(x)), train=True, p=0.001)
        x = torch.dropout(F.relu(self.fc6(x)), train=True, p=0.001)

        x = self.fcout(x)
        return x

class QuantumGNN(nn.Module):
    def __init__(self, q_num=None, xtalk_graph=None):
        super(QuantumGNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_num = q_num
        self.xtalk_graph = xtalk_graph

        self.posEmbed = nn.Linear(self.q_num, self.q_num)

        # 目前最靠谱
        self.fc1 = nn.Linear(self.q_num, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)

        self.fcout = nn.Linear(256, 1)
        

        # # 目前最靠谱
        # self.multiHead = nn.ModuleList([HeadMLP(self.q_num) for _ in range(8)])
        # self.fc2 = nn.Linear(32, 32)

        # self.fcout = nn.Linear(32, 1)

    def forward(self, x):
        xPos = x[:, -1].int()
        x = x[:, :-1]

        pos = torch.zeros_like(x)
        # 使用高级索引来设置 pos[i, xPos[i]] = 1
        pos[torch.arange(pos.size(0)), xPos] = 1

        x = x + self.posEmbed(pos)
        # x = torch.cat([x, self.posEmbed(pos)], dim=1)

        # 目前最靠谱
        x = torch.dropout(F.relu(self.fc1(x)), train=True, p=0.1)
        x = torch.dropout(F.relu(self.fc2(x)), train=True, p=0.1)
        x = torch.dropout(F.relu(self.fc3(x)), train=True, p=0.1) + x
        x = torch.dropout(F.relu(self.fc4(x)), train=True, p=0.1) + x
        x = torch.dropout(F.relu(self.fc5(x)), train=True, p=0.1) + x
        x = torch.dropout(F.relu(self.fc6(x)), train=True, p=0.1) + x

        x = self.fcout(x)
        
        # # 目前最靠谱
        # head_outputs = [head(x) for head in self.multiHead]
        # x = torch.cat(head_outputs, dim=1)
        # x = torch.dropout(F.relu(self.fc2(x)), train=True, p=0.0001)

        # x = self.fcout(x)
        
        return x

    def weight_loss(self, pred, targ):
        # loss = torch.mean(torch.abs(pred - targ) / (targ + 1e-6))
        loss = torch.mean(torch.abs(pred - targ))
        return loss

    def train_model(self, model, data_loader, minMaxErr):
        print('train')
        
        # mseloss = nn.MSELoss()
        # l1loss = nn.L1Loss()
        # epoches = 3000
        epoches = 1000

        losses = []
        optimizer = optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.001)
        # optimizer = optim.Adam(model.parameters(), weight_decay=1e-9, lr=0.001)

        # optimizer = optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.001)
        # scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=750, gamma=0.9)
        
        for epoch in range(epoches):

            a_i = None

            for batch_data in data_loader:
                optimizer.zero_grad()
                x = batch_data[0].to(self.device)
                model = model.train().to(self.device)
                output = model(x)           
                
                if output.size()[0] == output.size()[1] == 1:
                    output = output.squeeze(-1)
                else:
                    output = output.squeeze()
                target = batch_data[1].to(self.device)
                # loss = mseloss(output, target)
                # loss = l1loss(output, target)
                loss = self.weight_loss(output, target)

                if a_i is None:
                    a_i = output.cpu().detach()
                    b_i = batch_data[1].to(self.device).cpu()
                else:
                    a_i = torch.cat([a_i, output.cpu().detach()])
                    b_i = torch.cat([b_i, batch_data[1].to(self.device).cpu()])

                loss.backward()
                optimizer.step()
                # scheduler.step()
                
            # 计算每个epoch的平均损失
            if (not(epoch % 20) or loss < 0.002):
                print(a_i.size()[0], a_i.min() * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0], b_i.size()[0], b_i.min() * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0])
                print(a_i.size()[0], a_i.max() * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0], b_i.size()[0], b_i.max() * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0])
                losses.append(loss.cpu().detach().numpy())
                print(f'Epoch {epoch + 1}, Loss: {loss.cpu().detach().numpy()}')
                if loss < 0.002 and epoch > 400:
                    break
            
        a_i = (a_i * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]).numpy()
        b_i = (b_i * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]).numpy()
        return a_i, b_i
            
    def test_model(self, model, data_loader, minMaxErr):
        print('test')
        
        # mseloss = nn.MSELoss()
        # l1loss = nn.L1Loss()
        total_loss = 0.0
        a_i = None

        for batch_data in data_loader:
            
            x = batch_data[0].to(self.device)
            model = model.eval().to(self.device)
            output = model(x)

            # 计算损失
            if output.size()[0] == output.size()[1] == 1:
                output = output.squeeze(-1)
            else:
                output = output.squeeze()
            target = batch_data[1].to(self.device)
            # loss = mseloss(output, target)
            # loss = l1loss(output, target)
            loss = self.weight_loss(output, target)

            if a_i is None:
                print(loss)
                a_i = output.cpu().detach()
                b_i = batch_data[1].to(self.device).cpu()
            else:
                print(loss)
                a_i = torch.cat([a_i, output.cpu().detach()])
                b_i = torch.cat([b_i, batch_data[1].to(self.device).cpu()])

            total_loss += loss.item()
            print(loss.cpu().detach().numpy())
            
        average_loss = total_loss / len(data_loader)
        print(f'test, Average Loss: {average_loss}')

        a_i = (a_i * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]).numpy()
        b_i = (b_i * (minMaxErr[1] - minMaxErr[0]) + minMaxErr[0]).numpy()
        return a_i, b_i