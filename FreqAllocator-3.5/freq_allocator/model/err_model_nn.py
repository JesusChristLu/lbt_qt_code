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

class QuantumGNN(nn.Module):
    def __init__(self, q_num=None, xtalk_graph=None):
        super(QuantumGNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_num = q_num
        self.xtalk_graph = xtalk_graph

        self.posEmbed = nn.Linear(self.q_num, self.q_num)

        # # 目前最靠谱
        # self.fc1 = nn.Linear(self.q_num, 256)

        # self.fc4 = nn.Linear(256, 128)
        # self.fc6 = nn.Linear(128, 64)
        # self.fc8 = nn.Linear(64, 32)
        # self.fc10 = nn.Linear(32, 16)
        # self.fc12 = nn.Linear(16, 5)

        # self.fcout = nn.Linear(5, 1)

        # 目前最靠谱
        self.fc1 = nn.Linear(self.q_num, 100)

        self.fc4 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 25)
        self.fc8 = nn.Linear(25, 12)
        self.fc10 = nn.Linear(12, 6)
        self.fc12 = nn.Linear(6, 3)

        self.fcout = nn.Linear(3, 1)

    def forward(self, x):
        xPos = x[:, -1].int()
        x = x[:, :-1]

        pos = torch.zeros_like(x)
        # 使用高级索引来设置 pos[i, xPos[i]] = 1
        pos[torch.arange(pos.size(0)), xPos] = 1

        x = x + self.posEmbed(pos)
        # x = torch.cat([x, self.posEmbed(pos)], dim=1)

        # # 目前最靠谱
        # x = torch.dropout(F.leaky_relu(self.fc1(x)), train=True, p=0.5)

        # x = torch.dropout(F.leaky_relu(self.fc4(x)), train=True, p=0.2)
        # x = torch.dropout(F.leaky_relu(self.fc6(x)), train=True, p=0.2)
        # x = torch.dropout(F.leaky_relu(self.fc8(x)), train=True, p=0.2)
        # x = torch.dropout(F.leaky_relu(self.fc10(x)), train=True, p=0.1)
        # x = torch.dropout(F.leaky_relu(self.fc12(x)), train=True, p=0.0001)

        # x = self.fcout(x)

        # 目前最靠谱
        x = torch.dropout(F.leaky_relu(self.fc1(x)), train=True, p=0.1)

        x = torch.dropout(F.leaky_relu(self.fc4(x)), train=True, p=0.1)
        x = torch.dropout(F.leaky_relu(self.fc6(x)), train=True, p=0.1)
        x = torch.dropout(F.leaky_relu(self.fc8(x)), train=True, p=0.1)
        x = torch.dropout(F.leaky_relu(self.fc10(x)), train=True, p=0.01)
        x = torch.dropout(F.leaky_relu(self.fc12(x)), train=True, p=0.001)

        x = self.fcout(x)
        
        return x

    def weight_loss(self, pred, targ):
        loss = torch.mean(torch.abs(pred - targ))
        # loss = torch.mean(torch.abs(pred - targ) / (targ + 1e-6))
        # loss = 0.0001 * torch.mean(torch.abs(pred - targ) / (targ + 1e-6)) + torch.mean(torch.abs(pred - targ))
        return loss

    def train_model(self, model, data_loader, minMaxErr):
        print('train')
        
        # mseloss = nn.MSELoss()
        # l1loss = nn.L1Loss()
        # epoches = 2000
        epoches = 500

        losses = []
        optimizer = optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.001)

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
            if (not(epoch % 50) or loss < 0.002):
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