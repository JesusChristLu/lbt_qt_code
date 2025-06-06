import torch
import torch.nn as nn
import numpy as np

class LorenzNet(nn.Module):
    def __init__(self):
        super(LorenzNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(16, 1)  # 隐藏层到输出层的全连接层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU作为激活函数
        x = self.fc2(x)
        return x

def lorenz_function(fi, fj, gamma):
    return (1 / np.pi) * (gamma / ((fi - fj) ** 2 + (gamma) ** 2))

# 初始化神经网络
lorenz_net = LorenzNet()

# 定义优化器和损失函数
optimizer = torch.optim.Adam(lorenz_net.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 模拟数据
fi = 1 + 10 * torch.rand(30).view(-1, 1)  # 这里是示例数据，请替换为你的数据
fj = 1 + 10 * torch.rand(30).view(-1, 1)  # 这里是示例数据，请替换为你的数据
gamma = 1.0  # 这里是示例数据，请替换为你的参数

# 训练模型
for epoch in range(10000):  # 假设训练1000次
    optimizer.zero_grad()  # 梯度清零
    output = lorenz_net(torch.cat((fi, fj), dim=1))  # 将 fi 和 fj 拼接作为网络的输入
    target = torch.tensor([lorenz_function(fi[i], fj[i], gamma) for i in range(len(fi))]).view(-1, 1)  # 计算目标值
    loss = criterion(output, target)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

    if epoch % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, 1000, loss.item()))

# 使用训练好的模型进行预测
with torch.no_grad():
    predicted = lorenz_net(torch.cat((fi, fj), dim=1))
    print("Predicted values:", predicted)