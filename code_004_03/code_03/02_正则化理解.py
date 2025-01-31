import torch
import torch.nn as nn
from torch import optim


# 定义一个简单的神经网络
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # 创建一个包含多个层的序列模型
        self.model = nn.Sequential(
            nn.Linear(19, 128),  # 输入层：19个特征，输出层：128个节点
            nn.ReLU(),  # 激活函数：ReLU
            nn.Linear(128, 256),  # 隐藏层：128个输入，256个输出
            nn.ReLU(),  # 激活函数：ReLU
            nn.Linear(256, 3)  # 输出层：256个输入，3个输出
        )

    def forward(self, x):
        # 前向传播过程
        return self.model(x)


# 自定义L1正则化损失函数
class RegL1Loss(nn.Module):
    def __init__(self, lam, params):
        super(RegL1Loss, self).__init__()
        self.lam = lam  # 正则化系数
        self.params = list(params)  # 需要正则化的参数列表
        self.n_param = len(self.params)  # 参数数量

    def forward(self):
        # 计算所有参数的L1范数
        ls = 0.0
        for param in self.params:
            ls += torch.mean(torch.abs(param))  # 每个参数的平均绝对值
        # 返回加权后的L1范数
        return self.lam * ls / self.n_param


if __name__ == '__main__':
    # 创建网络实例
    net = Network()

    # 打印网络的第一个参数及其名称
    print(next(net.named_parameters()))

    # 创建MSE损失函数
    loss_fn = nn.MSELoss()

    # 创建L1正则化损失函数实例
    reg_l1_loss_fn = RegL1Loss(lam=0.1, params=net.parameters())

    # 求解L1正则化损失
    _l1_loss = reg_l1_loss_fn()  # 注意这里没有传入输入和目标，因为我们只关心参数的L1范数
    print(_l1_loss)

    # 创建SGD优化器，并设置weight_decay为L2正则化系数
    opt = optim.SGD(params=net.parameters(), lr=0.005, weight_decay=0.5)

    # 注释掉的部分，如果要计算总的损失（实际损失+正则化损失），可以这样做：
    # _loss = loss_fn(prediction, target)  # 获取基于预测和实际值的损失
    # total_loss = _loss + _l1_loss  # 总损失等于实际损失加上L1正则化损失
