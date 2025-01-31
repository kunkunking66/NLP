import torch
import torch.nn as nn
from torch.nn import functional as F

# 设置随机种子以保证实验结果的一致性
torch.random.manual_seed(10)
torch.manual_seed(10)


# 测试函数t1：演示BatchNorm1d的使用
# BatchNorm1d：用于处理一维数据，即形状为 (N, C) 或 (N, C, L) 的数据，其中 N 是批量大小，C 是通道数或特征数，L 是序列长度或其他维度的长度
# BatchNorm1d：常用于全连接层（Dense layers）或一维卷积层（Conv1d layers）之后的归一化
def t1():
    # x = torch.rand(8, 3)  # 创建一个形状为[8, 3]的随机张量
    # torch.FloatTensor：创建一个浮点类型的张量
    x = torch.FloatTensor(8, 3).uniform_(-10, 10)  # 其中包含的元素是从均匀分布（uniform_）中随机生成的，范围在 -10 到 10 之间
    w = nn.Parameter(torch.rand(3, 4))  # 创建一个形状为[3, 4]的可学习参数张量
    bn = nn.BatchNorm1d(num_features=4)  # 创建一个BatchNorm1d层

    # 打印BatchNorm1d层的running_mean和running_var
    # 这些值通常被初始化为0和1（或非常接近的值），因为 BatchNorm 层在 PyTorch 中默认使用0作为初始均值和1作为初始方差。
    print(bn.running_mean)
    print(bn.running_var)

    # 矩阵相乘用于将输入数据映射到新的特征空间，通常是更高维的空间，以便进行进一步的处理或分类
    # 以列为单位
    # 1.特征维度：在 z 张量中，每一列代表一个新特征空间中的所有样本。例如，z 的第一列 [1, 5, 9, 13, 17, 21, 25, 29] 表示所有8个样本在第一个新特征空间上的值。
    # 2.均值计算：当我们计算 z 的均值时，dim=0 意味着我们沿着行（即样本维度）计算均值。由于 z 是一个二维张量，dim=0 就是沿着行的方向，这样我们就得到了每一列的均值，即每个特征空间的均值。
    # 3.归一化：在归一化过程中，我们通常希望对每个特征（列）单独进行归一化，因为不同特征的分布和尺度可能不同。因此，我们计算每个特征的均值和标准差，并使用这些统计量对每个特征进行归一化。

    z = x @ w  # 矩阵相乘
    # z = x @ w 其中 x 是形状为 [8, 3] 的输入张量，w 是形状为 [3, 4] 的权重矩阵。结果 z 是一个形状为 [8, 4] 的张量，表示8个样本在4个新特征空间上的表示
    z_mean = torch.mean(z, dim=0, keepdim=True)  # dim=0：指定沿着第 0 维计算均值，这意味着将对每一列的所有元素求平均
    z_std = torch.std(z, dim=0, keepdim=True, unbiased=False)  # 计算z在第0维上的标准差
    print(z)
    print(z_mean)  
    print(z_std)  
    print((z - z_mean) / z_std)  # 打印标准化后的z

    bnz = bn(z)  # 应用BatchNorm1d
    print(bnz)  
    o = F.relu(bnz)  # 应用Sigmoid激活函数
    print(o)  
    print(F.sigmoid(z))    # 打印未经过BN的Sigmoid结果
    print(F.sigmoid(bnz))  # 打印经过BN的Sigmoid结果

    # 打印BatchNorm1d层的running_mean和running_var
    print(bn.running_mean)
    print(bn.running_var)

    # # 获取BatchNorm1d的gamma和beta
    # gamma = bn.weight
    # beta = bn.bias
    # print("gamma:", gamma)
    # print("beta:", beta)

    # # 应用gamma和beta
    # normalized_z_with_params = (z - z_mean) / z_std
    # normalized_z_with_params = gamma * normalized_z_with_params + beta.unsqueeze(0)
    # print("Normalized z with gamma and beta:", normalized_z_with_params)


# 测试函数t2：演示BatchNorm2d的使用
# BatchNorm2d：用于处理二维数据，即形状为 (N, C, H, W) 的数据，其中 N 是批量大小，C 是通道数，H 是高度，W 是宽度
# BatchNorm2d：常用于二维卷积层（Conv2d layers）之后的归一化
def t2():
    # z = torch.rand(8, 32, 128, 126) * 0.1 + 5.0  # 创建一个形状为[8, 32, 128, 126]的随机张量
    z = torch.rand(8, 32, 1280, 1260)   # 创建一个形状为[8, 32, 128, 126]的随机张量
    bn = nn.BatchNorm2d(num_features=32)  # 创建一个BatchNorm2d层

    # 打印BatchNorm2d层的weight、bias、running_mean和running_var
    print(bn.weight.shape)
    print(bn.bias.shape)
    print(bn.running_mean.shape)
    print(bn.running_var.shape)

    bnz = bn(z)
    print(F.sigmoid(z))  # 打印未经过BN的Sigmoid结果
    print(F.sigmoid(bnz))  # 打印经过BN后的Sigmoid结果

    # 打印BatchNorm2d层的running_mean和running_var
    print(bn.running_mean)
    print(bn.running_var)


# 测试函数t3：演示不同类型的归一化方法
def t3():
    z = torch.rand(8, 32, 128, 126) * 3.0  # 创建一个形状为[8, 32, 128, 126]的随机张量

    # 计算BatchNorm的均值
    bn_mean = torch.mean(z, dim=(0, 2, 3), keepdim=True)
    bn_norm = nn.BatchNorm2d(num_features=32)
    print(bn_mean.shape)

    # 计算LayerNorm的均值
    ln_mean = torch.mean(z, dim=(1, 2, 3), keepdim=True)
    ln_norm = nn.LayerNorm(normalized_shape=[1, 2, 3])
    print(ln_mean.shape)

    # 计算InstanceNorm的均值
    in_mean = torch.mean(z, dim=(2, 3), keepdim=True)
    in_norm = nn.InstanceNorm2d(num_features=32)
    print(in_mean.shape)

    # 重塑张量以适应GroupNorm
    gz = z.reshape(8, 2, 16, 128, 126)
    print(gz.shape)
    gz_mean = torch.mean(gz, dim=(2, 3, 4), keepdim=True)
    gb_norm = nn.GroupNorm(num_groups=2, num_channels=32)
    print(gz_mean.shape)

    # Softmax计算权重
    lambda_k = nn.Parameter(torch.tensor([0.5, 0.6, 0.8]))
    weight = F.softmax(lambda_k, dim=0)
    print(weight)

    # 计算加权均值
    sn_mean = weight[0] * bn_mean + weight[1] * ln_mean + weight[2] * in_mean
    print(sn_mean.shape)

    # 打印reshape后的均值
    print(ln_mean.view(-1))
    print(torch.mean(in_mean, dim=(1, 2, 3)).view(-1))


if __name__ == '__main__':
    t1()
