import torch
import torch.nn as nn


def t1():
    # 它返回一个形状为[20]的一维张量（Tensor），其中的元素是从标准正态分布（均值为0，标准差为1）中随机采样得到的
    x = torch.randn(20)
    p = 0.2  # Dropout的概率
    # 创建一个Dropout层
    d1 = nn.Dropout(p=p)

    # 打印分割线和原始输入
    print("=" * 100)
    print(x)

    # 打印调整后的输入（在推理模式下使用）
    print(x / (1 - p))

    # 打印输入的均值
    print(torch.mean(x))

    # 打印调整后输入的均值
    # 在训练阶段，Dropout层会随机将输入张量中的一些元素置零。为了保持输出的期望值不变，我们需要对未被置零的元素进行缩放。
    # 这是因为Dropout操作会改变网络的输出分布，而这种缩放是必要的，以确保在训练和推理阶段网络的输出具有相同的期望值。
    print(torch.mean(x / (1 - p)))

    # 设置Dropout层为训练模式
    d1.train()

    # 在训练模式下通过Dropout层
    x1 = d1(x)

    # 打印训练阶段的执行结果
    print(x1)
    # 训练阶段，随机选择占比为p的神经元直接重置为0，其它没有被重置为0的神经元的对应特征值为 x / (1 - p)

    # 打印训练阶段调整后的均值
    print(torch.mean(x1))

    # 再次打印训练阶段的结果，以确认是否一致
    print(d1(x))

    # 设置Dropout层为推理模式
    d1.eval()

    # 在推理模式下通过Dropout层
    print(d1(x))  # 推理阶段，Dropout不进行任何操作


def t2():
    # 创建一个形状为[1, 10, 2, 2]的随机张量作为Dropout2D的输入
    x = torch.rand(1, 10, 2, 2)

    # 创建一个Dropout2D层，概率为0.5
    d2 = nn.Dropout2d(p=0.5)

    # 设置Dropout2D层为训练模式
    d2.train()

    # 在训练模式下通过Dropout2D层
    x1 = d2(x)

    # 设置Dropout2D层为推理模式
    d2.eval()

    # 在推理模式下通过Dropout2D层
    x2 = d2(x)

    # 打印原始输入、训练阶段的执行结果、推理阶段的执行结果，并用分割线隔开
    print(x)
    print("=" * 100)
    print(x1)
    print("=" * 100)
    print(x2)
    print("=" * 100)

    # 创建一个普通的Dropout层，概率为0.5
    d = nn.Dropout(p=0.5)

    # 设置Dropout层为训练模式
    d.train()

    # 在训练模式下通过Dropout层
    print(d(x))

    # 设置Dropout层为推理模式
    d.eval()

    # 在推理模式下通过Dropout层
    print(d(x))


if __name__ == '__main__':
    t1()
