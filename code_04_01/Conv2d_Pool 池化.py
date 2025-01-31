import torch
import torch.nn as nn
# 更方便地访问构建神经网络所需的类和函数，例如层（如 nn.Linear、nn.Conv2d）、激活函数（如 nn.ReLU）、损失函数（如 nn.CrossEntropyLoss）等


def t1():
    m = nn.Conv2d(in_channels=16, out_channels=33, kernel_size=(3, 2), stride=2)
    # 为了捕获更复杂的特征或提高模型的容量，可能会选择更多的输入通道 一般的图像是RGB3个通道 灰度图像是一个通道
    # 输出通道=卷积核的数量 为了捕捉更多的特征
    # kernel_size：卷积核的大小，可以是一个整数或者一个元组。如果是一个整数，那么卷积核在高度和宽度上的大小相同；如果是一个元组，那么第一个元素是卷积核的高度，第二个元素是宽度
    # stride：卷积的步长，也就是卷积核移动的间隔。步长为2意味着每次卷积核向右或向下移动两个像素

    # 可以将张量x理解为包含八个256x256像素的图片
    x = torch.rand(8, 16, 256, 256)  # [N,C,H,W]
    # torch.rand 函数用于生成一个填充了从均匀分布采样的值的张量，其取值范围是 [0, 1)
    # 张量 x 有8个样本，每个样本有16个通道，每个通道的图像大小为256x256。这种类型的张量常用于训练深度学习模型，特别是在处理图像数据时
    # 卷积操作后输出的高度和宽度
    # output_height = math.floor((input_height - kernel_height) / stride_height) + 1
    # output_width = math.floor((input_width - kernel_width) / stride_width) + 1

    # 执行前向传播 将张量x通过卷积层m来处理
    z = m(x)  # [8, 16, 256, 256] -> [8,33,127,128]
    # 将输入张量 x 传递给卷积层模型 m 进行前向传播
    print(z.shape)
    for p in m.parameters():
        print(p.shape)

    print("=" * 100)
    m = nn.Conv2d(in_channels=16, out_channels=33, kernel_size=(3, 2), stride=1, padding='same')

    x = torch.rand(8, 16, 256, 256)  # [N,C,H,W]
    z = m(x)  # [8, 16, 256, 256] -> [8,33,127,128]
    print(z.shape)
    for p in m.parameters():
        print(p.shape)

    print(z)
    n = nn.ReLU()

    z = n(z)
    # 将张量z传递给激活函数n进行处理
    print(z)
    print(z.shape)

    # 开始进行池化和卷积操作

    # 创建一个最大池化层实例，内核大小为 2x2
    p = nn.MaxPool2d(2)
    zp1 = p(z)
    print(zp1.shape)

    # 创建另一个最大池化层实例，内核大小为 3x3，步长为 2，并且四周填充了 1 个像素
    p = nn.MaxPool2d(3, 2, padding=1)
    zp2 = p(z)
    print(zp2.shape)

    # 创建一个卷积层实例，输入通道为 33，输出通道为 33，内核大小为 3x3，步长为 2，并且四周填充了 1 个像素
    p = nn.Conv2d(33, 33, kernel_size=3, stride=2, padding=1)
    zp3 = p(z)
    print(zp3.shape)

    # 创建一个最大池化层实例，内核大小为 3x3，步长为 1，并且四周填充了 1 个像素
    p = nn.MaxPool2d(3, 1, padding=1)
    zp4 = p(z)
    print(zp4.shape)

    # AdaptiveMaxPool2d
    p = nn.AdaptiveMaxPool2d(output_size=(4, 4))
    # 根据指定的输出尺寸自动调整池化操作，以确保输出特征图的大小正好是所需的尺寸。这使得网络能够更灵活地处理不同尺寸的输入，而不需要手动计算池化层的参数
    zp5 = p(z)
    print(zp5.shape)


if __name__ == '__main__':
    t1()
