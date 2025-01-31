import os  # 导入操作系统模块，用于文件路径操作
import numpy as np  # 导入NumPy库，用于数值运算
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于绘图

from torchvision import datasets, transforms  # 导入torchvision库中的数据集和转换工具
from torch.utils.data import DataLoader  # 导入PyTorch的DataLoader类，用于创建数据加载器

if __name__ == '__main__':
    # 定义一个MNIST数据集对象，指定下载或加载位置、是否为训练集、转换方法和是否下载
    dataset = datasets.MNIST(
        root='../datas/MNIST',  # 数据集存储的位置
        train=True,  # 加载训练集
        # 将图像数据转换为张量这个转换会将图像的像素值从[0, 255]缩放到[0.0, 1.0]之间，
        # 并将图像的维度从[H x W]（高度x宽度）转换为[C x H x W]（通道x高度x宽度），其中C=1，因为MNIST图像是灰度图。
        transform=transforms.ToTensor(),
        download=True  # 如果数据不存在则下载数据
    )
    print(type(dataset))  # 输出dataset对象的类型，应该是torchvision.datasets.mnist.MNIST

    # 定义一个数据加载器，用于从数据集中批量读取数据
    data_loader = DataLoader(dataset, batch_size=4)  # batch_size设置为4
    print(type(data_loader))  # 输出data_loader对象的类型，应该是torch.utils.data.dataloader.DataLoader

    # 初始化计数器
    k = 0

    # 遍历数据加载器返回的批次数据
    for batch_img, batch_label in data_loader:
        # 输出当前批次图像和标签的形状
        print(batch_img.shape, batch_label.shape)

        # 解包batch_img的形状[n, c, h, w]
        n, c, h, w = batch_img.shape

        # 对每个样本进行处理
        for i in range(n):
            # 获取第i个图像，并将其转换为NumPy数组
            img = batch_img[i].detach().numpy()  # [c, h, w]

            # 将第一个通道的灰度图像转换为0-255范围内的整数
            gray_img = (img[0] * 256).astype(np.uint8)  # [h, w]

            # 获取第i个标签的标量值
            label = batch_label[i].item()  # 标量值

            # 构造输出文件路径
            output_path = f'../datas/MNIST/MNIST/images/{label}/{k}.png'

            # 如果目标目录不存在，则创建目录
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))

            # 使用Matplotlib保存灰度图像
            plt.imsave(output_path, gray_img, cmap='gray')

            # 更新计数器
            k += 1

        # 当计数器超过200时，停止循环
        if k > 200:
            break
