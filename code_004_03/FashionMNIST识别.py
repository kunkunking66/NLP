import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data


# 定义CNN模型
# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 训练函数
def train(model, data_loader, criterion, optimizer, num_epochs=5):
    # 将模型设置为训练模式。这对于某些特定的层（如Dropout层和BatchNorm层）是必要的，因为它们在训练和评估时的行为不同
    model.train()
    for epoch in range(num_epochs):  # 循环迭代指定的训练轮数
        for batch_idx, (data, target) in enumerate(
                data_loader):  # 循环迭代 data_loader 提供的数据批次。每个批次包含输入数据 data 和对应的目标标签 target
            optimizer.zero_grad()  # 每次迭代前清零梯度。这是必要的，因为PyTorch默认会累加梯度
            output = model(data)  # 将输入数据 data 通过模型进行前向传播，得到模型的预测输出
            loss = criterion(output, target)  # 用损失函数计算模型输出和真实标签之间的差异（损失）
            loss.backward()  # 对损失进行反向传播，计算梯度
            optimizer.step()  # 根据计算出的梯度，使用优化器更新模型的权重
            if batch_idx % 100 == 0:  # 每100个批次打印一次当前的训练轮次、批次索引和损失值，以便监控训练过程
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}')


def test(model, data_loader):
    #  将模型设置为评估模式。这对于某些特定的层（如Dropout层和BatchNorm层）是必要的，因为它们在训练和评估时的行为不同
    model.eval()
    correct = 0  # 用于记录预测正确的样本数量
    total = 0  # 用于记录测试样本的总数
    with torch.no_grad():  # 语句块内的所有操作都不会计算梯度，这可以减少内存消耗并加速计算，因为在评估模式下不需要梯度信息
        for data, target in data_loader:  # 循环迭代 data_loader 提供的测试数据批次。每个批次包含输入数据 data 和对应的目标标签 target。
            output = model(data)  # 将输入数据 data 通过模型进行前向传播，得到模型的预测输出
            _, predicted = torch.max(output.data, 1)  # 获取模型输出中概率最高的类别索引，即模型的预测类别。
            total += target.size(0)  # 增加测试样本的总数
            correct += (predicted == target).sum().item()  # 比较预测类别和真实标签，增加预测正确的样本数量
    accuracy = correct / total  # 计算测试集上的准确率
    print(f'Test Accuracy: {accuracy * 100:.2f}%')  # 打印测试集上的准确率，保留两位小数


if __name__ == '__main__':
    # 数据加载和预处理
    transform = transforms.Compose([transforms.ToTensor()])  # 将PIL图像或NumPy ndarray 转换为torch.Tensor
    # 假设FashionMNIST数据集已经下载在'./datas/FashionMNIST'目录下
    dataset = datasets.FashionMNIST(root='./datas/FashionMNIST', train=True, transform=transform, download=True)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))  # len(dataset)返回整个数据集的大小，0.8 * len(dataset)计算出训练集应有的样本数量，int()函数确保结果是一个整数
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # random_split接受两个参数：第一个是原始数据集，第二个是一个列表，包含两个元素，分别指定了训练集和测试集的大小

    # batch_size=64 这指定了每个批次（batch）中的样本数量为64。这意味着DataLoader会每次提供64个样本给模型进行训练
    # shuffle=True：这表示在每个epoch开始时，数据集中的样本将被打乱。打乱数据可以帮助模型学习时的泛化能力，因为它确保了模型不会记住特定顺序的样本。
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 实例化模型、定义损失函数和优化器
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, train_loader, criterion, optimizer, num_epochs=5)

    # 测试模型
    test(model, test_loader)

    for i in range(min(10, len(train_dataset))):  # 打印前10个或者数据集中更少的样本
        data, target = train_dataset[i]
        print(f"Sample {i + 1}: Data shape = {data.shape}, Target = {target}")
