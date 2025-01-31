import torch         
import torch.nn as nn 

if __name__ == '__main__':
    conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5)) 
    # 创建一个二维卷积层conv2d，输入通道数为3（例如RGB图像），输出通道数(卷积核数量)为32，卷积核大小为5x5
    for p in conv2d.parameters():
        print(p.shape)
    
    # 初始化卷积层的权重
    nn.init.kaiming_uniform_(conv2d.weight, mode='fan_out', nonlinearity='relu')
    
    # 如果有偏置项，初始化偏置项
    print(f"conv2d.bias: {conv2d.bias}")
    if conv2d.bias is not None:
        nn.init.zeros_(conv2d.bias)
    print(f"conv2d.bias: {conv2d.bias}")

    if conv2d.bias is not None:
        nn.init.normal_(conv2d.bias)
    print(f"conv2d.bias: {conv2d.bias}")

    print(conv2d.weight.shape) 
    # 打印出这个卷积层权重张量的形状，输出应为 [32, 3, 5, 5]

    alpha = nn.Parameter(torch.ones(3, 5)) 
    nn.init.ones_(alpha)
    
    nn.init.kaiming_uniform_(alpha, mode='fan_out', nonlinearity='relu') 
    # 使用Kaiming均匀分布初始化alpha，mode='fan_out'意味着考虑的是权重张量后面的维度，
    # nonlinearity='sigmoid'表明后续激活函数为sigmoid
    
    nn.init.xavier_uniform_(alpha, gain=1.0)
    # # 使用Xavier均匀分布初始化alpha，gain设置为1.0，通常用于线性激活函数
    
    nn.init.uniform_(alpha, -0.3, 0.3) 
    # # 使用均匀分布[-0.3, 0.3]再次初始化alpha
    
    print(alpha) 
    # 输出alpha的值，由于最后的初始化方法是uniform_，所以输出的张量值将在[-0.3, 0.3]范围内
