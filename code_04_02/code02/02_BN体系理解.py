import torch         
import torch.nn as nn 

if __name__ == '__main__':
    # 创建一个形状为[8, 32, 54, 56]的随机张量，代表一个批次的数据
    # 8表示批次大小，32表示通道数，54x56是输入图像的尺寸
    x = torch.randn(8, 32, 54, 56)

    # 创建一个BatchNorm2d层，num_features=32表示输入张量的通道数，
    # momentum=1.0表示该层将使用当前批次的统计量来更新其状态。在实际应用中，momentum通常设为较小的值（如默认的0.1），以便累积历史统计信息。
    # running_mean=(1−momentum)⋅running_mean+momentum⋅batch_mean
    bn = nn.BatchNorm2d(num_features=32, momentum=1.0)   
    
    bn_x1 = bn(x)  # 将输入数据x通过BatchNorm2d层进行归一化处理
    
    # 计算输入数据x在除通道维度外的所有维度上的均值，保持维度信息
    bn_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)  
    
    # 计算输入数据x在除通道维度外的所有维度上的标准差，保持维度信息
    bn_std = torch.std(x, dim=(0, 2, 3), keepdim=True)
    
    # 使用上面计算的均值和标准差手动对输入数据x进行归一化处理
    bn_x2 = (x - bn_mean) / bn_std
    
    # 比较经过BatchNorm2d层处理的结果与手动归一化的结果之间的最大绝对差异
    print(torch.abs(bn_x1 - bn_x2).max())
    
    # 比较BatchNorm2d层累积的均值与当前批次计算得到的均值的最大绝对差异
    print(torch.abs(bn.running_mean - bn_mean.view(-1)).max())
    
    # 比较BatchNorm2d层累积的标准差与当前批次计算得到的标准差的最大绝对差异
    # 注意这里需要加上eps来防止除零错误，并取平方根来获得标准差
    print(torch.abs(torch.sqrt(bn.running_var + bn.eps) - bn_std.view(-1)).max())
