import torch
import torch.nn as nn


class FCTextClassifyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(FCTextClassifyModel, self).__init__()

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.features = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 4),
            nn.ReLU()
        )

        self.classify = nn.Sequential(
            nn.Linear(embedding_dim * 4, num_classes)
        )

    def forward(self, x, mask=None):
        """
        前向执行过程
        :param x: [N,T] LongTensor N个文本，每个文本T个单词序号id
        :param mask: [N,T] 数据的mask矩阵，如果样本对应token实际存在，那么值为1，否则值为0
        :return: [N,num_classes] N个文本，每个文本对应每个类别的预测置信度
        """
        # 提取的单词特征
        z1 = self.embedding_layer(x)  # [N,T] -> [N,T,E]

        # 提取文本特征
        z2 = self.features(z1)  # [N,T,E] --> [N,T,4E]
        if mask is not None:
            mask = mask[..., None]  # [N,T] -> [N,T,1]
            z2 = z2 * mask  # [N,T,4E] * [N,T,1] -> [N,T,4E]          # 将 z2（文本特征）与 mask 相乘，这样填充部分的特征值将被置为0
            lengths = torch.sum(mask, dim=1)  # [N,T,1] -> [N,1]      # 计算每个样本的实际数据长度
            z3 = torch.sum(z2, dim=1) / lengths                       # 对每个样本的实际数据部分求和，然后除以实际数据的长度，得到加权平均值。
        else:
            z3 = torch.mean(z2, dim=1)  # [N,T,4E] -> [N,4E]

        # 决策输出
        z4 = self.classify(z3)  # [N,4E] --> [N,num_classes]
        return z4
