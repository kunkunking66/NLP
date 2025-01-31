import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(24)


def t():
    linear = nn.Linear(in_features=4, out_features=5, bias=False)
    print(f"linear weight shape: {linear.weight.shape}")
    x = torch.rand(2, 4)  # [2, 4]
    y1 = linear(x)
    print(y1.shape)
    # y2 = x @ linear.weight.T
    y2 = torch.matmul(x, linear.weight.T)
    print(y2.shape)
    print(y1 - y2)

    # 3个文档，每个文档2个词
    x_idx = [
        [0, 1],  # 类别序号/下标
        [0, 3],
        [1, 2]
    ]
    # onehot
    x = [
        [1, 1, 0, 0],  # 文本1
        [1, 0, 0, 1],  # 文本2
        [0, 1, 1, 0]  # 文本3
    ]
    x = torch.tensor(x, dtype=torch.float32)  # [n,m] n个样本，m维大小(词表的大小)
    y1 = linear(x)  # [n,m]*[m,o] -> [n,o] o表示输出特征向量大小 --> 计算量n*m*o
    print(y1)
    print(linear.weight.T[0] + linear.weight.T[1])
    print("=" * 100)

    w = linear.weight.T  # [4, 5]  在这里，w就表示每个单词对应一个稠密的特征向量
    print(w.shape)
    print(w)
    r = w[np.reshape(x_idx, -1)].reshape(-1, 2, 5)  # 每个单词对应一个向量
    print("=" * 100)
    print(r.shape)
    print(r)
    r = r.sum(dim=1)  # 合并到一起，得到每个文本对应一个向量
    print(r.shape)
    print(r)

    print("=" * 100)
    # _weight: 如果不给定，那么内部会随机初始化一个
    embed = nn.Embedding(num_embeddings=4, embedding_dim=5, _weight=w)
    r2 = embed(torch.tensor(x_idx, dtype=torch.int64))
    print(r2.shape)
    print(r2)

    n = 100
    tx_idx = torch.tensor(x_idx, dtype=torch.int64)
    import time

    t1 = time.time()
    for i in range(n):
        linear(x)
    t2 = time.time()
    for i in range(n):
        embed(tx_idx).sum(dim=1)  # 得到文本向量
    t3 = time.time()
    print(t3 - t2, t2 - t1)


def t1():
    vocab_size = 10000
    linear = nn.Linear(in_features=vocab_size, out_features=128, bias=False)
    embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=128)
    # embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=128, _weight=linear.weight.T)

    x = torch.randint(0, vocab_size, size=(16,))  # [16,]
    x_onehot = F.one_hot(x, num_classes=vocab_size)  # [16,10000]
    x_onehot = x_onehot.to(torch.float32)

    y1 = linear(x_onehot)
    y2 = embed(x)
    print(torch.mean(torch.abs(y1 - y2)))

    n = 100
    import time

    t1 = time.time()
    for i in range(n):
        linear(x_onehot)
    t2 = time.time()
    for i in range(n):
        embed(x)
    t3 = time.time()
    print(t3 - t2, t2 - t1)


if __name__ == '__main__':
    t()
