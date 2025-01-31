import torch
import torch.nn as nn
from torch import Tensor


def qkv_attention_value(q, k, v):
    """
    计算attention value
    :param q: [N,T1,E] or [N,h,T1,E]
    :param k: [N,T2,E] or [N,h,T2,E]
    :param v: [N,T2,E] or [N,h,T2,E]
    :return: [N,T1,E] or [N,h,T1,E]
    """
    # 2. 计算q和k之间的相关性->F函数
    k = torch.transpose(k, dim0=-2, dim1=-1)  # [??, T2, E] --> [??, E, T2]
    # matmul([??,T1,E], [??,E,T2])
    scores = torch.matmul(q, k)  # [??,T1,T2]

    # 3. 转换为权重
    alpha = torch.softmax(scores, dim=-1)  # [??,T1,T2]

    # 4. 值的合并
    # matmul([??,T1,T2], [??,T2,E])
    v = torch.matmul(alpha, v)  # [??,T1,E]
    return v


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)

    def forward(self, spu, view_spus):
        k = self.wk(view_spus)
        v = self.wv(view_spus)
        new_v = qkv_attention_value(
            q=torch.unsqueeze(spu, dim=1),
            k=k,
            v=v
        )
        return new_v[:, 0, :]


class Network(nn.Module):
    def __init__(self, spu_nums, hidden_size):
        super(Network, self).__init__()
        self.view_emb = nn.Embedding(num_embeddings=spu_nums, embedding_dim=hidden_size)
        self.spu_emb = nn.Embedding(num_embeddings=spu_nums, embedding_dim=hidden_size)
        self.attn = Attention(hidden_size)

        self.classify = nn.Linear(2 * hidden_size, 2)

    def forward(self, view_spu_ids: Tensor, spu_id: Tensor):
        """
        NOTE: 不考虑填充
        :param view_spu_ids: [N,T] N个样本，每个样本中包含的用户浏览商品id token列表 long
        :param spu_id: [N] N个样本，商品id token long
        :return: [N,2] N个样本，预测当前样本对应的用户是否会点击对应的商品
        """
        z1 = self.view_emb(view_spu_ids)  # [N,T] -> [N,T,hidden_size]
        z2 = self.spu_emb(spu_id)  # [N] -> [N,hidden_size]

        # 一定有一个代码块 --> 将T个向量合并成一个
        # z1 = torch.mean(z1, dim=1)  # [N,T,hidden_size] --> [N,hidden_size]
        z1 = self.attn(z2, z1)# [N,T,hidden_size] --> [N,hidden_size]
        print(z1)

        z = torch.concat([z1, z2], dim=1)  # 将z1和z2合并, [N, hidden_size*2]
        z = self.classify(z)
        return z


def t0():
    net = Network(100, 4)

    _x1 = torch.tensor([
        [1, 2, 3],
        [1, 2, 3]
    ], dtype=torch.long)
    _x2 = torch.tensor([
        4, 5
    ], dtype=torch.long)
    _r = net(_x1, _x2)
    print(_r.shape)

    """
    神经网络中特征融合主要三种方式：
    -1. concat：拼接
    -2. sum/mean: 求和/均值
    -3. attention: 注意力机制
    Attention的使用value的合并方式，将多个value向量合并成一个向量的一种方式
    样本1：
        特征1: 1,2,3
        特征2: 4
    样本2:
        特征1: 1,2,3
        特征2：5
    样本3:
        特征1: 4,5,6,7,8
        特征2: 12
    """


if __name__ == '__main__':
    t0()
