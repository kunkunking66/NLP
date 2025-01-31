import copy
import json
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# 用于处理文本分类任务中的数据，包括数据加载、长度计算和数据填充
class ClassifyDataset(Dataset):
    # 这部分代码使用 with 语句打开文件，确保文件在读取后能够正确关闭。
    # open(file_path, 'rb') 以二进制读取模式打开文件。
    # pickle.load(reader) 从文件中加载序列化的数据。这里假设数据是以 pickle 格式序列化的 Python 对象。
    def __init__(self, file_path, pad_token_idx=0):
        super(ClassifyDataset, self).__init__()
        self.PAD_IDX = pad_token_idx
        with open(file_path, 'rb') as reader:
            self.datas = pickle.load(reader)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        cat, y, x = self.datas[index]
        return copy.deepcopy(x), y, len(x)

    def collate_fn(self, batch):
        """
        数据聚合
        :param batch: list列表，列表中的每个元素都是调用了dataset的__getitem__方法得到的
        :return:
        """
        # 这行代码将 batch 中的样本数据解包，分别聚合成三个列表：x（所有样本的数据），y（所有样本的标签），和 lengths（所有样本的长度）
        x, y, lengths = list(zip(*batch))
        max_length = max(lengths)
        mask = np.zeros(shape=(len(x), max_length), dtype=np.float32)
        for i in range(len(x)):
            x[i].extend([self.PAD_IDX] * (max_length - lengths[i]))  # 数据填充
            mask[i][:lengths[i]] = 1
        x = torch.tensor(x, dtype=torch.long)  # [N,T]
        y = torch.tensor(y, dtype=torch.long)  # [N,]
        lengths = torch.tensor(lengths, dtype=torch.long)  # [N,]
        mask = torch.from_numpy(mask)
        return x, y, mask


# 在训练过程中批量加载数据
# # shuffle=True: 是否在每个epoch开始时打乱数据。设置为 True 可以确保数据的随机性，有助于模型训练的泛化能力
# # num_workers=2: 加载数据时使用的子进程数量。增加子进程数量可以提高数据加载的效率，但也可能增加内存消耗
def create_dataloader(file_path, batch_size, shuffle=False, num_workers=2, prefetch_factor=2):
    # 1. 构建DataSet对象
    dataset = ClassifyDataset(file_path)
    # 2. 构建dataloader对象
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=dataset.collate_fn  # 批次数据聚合函数
    )
    return dataloader


if __name__ == '__main__':
    # dataset = ClassifyDataset("./datas/train.pkl")
    # print(dataset[25])

    tokens = json.load(open(r'datas/tokens.json', "r", encoding="utf-8"))
    data = create_dataloader("datas/train.pkl", 4)
    for x_batch, y_batch, lengths_batch in data:
        print(x_batch)
        print(y_batch)
        print(lengths_batch)
        x_text = [''.join([tokens[token_id] for token_id in token_ids if token_id > 0]) for token_ids in
                  x_batch.detach().numpy()]
        print(x_text)
        break

"""
<PUN>：这个标识符通常用来表示句子或短语之间的停顿，相当于中文中的句号、逗号、分号等标点符号。在文本分析中，它可以被用来识别句子的边界，从而帮助进行句子分割。
<NUM>：这个标识符用来标记文本中的数字。在处理文本时，识别数字是非常重要的，因为它们可能包含重要的信息，如价格、年份、数量等。在这段文本中，<NUM> 可能用来标记年份、价格或其他数值信息。
<UNK>：这个标识符通常用来表示未知的词或者无法识别的字符。在文本处理中，如果遇到无法识别或者不在词汇表中的词，可能会用 <UNK> 来代替。
"""
