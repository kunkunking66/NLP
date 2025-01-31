"""
数据处理相关方法
"""
import json
import pickle
import re
from pathlib import Path

import jieba
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# 正则表达式识别数值格式
re_num = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')


# 检查传入的参数num是否是一个数字
def is_number(num):
    return bool(re_num.match(num))


# 检查给定的字符 char 是否是标点符号
def is_punctuation(char):
    pattern = r'^\W$'
    if re.match(pattern, char):
        return True
    else:
        return False


# 检查传入的字符串 token 是否包含在预定义的符号列表 symbols 中的任何一个字符
def is_symbols(token):
    symbols = ['#', '$', '%']
    for char in token:
        if char in symbols:
            return True
    return False


# 分词 并返回一个列表
def split_review(review):
    return jieba.lcut(review)


def stage1():
    """
    第一步操作：分词、构建词汇表、统计词频、保存分类和标签，并将处理后的数据保存为CSV文件
    :return:
    """
    # 初始化集合和字典，用于存储分类、标签和词频统计
    # # 集合是一个无序的、不重复的元素序列。在这里，cats 用于存储不同的分类名称，确保每个分类名称只存储一次 labels 用于存储不同的标签
    cats = set()
    labels = set()
    token2count = {}

    # 读取数据文件
    # # . 表示当前目录 ..表示上一个目录
    in_file = "../code_05_Embed/datas/online_shopping_10_cats.csv"
    # # sep="," 指定了字段分隔符为逗号（这是 CSV 文件的默认分隔符，所以即使不写，pandas 也会使用逗号作为分隔符）
    # # [:] 是一个切片操作，它在这里的作用是返回 DataFrame 的一个完整副本。这通常用于确保对原始数据的修改不会影响到原始的 DataFrame 对象
    df = pd.read_csv(in_file, sep=",")[:]

    # 初始化存储处理后数据的列表
    split_datas = []

    # 这个循环遍历 df 中的每一行，并将每行的值解包到变量 cat、label 和 review 中
    for cat, label, review in tqdm(df.values):
        try:
            # 清理文本，去除可能的BOM字符和前后空白
            # # .strip('\ufeff')去除字符串 review 开头和结尾的字节顺序标记
            # # .strip()：这个操作会去除字符串 review 开头和结尾的空白字符，包括空格、制表符、换行符等
            review = review.strip('\ufeff').strip()
            # 分词
            review_tokens = split_review(review)
            # 更新分类和标签集合
            cats.add(cat)
            labels.add(label)
            # 统计一个分词后的评论（review_tokens）中每个单词（token）出现的次数，并将其存储在字典 token2count 中
            for token in review_tokens:
                token2count[token] = token2count.get(token, 0) + 1
            # 将处理后的评论添加到列表中
            split_datas.append((cat, label, ' '.join(review_tokens)))
        # 异常处理
        except Exception as e:
            # 如果处理过程中出现异常，打印异常信息和对应的评论
            print(f"异常:{e} -- {review}")

    # 数据保存
    # # 创建了一个指向当前工作目录下名为 datas 的目录的 Path 对象，并将其赋值给变量 output_dir
    # # 尝试创建目录，如果目录已存在则不抛出异常
    output_dir = Path("datas")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存分类集合到JSON文件
    with open(str(output_dir / 'cats.json'), 'w', encoding='utf-8') as writer:
        json.dump(list(cats), writer, ensure_ascii=False)

    # 保存标签集合到JSON文件
    with open(str(output_dir / 'labels.json'), 'w', encoding='utf-8') as writer:
        json.dump(list(labels), writer, ensure_ascii=False)

    # 将词频字典转换为列表，并按词频降序排序
    token2count = list(token2count.items())
    token2count.sort(key=lambda t: -t[1])

    # 筛选出至少出现5次的词汇，并添加特殊标记
    tokens = [token for token, count in token2count if count >= 5]
    for token in ['<NUM>', '<PUN>', '<SYMBOLS>', '<UNK>', '<PAD>']:
        tokens.insert(0, token)

    # 保存词汇集合到JSON文件
    with open(str(output_dir / 'tokens.json'), 'w', encoding='utf-8') as writer:
        json.dump(tokens, writer, ensure_ascii=False, sort_keys=True, indent=2)

    # 将处理后的数据转换为DataFrame，并保存到CSV文件
    df = pd.DataFrame(split_datas, columns=['cat', 'label', 'review'])
    df.to_csv(str(output_dir / "split_datas.csv"), index=False)


def stage2():
    # 加载tokens.json文件，该文件包含了所有词汇
    tokens = json.load(open(r"datas/tokens.json", "r", encoding="utf-8"))
    # 创建一个字典，将每个词汇映射到一个唯一的ID  range用于生成从0开始到len(tokens) - 1的整数序列的函数  zip用于将token的元素和整数进行配对  dict将其转化为字典
    token2id = dict(zip(tokens, range(len(tokens))))
    # 获取'<UNK>'标记的ID，用于表示未知词汇
    unk_token_id = token2id['<UNK>']
    # 加载cats.json文件，该文件包含了所有分类
    cats = json.load(open(r"datas/cats.json", "r", encoding="utf-8"))
    # 创建一个字典，将每个分类映射到一个唯一的ID
    cat2id = dict(zip(cats, range(len(cats))))
    # 加载labels.json文件，该文件包含了所有标签
    labels = json.load(open(r"datas/labels.json", "r", encoding="utf-8"))
    # 创建一个字典，将每个标签映射到一个唯一的ID
    label2id = dict(zip(labels, range(len(labels))))

    # 读取处理后的数据文件
    in_file = r"datas/split_datas.csv"
    df = pd.read_csv(in_file, sep=",")
    # 初始化存储处理后数据的列表
    datas = []
    # 遍历数据集的每一行
    for cat, label, review in tqdm(df.values):
        # 分词
        tokens = review.split(" ")
        # 初始化存储token ID的列表
        tokenids = []
        # 遍历每个token
        for token in tokens:
            # 如果token是数字，则使用'<NUM>'标记的ID
            if is_number(token):
                tokenids.append(token2id['<NUM>'])
            # 如果token是特殊符号，则使用'<SYMBOLS>'标记的ID
            elif is_symbols(token):
                tokenids.append(token2id['<SYMBOLS>'])
            # 如果token是标点符号，则使用'<PUN>'标记的ID
            elif is_punctuation(token):
                tokenids.append(token2id['<PUN>'])
            # 否则，查找token的ID，如果不存在则使用'<UNK>'标记的ID
            else:
                tokenids.append(token2id.get(token, unk_token_id))
        # 将分类ID、标签ID和token ID列表组合成一个样本，并添加到datas列表中
        datas.append([cat2id[cat], label2id[label], tokenids])

    # 使用train_test_split函数将数据划分为训练集和验证集，比例为80%训练集和20%验证集
    train_datas, eval_datas = train_test_split(datas, test_size=0.2, random_state=24)
    # 打印训练集和验证集的样本数量
    print(f"训练数据量:{len(train_datas)}")
    print(f"验证数据量:{len(eval_datas)}")
    # 将训练集数据保存到文件
    with open('datas/train.pkl', 'wb') as writer:
        pickle.dump(train_datas, writer)
    # 将验证集数据保存到文件
    with open('datas/eval.pkl', 'wb') as writer:
        pickle.dump(eval_datas, writer)


if __name__ == '__main__':
    stage1()
