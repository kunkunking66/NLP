import numpy as np  
from gensim import corpora, models
from sklearn.decomposition import PCA, NMF  
from gensim.models import TfidfModel

def t1():
    # 定义一个特征矩阵，其中每一行代表一个文本，每一列表示一个单词在所有文本中的出现次数
    # 总共有 7 个文本，5 个单词
    features = [
        [1, 1, 0, 0, 0],  # 文本1
        [1, 1, 1, 0, 0],  # 文本2
        [0, 1, 1, 0, 0],  # 文本3
        [0, 0, 0, 1, 1],  # 文本4
        [0, 0, 1, 0, 1],  # 文本5
        [0, 0, 1, 1, 0],  # 文本6
        [0, 0, 1, 1, 1]  # 文本7
    ]
    
    # 将列表转换为 NumPy 数组，并指定数据类型为 float32
    features = np.asarray(features, dtype=np.float32)

    # 注释掉的代码，用于生成随机数据，此处使用固定的特征矩阵
    # features = np.random.randn(700, 500)

    # 获取特征矩阵的形状，这里 m 表示特征的数量
    _, m = features.shape

    # 根据特征数量的不同，选择不同的主成分数量
    # 如果特征数量少于 10，则使用 2 个主成分；否则使用 100 个主成分
    pca = PCA(n_components=2 if m < 10 else 100)

    # 使用 PCA 对特征矩阵进行降维，并得到降维后的特征矩阵 x
    x = pca.fit_transform(features)
    
    print(x.shape)
    print(x)

    # 获取降维后的主成分矩阵 z
    z = pca.components_  # 各个单词的特征向量
    
    # 打印主成分矩阵 z 的形状和内容
    print(z.shape)
    print(z)

    # 如果特征数量少于 10，则计算各个主成分之间的欧几里得距离，并打印距离的倒数
    if m < 10:
        for i in range(5):
            print("=" * 50)
            vi = z[:, i]
            for j in range(i, 5):
                vj = z[:, j]
                # 计算两个主成分之间的欧几里得距离
                dist = np.sqrt(np.sum(np.square(vi - vj)))
                # 打印距离的倒数
                print(1.0 / (dist + 1.0))

    print("=" * 100)

    # 计算降维后再重建的特征矩阵与原始特征矩阵之间的差异
    diff_features = np.dot(x, pca.components_) - features
    
    # 打印差异矩阵
    print(diff_features)
    
    # 计算差异矩阵的平均绝对值
    print(np.mean(np.abs(diff_features)))
    
    # 计算差异矩阵的平均绝对值相对于原始特征矩阵的平均绝对值的比例
    print(np.mean(np.abs(diff_features)) / np.mean(np.abs(features)))


def t2():
    features = [
    ["stocks", "bonds"],  # 文本1
    ["stocks", "bonds", "commodities"],  # 文本2
    ["bonds", "commodities"],  # 文本3
    ["interest_rates", "currencies"],  # 文本4
    ["commodities", "currencies"],  # 文本5
    ["commodities", "interest_rates"],  # 文本6
    ["commodities", "interest_rates", "currencies"]  # 文本7
]
    
    # 创建字典和语料库
    dictionary = corpora.Dictionary(features)
    corpus = [dictionary.doc2bow(feature) for feature in features]
    
    # 训练LDA模型
    lda = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)
    
    # 获取LDA模型的形状
    print(lda.print_topics())
    

if __name__ == '__main__':
    t1()