import os
import pandas as pd
from tqdm import tqdm
import jieba
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import csr_matrix, hstack, vstack

# 定义文件路径和输出目录
base_path = os.getcwd()
data_file_path = os.path.join(base_path, 'datas/online_shopping_10_cats.csv')
output_dir = os.path.join(base_path, "output2/gensim/t1")
os.makedirs(output_dir, exist_ok=True)

# 分词函数
def split_words(doc):
    return jieba.lcut(str(doc).strip())

# 读取数据
df = pd.read_csv(data_file_path)
documents = df['review'].values
labels = df['label'].values

# 清理空值或NaN
mask = (pd.notnull(documents)) & (documents != '')
documents = documents[mask]
labels = labels[mask]

# 分词
docs = [split_words(document) for document in documents]

# 构建词汇表
dct = Dictionary(docs)
dct.save_as_text(os.path.join(output_dir, "dict.txt"))

# 转换语料库
corpus = [dct.doc2bow(doc) for doc in docs]

# 保存TF-IDF模型
model = TfidfModel(corpus)
model.save(os.path.join(output_dir, "tfidf2.pkl"))

# 将语料库转换为TF-IDF特征的稀疏矩阵
corpus_tfidf = model[corpus]
data = []
row_ind = []
col_ind = []
for i, doc in enumerate(corpus_tfidf):
    for j, (termid, _) in enumerate(doc):
        row_ind.append(i)
        col_ind.append(termid)
        data.append(_)

# 创建一个稀疏矩阵，其中每个文档是一行，每个词汇表中的单词是一列
X = csr_matrix((data, (row_ind, col_ind)), shape=(len(documents), len(dct)))
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 训练分类模型
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# 模型评估
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 新评论预测
new_reviews = [
    # ... （你的新评论数据） ...
]

# 分词
new_docs = [split_words(review) for review in new_reviews]

# 转换为词袋模型
new_corpus = [dct.doc2bow(doc) for doc in new_docs]

# 计算TF-IDF值
new_review_tfidf = model[new_corpus]
new_data = []
new_row_ind = []
new_col_ind = []
for i, doc in enumerate(new_review_tfidf):
    for j, (termid, _) in enumerate(doc):
        new_row_ind.append(i)
        new_col_ind.append(termid)
        new_data.append(_)

# 创建一个新的稀疏矩阵，用于新评论的预测
new_X = csr_matrix((new_data, (new_row_ind, new_col_ind)), shape=(len(new_reviews), len(dct)))

# 使用分类器进行预测
predicted_classes = classifier.predict(new_X)
print("Predicted Classes:", predicted_classes)

