import json
import re

import jieba
import torch

re_num = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')


def is_number(num):
    return bool(re_num.match(num))


def is_punctuation(char):
    pattern = r'^\W$'
    if re.match(pattern, char):
        return True
    else:
        return False


def is_symbols(token):
    symblos = ['#', '$', '%']
    for char in token:
        if char in symblos:
            return True
    return False


class Predictor(object):
    def __init__(self, mod_path, token_path):
        """
        初始化Predictor类，加载模型和词汇表，并完成模型恢复逻辑。
        
        :param mod_path: 模型文件路径
        :param token_path: 词汇表文件路径
        """
        super(Predictor, self).__init__()
        # 加载Torch Script模型到CPU
        mod = torch.jit.load(mod_path, map_location='cpu')
        mod.eval()  # 将模型设置为评估模式
        self.mod = mod  # 保存模型对象
        # 加载词汇表文件
        tokens = json.load(open(token_path, "r", encoding="utf-8"))
        self.token2id = dict(zip(tokens, range(len(tokens))))  # 将词汇映射到ID
        self.unk_token_id = self.token2id['<UNK>']  # 获取未知词汇的ID
        # 设置类别ID到标签的映射
        self.classid2label = {
            0: '消极评论',
            1: '积极评论'
        }
        print("模型恢复完成!")

    @staticmethod
    def _split_review(review):
        """
        使用jieba进行中文分词。
        
        :param review: 待分词的评论文本
        :return: 分词结果列表
        """
        return jieba.lcut(review)

    def token_2_ids(self, tokens):
        """
        将分词结果转换为对应的token ID列表。
        
        :param tokens: 分词结果列表
        :return: token ID列表
        """
        tokenids = []
        for token in tokens:
            if is_number(token):
                tokenids.append(self.token2id['<NUM>'])  # 数字用<NUM>标记
            elif is_symbols(token):
                tokenids.append(self.token2id['<SYMBOLS>'])  # 特殊符号用<SYMBOLS>标记
            elif is_punctuation(token):
                tokenids.append(self.token2id['<PUN>'])  # 标点符号用<PUN>标记
            else:
                tokenids.append(self.token2id.get(token, self.unk_token_id))  # 其他词汇映射到ID，未知词汇用<UNK>标记
        return tokenids

    def predict(self, text: str) -> dict:
        """
        预测方法，基于给定的待预测文本，返回最终预测结果。
        
        :param text: 待预测文本
        :return: 预测结果字典，包含标签、概率和文本
        """
        print(f"当前待预测文本:{text}")
        if text is None:
            return {'code': 1, 'msg': '待预测文本为空.'}
        if len(text) == 0:
            return {'code': 1, 'msg': '待预测文本为空字符串.'}
        tokens = self._split_review(review=text.strip())  # 对文本进行分词
        print(f"分词结果:{tokens}")
        token_ids = self.token_2_ids(tokens)  # 将分词结果转换为token ID列表

        x = torch.tensor([token_ids], dtype=torch.long)  # 将token ID列表转换为PyTorch张量
        r = self.mod(x)  # 使用模型进行预测
        prob = torch.softmax(r, dim=1)  # 计算概率
        r_idx = torch.argmax(prob, dim=1)[0].item()  # 获取预测类别的索引
        prob = prob[0][r_idx].item()  # 获取预测类别的概率
        return {
            'code': 0,
            'data': {
                'label': r_idx,  # 预测类别的索引
                'label_name': self.classid2label[r_idx],  # 预测类别的名称
                'prob': f'{prob:.3f}'  # 预测类别的概率
            }
        }


if __name__ == '__main__':
    # 本地恢复预测的测试
    _p = Predictor(mod_path="output/02/mod.pt", token_path='datas/tokens.json')
    # _r = _p.predict(text="我喜欢看书")
    _r = _p.predict(text="这本书挺好的，我喜欢看")
    print(_r)
