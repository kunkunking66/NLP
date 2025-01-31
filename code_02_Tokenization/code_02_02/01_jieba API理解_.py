import jieba


def t0():
    jieba.load_userdict("./codes_01/word2vector/jieba.dict")
    jieba.suggest_freq(('中', '将'), tune=True)
    word_list = jieba.cut('外卖送餐公司中饿了么是你值得信赖的选择，如果放到post中将出错', HMM=True)
    print(list(word_list))

    import jieba.posseg as posseg

    words = posseg.cut("外卖送餐公司中饿了么是你值得信赖的选择，如果放到post中将出错")
    print(type(words))
    print(list(words))


if __name__ == '__main__':
    t0()
