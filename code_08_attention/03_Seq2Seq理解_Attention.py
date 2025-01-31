import torch
import torch.nn as nn
import torch.nn.functional as F

torch.random.manual_seed(1)


def attention_value(q, k, v):
    """
    Attention的value提取
    :param q: [n,e] query向量  n: 批次大小(同时处理的句子数量)
    :param k: [n,t,e] key向量  t:序列长度(文本中单词或字符的数量)
    :param v: [n,t,e] value向量  e:嵌入维度(每个单词的特征向量的维度)
    :return: [n,e]针对每个样本提取一个e维的向量
    """
    # 1. 计算q和k之间的相关性
    q = torch.unsqueeze(q, dim=2)  # 在第二个维度上增加一个维度 q的形状从[n, e]->[n, e, 1]
    scores = torch.matmul(k, q)  # [n,t,e]*[n,e,1] --> [n,t,1]

    # 2. 权重计算
    # scores = scores[:, :, 0]  # [n,t]
    # pt = torch.argmax(scores, dim=1)  # [n]
    # alpha = F.one_hot(pt, k.shape[1])  # [n,t]
    # alpha = alpha[:, :, None]
    # 指定在 scores 的第 1 个维度（指代的是序列长度维度）上应用 softmax 函数
    # scores 中的每个时间步（每一个单词）的得分将被转换为一个概率值
    alpha = torch.softmax(scores, dim=1)  # [n,t,1]

    # 3. 将t个value的向量值进行融合(加权融合)
    vs = alpha * v  # [n,t,1]*[n,t,e] -> [n,t,e]
    vs = torch.sum(vs, dim=1)  # [n,t,e] -> [n,e]
    # vs = torch.mean(v, dim=1)
    return vs


class EncoderModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=None, num_layers=1, bidirectional=False):
        super(EncoderModule, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        hidden_size = hidden_size or embedding_dim
        self.rnn = nn.RNN(
            input_size=embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=bidirectional
        )
        self.output_dim = hidden_size * num_layers * (2 if bidirectional else 1)

    def forward(self, x):
        """
        编码器前向过程
        :param x: [N,T] token id tensor对象  N:批次大小  T:句子中单词数量
        :return: ([N,T,hidden_size],[N,L])
                [N,T,hidden_size] 针对每个文本、每个时刻使用hidden_size维大小的向量进行特征表示
                [N,L] 向量矩阵，针对每个文本用一个L维的向量进行表示        """
        x = self.embedding_layer(x)  # [N,T] -> [N,T,E]
        ho, hn = self.rnn(x)  # ho：输出序列 [N,T,hidden_size] hn：最终隐藏状态 [L,N,E]
        """
        dims 参数指定了新的维度顺序。在这个例子中，dims=[1, 0, 2] 表示将 hn 的维度重新排列为：
        第一个维度（1）：原来的 N（批次大小）
        第二个维度（0）：原来的 L（层数）
        第三个维度（2）：原来的 E（隐藏层大小）
        """
        hz = torch.permute(hn, dims=[1, 0, 2])  # [L,N,E] -> [N,L,E]
        hz = torch.reshape(hz, shape=(hz.shape[0], self.output_dim))  # [N,?,E] -> [N,?*E] ?*E就是L
        return ho, hz


class DecoderModule(nn.Module):
    def __init__(self,
                 vocab_size, embedding_dim, encoder_state_dim,
                 hidden_size=None, num_layers=1, eos_token_id=0, max_seq_length=20
                 ):
        super(DecoderModule, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.hidden_size = hidden_size or embedding_dim
        assert num_layers == 1, "当前解码器仅支持单层结构!"
        self.num_layers = num_layers
        self.rnn_state_proj = nn.Sequential(
            nn.Linear(in_features=encoder_state_dim, out_features=self.hidden_size * self.num_layers),
            nn.Tanh()
        )
        self.rnn = nn.RNN(
            input_size=embedding_dim + self.hidden_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers, batch_first=True, bidirectional=False
        )
        # 当前模拟代码中，类别数目和词汇表数目一致
        self.proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=vocab_size
        )
        # 解码器属性
        self.max_seq_length = max_seq_length
        self.eos_token_id = eos_token_id
        self.rnn_cell = nn.RNNCell(input_size=embedding_dim, hidden_size=self.hidden_size)
        self.rnn_cell.weight_ih = self.rnn.weight_ih_l0
        self.rnn_cell.weight_hh = self.rnn.weight_hh_l0
        self.rnn_cell.bias_ih = self.rnn.bias_ih_l0
        self.rnn_cell.bias_hh = self.rnn.bias_hh_l0

    def forward(self, x, encoder_state, encoder_outputs):
        """
        解码器的前向过程
        :param x: [N,T] 训练的时候，是token id列表，T为实际长度；预测的时候T为1
        :param encoder_state: [N,encoder_state_dim] 解码器的初始状态信息 ---> 一般来源于编码器的输出
        :param encoder_outputs: [N,T1,hidden_size] 编码器的输出状态信息，假定hidden_size和当前解码器中的状态信息一致
        :return: [N,T,vocab_size] N个文本，对应T个时刻，每个时刻预测的类别置信度值
            NOTE: 训练的时候返回值中的T和x中的T一致，推理预测的时候不一致
        """
        # 将编码器传递过来的状态信息进行转换，作为解码器的初始状态信息
        init_state = self.rnn_state_proj(encoder_state)  # [N,encoder_state_dim] -> [N,hidden_size*num_layers]
        init_state = torch.reshape(init_state, shape=(
        -1, self.hidden_size, self.num_layers))  # [N,hidden_size*num_layers] -> [N,hidden_size,num_layers]
        init_state = torch.permute(init_state,
                                   dims=[2, 0, 1])  # [N,hidden_size,num_layers] -> [num_layers,N,hidden_size]

        # embedding操作
        x = self.embedding_layer(x)  # [N,T] -> [N,T,E]

        if self.training:
            output = []
            hx = init_state[0]
            _, t, _ = x.shape
            for i in range(t):
                xi = x[:, i, :]
                atten_value = attention_value(
                    q=hx,  # 上一个时刻的状态信息
                    k=encoder_outputs,
                    v=encoder_outputs
                )
                xi = torch.concat([xi, atten_value], dim=1)
                hx = self.rnn_cell(xi, hx)  # [N,hidden_size] 当前时刻的rnn输出
                # atten_value = attention_value(
                #     q=hx,  # 上一个时刻的状态信息
                #     k=encoder_outputs,
                #     v=encoder_outputs
                # )
                # ho = torch.concat([hx, atten_value], dim=1)
                # ho = torch.unsqueeze(ho, dim=1)
                # output.append(ho)
                output.append(torch.unsqueeze(hx, dim=1))
            output = torch.concat(output, dim=1)
            scores = self.proj(output)  # [n,T,vocab_size]
            return scores
        else:
            # 需要进行遍历操作，每个时刻每个时刻进行预测，直到预测结果为eos_token_id或者预测的序列长度超过阈值的时候，结束预测
            outputs = []
            hx = init_state[0]  # 第一层的rnn的状态信息
            xi = x[:, 0, :]
            n, _ = xi.shape
            eos_token_ids, is_eos = None, None
            while len(outputs) < self.max_seq_length:
                # 当前rnn的输入: x和状态信息 --> 获取当前rnn的输出
                atten_value = attention_value(
                    q=hx,
                    k=encoder_outputs,
                    v=encoder_outputs
                )
                xi = torch.concat([xi, atten_value], dim=1)
                hx = self.rnn_cell(xi, hx)  # [N,E]
                oi = hx  # RNN的状态信息就是输出信息

                # 进一步的特征提取转换，获取当前时刻的预测token id
                scores_i = self.proj(oi)  # 得到当前时刻的预测置信度[N,vocab_size]
                token_ids_i = torch.argmax(scores_i, dim=1, keepdim=True)  # 当前预测id [N, 1]
                outputs.append(token_ids_i)

                # 判断当前时刻的预测值是不是都是结束符号，如果是，直接退出循环
                if eos_token_ids is None:
                    eos_token_ids = token_ids_i
                    is_eos = (eos_token_ids == self.eos_token_id).to(token_ids_i.dtype)
                eos_token_ids = eos_token_ids * is_eos + token_ids_i * (1 - is_eos)  # 合并数据
                is_eos = (eos_token_ids == self.eos_token_id).to(token_ids_i.dtype)  # 是eos的就是1，不是的就是0
                eos_number = torch.sum(is_eos).item()
                if eos_number >= n:
                    break

                # 更新下一个时刻的输入 --> 将当前时刻的预测token id作为下一个时刻的输入
                xi = self.embedding_layer(token_ids_i)[:, 0, :]
            outputs = torch.concat(outputs, dim=1)  # [N,T2]
            return outputs


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 encoder_num_layers=1, encoder_bidirectional=True, encoder_hidden_size=None,
                 decoder_num_layers=1, decoder_vocab_size=None, decoder_embedding_dim=None, decoder_hidden_size=None,
                 eos_token_id=0,

                 ):
        super(Seq2SeqModel, self).__init__()

        self.encoder = EncoderModule(
            vocab_size, embedding_dim, hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers, bidirectional=encoder_bidirectional
        )
        self.decoder = DecoderModule(
            decoder_vocab_size or vocab_size, decoder_embedding_dim or embedding_dim,
            hidden_size=encoder_hidden_size * (2 if encoder_bidirectional else 1),
            encoder_state_dim=self.encoder.output_dim, num_layers=decoder_num_layers,
            # eos_token_id=eos_token_id,
            eos_token_id=5,  # 临时更改，为了预测退出逻辑
            max_seq_length=200
        )
        self.eos_token_id = eos_token_id
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, encoder_input_ids, label_ids=None):
        """
        前向过程： 前向预测 + loss
        NOTE: loss仅在训练的时候计算
        :param encoder_input_ids: [N,T1] token id tensor列表
        :param label_ids: [N,T2] 训练时候给定的标签id列表，推理预测的时候为None
        :return: [N,T2,vocab_size], loss
        """
        # 1. 基于编码器提取特征
        o, c = self.encoder(encoder_input_ids)
        # 2. 解码器操作
        if self.training:
            # 获取解码器的信息：解码器的输入label_ids的偏移 + 编码器的状态信息
            eos_ids = torch.zeros(size=(label_ids.shape[0], 1), dtype=label_ids.dtype)
            torch.fill_(eos_ids, self.eos_token_id)
            shift_decoder_input_ids = torch.concat([eos_ids, label_ids], dim=1)  # [N,T2+1]
            shift_decoder_output_ids = torch.concat([label_ids, eos_ids], dim=1)  # [N,T2+1]
            scores = self.decoder(shift_decoder_input_ids, c, o)  # [N,T2+1,vocab_size]
            # 损失的计算
            loss = self.loss_fn(torch.permute(scores, dims=[0, 2, 1]), shift_decoder_output_ids)
            return scores, loss
        else:
            # 构建解码器第一个时刻的输入
            eos_ids = torch.zeros(size=(encoder_input_ids.shape[0], 1), dtype=torch.long)
            torch.fill_(eos_ids, self.eos_token_id)
            token_ids = self.decoder(eos_ids, c, o)  # [N,T2+1,vocab_size]
            return token_ids


def model_struct(net, dummy_input, output_file):
    import torch.onnx
    import netron

    # 设置模型为评估模式
    net.eval()

    # # 创建示例输入
    # dummy_input = torch.randint(50, size=(2, 4))

    # # 导出模型为 ONNX 格式
    # output_file = "encodermodel.onnx"
    torch.onnx.export(net, dummy_input, output_file,
                      export_params=True,  # 存储训练过的参数
                      opset_version=10,  # ONNX 版本
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=['input'],  # 输入名称
                      output_names=['output'],  # 输出名称
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 批次大小动态
                      )

    # print(f"ONNX model exported to {output_file}")

    # 使用 netron 查看 ONNX 模型
    netron.start(output_file)


def t0():
    net = EncoderModule(vocab_size=100, embedding_dim=3, num_layers=2, bidirectional=False)
    x = torch.randint(50, size=(2, 6))
    # print(x.shape)
    ho, hz = net(x)
    print(ho.shape)  # 打印RNN输出的形状
    print(hz.shape)  # 打印最终状态的形状
    model_struct(net, x, 'EncoderModule.onnx')

    net2 = DecoderModule(vocab_size=100, embedding_dim=3, encoder_state_dim=net.output_dim)
    y = torch.randint(50, size=(2, 6))
    r2 = net2(y, hz, ho)  # 传递最终状态和RNN输出给解码器
    print(r2.shape)


def t1():
    # 假定场景: 词典大小为26个字母 + 一个特殊值<EOS> + 一个特殊值<PAD>
    # 有一条样本，编码器的输入: a b c，解码器的最终输出: w x y z
    # 对数据做转换:
    # ** 编码器输入: a b c
    # ** 解码器输入: <EOS> w x y z
    # ** 解码器输出: w x y z <EOS>
    # 解码器理解成序列生成，生成序列的时候是不是要一个字符/token一个字符/token来生成，在生成当前token/字符的时候，和是之前的token有强烈的关联关系的
    # 词典映射关系: {<EOS>:0, a:1, b:2, c:3, ......, w:23, x:24, y:25, z:26, <PAD>:27}
    x_id = torch.tensor([
        [1, 2, 3],
        [1, 2, 5]
    ])
    label_ids = torch.tensor([
        [23, 24, 25, 26],
        [23, 24, 25, 26]
    ])
    net = Seq2SeqModel(
        vocab_size=28,
        embedding_dim=4,
        encoder_num_layers=1,
        encoder_hidden_size=16,
        decoder_hidden_size=16,
        eos_token_id=0
    )
    _scores, _loss = net(x_id, label_ids)
    print(_scores.shape)
    print(_loss)

    net.eval()
    _predict_token_ids = net(x_id)
    print(_predict_token_ids)


def t2():
    # 假定翻译场景（中译英）：编码器和解码器的词典大小不一样的
    # 有一条样本，编码器的输入: 小明 吃 苹果，解码器的最终输出：xiao  ming  eats  apples
    # 对数据做转换:
    # ** 编码器输入: 小明 吃 苹果
    # ** 解码器输入: <begin> xiao  ming  eats  apples
    # ** 解码器输出: xiao  ming  eats  apples <end>
    # 解码器理解成序列生成，生成序列的时候是不是要一个字符/token一个字符/token来生成，在生成当前token/字符的时候，和是之前的token有强烈的关联关系的
    net = Seq2SeqModel(
        vocab_size=10000,  # 总共有10000个中文词语
        embedding_dim=128,
        encoder_hidden_size=64, encoder_num_layers=2, encoder_bidirectional=True,
        decoder_num_layers=1,
        decoder_vocab_size=3000,  # 总共有3000个英文词语
        decoder_embedding_dim=64,
        eos_token_id=0
    )
    print(net)
    x_id = torch.tensor([
        [1, 2, 3],
        [1, 2, 3]
    ])
    label_ids = torch.tensor([
        [23, 24, 25, 26],
        [23, 24, 25, 26]
    ])
    _scores, _loss = net(x_id, label_ids)
    print(_scores.shape)
    print(_loss)


def t3():
    q = torch.rand(4, 32)
    k = torch.rand(4, 8, 32)
    v = torch.rand(4, 8, 32)
    att_v = attention_value(q, k, v)
    print(att_v.shape)


if __name__ == '__main__':
    t0()
    # t1()
    # t2()
    # t3()
    pass
