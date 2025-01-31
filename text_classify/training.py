import sys
import os

# 防止import导入包异常的情况
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
print(sys.path)

import json
from pathlib import Path

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from models.fc_model import FCTextClassifyModel
from utils import create_dataloader

"""
train_path: 训练数据的路径。
eval_path: 评估数据的路径。
batch_size: 训练时每个批次的大小。
vocab_size: 词汇表的大小，用于文本嵌入。
embedding_dim: 嵌入层的维度，默认为16。
num_classes: 分类任务的类别数，默认为2。
total_epoch: 训练的总轮数，默认为3。
# # # output_dir: 输出目录的路径，用于保存模型和日志，默认为output/01。# # # 
"""


def run(
        train_path, eval_path, batch_size, vocab_size,
        embedding_dim=16, num_classes=2, total_epoch=3,
        output_dir=Path("output/01")
):
    # 检查 output_dir 参数是否已经是 Path 对象（来自 pathlib 模块，用于路径操作）。如果不是，它将使用 Path 函数将其转换为 Path 对象
    # 这样可以确保后续操作中 output_dir 总是一个 Path 对象，便于进行路径操作
    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    # 通过 / 运算符（在 pathlib 中重载为路径连接）创建一个新的 Path 对象，表示 output_dir 下的 logs 子目录
    summary_dir = output_dir / "logs"
    # 创建 summary_dir 指定的目录。parents=True 参数表示如果父目录不存在，则一并创建。exist_ok=True 参数表示如果目录已存在，则不抛出异常
    summary_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. 获取训练数据DataLoader对象
    train_dataloader = create_dataloader(train_path, batch_size, shuffle=True, num_workers=2, prefetch_factor=2)
    # 2. 获取评估数据DataLoader对象
    eval_dataloader = create_dataloader(eval_path, batch_size * 2)

    # 3. 模型创建
    net = FCTextClassifyModel(vocab_size, embedding_dim=embedding_dim, num_classes=num_classes)
    opt = optim.SGD(net.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # 当模型文件存在的时候，需要进行参数恢复
    # # 列出指定模型目录 models_dir 中的所有文件名
    model_file_names = os.listdir(str(models_dir))
    # # 如果目录不为空（即存在模型文件），则执行参数恢复
    if len(model_file_names) > 0:
        # 这行代码对文件名进行排序，确保最新的模型文件排在最后。文件名被假设为包含数字，这个数字用于确定文件的顺序。这里使用 split 函数来分割文件名，并提取用于排序的数字部分
        model_file_names.sort(key=lambda t: int(t.split("_", maxsplit=2)[1].split(".")[0]))
        # 选择排序后列表中的最后一个文件名，即最新的模型文件
        model_file_name = model_file_names[-1]
        print("model_file_name:", model_file_name)
        ori_net = torch.load(str(models_dir / model_file_name), map_location='cpu')
        # ori_net.state_dict()返回的是一个dict字典对象，key为参数name，value为参数对应的tensor对象
        # 底层是基于key进行参数匹配，然后进行恢复
        net.load_state_dict(ori_net.state_dict(), strict=True)  # 参数恢复

    # 使用 TensorBoard 的 SummaryWriter 类来记录模型的训练过程和结构，以便在 TensorBoard 中进行可视化
    # 日志文件，基于TensorFlow的tensorboard库，如果运行报错，直接pip install tensorflow / pip install tensorboard
    # 日志数据查看:tensorboard --logdir /Users/hayden/Desktop/Hayden/99_Learning/02_DL/lecture06/text_classify/output/01/logs
    writer = SummaryWriter(log_dir=str(summary_dir))
    writer.add_graph(net, torch.randint(vocab_size, size=(4, 8)))

    # 4. 数据迭代训练
    for epoch in range(total_epoch):
        # 模型训练
        # # 将模型设置为训练模式
        net.train()
        bar = tqdm(train_dataloader)
        # 这行代码迭代 train_dataloader 中的每个批次数据，bar 是使用 tqdm 创建的进度条对象，x 是输入数据，y 是对应的标签，mask 是对应输入数据的掩码
        for x, y, mask in bar:
            # 这行代码执行模型的前向传播，net 是模型实例，x 是输入数据，mask 是掩码。模型输出 scores，它是一个形状为 [N, num_classes] 的张量，其中 N 是批次大小，num_classes 是类别数
            scores = net(x, mask)  # 获取模型前向预测结果 [N,num_classes]
            # 输出 scores 和真实标签 y 之间的损失。loss_fn 是损失函数，这里使用的是交叉熵损失（nn.CrossEntropyLoss）
            loss = loss_fn(scores, y)  # 求解损失
            opt.zero_grad()  # 将每个参数的梯度重置为0
            loss.backward()  # 求解每个参数的梯度值
            opt.step()  # 参数更新
            # 更新进度条的后缀信息，显示当前的 epoch 和该批次的损失值
            bar.set_postfix(epoch=epoch, train_loss=loss.item())
            # 使用 SummaryWriter 将损失值记录到 TensorBoard 日志中，以便后续可视化
            writer.add_scalar('train_loss', loss.item())

        # 模型评估
        # # 使用 torch.no_grad() 上下文管理器来关闭梯度计算，这可以减少内存消耗并加速评估过程。
        # # net.eval() 将模型设置为评估模式，这会影响模型中如 Dropout 和 BatchNorm 等层的行为。
        with torch.no_grad():
            net.eval()
            y_preds = []
            y_trues = []
            bar = tqdm(eval_dataloader)
            for x, y, mask in bar:
                scores = net(x, mask)  # 获取模型前向预测结果 [N,num_classes]
                y_pred = torch.argmax(scores, dim=1)  # [N,]
                y_trues.append(y)
                y_preds.append(y_pred)
            y_preds = torch.concat(y_preds, dim=0).numpy()
            y_trues = torch.concat(y_trues, dim=0).numpy()
            accuracy = accuracy_score(y_trues, y_preds)
            writer.add_scalar('accuracy', accuracy, global_step=epoch)
            print("Accuracy:", accuracy)

        # 模型阶段持久化
        if epoch % 1 == 0:
            # 将模型保存到文件中
            torch.save(net, str(models_dir / f'net_{epoch}.pkl'))
    writer.close()


def get_parser():
    import argparse

    # https://zhuanlan.zhihu.com/p/582298060?utm_id=0
    parser = argparse.ArgumentParser(description='入参')
    parser.add_argument('-json_path', type=str, default=r"./datas/tokens.json", help='给定json token路径')
    parser.add_argument('-train_path', type=str, default=r"./datas/train.pkl", help='训练数据路径')
    parser.add_argument('-eval_path', type=str, default=r"./datas/eval.pkl", help='测试数据路径')
    parser.add_argument('-batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('-embedding_dim', type=int, default=16, help='向量维度大小')
    parser.add_argument('-num_classes', type=int, default=2, help='分类类别数目')
    parser.add_argument('-total_epoch', type=int, default=20, help='总的训练epoch数量')
    parser.add_argument('-output_dir', type=str, default='./output/01', help='输出文件夹路径')
    return parser


def run_with_args():
    # 参数解析
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    # print(os.path.abspath(args.json_path))
    tokens = json.load(open(args.json_path, "r", encoding="utf-8"))
    run(
        train_path=args.train_path,
        eval_path=args.eval_path,
        batch_size=args.batch_size,
        vocab_size=len(tokens),
        embedding_dim=args.embedding_dim,
        num_classes=args.num_classes,
        total_epoch=args.total_epoch,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    run_with_args()
