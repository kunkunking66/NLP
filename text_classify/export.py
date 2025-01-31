"""
模型静态结构导出文件
静态模型文件的优点：
-1. 脱离训练代码环境，可以独立部署
-2. 可以通过netron查看模型的结构：https://netron.app/
"""
import os
import numpy as np
import torch
from torch.onnx import TrainingMode
from pathlib import Path
import pickle

output_dir = Path("output/01")
models_dir = output_dir / 'models'
model_file_names = os.listdir(str(models_dir))
if len(model_file_names) > 0:
    model_file_names.sort(key=lambda t: int(t.split("_", maxsplit=2)[1].split(".")[0]))
    in_path = models_dir / model_file_names[-1]
    print("model_file_name:", in_path)
else:
    pass


def to_jit():
    """
    将PyTorch模型转换为Torch Script结构的模型
    :return:
    """
    # 模型恢复
    # in_path = r"./output/02/models/net_9.pkl"
    mod = torch.load(in_path, map_location="cpu")
    mod.eval()
    print(mod)

    # 模型转换
    mod = torch.jit.trace(mod, example_inputs=torch.randint(100, size=(1, 20)))

    # 模型直接保存
    out_path = r"output/02/mod.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.jit.save(mod, out_path)


def tt_jit():
    """
    测试jit文件的预测结果
    :return:
    """
    # 模型恢复
    # in_path = r"./output/02/models/net_9.pkl"
    mod = torch.load(in_path, map_location="cpu", pickle_module=pickle)
    mod.eval()

    # 模型恢复
    jit_in_path = r"output/02/mod.pt"
    jit_mod = torch.jit.load(jit_in_path, map_location="cpu")
    jit_mod.eval()

    x = torch.randint(100, size=(2, 25))
    r1 = mod(x)
    r2 = jit_mod(x)
    print(torch.mean(torch.abs(r1 - r2)))


def to_onnx():
    """
    将PyTorch模型转换为ONNX结构的模型
    :return:
    """
    # 模型恢复
    # in_path = r"./output/02/models/net_9.pkl"
    mod = torch.load(in_path, map_location="cpu")
    mod.eval()
    print(mod)

    # 模型转换
    torch.onnx.export(
        mod,  # 模型对象
        args=(torch.randint(100, size=(1, 20)),),  # 输入特征
        f=r"./output/02/mod_01.onnx",  # 模型保存文件路径
        verbose=False,
        training=TrainingMode.EVAL,  # 模式/阶段
        input_names=['ip_name'],  # 自定义的输入特征名称
        output_names=['op_name'],  # 自定义的输出特征名称
        opset_version=12,
        dynamic_axes=None  # 当前输出的模型，在调用的时候，要求输入shape信息必须和args内的一致
    )

    torch.onnx.export(
        mod,  # 模型对象
        (torch.randint(100, size=(1, 20)),),  # 输入特征
        r"./output/02/mod_02.onnx",  # 模型保存文件路径
        verbose=False,
        training=TrainingMode.EVAL,  # 模式/阶段
        input_names=['ip_name'],  # 自定义的输入特征名称
        output_names=['op_name'],  # 自定义的输出特征名称
        opset_version=12,
        dynamic_axes={
            # 'ip_name': [0, 1],  # 输入特征种第0维和第1维是动态的
            'ip_name': {
                0: 'n',
                1: 't'
            },
            'op_name': {
                0: 'n'  # 输出特征种第0维是动态的，并且将这个维度的名称重命名为'n'
            }
        }
    )


def tt_onnx():
    """
    测试onnx文件的预测结果
    :return:
    """
    device = "cpu"  # 设置设备为CPU

    # 模型恢复
    # in_path = r"./output/02/models/net_9.pkl"  # 原始PyTorch模型的路径（此行已注释）
    mod = torch.load(in_path, map_location="cpu")  # 加载PyTorch模型，并将其映射到CPU上
    mod.eval()  # 将模型设置为评估模式

    # ONNX模型恢复
    import onnxruntime  # 导入ONNX Runtime库，用于加载和运行ONNX模型
    # 指定ONNX模型的路径
    onnx_in_path1 = r"./output/02/mod_01.onnx"
    onnx_in_path2 = r"./output/02/mod_02.onnx"

    # 根据设备选择执行提供者，支持GPU或CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']

    # 创建ONNX模型的推理会话
    mod1_session = onnxruntime.InferenceSession(onnx_in_path1, providers=providers)
    print(f"mod1_session.get_providers(): {mod1_session.get_providers()}")  # 打印可用的执行提供者
    mod2_session = onnxruntime.InferenceSession(onnx_in_path2, providers=providers)
    print(f"mod2_session.get_providers(): {mod2_session.get_providers()}")  # 打印可用的执行提供者

    # 生成随机输入数据
    x1 = torch.randint(1, 100, size=(1, 20))  # 生成形状为(1, 20)的随机整数张量
    x2 = torch.randint(1, 100, size=(2, 25))  # 生成形状为(2, 25)的随机整数张量

    # 使用原始PyTorch模型进行预测
    r11 = mod(x1)  # 对x1进行预测
    r12 = mod(x2)  # 对x2进行预测

    # 使用第一个ONNX模型进行预测
    r21 = mod1_session.run(['op_name'], input_feed={'ip_name': x1.detach().numpy()})[0]  # 对x1进行预测
    try:
        r22 = mod1_session.run(['op_name'], input_feed={'ip_name': x2.detach().numpy()})[0]  # 对x2进行预测
    except Exception as e:
        print(f"异常:{e}")  # 捕获并打印异常信息

    # 使用第二个ONNX模型进行预测
    r31 = mod2_session.run(['op_name'], input_feed={'ip_name': x1.detach().numpy()})[0]  # 对x1进行预测
    r32 = mod2_session.run(['op_name'], input_feed={'ip_name': x2.detach().numpy()})[0]  # 对x2进行预测

    # 计算并打印原始模型与ONNX模型预测结果之间的差异
    print(np.mean(r11.detach().numpy() - r31))  # 原始模型与第一个ONNX模型的差异
    print(np.mean(r11.detach().numpy() - r21))  # 原始模型与第二个ONNX模型的差异
    print(np.mean(r12.detach().numpy() - r32))  # 原始模型与第二个ONNX模型的差异


if __name__ == '__main__':
    # to_jit()  # 模型转换
    # tt_jit()  # 模型测试
    # to_onnx()  # onnx 模型转换
    tt_onnx()  # onnx测试
    # pass
