# NLP 学习笔记与实战项目

这个仓库记录了我在学习 NLP 过程中的所有笔记、代码和项目实践。我将学习过程划分为不同的模块，从基础的 PyTorch 和文本处理，到核心的 Transformer 模型，再到具体的实战项目。

希望这个仓库能帮助到同样在学习 NLP 的你！

## 环境配置与安装教程

为了保证项目代码的顺利运行，并避免与您电脑上其他的 Python 项目产生冲突，强烈建议您使用虚拟环境。

### 第一步：克隆（下载）本仓库

首先，打开您的终端（在 Windows 上是 `Command Prompt` 或 `PowerShell`，在 macOS 或 Linux 上是 `Terminal`），然后运行以下命令，将这个仓库复制到您的本地电脑：

```bash
git clone https://github.com/kunkunking66/NLP.git
cd NLP
```

### 第二步：PyTorch 版本与 CUDA 配置指南 (重要！)

这是整个环境配置中最关键、也最容易出错的一步。深度学习代码通常可以在 CPU 或 GPU 上运行。在 GPU 上运行速度会快非常多，而 PyTorch 使用 NVIDIA 的 CUDA 平台来利用 GPU 加速。

**配置成功的关键在于：PyTorch 版本、CUDA 工具包版本、NVIDIA 驱动版本三者必须兼容。**

#### 1. 检查你的硬件与驱动

首先，你必须确定你是否拥有一块支持 CUDA 的 NVIDIA 显卡。

打开终端，输入以下命令：

```bash
nvidia-smi
```

*   **如果命令成功执行**：你会看到一个表格，里面包含了你的显卡型号（如 `GeForce RTX 3060`）、驱动版本（`Driver Version`）和此驱动最高支持的 CUDA 版本（`CUDA Version`）。
    *   **请特别注意**：右上角显示的 `CUDA Version` (例如 `12.2`) 是指你的**驱动程序**最高能支持的 CUDA 版本，**不代表**你的电脑上已经安装了这个版本的 CUDA 工具包。你只需要保证你安装的 PyTorch 所需的 CUDA 版本 **小于或等于** 这个版本号即可。

*   **如果命令失败或提示找不到**：
    *   可能你没有 NVIDIA 显卡（例如是 AMD 或 Intel 的显卡）。
    *   可能你没有安装 NVIDIA 驱动程序。
    *   在这种情况下，你**只能安装 CPU 版本的 PyTorch**。

#### 2. 获取正确的 PyTorch 安装命令

**不要直接使用 `pip install torch`**，因为这很可能会安装一个不适合你硬件的版本（通常是 CPU 版）。请务g必遵循以下步骤：

1.  访问 [PyTorch 官网的“Get Started”页面](https://pytorch.org/get-started/locally/)。

2.  在该页面，你会看到一个交互式的配置工具。请根据你的情况进行选择：
    *   **PyTorch Build**: 选择 `Stable` (稳定版)。
    *   **Your OS**: 选择你的操作系统 (Linux, Mac, Windows)。
    *   **Package**: 推荐选择 `Conda`，因为它能更好地管理复杂的依赖关系。如果不想用 Conda，也可以选择 `Pip`。
    *   **Language**: 选择 `Python`。
    *   **Compute Platform**: 这是最关键的选择！
        *   如果 `nvidia-smi` 命令失败，或者你不想使用 GPU，请选择 `CPU`。
        *   如果 `nvidia-smi` 命令成功，请选择一个 `CUDA` 版本。**原则是：选择的版本号应小于或等于 `nvidia-smi` 右上角显示的 CUDA 版本**。例如，如果 `nvidia-smi` 显示 `CUDA Version: 12.1`，那么你可以安全地选择 `CUDA 12.1` 或 `CUDA 11.8` 的 PyTorch。通常选择最新的兼容版本即可。

3.  选择完毕后，下方会生成一行**安装命令**，例如：
    `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
    这才是为你量身定制的、最正确的安装命令。

### 第三步：创建虚拟环境并安装 PyTorch

现在，我们将结合前面的步骤，完成整个安装流程。

1.  **创建并激活虚拟环境 (推荐 Conda)**
    ```bash
    # 创建一个名为 nlp_env 的新环境
    conda create --name nlp_env python=3.8 -y

    # 激活环境
    conda activate nlp_env
    ```

2.  **执行安装命令**
    在**已激活**的 `nlp_env` 环境中，复制并粘贴你在上一步从 PyTorch 官网获取的安装命令，然后执行。Conda 会自动处理 PyTorch 以及其对应的 CUDA 工具包的下载和安装。

### 第四步：验证安装是否成功

在你的 Conda 环境中，进入 Python 解释器：

```bash
python
```

然后输入以下代码：

```python
import torch

# 1. 检查 PyTorch 版本
print(f"PyTorch Version: {torch.__version__}")

# 2. 这是最关键的检查：检查 CUDA 是否可用
is_cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda_available}")

if is_cuda_available:
    # 3. 如果 CUDA 可用，打印出 GPU 信息
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("PyTorch is running on CPU.")

```

*   **如果 `CUDA Available` 输出为 `True`**：恭喜你！你的 GPU 环境已配置成功。
*   **如果输出为 `False`**：说明 PyTorch 无法找到 GPU，目前只能在 CPU 上运行。最可能的原因是第二步中某个版本不匹配。请尝试卸载 PyTorch (`conda uninstall pytorch`) 并重新按照第二步的指导生成命令进行安装。

---

## 仓库内容结构

本仓库主要分为两大部分：**基础知识模块** 和 **实战项目**。

### Part 1: NLP 基础知识模块

这部分循序渐进地涵盖了 NLP 的核心概念和模型。

*   **`01-Torch_and_Autograd`**:
    *   **内容**: 介绍了深度学习框架 PyTorch 的基本使用，包括张量（Tensors）的创建与操作，以及作为其核心的自动求导（Autograd）机制。这是后续所有神经网络模型的基础。

*   **`02-Tokenization`**:
    *   **内容**: 讲解文本预处理最关键的一步——分词（Tokenization）。您将学到如何将原始文本字符串切分成一个个有意义的词元（tokens），为模型处理做好准备。

*   **`04-Normalization_and_CNN`**:
    *   **内容**: 介绍了两种在深度学习中非常重要的技术。
        *   **批量归一化 (Batch Normalization)**: 一种加速神经网络训练、提升模型稳定性的常用技巧。
        *   **卷积神经网络 (CNN)**: 虽然常用于图像领域，但 CNN 也能高效地提取文本中的 n-gram 等局部特征，在某些 NLP 任务中表现出色。

*   **`05-Embedding`**:
    *   **内容**: 讲解如何将离散的文本词元转换为计算机可以处理的连续、稠密的向量，即词嵌入（Word Embedding）。这是让模型“理解”词汇语义的关键第一步。

*   **`06-RNN`**:
    *   **内容**: 介绍了专为处理序列数据（如文本、时间序列）而设计的循环神经网络（Recurrent Neural Network），及其更强大的变体如 LSTM 和 GRU。

*   **`07-Seq2Seq`**:
    *   **内容**: 讲解经典的序列到序列（Sequence-to-Sequence）模型。这种“编码器-解码器”架构是机器翻译、文本摘要、对话系统等生成任务的基础。

*   **`08-Attention`**:
    *   **内容**: 深入剖析了注意力机制（Attention Mechanism）。它允许模型在生成输出时，动态地“关注”输入序列中最重要的部分，极大地提升了 Seq2Seq 等模型的性能。

*   **`09-Transformer`**:
    *   **内容**: 讲解当今 NLP 领域最核心、最具影响力的模型——Transformer。它完全基于自注意力机制，摆脱了 RNN 的序列依赖，实现了大规模并行计算，是 BERT 和 GPT 等所有现代预训练语言模型的基石。

### Part 2: 实战项目

在掌握了基础知识后，本部分将带您进入真实世界的 NLP 应用。

*   **意图识别 (Intent Recognition)**:
    *   **项目目标**: 训练一个分类模型，使其能够理解用户输入的句子并准确判断其意图。例如，将“帮我订一张明天到上海的机票”识别为“机票预订”意图。
    *   **流程**: 通常包括数据加载、文本预处理、利用 Embedding 将文本向量化、搭建分类模型（如 CNN, RNN 或 Transformer Encoder）进行训练和评估。

*   **文本分类 (Text Classification)**:
    *   **项目目标**: 训练一个模型，将一篇完整的文档或一条评论自动分配到预设的类别中。例如，将新闻文章分类为“体育”、“科技”或“财经”，或将用户评论分类为“好评”或“差评”。
    *   **流程**: 与意图识别类似，但通常处理的文本更长。这是垃圾邮件检测、情感分析等任务的核心技术。

### 学习资料

在部分文件夹中，您可能还会发现一些 `.pdf` 文件。这些是我在学习过程中收集的相关论文、经典教程或课程讲义，希望能为您的学习提供额外的参考和深度。

## 如何运行代码

1.  确保您已经按照前面的教程**激活了虚拟环境**并**安装了所有依赖**。
2.  使用 `cd` 命令进入您感兴趣的模块或项目文件夹。例如，要运行 `Transformer` 的代码：
    ```bash
    cd 09-Transformer
    ```
3.  找到该文件夹下的主 Python 脚本（通常名为 `main.py`, `train.py` 或以 Notebook `.ipynb` 形式提供），然后运行它：
    ```bash
    python main.py
    ```

## 贡献

非常欢迎您为这个仓库做出贡献！如果您在学习过程中发现了任何代码错误、有更好的实现方式，或者想补充新的学习笔记，请随时提交一个 Pull Request 或创建一个 Issue 与我交流。
