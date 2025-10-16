# Transformer
## 一、Transformer 的核心设计：解决 “长序列依赖” 与 “并行计算”

传统 RNN 处理序列时需 “逐元素迭代”（如从文本第一个词算到最后一个），不仅效率低，还难以捕捉长距离语义关联（如 “猫” 和隔 10 个词的 “抓老鼠”）。Transformer 用**自注意力机制**打破这一限制，同时通过 “并行计算” 大幅提升训练效率，其核心设计可拆解为 “整体结构” 和 “关键组件” 两部分：
### 环境准备

```bash
pip install torch torchtext
```
### 导入依赖

```python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import math
import random
```
## 1. 整体结构：编码器（Encoder）+ 解码器（Decoder）

Transformer 本质是 “编码器 - 解码器架构”，但不同任务可灵活选择使用部分结构（如文本理解用编码器，文本生成用解码器）：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161447577.png)
可以看到 **Transformer 由 Encoder 和 Decoder 两个部分组成**，Encoder 和 Decoder 都包含 6 个 block。Transformer 的工作流程大体如下：

第一步：获取输入句子的每一个单词的表示向量 X、，**X**由单词的 Embedding（Embedding就是从原始数据提取出来的Feature） 和单词位置的 Embedding 相加得到。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161526517.png)



第二步：将得到的单词表示向量矩阵 (如上图所示，每一行是一个单词的表示 **x**) 传入 Encoder 中，经过 6 个 Encoder block 后可以得到句子所有单词的编码信息矩阵 **C**，如下图。单词向量矩阵用  表示， n 是句子中单词个数，d 是表示向量的维度 (论文中 d=512)。每一个 Encoder block 输出的矩阵维度与输入完全一致。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161530022.png)
**第三步**：将 Encoder 输出的编码信息矩阵 **C**传递到 Decoder 中，Decoder 依次会根据当前翻译过的单词 1~ i 翻译下一个单词 i+1，如下图所示。在使用的过程中，翻译到单词 i+1 的时候需要通过 **Mask (掩盖)** 操作遮盖住 i+1 之后的单词。

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161546044.png)
上图 Decoder 接收了 Encoder 的编码矩阵 **C**，然后首先输入一个翻译开始符 "<Begin>"，预测第一个单词 "I"；然后输入翻译开始符 "<Begin>" 和单词 "I"，预测单词 "have"，以此类推。这是 Transformer 使用时候的大致流程，接下来是里面各个部分的细节。

#    2. Transformer 的输入

Transformer 中单词的输入表示 **x**由**单词 Embedding** 和**位置 Embedding** （Positional Encoding）相加得到。