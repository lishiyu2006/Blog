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


上图 Decoder 接收了 Encoder 的编码矩阵 **C**，然后首先输入一个翻译开始符 "< Begin >"，预测第一个单词 "I"；然后输入翻译开始符 "< Begin >" 和单词 "I"，预测单词 "have"，以此类推。这是 Transformer 使用时候的大致流程，接下来是里面各个部分的细节。

## 2. Transformer 的输入

Transformer 中单词的输入表示 **x**由**单词 Embedding** 和**位置 Embedding** （Positional Encoding）相加得到。

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510161555747.png)

### 2.1 单词 Embedding

单词的 Embedding 有很多种方式可以获取，例如可以采用 Word2Vec、Glove 等算法预训练得到，也可以在 Transformer 中训练得到。

### 2.2 位置 Embedding

Transformer 中除了单词的 Embedding，还需要使用位置 Embedding 表示单词出现在句子中的位置。**因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于 NLP 来说非常重要。**所以 Transformer 中使用位置 Embedding 保存单词在序列中的相对或绝对位置。

位置 Embedding 用 **PE**表示，**PE** 的维度与单词 Embedding 是一样的。PE 可以通过训练得到，也可以使用某种公式计算得到。在 Transformer 中采用了后者，计算公式如下：

$PE_{(pos, 2i)} = \sin\left(pos / 10000^{2i/d}\right)$ 
$PE_{(pos, 2i+1)} = \cos\left(pos / 10000^{2i/d}\right)$ 

其中，pos 表示单词在句子中的位置，d 表示 PE的维度 (与词 Embedding 一样)，2i 表示偶数的维度，2i+1 表示奇数维度 (即 2i≤d, 2i+1≤d)。使用这种公式计算 PE 有以下的好处：

- 使 PE 能够适应比训练集里面所有句子更长的句子，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。
- 可以让模型容易地计算出相对位置，对于固定长度的间距 k，**PE(pos+k)** 可以用 **PE(pos)** 计算得到。因为 Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)。

将单词的词 Embedding 和位置 Embedding 相加，就可以得到单词的表示向量 **x**，**x** 就是 Transformer 的输入。

## 3. Self-Attention（自注意力机制）