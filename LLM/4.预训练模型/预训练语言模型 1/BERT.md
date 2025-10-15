# BERT
参考：[读懂BERT，看这一篇就够了 - 知乎](https://zhuanlan.zhihu.com/p/403495863)
## 一、介绍
BERT(Bidirectional Encoder Representation from Transformers)是2018年10月由Google AI研究院提出的一种预训练模型

BERT的网络架构使用的是[《Attention is all you need》](https://zhida.zhihu.com/search?content_id=177795576&content_type=Article&match_order=1&q=%E3%80%8AAttention+is+all+you+need%E3%80%8B&zhida_source=entity)中提出的多层Transformer结构。
## **2 BERT模型结构**
下图展示的是BERT的总体结构图，多个Transformer Encoder一层一层地堆叠起来，就组装成了BERT了，在论文中，作者分别用12层和24层Transformer Encoder组装了两套BERT模型，两套模型的参数总数分别为110M和340M。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151751462.png)

BERT是用了Transformer的encoder侧的网络，encoder中的Self-attention机制在编码一个token的时候同时利用了其上下文的token，其中‘同时利用上下文’即为双向的体现，而并非想Bi-LSTM那样把句子倒序输入一遍。在BERT之前是GPT，GPT使用的是Transformer的decoder侧的网络，GPT是一个单向语言模型的预训练过程，更适用于文本生成，通过前文去预测当前的字。

### 2.1 Embedding
Embedding由三种Embedding求和而成：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151755416.png)
```python
"""
input_ids: 输入文本的token索引，形状为[batch_size, seq_len]
segment_ids: 段标识（0或1），形状为[batch_size, seq_len]
"""
```
- Token Embeddings是词向量，第一个单词是CLS标志，可以用于之后的分类任务

通过建立字向量表将每个字转换成一个一维向量，作为模型输入。特别的，英文词汇会做更细粒度的切分，比如playing 或切割成 play 和 ##ing，中文目前尚未对输入文本进行分词，直接对单子构成为本的输入单位。将词切割成更细粒度的 Word Piece 是为了解决未登录词的常见方法。

假如输入文本 ”I like dog“。下图则为 Token Embeddings 层实现过程。输入文本在送入 Token Embeddings 层之前要先进性 tokenization 处理，且两个特殊的 Token 会插入在文本开头 [CLS] 和结尾 [SEP]。[CLS]表示该特征用于分类模型，对非分类模型，该符号可以省去。[SEP]表示分句符号，用于断开输入语料中的两个句子。

Bert 在处理英文文本时只需要 30522 个词（词表大小），Token Embeddings 层会将每个词转换成 768 维向量，例子中 5 个Token 会被转换成一个 (5, 768) 的矩阵或 (1, 5, 768) 的张量。

```python
# 1. Token Embeddings：通过索引获取词向量
token_embeds = self.token_embeddings[input_ids]  # 形状：[batch_size, seq_len, hidden_size]
```
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151758706.png)

- Segment Embeddings用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务

Bert 能够处理句子对的分类任务，这类任务就是判断两个文本是否是语义相似的。句子对中的两个句子被简单的拼接在一起后送入模型中，Bert 如何区分一个句子对是两个句子呢？答案就是 Segment Embeddings。

Segement Embeddings 层有两种向量表示，前一个向量是把 0 赋值给第一个句子的各个 Token，后一个向量是把1赋值给各个 Token，问答系统等任务要预测下一句，因此输入是有关联的句子。而文本分类只有一个句子，那么 Segement embeddings 就全部是 0。

```python
# 2. Segment Embeddings：获取段向量（文本分类任务中全为0）
segment_embeds = self.segment_embeddings[segment_ids]  # 形状：[batch_size, seq_len, hidden_size]
```

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151804020.png)

- Position Embeddings和正常的Transformer不一样，不是三角函数而是学习出来的

由于出现在文本不同位置的字/词所携带的语义信息存在差异(如 ”你爱我“ 和 ”我爱你“)，你和我虽然都和爱字很接近，但是位置不同，表示的含义不同。

在 RNN 中，第二个 ”I“ 和 第一个 ”I“ 表达的意义不一样，因为它们的隐状态不一样。对第二个 ”I“ 来说，隐状态经过 ”I think therefore“ 三个词，包含了前面三个词的信息，而第一个 ”I“ 只是一个初始值。因此，RNN 的隐状态保证在不同位置上相同的词有不同的输出向量表示。

```python
# 3. Position Embeddings：获取位置向量
position_ids = np.arange(seq_len).reshape(1, -1)  # 生成位置索引：[1, seq_len]
position_embeds = self.position_embeddings[position_ids]  # 形状：[1, seq_len, hidden_size]
position_embeds = np.tile(position_embeds, (batch_size, 1, 1))  # 扩展到batch_size
```
`np.arange(seq_len)`：生成一个从 0 到 `seq_len-1` 的连续整数数组，长度为 `seq_len`（即序列中 token 的数量）。

`.reshape(1, -1)`：将生成的一维数组转换为二维数组，形状为 `(1, seq_len)`。其中 `1` 表示批次维度（这里临时用 1 表示单条数据），`-1` 表示自动计算该维度的长度（保持与原数组元素总数一致）。例如，`[0,1,2,3,4]` 会被转换为 `[[0, 1, 2, 3, 4]]`。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151808827.png)

```python
# 三者相加得到最终嵌入
final_embeds = token_embeds + segment_embeds + position_embeds
```

RNN 能够让模型隐式的编码序列的顺序信息（RNN 能让模型在处理序列（比如文字、语音、时间序列数据）时，**自动记住前面元素的信息，并把这些信息用在对后面元素的处理中**，无需人工额外标注 “顺序”，这就是 “隐式编码顺序信息”）
相比之下，Transformer 的自注意力层 (Self-Attention) 对不同位置出现相同词给出的是同样的输出向量表示。尽管 Transformer 中两个 ”I“ 在不同的位置上，但是表示的向量是相同的。

Transformer 中通过植入关于 Token 的相对位置或者绝对位置信息来表示序列的顺序信息。作者测试用学习的方法来得到 Position Embeddings，最终发现固定位置和相对位置效果差不多，所以最后用的是固定位置的，而正弦可以处理更长的 Sequence，且可以用前面位置的值线性表示后面的位置。

BERT 中处理的最长序列是 512 个 Token，长度超过 512 会被截取，BERT 在各个位置上学习一个向量来表示序列顺序的信息编码进来，这意味着 Position Embeddings 实际上是一个 (512, 768) 的 lookup 表，表第一行是代表第一个序列的每个位置，第二行代表序列第二个位置。

最后，BERT 模型将 Token Embeddings (1, n, 768) + Segment Embeddings(1, n, 768) + Position Embeddings(1, n, 768) 求和的方式得到一个 Embedding(1, n, 768) 作为模型的输入。

**[CLS]的作用**

BERT在第一句前会加一个[CLS]标志，最后一层该位对应向量可以作为整句话的语义表示，从而用于下游的分类任务等。因为与文本中已有的其它词相比，这个无明显语义信息的符号会更“公平”地融合文本中各个词的语义信息，从而更好的表示整句话的语义。 具体来说，self-attention是用文本中的其它词来增强目标词的语义表示，但是目标词本身的语义还是会占主要部分的，因此，经过BERT的12层（BERT-base为例），每次词的embedding融合了所有词的信息，可以去更好的表示自己的语义。而[CLS]位本身没有语义，经过12层，句子级别的向量，相比其他正常词，可以更好的表征句子语义。