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
- Token Embeddings是词向量，第一个单词是CLS标志，可以用于之后的分类任务

通过建立字向量表将每个字转换成一个一维向量，作为模型输入。特别的，英文词汇会做更细粒度的切分，比如playing 或切割成 play 和 ##ing，中文目前尚未对输入文本进行分词，直接对单子构成为本的输入单位。将词切割成更细粒度的 Word Piece 是为了解决未登录词的常见方法。

假如输入文本 ”I like dog“。下图则为 Token Embeddings 层实现过程。输入文本在送入 Token Embeddings 层之前要先进性 tokenization 处理，且两个特殊的 Token 会插入在文本开头 [CLS] 和结尾 [SEP]。[CLS]表示该特征用于分类模型，对非分类模型，该符号可以省去。[SEP]表示分句符号，用于断开输入语料中的两个句子。

Bert 在处理英文文本时只需要 30522 个词（词表大小），Token Embeddings 层会将每个词转换成 768 维向量，例子中 5 个Token 会被转换成一个 (5, 768) 的矩阵或 (1, 5, 768) 的张量。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151758706.png)
- Segment Embeddings用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务

Bert 能够处理句子对的分类任务，这类任务就是判断两个文本是否是语义相似的。句子对中的两个句子被简单的拼接在一起后送入模型中，Bert 如何区分一个句子对是两个句子呢？答案就是 Segment Embeddings。

Segement Embeddings 层有两种向量表示，前一个向量是把 0 赋值给第一个句子的各个 Token，后一个向量是把1赋值给各个 Token，问答系统等任务要预测下一句，因此输入是有关联的句子。而文本分类只有一个句子，那么 Segement embeddings 就全部是 0。

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510151804020.png)

- Position Embeddings和之前文章中的Transformer不一样，不是三角函数而是学习出来的

由于出现在文本不同位置的字/词所携带的语义信息存在差异(如 ”你爱我“ 和 ”我爱你“)，你和我虽然都和爱字很接近，但是位置不同，表示的含义不同。

在 RNN 中，第二个 ”I“ 和 第一个 ”I“ 表达的意义不一样，因为它们的隐状态不一样。对第二个 ”I“ 来说，隐状态经过 ”I think therefore“ 三个词，包含了前面三个词的信息，而第一个 ”I“ 只是一个初始值。因此，RNN 的隐状态保证在不同位置上相同的词有不同的输出向量表示。