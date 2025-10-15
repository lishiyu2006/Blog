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
