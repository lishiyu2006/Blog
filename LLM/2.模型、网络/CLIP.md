# CLIP Text Encode(prompt) - CLIP文本编码器

## A. 概述
SD使用的是OpenAi的CLIP预训练模型，即别人训练好的拿来就用。

我们需要给出提示词Prompt， 然后利用CLIP模型将文本转换成嵌入表示Context，作为UNet的一个输入。

CLIP的作用，就是将文本转换为语言信息并使其与图像信息在UNet中采用Attention更好的偶合到一起，成为了文本和图像之间的连接通道。

CLIP的训练 用到了Text-Image配对的数据集，大概4亿张，主要是通过网络爬取图片以及相应的标签。如下图所示：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510051655198.png)

## **B. 整体结构**

**CLIP的网络结构**由两部分组成：**图像 Image Encoder** **+** **文字 Text Encoder**。
## C.核心工作流程
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510051631035.png)

### 步骤 1：嵌入图像和文本（Embed image and text）

CLIP 模型里有专门的图像编码器（Image Encoder）和文本编码器（Text Encoder）。图像编码器会把输入的图像（如 “Image 1”）转化为图像嵌入向量（Image embedding），文本编码器则将输入的文本（如 “Text 1”）转化为文本嵌入向量（Text embedding），这样就把图像和文本都转化成了机器能处理的向量形式。

### 步骤 2：比较嵌入向量（Compare the embeddings）

得到图像嵌入向量和文本嵌入向量后，会对它们进行比较，判断两者是否相似。图中 “Prediction” 部分的 “1（Similar）” 表示预测为相似，“0（Not similar）” 表示预测为不相似，同时还有对应的 “Label（标签）”，标签里的 “1（Similar）” 是真实的相似情况，“0（Not similar）” 是真实的不相似情况，通过预测和标签的对比来评估模型判断的准确性。
##### 三种 “配对逻辑” 的拆解

我们可以把 “图像 - 文本” 的匹配关系拆成 **“图像对文本的匹配”** 和 **“文本对图像的匹配”** 两个视角，从而产生不同的标签组合：
1. 「1 - 1」：完美匹配
2. 「1 - 0」：图像认为匹配，文本认为不匹配（几乎不存在，因为数据是 “配对提供” 的）
### 步骤 3：更新模型（Update the models）

根据预测结果和真实标签之间的差异，来调整图像编码器和文本编码器的参数，让模型在之后能更准确地判断图像和文本是否匹配，不断优化模型的性能。

上面讲了Batch为1时的情况，当我们把训练的Batch提高到N时，其实整体的训练流程是不变的。**只是现在CLIP模型需要将N个标签文本和N个图片的两两组合预测出N2个可能的文本-图片对的余弦相似性**，即下图所示的矩阵。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510051643204.png)
在 CLIP 模型完成训练后：

输入配对的图片和文字，两个 encoder 就可以输出相似的 embedding 向量，余弦相似度接近于1；

输入不匹配的图片和文字，两个 encoder 输出向量的余弦相似度就会接近于 0。

## **D. SD中的应用**

上面讲到了CLIP模型的**Image Encoder** 和 **Text Encoder**两个模块，在**Stable Diffusion中只用到了Text Encoder模块**。

**CLIP Text Encoder模型将输入的文本Prompt进行编码，转换成Text Embeddings（文本的语义信息），作为UNet网络的Context输入，并在**UNet网络中的CrossAttention模块中，结合提取特征F**对生成图像的内容进行一定程度的控制与引导**；
## **E. Text Encoder 网络结构**

目前SD中用到的是CLIP ViT-L/14中的 Text-Encoder模型，网络结构如下：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510051705948.png)
