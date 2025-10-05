# CLIP Text Encode(prompt) - CLIP文本编码器
**CLIP（Contrastive Language - Image Pre - training，对比语言 - 图像预训练）模型**的核心工作流程。
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