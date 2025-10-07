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
##### 步骤3：三种 “配对逻辑” 的拆解

我们可以把 “图像 - 文本” 的匹配关系拆成 **“图像对文本的匹配”** 和 **“文本对图像的匹配”** 两个视角，从而产生不同的标签组合：
1. 「1 - 1」：完美匹配
2. 「1 - 0」：图像认为匹配，文本认为不匹配（几乎不存在，因为数据是 “配对提供” 的）
##### 步骤 3：更新模型（Update the models）

根据预测结果和真实标签之间的差异，来调整图像编码器和文本编码器的参数，让模型在之后能更准确地判断图像和文本是否匹配，不断优化模型的性能。


上面讲了Batch为1时的情况，当我们把训练的Batch提高到N时，其实整体的训练流程是不变的。**只是现在CLIP模型需要将N个标签文本和N个图片的两两组合预测出$N^2$个可能的文本-图片对的余弦相似性**，即下图所示的矩阵。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510051643204.png)
### **具体计算过程**：
#### 一、核心符号定义

- 设一批训练数据包含 N 个图文对：$(t_1, i_1), (t_2, i_2), ..., (t_N, i_N)$，其中 $t_i$ 为文本，$i_i$ 为对应的图像。
- 文本编码器输出文本特征向量：$T_i = \text{TextEncoder}(t_i)$（维度为 d 的向量）。
- 图像编码器输出图像特征向量：$I_i = \text{ImageEncoder}(i_i)$（维度同样为 d 的向量）。
- 对所有向量进行 $L_2$ 归一化：$\hat{T}_i = \frac{T_i}{\|T_i\|}$，$\hat{I}_i = \frac{I_i}{\|I_i\|}$（确保向量模长为 1，简化相似度计算）。
扩展：$L_2$ 归一化（也叫 “欧几里得归一化”）的核心是：**把向量的 “长度（模长）” 缩放为 1**，但保持向量的 “方向” 不变。
- $T_i$ 是文本编码器输出的原始文本向量，$\|T_i\|$ 是 $T_i$ 的 **$L_2$ 范数**（即向量的模长，计算方式为 $\|T_i\| = \sqrt{T_i^{(1)^2} + T_i^{(2)^2} + \dots + T_i^{(d)^2}}$，d 是向量维度）。
- $\hat{T}_i = \frac{T_i}{\|T_i\|}$ 表示：把原始向量 $T_i$ 的每个元素，都除以它的模长 $\|T_i\|$，得到新向量 $\hat{T}_i$。
- 对图像向量 $I_i$ 的处理 $\hat{I}_i = \frac{I_i}{\|I_i\|}$ 逻辑完全相同。
#### 二、相似度计算

文本与图像的语义相似度通过**向量点积**计算（归一化后等价于余弦相似度）：$s_{ij} = \hat{T}_i \cdot \hat{I}_j = \sum_{k=1}^d \hat{T}_i^{(k)} \cdot \hat{I}_j^{(k)}$其中 $s_{ij}$ 表示第 i 个文本与第 j 个图像的相似度得分，值越大表示语义越接近。

#### 三、对比损失函数（InfoNCE Loss）

CLIP 的训练目标是最小化**对比损失**，让匹配的图文对（\(i=j\)）的相似度 \(s_{ii}\) 远大于不匹配的（\(i \neq j\)）。损失函数定义为：

##### 1. 单样本损失（对第 i 个图文对）

对于文本 \(t_i\)，其正样本是对应的图像 \(i_i\)，负样本是其他所有图像（\(i_1, ..., i_{i-1}, i_{i+1}, ..., i_N\)）。文本视角的损失：\(L_i^{\text{text}} = -\log \left( \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^N \exp(s_{ij} / \tau)} \right)\)

同理，对于图像 \(i_i\)，其正样本是对应的文本 \(t_i\)，负样本是其他所有文本，图像视角的损失：\(L_i^{\text{image}} = -\log \left( \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^N \exp(s_{ji} / \tau)} \right)\)

其中 \(\tau\) 是**温度参数**（通常取 0.07），用于缩放相似度得分，控制分布的陡峭程度（\(\tau\) 越小，对差异的惩罚越敏感）。

##### 2. 批量总损失

对所有 N 个样本的文本和图像视角损失取平均，得到总损失：\(L = \frac{1}{2N} \sum_{i=1}^N \left( L_i^{\text{text}} + L_i^{\text{image}} \right)\)

代入单样本损失公式后展开：\(L = -\frac{1}{2N} \sum_{i=1}^N \left[ \log \left( \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^N \exp(s_{ij}/\tau)} \right) + \log \left( \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^N \exp(s_{ji}/\tau)} \right) \right]\)

#### 四、推导逻辑与目标

- 当模型训练理想时，匹配的图文对相似度 \(s_{ii}\) 应显著大于所有不匹配的 \(s_{ij}\)（\(i \neq j\)）。
- 此时分子 \(\exp(s_{ii}/\tau)\) 远大于分母中的其他项，导致 \(\log\) 内的值接近 1，损失 L 趋近于 0。
- 反之，若模型无法区分匹配 / 不匹配对，损失会增大，通过反向传播迫使模型调整文本编码器和图像编码器的参数，最终让语义相关的图文对在向量空间中更接近。

 在 CLIP 模型完成训练后：

输入配对的图片和文字，两个 encoder 就可以输出相似的 embedding 向量，余弦相似度接近于1；

输入不匹配的图片和文字，两个 encoder 输出向量的余弦相似度就会接近于 0。

## **D. SD中的应用**

上面讲到了CLIP模型的**Image Encoder** 和 **Text Encoder**两个模块，在**Stable Diffusion中只用到了Text Encoder模块**。

**CLIP Text Encoder模型将输入的文本Prompt进行编码，转换成Text Embeddings（文本的语义信息），作为UNet网络的Context输入，并在**UNet网络中的CrossAttention模块中，结合提取特征F**对生成图像的内容进行一定程度的控制与引导**；
## **E. Text Encoder 网络结构**

目前SD中用到的是CLIP ViT-L/14中的 Text-Encoder模型，网络结构如下：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510051705948.png)

由上图可见，Text Encoder 是由 Transformer 中的 SelfAttention + FeedForward 组成，一共有12个 TextEncoder_Block 模块，模型参数大小为123M,其中特征维度为768，token数量为77，故输出的 Text_Embedding 的维度为77x768。

## **F. Text Encoder代码**
```python
import torch 
import torch.nn as nn
from transformers import CLIPTokenizer,CLIPTextModel

class Text_Encoder(nn.Module):
    '''
    clip-vit-large-patch14为模型参数,需要提前单独下载并保存于本地
    '''
    def __init__(self,version='/本地路径/clip-vit-large-patch14',device='cuda',max_length=77,freeze=True):
        super(Text_Encoder,self).__init__()
        # 定义文本的tokenizer和transformer
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        
        self.device = device 
        self.max_length = max_length
        # 冻结模型参数
        if freeze:
            self.freeze()
            
    
    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False                      
            
            
    def forward(self,text):
        # 对输入图片进行分词并编码,长度不足时直接padding到77
        batch_encoding = self.tokenizer(text,truncation=True,max_length=self.max_length,return_length=True,
                                        return_overflowing_tokens=False,padding='max_length',return_tensors='pt')
        # 拿出input_ids然后传入transformer进行特征提取
        tokens = batch_encoding['input_ids'].to(self.device)
        outputs = self.transformer(input_ids=tokens,output_hidden_states=False)
        out = outputs.last_hidden_state
        return out 
```

### **G. 注意事项**

<1>
CLIP在训练时设定的最大Token数量为77，故SD在前向推理时：
- 如输入的Prompt的Token数量超过77，则会采取**切片操作**，只取前77个；  
- 如输入Token数量小于77，则采取**Padding操作**，得到77x768;

<2>
在SD模型训练过程中，CLIP 的Text Encoder的模型参数是冻结Freeze的，无需重新训练；
**原因**：预训练的CLIP模型已经足以满足后面的任务需求。