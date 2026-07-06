# LLM

LLM(如Qwen)的Embedding生成原理:
模型结构: 基于Transformer解码器，使用单向自注意力(仅关注左侧上下文)
预训练任务: 自回归语言建模(预测下一个token)。

Embedding来源:
通常取最后一层所有token的隐藏状态，或最后一个token的隐藏状态作为序列表示(需根据任务调整)