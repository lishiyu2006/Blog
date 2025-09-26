# 扩散模型DDPM
## 结构：级联去噪模型
## 步骤：
1. 前向扩散 
2. 反向去噪![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509262221782.png)
attention：无论是前向过程还是反向过程都是一个参数化的[马尔可夫链](https://zhuanlan.zhihu.com/p/448575579)（Markov chain），其中反向过程可用于生成数据样本（它的作用类似GAN中的生成器，只不过GAN生成器会有维度变化，而DDPM的反向过程没有维度变化）。
## 可以使用的神经网络：U-Net，Transformer
### 前向过程

