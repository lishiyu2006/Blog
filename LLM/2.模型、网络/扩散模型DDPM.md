# 扩散模型DDPM
## 结构：级联去噪模型
## 步骤：
1. **前向扩散**:前向过程是加噪的过程，前向过程中图像 $x_t$ 只和上一时刻的 $x_{t-1}$ 有关, 该过程可以视为马尔科夫过程
2. **反向去噪**:逆向过程是去噪的过程，如果得到好的逆向过程就可以通过随机噪声 逐步还原出一张图像。DDPM使用神经网络  拟合逆向过程  。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509262221782.png)
attention：无论是前向过程还是反向过程都是一个参数化的[马尔可夫链](https://zhuanlan.zhihu.com/p/448575579)（Markov chain），其中反向过程可用于生成数据样本（它的作用类似GAN中的生成器，只不过GAN生成器会有维度变化，而DDPM的反向过程没有维度变化）
## 可以使用的神经网络：U-Net，Transformer

这里采用Unet实现正向的预测，整个训练过程其实就是在训练Unet网络的参数

### Unet职责:
无论在前向过程还是反向过程，Unet的职责都是根据当前的样本和时间t预测噪声。

### Gaussion Diffusion职责(**高斯扩散**):
