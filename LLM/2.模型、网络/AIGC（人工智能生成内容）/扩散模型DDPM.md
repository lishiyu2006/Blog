# 扩散模型DDPM
## 结构：级联去噪模型
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509262221782.png)
## 前提
1.任何一张图像 x（无论是 x_0, x_t 还是噪声 ε）都会被转换成一个张量。
一个典型的图像张量包含以下维度信息：

**(Batch Size, Channels, Height, Width)**

- **Batch Size (B)**：一批处理多少张图片。在训练时通常大于1（如16, 32），在生成单张图时为1。
    
- **Channels (C)**：颜色通道。
    - 对于彩色图（RGB），Channels = 3。
    - 对于灰度图，Channels = 1。
        
- **Height (H)**：图像的高度（像素数）。
    
- **Width (W)**：图像的宽度（像素数）。 

2.1) $q$ 通常用来表示一个**预先定义好的、固定的、不需要学习**的概率分布。它代表了某种“事实”或“数学真理”。例如:
$q(x_t | x_{t-1})$ (前向过程)
$q(x_{t-1} | x_t, x_0)$ (真实的逆向过程 - 条件)
$q(x_{t-1} | x_t)$ (真实的逆向过程 - 无条件)

2.2) $p$ (通常带有下标 $θ$) 用来表示一个**由神经网络定义的、需要通过训练学习**的概率分布。它是一个**近似模型**。例如:
$p_θ(x_{t-1} | x_t)$ (学习的逆向过程)
## 步骤：
## 一.前向过程
 **前向扩散**:前向过程是加噪的过程，前向过程中图像 $x_t$ 只和上一时刻的 $x_{t-1}$ 有关, 该过程可以视为[马尔可夫链](https://zhuanlan.zhihu.com/p/448575579)![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291011820.png)
### 微观：
首先推出 $x_t$ 和 $x_{t-1}$ 的关系是：
$x_t = \sqrt{α_t} * x_{t-1} + \sqrt{1 - α_t} * ε_{t-1}$                                                                             （1）
因为是一个马尔科夫过程，所以最后都会归为一个稳定的状态，由此推断 $x_0$ 和 $x_t$ 也与一个稳定的公式，验证得到：
$x_t = \sqrt{\bar{α}_t} * x_0 + \sqrt{1 - \bar{α}_t} * ε$                                                                                     （2）
**参数解释**：
1. $x_0$ (原始图像)
    - **是什么**：这是你的“基酒”，是清晰、无噪声的原始图片（比如一张猫的照片）。
        
2.  $\epsilon_t$ (噪声)
    - **是什么**：这是“调味剂”，是一个从标准正态分布（高斯噪声）中随机抽取的、和 x_0 尺寸完全相同的噪声张量。
    - **怎么得到**：
		- 噪声是从标准高斯分布（正态分布$\epsilon \sim N(0, 1)$）中随机采样的，类似“随机抽数”。
		- 高斯分布是一种常见的数据分布（如身高、成绩的分布），从中采样能得到符合“自然随机性”的数值。
		
3. $\sqrt{\bar{α}_t}$ (图像权重)
    - **是什么**：这是“基酒”的**份量**。这个值由时间步 t 和预设的 Beta Schedule 共同决定。
    - **规律**：
        - 当 t 很小（刚开始加噪）时，$\bar{α}_t$ 接近 1，所以这个权重也接近 1。
        - 当 t 很大（接近纯噪声）时，$\bar{α}_t$ 近 0，所以这个权重也接近 0。
    - **怎么得到**： 
        - 系数是人工预设的固定值，非模型学习而来，目的是让加噪过程平稳可控。
        - 先设定一个微小的“每步噪声强度” $\beta_t$（如0.001~0.02），$\beta_t$ 可固定或随步数轻微增大。
        - 计算关系：$\alpha_t = 1 - \beta_t$，再通过累积得到 $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$（用于简化前向计算）。
        - 其中 $\beta_t$ 怎么算：
	        - 当前图像 $x_t$ 的条件概率分布，这是一个高斯分布，其均值为 $\sqrt{1-\beta_t}x_{t-1}$ ，方差为 $\beta_t$ ，其中 $\beta_t$ 是一个预定的噪声方差系数，$I$ 是单位矩阵。
	        - 其中不同 $t$ 的 $\beta_t$ 是预先定义好的，由时间 $1$~$T$ 逐渐的递增，可以是Linear，Cosine等，满足 : $β_1 < β_2 < ... < β_T$。
            
4. $\sqrt{1 - \bar{α}_t}$ (噪声权重)
    - **是什么**：这是“调味剂”（噪声）的**份量**。
    - **规律**：     
        - 当 t 很小的时候，这个权重接近 0。
        - 当 t 很大的时候，这个权重接近 1。

### 宏观：
前向过程的图像 $x_t$ 值和上一时刻的 $x_{t-1}$ 有关，他们之间的关系可以表示成一个多维高斯分布（条件概率），它不是描述一个单一数值的概率，而是描述一个包含成千上万个数值的**向量**的概率：
$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$
每一步 $q(x_t|x_{t-1})$ 表示:第 $t$ 步的噪声图像 $x_t$ ，是通过将第 $t-1$ 步的图像 $x_{t-1}$ 的像素值稍微缩小一点，然后给每个像素独立地添加一个强度为 $\beta_t$ 的[高斯噪声](../../高斯噪声.md)得到了一个高斯分布，再将 $x_t$ 输入就得到了一个条件概率。
因为他是一个 [马尔可夫链](https://zhuanlan.zhihu.com/p/448575579) ，所以得出的稳定的概率是**从原始图像 $x_0$ 出发，经历 $T$ 步，得到一整条特定的加噪图像序列 $x₁, x₂, ..., xᴛ$ 的联合概率**。
$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})$ 

eg.一维高斯分布函数
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291633054.png)
二维高斯分布函数
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291632813.png)
多维...
## 二.逆向过程

 **反向去噪**:DDPM使用神经网络拟合逆向过程，因为实际的去噪是未知的，想要得到好的去噪的效果，要先投喂数据，然神经网络记住这些样例的特征，让后去噪的时候就可以达成生成类似照片的效果。如果得到好的逆向过程就可以通过随机噪声，逐步还原出一张图像。

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291700117.png)

逆向过程是去噪的过程,如果得到逆向过程 $q(x_{t-1}|x_t)$ ，就可以通过随机噪声 逐步还原出一张图像。DDPM使用神经网络  拟合逆向过程。
真实过程是$q(x_{t-1}|x_t, x_0) = N(x_{t-1} | \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$ ,可以推导出:逆向过程 
$p_θ(x_{t-1}|x_t) = N(x_{t-1} | \mu_θ(x_t, t), Σ_θ(x_t, t))$

这里的均值 $\tilde{\mu}_t$ 和方差 $\tilde{\beta}_t$ 是属于**逆向过程后验分布**（reverse process posterior）的参数，它们是通过数学推导得出的“真实”去噪步骤的参数，而不是我们之前在前向（加噪）过程中定义的原始参数 $β_t$。

|      | 第一行                            | 第二行                            |
| ---- | ------------------------------ | ------------------------------ |
| 角色   | 标准答案 **(Ground Truth)**        | 学生的回答 **(Model's Prediction)** |
| 前提条件 | 知道问题 $(x_t)$ 和答案 $(x_0)$       | 只知道问题 $(x_t)$                  |
| 均值   | $\tilde{\mu}_t$ (波浪号): 真实的均值   | $\mu_θ$: 预测的均值                 |
| 方差   | $\tilde{\beta}_t$ (波浪号): 真实的方差 | $Σ_θ$: 预测的方差                   |

$x_{t-1} = (1 / \sqrt{α_t}) * (x_t - ((1 - α_t) / \sqrt{1 - ᾱ_t}) * ε_θ(x_t, t)) + σ_t * z$                                  （3）
**参数解释**:
- $x_{t-1}$: **目标**。我们想要生成的、噪声更少一点的上一时刻的图像。
    
- $x_t$: **输入**。当前时刻的、带有较多噪声的图像。
    
- $ε_θ(x_t, t)$: 模型的核心——噪声预测器。
    - 这是一个由参数 $θ$ 构成的**神经网络**（通常是U-Net架构）。
    - 它的任务是接收当前带噪图像 $x_t$ 和当前的时间步 $t$ 作为输入，然后输出一个**预测的噪声**。
    - 这个神经网络是通过大量训练学成的，它能够精准地识别出在任意时刻 $t$ 的图像 $x_t$ 中所包含的噪声模式。
        
- $(1 - α_t) / \sqrt{1 - ᾱ_t}$: **噪声的缩放系数**。
    - 这是一个基于预设的噪声表（$α_t$ 和 $ᾱ_t$）计算出的系数。
    - 它的作用是正确地缩放神经网络预测出的噪声 $ε_θ$，使其与我们理论上要从 $x_t$ 中减去的噪声量相匹配。你可以把它理解为一个“校准”步骤。
        
- $1 / \sqrt{α_t}$: **图像的缩放系数**。
    - 在前向加噪过程中，每一步图像的信号强度都会被 $\sqrt{α_t}$ 稍微衰减一点。
    - 因此，在反向去噪时，我们需要乘以 $1 / \sqrt{α_t}$ 把它“放大”回来，恢复其在上一时刻应有的尺度。
        
- $σ_t * z$: **随机噪声项**。
    - $z$: 一个从标准正态分布中采样的新随机噪声,$z \sim N(0, 1)$。
    - $σ_t$: 是噪声估计函数（一般使用NN模型）,控制这个新噪声的强度。
    - 这一项非常重要！它为生成过程引入了**随机性**。这意味着即使从同一个 $x_t$ 开始去噪，每次生成的 $x_{t-1}$ 也会有微小的、合理的差异。这正是扩散模型能够生成多样化结果的关键。如果没有这一项，生成过程将是完全确定的。

attention：
1.无论是前向过程还是反向过程都是一个参数化的[马尔可夫链](https://zhuanlan.zhihu.com/p/448575579)（Markov chain）（即当经过一定的训练负责还原或者加噪1%的这部分会最终稳定下来），其中反向过程可用于生成数据样本（它的作用类似GAN中的生成器，只不过GAN生成器会有维度变化，而DDPM的反向过程没有维度变化）
## 训练
DDPM 论文中通过神经网络（通常是 U-Net）拟合噪声预测模型 $ε_θ(x_t, t)$ ，以计算 $x_{t-1}$ 。那么损失函数可以使用MSE误差，表示如下：
$Loss = ||ε - ε_θ(x_t, t)||²$
$=||ε - ε_θ(\sqrt{ᾱ_t}x_0 + \sqrt{1 - ᾱ_t}ε, t)||²$
整个训练过程可以表示如下：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291902220.png)

论文中的DDPM训练过程如下所示：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291903786.png)
## DDPM如何生成图片
在得到预估噪声  后，就可以按公式（3）逐步得到最终的图片  ，整个过程表示如下：

网上有很多DDPM的实现，包括[论文中基于tensorflow的实现](https://link.zhihu.com/?target=https%3A//github.com/hojonathanho/diffusion)，还有[基于pytorch的实现](https://link.zhihu.com/?target=https%3A//github.com/xiaohu2015/nngen/blob/main/models/diffusion_models/ddpm_mnist.ipynb)，但是由于代码结构复杂，很难上手。为了便于理解以及快速运行，我们将代码合并在一个文件里面，基于tf2.5实现，直接copy过去就能运行。代码主要分为3个部分：DDPM前向和反向过程（都在GaussianDiffusion一个类里面实现）、模型训练过程、新图片生成过程。

**DDPM前向和后向过程代码如下：**
```python
import pandas as pd
import numpy as np
import os
import numpy as np
import sys
import pandas as pd
from numpy import arange
import math
import pyecharts
import sys,base64,urllib,re
import multiprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score
import warnings 
from optparse import OptionParser
import logging
import logging.config
import time
import tensorflow as tf
from sklearn.preprocessing import normalize
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LeakyReLU, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets
from tensorflow import keras
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# beta schedule
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
            
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        
        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(self.alphas_cumprod), dtype=tf.float32)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - self.alphas_cumprod), dtype=tf.float32)
        self.log_one_minus_alphas_cumprod = tf.constant(np.log(1. - alphas_cumprod), dtype=tf.float32)
        self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1. / alphas_cumprod), dtype=tf.float32)
        self.sqrt_recipm1_alphas_cumprod = tf.constant(np.sqrt(1. / alphas_cumprod - 1), dtype=tf.float32)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(self.posterior_variance, 1e-20)), dtype=tf.float32)
        
        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), dtype=tf.float32)
        
        self.posterior_mean_coef2 = tf.constant(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod), dtype=tf.float32)
    
    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = tf.gather(a, t)
        assert out.shape == [bs]
        return tf.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = tf.random.normal(shape=x_start.shape)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # predict noise using model
        pred_noise = model([x_t, t])
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, -1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_sample(self, model, x_t, t, clip_denoised=True):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)
        noise = tf.random.normal(shape=x_t.shape)
        # no noise when t == 0
        nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [x_t.shape[0]] + [1] * (len(x_t.shape) - 1))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
        return pred_img
    
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        # start from pure noise (for each example in the batch)
        img = tf.random.normal(shape=shape)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, tf.fill([batch_size], i))
            imgs.append(img.numpy())
        return imgs
    
    def sample(self, model, image_size, batch_size=8, channels=3):
        return self.p_sample_loop(model, shape=[batch_size, image_size, image_size, channels])
    
    # compute train losses
    def train_losses(self, model, x_start, t):
        # generate random noise
        noise = tf.random.normal(shape=x_start.shape)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        model.train_on_batch([x_noisy, t], noise)
        predicted_noise = model([x_noisy, t])
        loss = model.loss(noise, predicted_noise)
        return loss

# Load the dataset
def load_data():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    return (x_train, y_train)

print("forward diffusion: q(x_t | x_0)")
timesteps = 500
X_train, y_train = load_data()
gaussian_diffusion = GaussianDiffusion(timesteps)
plt.figure(figsize=(16, 8))
x_start = X_train[7:8]
for idx, t in enumerate([0, 50, 100, 200, 499]):
    x_noisy = gaussian_diffusion.q_sample(x_start, t=tf.convert_to_tensor([t]))
    x_noisy = x_noisy.numpy()
    x_noisy = x_noisy.reshape(28, 28)
    plt.subplot(1, 5, 1 + idx)
    plt.imshow(x_noisy, cmap="gray")
    plt.axis("off")
    plt.title(f"t={t}")
```
运行上面代码，我们可以得到前向过程的效果如下图所示：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291858231.png)
从图中可以看出，随着不断加噪，图片变得越来越模糊，最后变成随机噪声。
**接下来是模型训练过程，我们先使用一个简单的残差网络模型，代码如下：**
```python
# ResNet model
class ResNet(keras.layers.Layer):
    
    def __init__(self, in_channels, out_channels, name='ResNet', **kwargs):
        super(ResNet, self).__init__(name=name, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def get_config(self):
        config = super(ResNet, self).get_config()
        config.update({'in_channels': self.in_channels, 'out_channels': self.out_channels})
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
    def build(self, input_shape):
        self.conv1 = Sequential([
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same')
        ])
        self.conv2 = Sequential([
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same', name='conv2')
        ])

    def call(self, inputs_all, dropout=None, **kwargs):
        """
        `x` has shape `[batch_size, height, width, in_dim]`
        """
        x, t = inputs_all
        h = self.conv1(x)
        h = self.conv2(h)
        h += x
        
        return h

def build_DDPM(nn_model):
    nn_model.trainablea = True
    inputs = Input(shape=(28, 28, 1,))
    timesteps=Input(shape=(1,))
    outputs = nn_model([inputs, timesteps])
    ddpm = Model(inputs=[inputs, timesteps], outputs=outputs)
    ddpm.compile(loss=keras.losses.mse, optimizer=Adam(5e-4))
    return ddpm

# train ddpm
def train_ddpm(ddpm, gaussian_diffusion, epochs=1, batch_size=128, timesteps=500):
    
    #Loading the data
    X_train, y_train = load_data()
    step_cont = len(y_train) // batch_size
    
    step = 1
    for i in range(1, epochs + 1):
        for s in range(step_cont):
            if (s+1)*batch_size > len(y_train):
                break
            images = X_train[s*batch_size:(s+1)*batch_size]
            images = tf.reshape(images, [-1, 28, 28 ,1])
            t = tf.random.uniform(shape=[batch_size], minval=0, maxval=timesteps, dtype=tf.int32)
            loss = gaussian_diffusion.train_losses(ddpm, images, t)
            if step == 1 or step % 100 == 0:
                print("[step=%s]\tloss: %s" %(step, str(tf.reduce_mean(loss).numpy())))
            step += 1

print("[ResNet] train ddpm")
nn_model = ResNet(in_channels=1, out_channels=1)
ddpm = build_DDPM(nn_model)
gaussian_diffusion = GaussianDiffusion(timesteps=500)
train_ddpm(ddpm, gaussian_diffusion, epochs=10, batch_size=64, timesteps=500)

print("[ResNet] generate new images")
generated_images = gaussian_diffusion.sample(ddpm, 28, batch_size=64, channels=1)
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)

imgs = generated_images[-1].reshape(8, 8, 28, 28)
for n_row in range(8):
    for n_col in range(8):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")

print("[ResNet] show the denoise steps")
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(16, 16)

for n_row in range(16):
    for n_col in range(16):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        t_idx = (timesteps // 16) * n_col if n_col < 15 else -1
        img = generated_images[t_idx][n_row].reshape(28, 28)
        f_ax.imshow((img+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")
```
运行上面代码，我们能得到训练Loss如下：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291859988.png)
训练完后生成的图片如下图所示：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291859839.png)
可以看到效果非常差，基本看不出是手写数字
实际应用中一般是基于U-Net模型，模型结构如下：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291859252.png)
**使用U-Net进行训练的代码如下：**
```python
"""
U-Net model
as proposed in https://arxiv.org/pdf/1505.04597v1.pdf
"""

# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)   
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = tf.exp(
        -math.log(max_period) * tf.experimental.numpy.arange(start=0, stop=half, step=1, dtype=tf.float32) / half
    )
    args = timesteps[:, ] * freqs
    embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
    if dim % 2:
        embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

# upsample
class Upsample(keras.layers.Layer):
    def __init__(self, channels, use_conv=False, name='Upsample', **kwargs):
        super(Upsample, self).__init__(name=name, **kwargs)
        self.use_conv = use_conv
        self.channels = channels
    
    def get_config(self):
        config = super(Upsample, self).get_config()
        config.update({'channels': self.channels, 'use_conv': self.use_conv})
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
    def build(self, input_shape):
        if self.use_conv:
            self.conv = keras.layers.Conv2D(filters=self.channels, kernel_size=3, padding='same')

    def call(self, inputs_all, dropout=None, **kwargs):
        x, t = inputs_all
        x = tf.image.resize_with_pad(x, target_height=x.shape[1]*2, target_width=x.shape[2]*2, method='nearest')
#         if self.use_conv:
#             x = self.conv(x)
        return x

# downsample
class Downsample(keras.layers.Layer):
    def __init__(self, channels, use_conv=True, name='Downsample', **kwargs):
        super(Downsample, self).__init__(name=name, **kwargs)
        self.use_conv = use_conv
        self.channels = channels
    
    def get_config(self):
        config = super(Downsample, self).get_config()
        config.update({'channels': self.channels, 'use_conv': self.use_conv})
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
    def build(self, input_shape):
        if self.use_conv:
            self.op = keras.layers.Conv2D(filters=self.channels, kernel_size=3, strides=2, padding='same')
        else:
            self.op = keras.layers.AveragePooling2D(strides=(2, 2))

    def call(self, inputs_all, dropout=None, **kwargs):
        x, t = inputs_all
        return self.op(x)

# Residual block
class ResidualBlock(keras.layers.Layer):
    
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        time_channels, 
        use_time_emb=True,
        name='residul_block', **kwargs
    ):
        super(ResidualBlock, self).__init__(name=name, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.use_time_emb = use_time_emb
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'time_channels': self.time_channels, 
            'in_channels': self.in_channels, 
            'out_channels': self.out_channels,
            'use_time_emb': self.use_time_emb
        })
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
    def build(self, input_shape):
        self.dense_ = keras.layers.Dense(units=self.out_channels, activation=None)
        self.dense_short = keras.layers.Dense(units=self.out_channels, activation=None)
        
        self.conv1 = [
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same')
        ]
        self.conv2 = [
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same', name='conv2')
        ]
        self.conv3 = [
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=1, name='conv3')
        ]
        
        self.activate = keras.layers.LeakyReLU()

    def call(self, inputs_all, dropout=None, **kwargs):
        """
        `x` has shape `[batch_size, height, width, in_dim]`
        `t` has shape `[batch_size, time_dim]`
        """
        x, t = inputs_all
        h = x
        for module in self.conv1:
            h = module(x)
        
        # Add time step embeddings
        if self.use_time_emb:
            time_emb = self.dense_(self.activate(t))[:, None, None, :]
            h += time_emb
        for module in self.conv2:
            h = module(h)
        
        if self.in_channels != self.out_channels:
            for module in self.conv3:
                x = module(x)
            return h + x
        else:
            return h + x

# Attention block with shortcut
class AttentionBlock(keras.layers.Layer):
    
    def __init__(self, channels, num_heads=1, name='attention_block', **kwargs):
        super(AttentionBlock, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.num_heads = num_heads
        self.dense_layers = []
        
    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update({'channels': self.channels, 'num_heads': self.num_heads})
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
    def build(self, input_shape):
        for i in range(3):
            dense_ = keras.layers.Conv2D(filters=self.channels, kernel_size=1)
            self.dense_layers.append(dense_)
        self.proj = keras.layers.Conv2D(filters=self.channels, kernel_size=1)
    
    def call(self, inputs_all, dropout=None, **kwargs):
        inputs, t = inputs_all
        H = inputs.shape[1]
        W = inputs.shape[2]
        C = inputs.shape[3]
        qkv = inputs
        q = self.dense_layers[0](qkv)
        k = self.dense_layers[1](qkv)
        v = self.dense_layers[2](qkv)
        attn = tf.einsum("bhwc,bHWc->bhwHW", q, k)* (int(C) ** (-0.5))
        attn = tf.reshape(attn, [-1, H, W, H * W])
        attn = tf.nn.softmax(attn, axis=-1)
        attn = tf.reshape(attn, [-1, H, W, H, W])
        
        h = tf.einsum('bhwHW,bHWc->bhwc', attn, v)
        h = self.proj(h)
        
        return h + inputs

# upsample
class UNetModel(keras.layers.Layer):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4,
        name='UNetModel',
        **kwargs
    ):
        super(UNetModel, self).__init__(name=name, **kwargs)
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.time_embed_dim = self.model_channels * 4
    
    def build(self, input_shape):
        
        # time embedding
        self.time_embed = [
            keras.layers.Dense(self.time_embed_dim, activation=None),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(self.time_embed_dim, activation=None)
        ]
        
        # down blocks
        self.conv = keras.layers.Conv2D(filters=self.model_channels, kernel_size=3, padding='same')
        self.down_blocks = []
        down_block_chans = [self.model_channels]
        ch = self.model_channels
        ds = 1
        index = 0
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                
                layers = [
                    ResidualBlock(
                        in_channels=ch, 
                        out_channels=mult * self.model_channels, 
                        time_channels=self.time_embed_dim,
                        name='resnet_'+str(index)
                    )
                ]
                index += 1
                ch = mult * self.model_channels
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=self.num_heads))
                self.down_blocks.append(layers)
                down_block_chans.append(ch)
        
            if level != len(self.channel_mult) - 1: # don't use downsample for the last stage
                self.down_blocks.append(Downsample(ch, self.conv_resample))
                down_block_chans.append(ch)
                ds *= 2
                
        # middle block
        self.middle_block = [
            ResidualBlock(ch, ch, self.time_embed_dim, name='res1'),
            AttentionBlock(ch, num_heads=self.num_heads),
            ResidualBlock(ch, ch, self.time_embed_dim, name='res2')
        ]
        
        # up blocks
        self.up_blocks = []
        index = 0
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                layers = []
                layers.append(
                    ResidualBlock(
                        in_channels=ch + down_block_chans.pop(), 
                        out_channels=self.model_channels * mult, 
                        time_channels=self.time_embed_dim,
                        name='up_resnet_'+str(index)
                    )
                )
                
                layer_num = 1
                ch = self.model_channels * mult
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=self.num_heads))
                if level and i == self.num_res_blocks:
                    layers.append(Upsample(ch, self.conv_resample))
                    ds //= 2
                self.up_blocks.append(layers)
                
                index += 1
            
        
        self.out = Sequential([
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same')
        ])

    def call(self, inputs, dropout=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x H x W x C] Tensor of inputs. N, H, W, C
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x, timesteps = inputs
        hs = []
        
        # time step embedding
        emb = timestep_embedding(timesteps, self.model_channels)
        for module in self.time_embed:
            emb = module(emb)
        
        # down stage
        h = x
        h = self.conv(h)
        hs = [h]
        for module_list in self.down_blocks:
            if isinstance(module_list, list):
                for module in module_list:
                    h = module([h, emb])
            else:
                h = module_list([h, emb])
            hs.append(h)
            
        # middle stage
        for module in self.middle_block:
            h = module([h, emb])
        
        # up stage
        for module_list in self.up_blocks:
            cat_in = tf.concat([h, hs.pop()], axis=-1)
            h = cat_in
            for module in module_list:
                h = module([h, emb])
        
        return self.out(h)

print("[U-Net] train ddpm")
nn_model = UNetModel(
    in_channels=1,
    model_channels=96,
    out_channels=1,
    channel_mult=(1, 2, 2),
    attention_resolutions=[]
)
ddpm = build_DDPM(nn_model)
gaussian_diffusion = GaussianDiffusion(timesteps=500)
train_ddpm(ddpm, gaussian_diffusion, epochs=10, batch_size=64, timesteps=500)

print("[U-Net] generate new images")
generated_images = gaussian_diffusion.sample(ddpm, 28, batch_size=64, channels=1)
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)

imgs = generated_images[-1].reshape(8, 8, 28, 28)
for n_row in range(8):
    for n_col in range(8):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")

print("[U-Net] show the denoise steps")
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(16, 16)

for n_row in range(16):
    for n_col in range(16):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        t_idx = (timesteps // 16) * n_col if n_col < 15 else -1
        img = generated_images[t_idx][n_row].reshape(28, 28)
        f_ax.imshow((img+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")
```
运行上面代码，训练Loss如下：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291900921.png)
训练好后生成的图片如下：
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291900793.png)
可以看到明显好于前面基于ResNet实现的效果，而整个反向过程（去噪过程）的效果如下图所示。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509291900494.png)
