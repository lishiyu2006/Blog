## 全连接神经网络（Full Connect Neural Network）
## 全连接神经网络原理
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509302019307.png)

就是这么一个东西，左边输入，中间计算，右边输出。

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509302020910.png)

不算输入层，上面的网络结构总共有两层，隐藏层和输出层，它们“圆圈”里的计算都是公式(4.1)和(4.2)的计算组合：

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509302022631.png)

每一级都是利用前一级的输出做输入，再经过圆圈内的组合计算，输出到下一级。
看到这里，可能很多人会疑惑，为什么要加上f(z)这个运算，这个运算的目的是为了将输出的值域压缩到（0，1），也就是所谓的归一化，因为每一级输出的值都将作为下一级的输入，只有将输入归一化了，才会避免某个输入无穷大，导致其他输入无效，最终网络训练效果非常不好。
为了解决这个