# 全连接神经网络（Full Connect Neural Network）
## 全连接神经网络原理
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509302019307.png)

就是这么一个东西，左边输入，中间计算，右边输出。

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509302020910.png)

不算输入层，上面的网络结构总共有两层，隐藏层和输出层，它们“圆圈”里的计算都是公式(4.1)和(4.2)的计算组合：

![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509302022631.png)

每一级都是利用前一级的输出做输入，再经过圆圈内的组合计算，输出到下一级。
看到这里，可能很多人会疑惑，为什么要加上f(z)这个运算，这个运算的目的是为了将输出的值域压缩到（0，1），也就是所谓的归一化，因为每一级输出的值都将作为下一级的输入，只有将输入归一化了，才会避免某个输入无穷大，导致其他输入无效，最终网络训练效果非常不好。
为了解决这个问题设计了，一个反向传播算法，反向传播的过程：![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509302035780.png)
1. 前向传播 (Forward Propagation): 将训练数据输入网络，从输入层开始，逐层计算，直到输出层得到预测结果。
2. 反向传播 (Backward Propagation):
计算误差: 将网络的预测结果与真实标签进行比较，计算出损失函数的值。
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202509302051048.png)
公式经过化简，我们可以看到A、B、C、D、E、F都是常系数，未知数就是w 和b ，也就是为了让Loss 最小，我们要求解出最佳的w 和b 。这时我们稍微想象一下，如果这是个二维空间，那么我们相当于要找一条曲线，让它与坐标轴上所有样本点距离最小。如下![Uploading file...kxz9e]()

梯度计算: 利用微积分中的链式法则，从输出层开始，逐层向后计算损失函数对网络中每个权重和偏置的梯度（偏导数）。 “反向传播”这个名字也正源于此，即误差从后向前传播。

权重更新: 使用计算出的梯度来更新网络中的权重和偏置，通常会结合梯度下降等优化算法。 更新的目的是让网络在下一次前向传播中做出更准确的预测。


