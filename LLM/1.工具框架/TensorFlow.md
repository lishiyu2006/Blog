# TensorFlow

## 含义
Tensor表示张量，表示里面的数据是用张量的表示的。
一、TensorFlow的计算模型——计算图
计算图是TensorFlow中最基本的一个概念，TensorFlow中的所有计算都会被转化为计算图上的节点，依据节点中传递的值，进行数值的“Flow”。

1、计算图的概念
TensorFlow是一个通过计算图的形式来表述计算的编程系统，TensorFlow中的每一个数都是计算图上的一个节点，而节点之间的边描述了节点之间的计算关系。

```python
# 定义tensor
v1 = tf.constant(1,name='v1',shape=(),dtype=tf.float32)
v2 = tf.constant(2,name='v2',shape=(),dtype=tf.float32)

# 定义一个tensor运算
add = v1 + v2

# 创建会话，运行运算，这部分会在之后详细介绍
with tf.Session() as sess:
    print(sess.run(add))   #输出：3
```
![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510072138661.png)

## 张量的三个属性

tensor具有三个很重要的属性：name、shape和type。

### name
name是tensor的唯一标识符。
```python
# 定义tensor，第一个参数是值；第二个参数是tensor的name；
#第三个参数是tensor的shape；第四个参数是tensor的type 
v1 = tf.constant(1,name='v1',shape=(),dtype=tf.float32) 
v2 = tf.constant(2,shape=(),dtype=tf.float32) 

# 定义一个tensor运算 
add = v1 + v2

print(v1) #Tensor("v1:0", shape=(), dtype=float32) 
print(v2) #Tensor("Const:0", shape=(), dtype=float32)
```
还是上面一样的代码，我们将`v2`常量定义的时候，不去定义它的name属性，然后直接打印`v1`和`v2`，我们可以看到当我们不指定name的时候，TensorFlow会自动帮我们定义name属性。

tensor通过`node:src_output`的形式展示，其中node就是tensor的name，src_output是当前tensor的第几个输出。例如：`v1:0`就是v1节点的第一个输出（src_output从0开始）
  ![image.png](https://raw.githubusercontent.com/lishiyu2006/picgo/main/cdning/202510072145894.png)
### shape
shape是纬度，它描述了tensor的纬度信息
