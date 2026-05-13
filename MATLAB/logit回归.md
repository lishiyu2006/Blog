# logit回归

## 含义

Logit回归（Logistic Regression）是一种广义线性回归分析模型，专门用于解决**二分类或多分类问题**，预测事件发生的概率（0到1之间）

## 本质

利用[Sigmoid函数]([激活函数 | 下午好](https://lishiyu2006.github.io/Blog/%E9%9A%8F%E4%BE%BF%E5%90%83%E7%82%B9/LLM/2.%E5%9F%BA%E7%A1%80%E7%BD%91%E7%BB%9C%E7%BB%84%E4%BB%B6/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.html))（$(g(z) = \frac{1}{1 + e^{-z}}$)）将线性回归的输出 z（可以是 $-\infty$ 到 $+\infty$）转换为概率值 P ( 0 到 1 ) 。


