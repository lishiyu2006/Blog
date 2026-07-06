#### 拆分列表、元组

###### 单个拆分

```python
 data = [ 'ACME', 50, 91.1, (2012, 12, 21) ]
 name, shares, price, date = data
//或者_, shares, price, _ = data
```

> [!NOTE]
>
> 单个拆分时，要一一对应，或者用别的名称占位，如_或者ign(ignore)
>
> 也可用于字符串

###### 多个拆分

```python
first, *middle, last = data
//mindlle将继承中间的数据，一定是列表类型
//这种方式也可用于for循环中
```

###### 选定n个拆分

1. 引用库

```python
from collections import deque//导入包含生成固定长度队列的库
```

