# 枚举 and 拉链
## 枚举（emumerate）
定义：相当于给每个
```python
fruits={'苹果','香蕉','橙子'}
for index,fruit in enumerate(fruits):
	print('第{index+1}个水果是{fruit}')
```
## 拉链（zip）
```python
>>> a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 返回一个对象
>>> zipped
<zip object at 0x103abc288>
>>> list(zipped)  # list() 转换为列表
[(1, 4), (2, 5), (3, 6)]
>>> list(zip(a,c))              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]

>>> a1, a2 = zip(*zip(a,b))          # 与 zip 相反，zip(*) 可理解为解压，返回二维矩阵式
>>> list(a1)
[1, 2, 3]
>>> list(a2)
[4, 5, 6]
>>>
```
这个原理接近于压缩软件的原理

将a和b里的第i个元素组成一个元组,用一个元组迭代器,取出