# List

### .concat()

```python
import pandas as pd

# 合并多个 DataFrame（纵向堆叠）
df_list = [df1, df2, df3]
combined = pd.concat(df_list, ignore_index=True)

# 合并 Series（变成一列）
s1 = pd.Series(['A', 'B'])
s2 = pd.Series(['C', 'D'])
all_genes = pd.concat([s1, s2], ignore_index=True)
# 结果: ['A', 'B', 'C', 'D']
```

~~~python

squares = [1,2,3]
squares += [4,5,6]

#squares结果是：[1,2,3,4,5,6]
~~~