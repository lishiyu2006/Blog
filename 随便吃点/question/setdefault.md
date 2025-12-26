# setdefault

`setdefault()` 是 Python 字典（`dict`）的一个内置方法，用于**获取指定键的值**；如果该键不存在，则**为该键设置一个默认值**并返回该默认值。它的核心作用是 “查询键值，不存在则添加默认值”，避免了单独判断键是否存在的冗余代码。

```python
dict.setdefault(key, default_value)
```

- **`key`**：要查询的键。
- **`default_value`**：可选参数，当 `key` 不存在时，为该键设置的默认值（默认值为 `None`）。