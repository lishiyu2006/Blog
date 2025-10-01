# python之禅
每个缩进级别使用 4 个空格。

续行应使用 Python 的圆括号、方括号和大括号内的隐式行连接或悬挂缩进垂直对齐换行元素。使用悬挂缩进时，应考虑以下事项：第一行不应包含任何参数，并且应使用进一步的缩进来清楚地将其自身与续行区分开来：
```python
# Correct:

# Aligned with opening delimiter.
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# Add 4 spaces (an extra level of indentation) to distinguish arguments from the rest.
def long_function_name(
        var_one, var_two, var_three,
        var_four):
    print(var_one)

# Hanging indents should add a level.
foo = long_function_name(
    var_one, var_two,
    var_three, var_four)
```