# python之禅
每个缩进级别使用 4 个空格。

续行应使用 Python 的圆括号、方括号和大括号内的隐式行连接或悬挂缩进垂直对齐换行元素。使用悬挂缩进时，应考虑以下事项：第一行不应包含任何参数，并且应使用进一步的缩进来清楚地将其自身与续行区分开来：
```python
# 正确：

# 与开头分隔符对齐。
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# 添加 4 个空格（额外的缩进级别）以区分参数与其他参数。
def long_function_name(
        var_one, var_two, var_three,
        var_four):
    print(var_one)

# 悬挂凹痕应增加一个水平。
foo = long_function_name(
    var_one, var_two,
    var_three, var_four)
```

```
# Wrong:

# Arguments on first line forbidden when not using vertical alignment.
foo = long_function_name(var_one, var_two,
    var_three, var_four)

# Further indentation required as indentation is not distinguishable.
def long_function_name(
    var_one, var_two, var_three,
    var_four):
    print(var_one)
```

对于连续行来说，4 空格规则是可选的。

选修的：
```
# Hanging indents *may* be indented to other than 4 spaces.
foo = long_function_name(
  var_one, var_two,
  var_three, var_four)
```

当 `if` 语句的条件部分足够长，需要将其写在多行中时，值得注意的是，两个字符的关键字（例如 `if` ）加上一个空格，再加上一个左括号的组合，会为多行条件的后续行创建自然的 4 个空格缩进。这可能会与嵌套在 `if` 语句中的缩进代码组产生视觉冲突，后者也会自然地缩进 4 个空格。本 PEP 并未明确说明如何（或是否）在视觉上进一步区分此类条件行与 `if` 语句中的嵌套代码组。在这种情况下，可接受的选项包括但不限于：

```
# No extra indentation.
if (this_is_one_thing and
    that_is_another_thing):
    do_something()

# Add a comment, which will provide some distinction in editors
# supporting syntax highlighting.
if (this_is_one_thing and
    that_is_another_thing):
    # Since both conditions are true, we can frobnicate.
    do_something()

# Add some extra indentation on the conditional continuation line.
if (this_is_one_thing
        and that_is_another_thing):
    do_something()
```

