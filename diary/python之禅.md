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

```python
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
```python
# Hanging indents *may* be indented to other than 4 spaces.
foo = long_function_name(
  var_one, var_two,
  var_three, var_four)
```

当 `if` 语句的条件部分足够长，需要将其写在多行中时，值得注意的是，两个字符的关键字（例如 `if` ）加上一个空格，再加上一个左括号的组合，会为多行条件的后续行创建自然的 4 个空格缩进。这可能会与嵌套在 `if` 语句中的缩进代码组产生视觉冲突，后者也会自然地缩进 4 个空格。本 PEP 并未明确说明如何（或是否）在视觉上进一步区分此类条件行与 `if` 语句中的嵌套代码组。在这种情况下，可接受的选项包括但不限于：

```python
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

多行结构中的右括号/方括号/圆括号可以排列在列表最后一行的第一个非空白字符下方，如下所示：

```python
my_list = [
    1, 2, 3,
    4, 5, 6,
    ]
result = some_function_that_takes_arguments(
    'a', 'b', 'c',
    'd', 'e', 'f',
    )
```

或者它可以排列在多行构造开始的行的第一个字符下面，如下所示：

```python
my_list = [
    1, 2, 3,
    4, 5, 6,
]
result = some_function_that_takes_arguments(
    'a', 'b', 'c',
    'd', 'e', 'f',
)
```

### [制表符还是空格？](https://peps.python.org/pep-0008/#tabs-or-spaces)

空格是首选的缩进方法。

制表符应该仅用于与已经使用制表符缩进的代码保持一致。

Python 不允许混合使用制表符和空格进行缩进。

### [最大行长度](https://peps.python.org/pep-0008/#maximum-line-length)

将所有行限制为最多 79 个字符。

对于结构限制较少（文档字符串或注释）的流动长文本块，行长应限制为 72 个字符。

限制所需的编辑器窗口宽度使得可以并排打开多个文件，并且在使用在相邻列中显示两个版本的代码审查工具时效果很好。

大多数工具的默认换行会破坏代码的视觉结构，使其更加难以理解。这些限制是为了避免在窗口宽度设置为 80 的编辑器中换行，即使工具在换行时在最后一列放置了标记符号。某些基于 Web 的工具可能根本不提供动态换行功能。

有些团队强烈倾向于更长的行长度。对于由能够就此问题达成一致的团队独家或主要维护的代码，可以将行长度限制增加到 99 个字符，前提是注释和文档字符串仍然在 72 个字符处换行。

Python 标准库比较保守，要求将行限制为 79 个字符（将文档字符串/注释限制为 72 个）。

换行较长的行的首选方法是使用 Python 隐含的圆括号、方括号和花括号内的续行功能。长行可以通过将表达式括在圆括号中来拆分为多行。应优先使用圆括号而不是反斜杠进行续行。

有时反斜杠可能仍然适用。例如，在 Python 3.10 之前，较长的多个 `with` 语句无法使用隐式继续，因此在这种情况下反斜杠是可以接受的：