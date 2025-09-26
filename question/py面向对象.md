# py面向对象

1.定义:
```python
class name():
	# 构造方法,初始化对象,(表示属性装配),初始化调用方法
	def __init__(self,name,): 
        # self 表示类的实例本身
        self.attribute1 = parameter1  # 实例属性1
        self.attribute2 = parameter2  # 实例属性2
        self.name = name
        
    def name_method(self)
        print(f"Hello, I'm {self.name}")

p = Person("Alice")
p.greet()  # 调用时无需传递 self，输出：Hello, I'm Alice
```
