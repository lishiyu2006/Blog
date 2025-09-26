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
2.其他用法:
(1).类方法（用 `@classmethod` 装饰）
```
class Person: 
	species = "Human" # 类属性 
	
	# 类方法：第一个参数是 cls @classmethod 
	def show_species(cls): 
		print(f"This is a {cls.species}") 
		
Person.show_species() # 直接通过类调用，输出：This is a Human
```
3.静态方法（用 `@staticmethod` 装饰） 
