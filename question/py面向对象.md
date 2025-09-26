# py面向对象

## 1.定义:
```python
class Person():
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
## 2.其他用法:
### (1).类方法（用 `@classmethod` 装饰）
第一个参数**必须**是 `cls`（约定名称），指代当前类本身，而非实例。调用时不需要手动传递 `cls`
```python
class Person: 
	species = "Human" # 类属性，相当于上面定义了一个self.species,所以不用__init__
	
	# 类方法：第一个参数是 cls @classmethod 
	def show_species(cls): 
		print(f"This is a {cls.species}") 
		
Person.show_species() # 直接通过类调用，输出：This is a Human
```
### (2).静态方法（用 `@staticmethod` 装饰） 
不需要特殊参数（既不用 `self` 也不用 `cls`），更像一个普通函数，只是定义在类的命名空间里。

```python
class Calculator:
    # 静态方法：无特殊参数
    @staticmethod
    def add(a, b):
        return a + b

# 可以通过类或实例调用
print(Calculator.add(2, 3))  # 输出：5
```
