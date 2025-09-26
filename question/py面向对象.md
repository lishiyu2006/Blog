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

## 其他参数
### 一~类的私有性：
```python
class people: 
	#定义基本属性 
	name = ''
	age = 0
	#定义私有属性,私有属性在类外部无法直接进行访问
	__weight = 0
```

#### 1.类外部直接访问时受限
私有属性的主要作用就是限制在类的外部直接访问
```python
class MyClass:
    def __init__(self):
        self.__private = "我是私有属性"  # 私有实例属性

obj = MyClass()
print(obj.__private)  # 报错：AttributeError: 'MyClass' object has no attribute '__private'
```

#### 2.子类中不能直接访问父类的私有属性
父类的私有属性会被 "隐藏"，子类无法直接通过属性名访问，即使子类继承了父类也不行。
```python
class Parent:
    def __init__(self):
        self.__private = "父类私有属性"

class Child(Parent):
    def get_parent_private(self):
        return self.__private  # 报错：无法直接访问父类的私有属性

child = Child()
child.get_parent_private()  # AttributeError: 'Child' object has no attribute '__private'
```

### 二~父类（基类）和子类（派生类）
#### 基本语法：
```python
# 定义父类（基类）
class ParentClass:
    # 父类的属性和方法
    pass

# 定义子类（派生类），继承自父类
class ChildClass(ParentClass):
    # 子类的属性和方法（可以继承父类的，也可以新增或重写）
    pass
```
### 示例 1：简单的父类和子类
父类有分支子类；
这些分支子类有重写，继承（**多层继承**：子类可以继承自 "子类的子类"；**多继承**：一个子类可以同时继承多个父类（用逗号分隔））等功能。
```python
# 父类：动物
class Animal:
    # 类属性
    category = "生物"
    
    # 初始化方法
    def __init__(self, name):
        self.name = name  # 实例属性
    
    # 父类的方法
    def eat(self):
        print(f"{self.name} 在吃东西")

# 子类：狗（继承自动物）
class Dog(Animal):
    # 子类新增的方法
    def bark(self):
        print(f"{self.name} 在汪汪叫")

# 子类：猫（继承自动物）
class Cat(Animal):
    # 子类重写父类的方法
    def eat(self):
        print(f"{self.name} 优雅地吃着猫粮")
    
    # 子类新增的方法
    def meow(self):
        print(f"{self.name} 在喵喵叫")

# 使用子类
dog = Dog("旺财")
dog.eat()   # 继承父类的方法：旺财 在吃东西
dog.bark()  # 子类自己的方法：旺财 在汪汪叫

cat = Cat("咪宝")
cat.eat()   # 重写后的方法：咪宝 优雅地吃着猫粮
cat.meow()  # 子类自己的方法：咪宝 在喵喵叫

# 访问父类的类属性
print(dog.category)  # 生物（子类继承了父类的类属性）
```
### 示例 2：多层继承和多继承
```python
# 祖父类
class GrandParent:
    def grand_method(self):
        print("这是祖父类的方法")

# 父类（继承自祖父类）
class Parent(GrandParent):
    def parent_method(self):
        print("这是父类的方法")

# 子类（继承自父类，同时继承祖父类的方法）
class Child(Parent):
    def child_method(self):
        print("这是子类的方法")

# 多继承示例（同时继承两个父类）
class Father:
    def father_say(self):
        print("我是父亲")

class Mother:
    def mother_say(self):
        print("我是母亲")

class Child(Father, Mother):
    pass  # 继承两个父类的所有方法

child = Child()
child.father_say()  # 我是父亲
child.mother_say()  # 我是母亲
```
`pass` 是一个**空语句**，它的核心作用是 “占位”—— 表示 “这里需要有代码，但暂时什么都不做