from dataclasses import dataclass
from typing import Callable, Any

from my_package import my_module1


@dataclass
class Student:
    name: str
    age: int
    gender: str

    # 私有变量
    __private = "aaa"

    def __init__(self, name=None, age=None, gender=None):
        self.name = name
        self.age = age
        self.gender = gender

    def __str__(self):
        return f'{self.name} {self.age} {self.gender}'

    def print_name(self):
        print(f'Hi, {self.name} {self.__private}')

    def print_msg(self, msg):
        print(f'msg: {msg}')


def print_hi(name):
    print(f'Hi, {name}')


def mul_result():
    return 1, 2, 3


def test_function_param(multiply):
    result = multiply(3, 2)
    return result


# 定义基类
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def move(self):
        pass


# 定义子类
class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"


# 定义子类
class Cat(Animal):
    def _wrapped_call_impl(self, *args, **kwargs):
        return self.speak()

    """
      __call__ 方法的用处
       使实例可调用：
       当你实现 __call__ 方法后，你的对象可以像函数一样被调用，这使得代码更简洁。
       例如，你可以直接通过 cat() 来获取猫的叫声，而不必显式调用 cat.speak()。
       提高代码可读性：
       在某些情况下，使用可调用对象可以使代码更自然，尤其是在需要传递函数或对象作为参数时。
       实现更复杂的逻辑：
       你可以在 __call__ 方法中实现复杂的逻辑，例如根据输入参数返回不同的结果。
    """
    __call__: Callable[..., Any] = _wrapped_call_impl

    def speak(self):
        return f"{self.name} says Meow!"


class HotCat(Cat):
    pass

def speak(animal: Animal):
    print(animal.speak())


if __name__ == '__main__':
    u = "AddaSs"
    a = u
    v = u[:3]
    w = u[3:]
    a = a.lower()
    print(u)
    print(a)
    print(v)
    print(w)
    # z = my_module1.add(1, 2)
    # print(z)
    #
    # x = test_function_param(lambda x, y: x * y)
    # print(x)
    #
    # a, b, c = mul_result()
    # print(a, b, c)
    #
    # for x in range(5, 10, 2):
    #     print(x)
    #
    z = my_module1.add(1, 2)  # type: int
    print(z)
    print(type(z))
    # print("------------")
    # my_list = ["a", "b", 66, True]
    # print(my_list[-2])
    # print(type(my_list))
    # my_list.append("c")
    # for e in my_list:
    #     print(e)
    # print("------------")
    # my_tuple = (1, 2, 3)
    # print(my_tuple)
    # print(type(my_tuple))
    # print("------------")
    # mySet = {"a", "b", 66, True, "a"}
    # print(mySet)
    # print(type(mySet))
    # print("------------")
    # myDict1 = {"a": 99, "b": 88}
    # print(myDict1)
    # print(type(myDict1))
    # myDict2 = {}
    # print(myDict2)
    # print(type(myDict2))

    stu_1 = Student()
    stu_1.name = "李雷"
    stu_1.age = 25
    stu_1.gender = "M"
    print(stu_1)
    stu_2 = Student()
    stu_2.name = "韩梅梅"
    stu_2.age = 35
    stu_2.gender = "F"
    stu_2.print_name()
    stu_2.print_msg("2222")

    # 使用子类
    dog = Dog("Buddy")
    cat = Cat("Whiskers").__call__()
    hotCat = HotCat("hahaha")(cat)
    speak(dog)
    print(cat)
    print(hotCat)
    # with open("D:\Work\Test\ddd.txt", "r", encoding="utf-8") as f:
    #     for line in f:
    #         print(line)
