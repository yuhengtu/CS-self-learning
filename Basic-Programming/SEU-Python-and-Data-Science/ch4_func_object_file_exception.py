#函数round(number, ndigits=None)
# def 函数名(形参表):
#     函数体
#     return 返回值

# None是Python中一个特殊的常量，它不代表0，也不代表空，就代表没有值
# 当函数没有return语句，或者return后没有返回值时，该函数默认返回None
# 也可以给变量赋值为None，但不能和其它数据进行混合运算，因此通常只用于参数的默认值中
print(print(1)) # print函数就返回 None

# 质数
def isprime(n):
    if n == 2:
        return True
    elif n % 2 == 0:
        return False
    k = int(n ** 0.5)
    for i in range(3, k + 1, 2):
        if n % i == 0:
            return False
    return True
for i in range(2, 100):
    if isprime(i):
        print(i, end='\t')

#my_pow(2, 3)  位置参数
# 在参数列表当中，位于 / 之前的参数，只能通过位置参数来传递(3.8+)
def my_pow(x, /, n):
    return x ** n
print(my_pow(2, 3))
print(my_pow(2, n=3))
# print(my_pow(x=2, n=3)) # 错误

# 默认参数
def my_pow(x, n=2):
    return x ** n
print(my_pow(2))
# 通过关键字参数提交时，可以无视原有的参数顺序。
# 关键字参数不能放在位置参数的左侧
def total_score(mid_score, end_score, rate=0.3):
    return mid_score * rate + end_score * (1 - rate)
print(total_score(mid_score=80, end_score=90))
print(total_score(end_score=90, mid_score=80))


# 可变参数 可以传入任意个位置参数
def square_total(*numbers): # 变量名左侧的星号代表可变参数
    total = 0
    for i in numbers: # 输入函数的多个参数会组装为一个元组
        total += i * i
    return total
print(square_total())
print(square_total(1))
print(square_total(1, 2))#1平方+2平方

#可变关键字 ** 装成字典
def f(age,name,gender,**hobby):
    print(f'学生{name}，今年{age}岁，性别{gender}')
    for k,v in hobby.items():
        print(f'爱好{k},水平{v}')
f(18,'张三','男')
f(18,'李四','女',piano='10级',dance='10级')

# 命名关键字参数 *前为默认参数 *后面的参数被视为命名关键字参数 命名关键词也可以提供默认值
def person(name, age, *, city, job):
    print(name, age, city, job)
person('TOM', 20, city='nanjing', job='worker')

#函数内定义的是局部变量 局部变量屏蔽外部 global x访问全局变量

# 如果在定义函数A中的过程中调用了函数A，则称为直接递归调用；如果在定
# 义函数A中调用函数B，在定义函数B中调用函数A，则称为间接递归。
# 运行效率极低
def factorial(n):
    if n <= 1: # 递归终止条件
        return 1
    return n * factorial(n - 1) # 展开递归

def fib(n):
    if n <= 2: # 递归终止条件
        return 1
    return fib(n - 2) + fib(n - 1) # 展开递归

def backward(n):
    if n > 0:
        print(n % 10, end='')
        backward(n // 10)
backward(12345)

#汉诺塔问题
def hanoi(n, start, mid, end):
    if n == 1:
        print(start, '->', end)
        return
    hanoi(n - 1, start, end, mid)
    print(start, '->', end)
    hanoi(n - 1, mid, start, end)
hanoi(3, 'A', 'B', 'C')

# 高阶函数、匿名函数、返回函数(*)、闭包和装饰器(*)、偏函数
# 如果一个函数接受其它函数作为参数，那么该函数就称为高阶函数 和数学问题有很好的结合
print(type(print)) # 内置函数类型
myprint = print # 函数可以给对象赋值
myprint(2) # myprint具备了print函数的功能

# 曲线面积（定积分） 
def curse_area(f, a, b, n=10000): # 默认分为10000份
    h = (b - a) / n # 梯形的高
    total = (f(a) + f(b)) / 2 # 首尾只求和一次
    for i in range(1, n):
        total += f(a + i * h) # 中间的底需要求和两次
    return total * h # 梯形面积要乘以高
def f1(x):
    return x
def f2(x):
    return x * x
print(curse_area(f1, 0, 1))#函数 下限 上限 
print(curse_area(f2, 0, 1))

# map函数 结果作为迭代器返回
def square(x):
    return x * x
a = [1, 2, 3, 4, 5]
b = list(map(square, a)) # 省去循环过程
print(b)

#  filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素。
def is_odd(n):
    return n % 2 == 1
a = [1, 2, 3, 4, 5]
print(list(filter(is_odd, a)))

# 匿名函数lambda 不能复杂 lambda 参数表 : 表达式
a = [1, 2, 3, 4, 5]
print(list(filter(lambda n: n % 2 == 1, a)))

# sorted函数可以根据指定的排序依据，对序列进行排序，并返回排序之后的列表。
# 排序的两个关键参数：reverse代表是否降序，默认升序，key代表排序依据。
a = [5, -2, 1, -4, 3]
b = sorted(a)
c = sorted(a, reverse=True)
d = sorted(a, key=lambda x: x * x)#按平方值比较
print(b, c, d)

# 迭代器  与  可迭代对象（for之后的）
# 能支持for循环的都是可迭代对象，能支持next函数（可惰性计算(lazyevaluation的序列）的是迭代器
# 惰性 催一次算一次 
myList = [1, 2, 3]
myIterator = iter(myList) # 将可迭代对象转为迭代器
print(next(myIterator))
print(next(myIterator))
print(next(myIterator))
# print(next(myIterator)) 到头了报错 一次性 不能往回走
# enumerate、 zip、 reversed和其他⼀些内置函数会返回迭代器
# range函数也是可迭代对象（懒序列），但不是迭代器

# 生成器 一种可以主动控制的迭代器
def myGen():
    x = range(1, 3)
    for i in x:
        yield i + 2
print(myGen())
a=myGen()
print(next(a))
print(next(a))

numbers = [1, 2, 3, 4, 5]
squares = (n ** 2 for n in numbers)
print(type(squares))
print(next(squares))
print(next(squares))
print(next(squares))
print(next(squares))
print(next(squares))

# import datetime
# a, b = range(0, 50000000), []
# start = datetime.datetime.now()
# for i in a:
#     b.append(i)#真正5kw个空间 花时间 
# end = datetime.datetime.now()
# print(end - start)#操作花费了多少时间

start = datetime.datetime.now()
c = (i for i in a)#迭代器 惰性序 列 不花时间
end = datetime.datetime.now()
print(end - start)

# 装饰器 修饰 不影响代码原本功能 用于计时更简便
import datetime
def count_time(func):
    def wrapper(*args, **kw): # 闭包(closure) python里参数不是实参就是关键字 这样可以获取一切参数
        start = datetime.datetime.now()
        t = func(*args, **kw)
        end = datetime.datetime.now()
        print(end - start)
        return t
    return wrapper

@count_time # 在f之前 写上 可以计时f
def f():
    a = [i * i for i in range(50000000)]
    return a
f() #等价于count_time(f)()

# 编写“⽤户登陆”装饰器，功能是在函数执⾏之前进⾏⽤户名和密码的检查，如果检查成功则运⾏，否则给出错误提示
import datetime
def check_login(func):
    def wrapper(*args, **kw): # 闭包(closure) python里参数不是实参就是关键字 这样可以获取一切参数
        if input('请输入用户名')!='admin' or input('请输入密码')!='123':
            print('非法操作')
            return None
        else:
            t = func(*args,**kw)
            return t
    return wrapper
@check_login
def ff():
    print('下面开始机密操作')
ff()

# python一切皆是对象 类中的函数称为方法  __name：private  _protective
class Student():#括号内为父类
    def __init__(self, name, score): # 构造初始化函数 self必须有 代表创建的对象 cpp的this指针
        self.name = name
        self.score = score
    def print_score(self):
        print(f'{self.name}:{self.score}')
bart = Student('Bart Simpson', 59)
lisa = Student('Lisa Simpson', 87)
bart.print_score()
lisa.print_score()

# 文本 或 二进制编码（图片视频） win分大小写 mac不分大小写
# win_path = 'f:\\documents\\python\\5-1.py' # 不能直接书写\
# win_path2 = r'f:\documents\python\5-1.py' # 原始字符串
# mac_path = '/Users/xiaxiaojun/Documents/Python'

#相对路径
# import os 操作系统
# print(os.getcwd())  current working directory py正在工作的目录
# os.chdir('f:\\documents')
# print('..\\文件与异常处理.doc', 'data.txt', 'python\\5-1.py')  ..为上一步文件夹

# file对象名 = open(路径字符串, 模式字符串)
# # 文件读写操作
# file对象名.close()

# r 只读(read) 正常 报错
# w 只写(write) 覆盖/清空 新建
# a 追加(append) 续写 新建
# x 存在(exists)则报错 新建
# + 可读可写
# b 二进制(binary)
# t 文本（text，默认)
# 通常先写操作模式再写文件模式，如'r' 'w' 'rb+' 'ab'

# 写文本文件 只写字符串
# 写文件通过 write 和 writelines 方法实现
# 这两个方法都不会自动分隔或换行，需要手动添加
# 写文件切记close
# file = open('new.txt', 'w')
# file.write('Hello')
# file.write('I am Jack,f{i}\t\n')
# file.writelines(['nice to meet you', 'have a nice day'])
# file.close()

# 读文件主要通过 read / readline / readlines 方法实现
# 文件对象本身也是迭代器，可以直接遍历每一行
# file = open('new.txt', 'r')
# file.read(5) 最多读取5个字符
# file.read() 读出所有内容 会卡死
# file.readline() 读出一行（带回车符）
# file.readlines() 读出所有行，存为列表
# for line in file: # 这里的行也带回车符
#     print(line, end='')  取消print自带换行 因为之前读出的内容末尾带有换行

# 异常
# ZeroDivisionError 除数为零
# OSError 系统级别的操作失败
# IndexError 序列中没有此索引(index)
# KeyError 映射中没有这个键
# NameError 未声明/初始化对象(没有属性)
# ValueError 数值/计算错误
# TypeError 类型错误

# try:
# 待运行语句
# except 异常名称 as 异常内容（可选）:
# 捕获异常时处理
# ……
# else:
# 未发生异常时处理（可选）
# finally:  不论有无异常都执行 
# 收尾语句（可选）

#常用于函数中 如求a的b次方 不能输入字符串
#如文件找不到 归于oserror 
#工程代码

# 输入两个整数a和b，求其商（续）
try:
    a = int(input())
    b = int(input())
    print(a / b)
except ZeroDivisionError as result:
    print(f'触发了零除异常，提示：{result}')
except ValueError as result:
    print(f'你的输入不合理，提示：{result}')
# except:
#   print('发生了未知错误，请联系管理员')
# else:
#   print('未发生任何错误')
# finally:
#   print('程序结束')
