#3.10加入switch 豆瓣 py2于2020年停止更新

#str  int  float:1e6  bool:True&False 3个大写还有None  complex：1+2j
#list tuple dict set class

print(r'd:\new\text')#r取消转义

w = 2
print('%.2f' %w)
print(f'w = {w:.2f}')
a = "早晨"
b = "下午"
c = 2
print(f"从{a}起床到{b},我一共只吃了{c}个苹果")

#8进制:0o  16:0x  2:0b

import keyword
print(keyword.kwlist)#打印关键字sum len max min

m=2
print(type(m))
#不可以print(12+‘34’)，强类型语言

#//地板除向下取整  **乘方
print(-7**2)
print(round(3.1415926, 2)) # 四舍五⼊函数
print(divmod(7, 2)) # 返回商和余数
print(abs(-2)) # 返回绝对值
print(pow(2, 3)) # 幂运算
print(max(3, 1, 2)) # 最⼤值
print(min(3, 1, 2)) # 最⼩值
print(sum((3, 1, 2))) # 总和，注意参数格式内置数学模块

import math
print(dir(math)) # 列出模块所有的功能和数据
print(math.sin(math.pi / 2))
print(math.ceil(3.5)) # 向上取整

x,y=3,4
x,y=y,x #swap
print(x,y)

x,*y,z=1,2,3,4,5#y接受多个数据存为列表
print(y)

print(ord('A'), ord('a'))
print(chr(65), chr(97))
print(ord('男'), ord('⼥'))#字符编码Unicode，utf8

s = 'nanjing'
print('nan' in s)
a = [1, 2, 3, 4]
print(5 not in a)

a = b = 2
print(a is b)
a = 3
print(a is not b)

print(type(eval('1+1')))

print(int(3.5), int(-3.5))#向零取整
print(bool(0), bool(''))#非零非空为True

# 短路求值
# 计算a and b时，如a可转换为True，则结果是b，否则结果是a
# 计算a or b时， 如a可以转换为True，则结果是a，否则结果是b
print(1 and 0, 1 and 'x')
print(0 and 0, '' and 'x')
print(1 or 0, 1 or 'x')
print(0 or 0, '' or 'x')

# #请⽤户输⼊⽤户名，如果⽤户不输⼊，则默认为Anonymous
# username = input('请输⼊⽤户名： ') or 'Anonymous'
# print('欢迎你', username)

# while (input_str := input('请输入：')) != 'exit':#海象运算符py3.8
#   print(f"您输入的是{input_str}")

# r = float(input('请输⼊圆的半径： ')) #input:str型，转float
# area = 3.14 * r ** 2
# print(area)