'''
作者：张三
'''

#强烈建议使⽤4个空格作为缩进

# 续行符\
# if year % 4 == 0 and year % 100 != 0 \
# or year % 400 == 0:

# Python之禅
#import this
# PEP8-Style Guide for Python Code官⽅代码的格式化规范
# https://peps.python.org/pep-0008/
# ⾕歌公司的Python代码规范
# https://github.com/google/styleguide/blob/ghpages/pyguide.md

# # ⼀次性输⼊多个字符串，再分别进⾏进⾏转换
# x, y, z = input('').split()
# x = float(x)
# print(x, y, z)
# # 也可以通过内置函数map⼀次性进⾏处理转换
# a, b, c = map(float, input('请输⼊三⻆形的三条边： ').split())#转float类型
# print(a, b, c)

# if 表达式1:
# 分支语句1
# elif 表达式2:
# 分支语句2
# elif 表达式n:
# 分支语句n
# else:
# 备选语句
# 执行了一个后面的就不管了

# a = int(input('请输入一个整数：'))
# print('是偶数' if a % 2 == 0 else '不是偶数')

# #3.10 switch语句
# match month:
#   case 1 | 3 | 5 | 7 | 8 | 10 | 12:
#     days = 31
#   case 4 | 6 | 9 | 11:
#     days = 30
#   case 2:
#     pass # 省略

# while 条件表达式:
# 循环体语句
# else: # else可选
# 备选语句
# 正常结束才执行else，break则不执行else

#range(5):0-4
#range(1,5):1-4
#for速度比while快得多
total = 0
for i in range(1, 101):
  total += i
print(total)

for i in range(1, 10):
  if i < 5:
    continue#进入下一轮循环
  print(i)
#break离开循环

#字符串0-5，-6~-1，不可改，生成副本改
# slide切片，越界不报错，头：尾：步长
# 步长为正时：不写起点默认为负无穷 ，不写终点默认为正无穷
# 步长为负时：不写起点默认为正无穷 ，不写终点默认为负无穷
s = 'Hello Mike'
a = s[::-1]
print(s[0:5]) # Hello 0-4
print(s[-1:-5:-1]) # ekiM
print(s[:])
print(s[0:])
print(s[:10])
print(s[::-1])
print(s[-1:-11:-1])
print(s[9:-11:-1])
print(s[:-11:-1])

print('a'*3,'i'+'love')

x = '好好学习，天天向上'
print(len(x))
print(str(120 + 5))
print(hex(20))#出字符串，不是数
print(oct(20))
print(bin(20))

#大小写转换
# lower = input('请输入一个小写字母：')
# upper = chr(ord(lower) - 32)
# print(upperi)

# find方法：用来查找一个字符串在另一个字符串指定范围（默认是整个字符串）
# 中首次出现的位置，如果不存在则返回-1。
# n index方法：用来查找一个字符串在另一个字符串指定范围（默认是整个字符串）
# 中首次出现的位置，如果不存在则抛出异常。
# n count方法：用来返回一个字符串在另一个字符串中出现的次数，如果不存在则返
# 回0。
s = 'bird,fish,monkey,rabbit'
print(s.find('fish'))
print(s.index('fish'))
print(s.count('i'))

s = 'bird,fish,monkey,rabbit'
print(s.split(','))
s = 'how are you?'
print(s.split())

s = 'bird,fish,monkey,rabbit'
ls = s.split(',')
s2 = ','.join(ls)
print(s2)

#大小写 生成新字符串
s = 'hello, I am Jack'
print(s.upper())
print(s.lower())
print(s.capitalize())
print(s.title())
print(s.swapcase())#大变小，小变大

s = '你是我的小呀小苹果'
print(s.replace('小', 'small'))

#爬虫常用  左删 右删 左右删 中间删不掉
s = ' abc '
print(s.lstrip())
print(s.rstrip())
print(s.strip())
s = '===a=b=c==='
print(s.lstrip('='))
print(s.rstrip('='))
print(s.strip('='))

#前后缀
s = 'Python作业.py'
print(s.startswith('Python'))
print(s.endswith('py'))

# isupper方法：判断是否都为大写字母
# islower方法：判断是否都为小写字母
# isalpha方法：判断是否都为字母
# isdigit方法：判断是否都为数字字符
# isalnum方法 ：判断是否都为数字字符或字母

#3种字符串格式化方法 % format f（3.8）
print("我是{}，今年{}岁".format('张三', 20)) # 默认顺序
print("我今年{1}，名叫{0}".format('张三', 20)) # 指定顺序
name='张三'
age=10
print(f'我名字是{name},今年{age*2}岁',end='')#不换行
print(f'我名字是{name},今年{age*2}岁')

#右对齐{:5}  左对齐{:<5}  居中{:^5}
print("{:*^20}".format('Mike'))#补星号
print("{:*<20}".format('Mike'))
print("{:.4f}".format(3.1415926)) # 小数点后4位
print("{:.4}".format(3.1415926)) # 有效数字4位
print("{:5d}".format(24)) # d代表十进制整数
print("{:x>5d}".format(24))





