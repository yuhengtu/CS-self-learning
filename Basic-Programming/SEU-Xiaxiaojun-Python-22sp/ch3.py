#列表  字符串不能改，列表可以改
#有players1 + players2；players1 * 2；player in players
#有len(players)；players.index('郭靖')查找；players.count('郭靖')；players.sort()；players.reverse()
#sort&sorted  sort就地改  sorted原本的不改，生成一个新改过的列表   reverse和reversed相同
student = ['张三', 18, '男'] #混搭建议用字典
scores = [98, 96, 95, 94, 92]
print(scores[0], scores[4])
print(scores[-5], scores[-1])
#可嵌套  students1 = [student1, student2]  students2 = [['张三', 18, '男'], ['李四', 19, '女']]
#student[1][1]=19

a = [1, 2, 3]
a[2] = a # 如何理解
print(a)
print(a[2])
print(a[2][2][2][2][2][2][2][2])#无穷递归
print(...)
print(a[2] is a)

players = ['郭靖', '杨过', '令狐冲', '萧峰', '黄蓉']
print(len(players))
players.append('洪七公')#加在最后
players.insert(2, '段誉')#加在指定位置
print(players)
del players[2]
print(players)
player1 = players.pop() #pop取出 
player2 = players.pop(2)
print(player1, player2)
print(players)
players.remove('黄蓉')#只删除第一个 容易出错
#易错
nums = [3, 2, 1, 1, 4, 1, 5]
for i in nums:
    if i == 1:
        nums.remove(i)
    print(nums) 
#遍历
for i in range(len(players)):
    print(players[i])
for player in players:
    print(player)
#切片
print(players[1:3])
players[1:1] = ['洪七公'] # 插入
print(players)
players[1:2] = ['欧阳锋'] # 替换
print(players)
players[1:2] = [] # 删除
print(players)

#extend
persons = ['张三', '李四', '王五']
friends = ['TOM', 'MARY', 'JACK']
persons.extend(friends) # persons += friends
print(persons)

#三种复制 引用，浅复制和深复制
persons = ['张三', '李四', '王五']
persons1 = persons # 这里是名字的引用
persons1[0] = 'TOM'
print(persons)#两个一起变

persons = ['张三', '李四', '王五']
persons1 = persons.copy() # 这里浅复制
persons1[0] = 'TOM'
print(persons)

import copy
persons = ['张三', '李四', [1, 2, 3]]
persons1 = copy.deepcopy(persons) # 这里深复制，完全隔离
persons1[2].append(4)
print(persons)
print(persons[0] is persons1[0])
print(persons[2] is persons1[2])

#数值列表
# lst = eval(input('请输入您的数值列表：')) # 输入时要带[]
# print(lst)
# print(max(lst), min(lst), sum(lst))

# 不常用
# lst1 = list(range(1, 11))#等差数列
# lst2 = []
# for i in range(1, 11):
#     lst2.append(i * i)
# print(lst1)
# print(lst2)

#列表生成式   列表生成式语法：[表达式 for 变量 in 序列 if 条件]
lst1 = [i * i for i in range(1, 11)]
lst2 = [i * i for i in range(1, 11) if i % 2 == 0]
print(lst1)
print(lst2)
s = 'ABC'
t = '123'
lst3 = [i + j for i in s for j in t] # 循环嵌套
print(lst3) 
#一行输出九九乘法表
[print(f'{j}*{i}={j*i}', end='\t' if j < i else '\n')
for i in range(1, 9 + 1) 
for j in range(1, i + 1)]

# random.random() 0-1之间的随机实数
# random.uniform(a,b) a-b之间的随机实数
# random.randint(a, b) a-b之间的随机整数 常用 ab均取得到
# random.choice(sequence) 序列当中随机选择一个
# random.shuffle(x) 随机乱序
# random.sample(sequence, k) 序列当中随机选择k个

nums = [3, 2, 1, 1, 4, 1, 5]
# 思考如何删除列表当中所有的1
# 方法一：循环判断生成新列表
# 方法二：列表生成式生成新列表
# 方法三：filter函数(*)

# 元组（tuple）与列表类似，也是用来存放一组相关的数据
# 两者的不同之处主要有两点：
# 创建元组使用圆括号()，列表使用方括号[]
# 元组的元素不能修改，因此无相关成员函数
# 元组比列表执行效率更高、更安全
players = ('郭靖', '杨过', '令狐冲', '萧峰', '黄蓉')
#players.append('欧阳锋') # 无法添加
scores = 90, 92, 93
print(type(scores))
nums1 = (1) # 数字1  不是元组
nums2 = (1,) # 单元素的元组

#字典 映射  key指向value  key，键，唯一且不可改 哈希
#key重复不报错 新增的代替原有的
country_areas = {'俄罗斯': 1708, '加拿大': 997, '中国': 960}
print(type(country_areas))
print(country_areas)
country_tuple = ('俄罗斯', 1708), ('加拿大', 997), ('中国', 960)
country_areas2 = dict(country_tuple)#元组转字典
print(country_areas2)
# 字典名[key] = value # key存在为修改，不存在为添加
country_areas = {} # 空字典
country_areas['俄罗斯'] = 1708
country_areas['加拿大'] = 997
country_areas['中国'] = 960

del country_areas['俄罗斯']
area = country_areas.pop('加拿大')
country_area = country_areas.popitem()#删除最后一个
print(country_areas, country_area, area)

country_areas.clear()
print(country_areas)
del country_areas
#print(country_areas)

country_areas1 = {'俄罗斯': 1708, '加拿大': 997, '中国': 960}
country_areas2 = {'俄罗斯': 1709, '美国': 937, '日本': 38}
country_areas1.update(country_areas2)
print(country_areas1)
country_areas2.update(country_areas1)
print(country_areas2)
# update方法会修改字典本身，如果希望不修改两个字典a和b，
# 而得到一个新的字典，如何进行
# 方法1：创建空字典，两次更新字典
# 方法2：复制字典a，更新字典b
# 方法3：用**操作符进行字典解包
# 方法4：用 | 运算符直接实现（Python 3.9）
c1 = country_areas1|country_areas2#py3.9
c2 = {**country_areas1,**country_areas2}
print(c1,c2) 
#只能查找键
country_areas = {'俄罗斯': 1708, '加拿大': 997, '中国': 960}
print('中国' in country_areas)
print('美国' in country_areas)
print(country_areas.get('中国'))
print(country_areas.get('美国')) # 键不存在，默认返回None
print(country_areas.get('美国', 937)) # 键不存在，返回指定参数
# 问题：输入一个句子，统计各字母出现的次数（大小写视为一致），
# 输出统计结果（未出现的不显示）
# count = {}
# s = input('请输入一个英文句子：')
# for ch in s.upper(): # 判断句子的大写形式
# if ch.isalpha(): # 是否为字母
# count[ch] = count.get(ch, 0) + 1 # 如果找不到ch就返回0
# print(count)

#遍历
country_areas = {'俄罗斯': 1708, '加拿大': 997, '中国': 960}
for key in country_areas.keys():
    print('{}的面积是{}万平方公里'.format(key,country_areas[key]))
for k, v in country_areas.items():
    print('{}的面积是{}万平方公里'.format(k, v))
#排序
country_areas = {'RUS': 1708, 'CAN': 997, 'CN': 960}
ls = sorted(country_areas.keys())#按键来排序
for country in ls:
    print(f'{country}的面积是{country_areas[country]}万平公里')
#key和value互换
area_countries = {v:k for k,v in country_areas.items()}#随后可按面积排序
print(area_countries)

#集合 不常用 完全无序 不可变 自动去重
country_set = {'俄罗斯', '加拿大', '中国'}#集合
scores = [92, 94, 96, 100, 100, 92]
scores_set = set(scores)
names1 = {} # 不是空集合，因为空大括号代表空字典
names2 = set() # 创建空集合
print(country_set) # 集合输出很可能乱序
print(scores_set)
# print(scores[0]) # 错误
for score in scores:
    print(score, end='\t')

setI = {'Python', 'C++', 'C', 'Java', 'C#'}
setT = {'Java', 'C', 'Python', 'C++', 'VB.NET'}
print("IEEE2018排行榜前五的编程语言有:", setI)
print("TIOBE2018排行榜前五的编程语言有:", setT)
print("前五名上榜的所有语言有:", setI | setT)
print("在两个榜单同时进前五的语言有:", setI & setT)
print("只在IEEE榜进前五的语言有:", setI - setT)
print("只在一个榜单进前五的语言:", setI ^ setT)


