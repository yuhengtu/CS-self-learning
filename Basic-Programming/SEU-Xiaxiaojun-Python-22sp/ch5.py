import numpy as np
# numpy 基于C 快 python胶水语言，库可以是任何语言写的
# jupyter 用arr1输出out更美观 不用print 可配置插件
# ndim维度，shape，dtype    size 元素数量  flags 内存信息描述 默认C⻛格的数组保存⽅法
# itemsize 元素内存⼤⼩ 和dtype对应
# strides 跨度 从当前维度到下⼀维度所需要“跨过”的字节数
# 1 2 3
# 4 5 6 stride：（24，8），一个元素8字节
# 轴 axis012 行列层

# 创建
# np.array 原⽣列表或元组 将原⽣列表或元组等转换为数组
# np.arange 起点/终点/步⻓ 可以使⽤实数
# np.linspace 列起点/终点/项数
x = np.linspace(10, 20, 5)
print(x)
y = np.linspace(10, 20, 5, endpoint=False)
print(y)
# np.zeros((4,5)) 全零实数数组
# np.ones  全⼀实数数组
# np.full((4,5),2)  指定数值填充
# np.eye(3) 单位矩阵
# np.logspace 等⽐数列起点 项数/底/终点/ 底默认为10

# rand 0和1之间的随机实数 
# normal 正态分布的随机数
# randint 指定范围内的随机整数 
# uniform 均匀分布
# randn 标准正态的随机数 
# poisson 泊松分布
# choice 随机抽取样本 
# shuffle 随机打乱顺序
x = np.random.randint(0, 100, (3, 5))
print(x)
y = np.random.normal(0, 1, (3, 4))
print(y)

# 改变是否就地改变;得到的新数组空间是否独⽴
a = np.arange(1, 7)
a.reshape(2, 3) # 不是就地修改,a不变
print(a)
b = a.reshape(2, 3) # a和b共享内存，本质上是改变了strides
print(b)
b[0][0] = 100
print(a)

a = np.arange(1, 7)
a.resize(2, 3) # 就地修改
print(a)
a.resize(2, 4) # 不⾜就补0或空
print(a)
a.resize(2, 2) # 超过则丢弃
print(a)

# 互换轴，转置  b不是新数据，本质数据不动更改stride
a = np.arange(1, 7)
a.resize(2, 3) 
b = a.swapaxes(0, 1)
print(a, a.strides) #24，8
print(b, b.strides) #8，24
# 三维
a = np.arange(1, 9).reshape(2, 2, 2)
b = a.swapaxes(2, 0) # 换0和2轴
print(b,b.strides)
# [[[1 5]
#   [3 7]]

#  [[2 6]
#   [4 8]]]
# (8,16,32),下一层隔1个数 1->2;下一行隔2个数 1->3;下一列隔4个数 1->5

# 转置
a = np.arange(1, 7).reshape(2, 3)
b = a.T
c = np.transpose(a) # 这⾥都是⾏列互换
print(b)
print(c)

# 轴的滚动 改stride
a = np.arange(1, 9).reshape(2, 2, 2)
b = np.rollaxis(a, 2) # 2往前滚，下标顺序由012改成201
c = np.rollaxis(a, 2, 1) # 轴2挪到第一位（012），下标顺序由012改成021
print(a, b, c, sep='\n\n')

a = np.arange(1, 7).reshape(2, 3)
# 压平
b = a.flatten()
print(b)
# 转list
c = a.tolist()
print(c)

# 原生切片开新空间，numpy中切片尽量不开新空间
a = np.arange(1, 7)
b = a[2:4]
c = a[:]
d = a[...]
# c和d一样 
b[0] = 100
print(a)

a = np.arange(1, 17).reshape(4, 4)
print(a)
print(a[0, 2:4])
print(a[2:, 2:])
print(a[0::2, ::2]) #间隔2

# 花式索引 开新空间新的数组（尽量不开，这里得开）
a = np.arange(1, 17).reshape(4, 4)
print(a)
b = a[[0, 3, 1], 1:]
print(b)
b[0] = 100
print(a)

# 布尔索引，退回1维，新开空间
a = np.arange(1, 17).reshape(4, 4)
b = a[a > 10]
print(b)

# 广播
a = np.arange(1, 7).reshape(2, 3)
b = a + 1
print(b)

arr1 = np.arange(1, 13).reshape(4, 3)
arr2 = np.array([10, 10, 10])
print(arr1 + arr2)

# 切割
x = np.arange(11, 19)
x1, x2, x3 = np.split(x, [3, 5]) # 在3，5位置切割，前闭后开
print(x1, x2, x3)
# 二维切割 vsplit和hsplit
x = np.arange(1, 49).reshape(6, 8)
y = np.vsplit(x, 3) # 分成3份 vertical横着切，6行切三份 得到列表
print(y)
z = np.vsplit(x, [2, 4]) # 试试hspilt
print(z)

# 高维 指定轴
x = np.arange(1, 25).reshape(2, 3, 4)
upper, lower = np.split(x, [1], axis=0) # 0轴为层  层行列
print(upper, lower)
upper, lower = np.split(x, [1], axis=1) # 每一层中按行切 
print(upper, lower)

# 堆叠 concatenate
x = np.arange(1, 25).reshape(2, 3, 4)
upper, lower = np.split(x, [1], axis=0)
a = np.concatenate((upper, lower), axis=0) # 还原（2，3，4）
a = np.concatenate((upper, lower), axis=0) # 还原（1，6，4）
a = np.concatenate((upper, lower), axis=0) # 还原（1，3，8）
# 二维专用
b = np.vstack([upper, lower])
c = np.hstack([upper, lower])
print(a, b, c, sep='\n\n')
print(b.shape, c.shape) 

# 操纵 
x = np.arange(8)
print(np.delete(x, 2))
# 非就地改 新开空间
y = np.arange(9).reshape(3, 3)
print(np.delete(y, [1], axis=0))
print(np.insert(x, 0, -1))
print(np.insert(y, 1, [0, 0, 0], axis=0))

