# Debugging

the most recent function call at the bottom.

File "<file name>", line <number>, in <function>



python3 -m doctest file.py -v，详细显示doctest测试结果



print('DEBUG, a: ', a)

```python
debug = True #全局

if debug:        
  print('DEBUG: i is', i)
```



assert进行确认（False会报错）

```python
def double(x):    
  assert isinstance(x, int), #"The input to double(x) must be an integer"    
  return 2 * x

assert fib(8) == 13, '第八个斐波那契数应该是 13'
def fib_test():
    assert fib(2) == 1, '第二个斐波那契数应该是 1'
    assert fib(3) == 1, '第三个斐波那契数应该是 1'
    assert fib(50) == 7778742049, '在第五十个斐波那契数发生 Error'
```

测试在 `_test.py` 中编写的。



`SyntaxError`: 语法，check before executing any code；当指向正确的一行，大概率是少了后括号；`EOL` stands for "End of Line."

`IndentationError`: 缩进，用空格，不用Tab

`TypeError`: ... NoneType ... , forgot a return statement

`NameError`: 拼错，大小写

`IndexError`: tuple, list, string越界



# Ch1(SICP)

visualize：https://pythontutor.com/cp/composingprograms.html#mode=edit



doctoring/doctest（help(函数名)可查看，[docstring准则](http://www.python.org/dev/peps/pep-0257/)）

<img src="../Library/Application Support/typora-user-images/image-20231128205827019.png" alt="image-20231128205827019" style="zoom:25%;" />

python -m doctest file.py，运行doctest

```python
>>> from doctest import testmod
>>> testmod()
TestResults(failed=0, attempted=2)

# 单个函数测试（unit test）,第二个参数是globals()；第三个参数 True 表示想要“详细”输出
>>> from doctest import run_docstring_examples
>>> run_docstring_examples(sum_naturals, globals(), True)
Finding tests in NoName
Trying:
    sum_naturals(10)
Expecting:
    55
ok
Trying:
    sum_naturals(100)
Expecting:
    5050
ok
```

test-driven development，写test -> 写code -> 写special test

python -i file.py



y, x = x, y

//整数除法，/浮点除法，8/4 = 2.0（ `truediv` 和 `floordiv` 函数的简写）

非零非空为True，负数也是True，not None = True

迭代iteration（循环，重复）!= 递归（recursion）

名称可以与函数绑定，f = max；f = 5，取消函数绑定，与值5绑定；函数定义中的名称叫 **内在名称（intrinsic name）**，帧中的名称叫 **绑定名称（bound name）**。不同的名称可能指的是同一个函数，但该函数本身只有一个内在名称。



assignment赋值：无evaluation（求解）有execution（执行）

**纯函数（Pure functions）**：abs

**非纯函数（Non-pure functions）**：return None，交互界面不打印None，have side effect，不可嵌套

```
>>> print(print(1), print(2))
1
2
None None
```

formal parameter **形式参数**



环境，包含frame/global frame/local frame，存储绑定bind（import/赋值）

绑定：名字和值/函数

`func` 函数名称（形式参数）；`mul` 内置函数使用 `...` ，可以接受任意数量的参数

![image-20231129140650574](../Library/Application Support/typora-user-images/image-20231129140650574.png)

使用参数 -2 调用 `square` 函数，它会创建一个新的local帧，将形式参数 `x` 与 -2 绑定

![image-20231129153219043](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311291532071.png)

环境、名称和函数的概念框架构成了*model of evaluation*

**名称求解（Name Evaluation）**：名称会优先求解为local frame中与它绑定的值。以下例子有3个x，开了三个local frame，确保名称在程序执行期间的正确时间解析为正确的值

![image-20231129153246151](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311291532658.png)



[Python 代码风格指南](http://www.python.org/dev/peps/pep-0008)，函数名和参数名小写；缩进用4个空格；每个函数应该只负责一个任务，DRY（*Don't repeat yourself*），通用性（与其定义平方，不如定义任意次方）

 有时，在调用 `print` 等非纯函数时，拥有一个主体为表达式的函数确实有意义。

```
>>> def print_square(x):
        print(square(x))
```

执行比较并返回布尔值的函数通常以 `is` 开头（例如 `isfinite, isdigit, isinstance` 等）



higher-order functions：接收其他函数作为参数/把函数当作返回值

抽象出“求和”这一共性

```python
def summation(n, term):
	    total, k = 0, 1
	    while k <= n:
	        total, k = total + term(k), k + 1
	    return total
	
def cube(x):
	    return x*x*x
	
def sum_cubes(n):
	    return summation(n, cube)
	
result = sum_cubes(3)
```

抽象出“迭代改进iterative improvement“

```python
>>> def improve(update, close, guess=1):
        while not close(guess):
            guess = update(guess)
        return guess
      
# 用于求解黄金比例，它可以通过反复叠加任何正数的倒数加上 1 来计算，而且它比它的平方小 1
>>> def golden_update(guess):
        return 1/guess + 1

>>> def square_close_to_successor(guess):
        return approx_eq(guess * guess, guess + 1)

>>> def approx_eq(x, y, tolerance=1e-15):
        return abs(x - y) < tolerance
  
>>> improve(golden_update, square_close_to_successor)
1.6180339887498951
```



嵌套函数（Nested function），嵌套定义之间共享名称的规则称为词法作用域 Lexical scope

```python
# x是平方根初始猜测值，x * x = a
>>> def average(x, y):
        return (x + y)/2

>>> def sqrt(a):
        def sqrt_update(x):
            return average(x, a/x)
        def sqrt_close(x):
            return approx_eq(x * x, a)
        return improve(sqrt_update, sqrt_close)
```

父级注释（a parent annotation）

![image-20231201173528796](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312011735881.png)

带有词法作用域的编程语言：

- 局部函数的名称不会影响定义它的函数的外部名称（局部环境中）
- 局部函数可以访问外层函数的环境
- 局部定义的函数通常被称为闭包（closures）



函数组合（composition）

```python
>>> def compose1(f, g):
        def h(x):
            return f(g(x))
        return h
```



牛顿迭代法（Newton's method），求零点，并不总是收敛

![image-20231201175826977](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312011758062.png)![image-20231201180037352](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312011800398.png)

```python
>>> def newton_update(f, df):
        def update(x):
            return x - f(x) / df(x)
        return update

>>> def find_zero(f, df):
        def near_zero(x):
            return approx_eq(f(x), 0)
        return improve(newton_update(f, df), near_zero)

# 64 的平方根，相当于 x^{2}− 64的零点
>>> def square_root_newton(a):
        def f(x):
            return x * x - a
        def df(x):
            return 2 * x
        return find_zero(f, df)

>>> square_root_newton(64)
8.0

# n次方根，x^{n}− a = 0
>>> def power(x, n):
        """返回 x * x * x * ... * x，n 个 x 相乘"""
        product, k = 1, 0
        while k < n:
            product, k = product * x, k + 1
        return product

>>> def nth_root_of_a(n, a):
        def f(x):
            return power(x, n) - a
        def df(x):
            return n * power(x, n-1)
        return find_zero(f, df)

>>> nth_root_of_a(3, 64)
4.0
```



柯里化（Curring），函数链，g(x)(y)；编程语言如 Haskell，只允许使用单个参数的函数

map 模式（map pattern）就可以将单参数函数应用于一串值

```python
>>> def curried_pow(x):
        def h(y):
            return pow(x, y)
        return h
>>> curried_pow(2)(3)
8

# 计算 2 的前十次方
>>> def map_to_range(start, end, f):
        while start < end:
            print(f(start))
            start = start + 1
          
>>> map_to_range(0, 10, curried_pow(2))
1
2
4
8
16
32
64
128
256
512
```

定义函数来自动进行柯里化，以及逆柯里化变换（uncurrying transformation）

```python
>>> def curry2(f):
        """返回给定的双参数函数的柯里化版本"""
        def g(x):
            def h(y):
                return f(x, y)
            return h
        return g
      
>>> def uncurry2(g):
        """返回给定的柯里化函数的双参数版本"""
        def f(x, y):
            return g(x)(y)
        return f
      
>>> pow_curried = curry2(pow)
>>> pow_curried(2)(5)
32
>>> map_to_range(0, 10, pow_curried(2))
1
2
4
8
16
32
64
128
256
512
>>> uncurry2(pow_curried)(2, 5)
32
```



Lambda 表达式，匿名函数，以希腊字母 λ（lambda）命名

```
lambda              x         :              f(g(x))
"A function that    takes x   and returns    f(g(x))"
```

复合 lambda 表达式很难辨认，从左往右传入参数，少用

```python
# 函数compose1接受两个函数 f 和 g 作为参数，然后返回一个新的函数，这个新的函数接受一个参数 x
>>> compose1 = lambda f,g: lambda x: f(g(x))
```



编程语言会对计算元素的操作方式施加限制。拥有最少限制的元素可以获得 first-class status，权利包括：

1. 可以与名称绑定
2. 可以作为参数传递给函数
3. 可以作为函数的结果返回
4. 可以包含在数据结构中

Python 授予函数完全的一等地位



装饰器（decorator）

triple 的 `def` 语句有一个注解（annotation） `@trace`

名称 triple 不会绑定到这个函数上，会被绑定到在新定义的 `triple` 函数调用 `trace` 后返回的函数值上。

```python
>>> def trace(fn):
        def wrapped(x):
            print('-> ', fn, '(', x, ')')
            return fn(x)
        return wrapped

>>> @trace
    def triple(x):
        return 3 * x

>>> triple(12)
->  <function triple at 0x102a39848> ( 12 )
36

# 等价于
>>> def triple(x):
        return 3 * x
>>> triple = trace(triple)
```

装饰器符号 `@` 也可以后跟一个调用表达式。跟在 `@` 后面的表达式会先被解析（就像上面的 'trace' 名称一样），然后是 `def` 语句，最后将装饰器表达式的运算结果应用到新定义的函数上，并将其结果绑定到 `def` 语句中的名称上。



递归

```python
>>> def sum_digits(n):
        """返回正整数 n 的所有数字位之和"""
        if n < 10:
            return n
        else:
            all_but_last, last = n // 10, n % 10
            return sum_digits(all_but_last) + last
          
>>> sum_digits(738)
18
```

![image-20231201185350326](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312011853436.png)



```python
>>> def cascade(n):
        """打印数字 n 的前缀的级联"""
        if n < 10:
            print(n)
        else:
            print(n)
            cascade(n//10)
            print(n)

>>> cascade(2013)
2013
201
20
2
20
201
2013
```



互递归 mutually recursive

```python
# 如果一个数比一个奇数大 1，那它就是偶数
# 如果一个数比一个偶数大 1，那它就是奇数
# 0 是偶数

def is_even(n):
  if n == 0:
    return True
	else:
 	  return is_odd(n-1)
 	
def is_odd(n):
  if n == 0:
    return False
  else:
    return is_even(n-1)

result = is_even(4)
```

```python
# 两人博弈，桌子上最初有 n 个石子，玩家轮流从桌面上拿走一个或两个石子，拿走最后一个石子的玩家获胜。
# Alice 总是取走一个石子; 如果桌子上有偶数个石子，Bob 就拿走两个石子，否则就拿走一个石子
# 给定 n 个初始石子且 Alice 先开始拿，谁会赢得游戏？
>>> def play_alice(n):
        if n == 0:
            print("Bob wins!")
        else:
            play_bob(n-1)

>>> def play_bob(n):
        if n == 0:
            print("Alice wins!")
        elif is_even(n):
            play_alice(n-2)
        else:
            play_alice(n-1)

>>> play_alice(20)
Bob wins!
```





树递归（tree recursion），函数会多次调用自己

```python
# 斐波那契数列
def fib(n):
  if n == 1:
    return 0
  if n == 2:
    return 1
  else:
    return fib(n-2) + fib(n-1)

result = fib(6)
```

```python
# 求正整数 n 的分割数，最大部分为 m，即 n 可以分割为不大于 m 的正整数的和，并且按递增顺序排列。
# 例如，使用 4 作为最大数对 6 进行分割的方式有 9 种。
1.  6 = 2 + 4
2.  6 = 1 + 1 + 4
3.  6 = 3 + 3
4.  6 = 1 + 2 + 3
5.  6 = 1 + 1 + 1 + 3
6.  6 = 2 + 2 + 2
7.  6 = 1 + 1 + 2 + 2
8.  6 = 1 + 1 + 1 + 1 + 2
9.  6 = 1 + 1 + 1 + 1 + 1 + 1

# 使用最大数为 m 的整数分割 n 的方式的数量 = 
# 使用最大数为 m 的整数分割 n-m 的方式的数量 + 使用最大数为 m-1 的整数分割 n 的方式的数量
>>> def count_partitions(n, m):
        """计算使用最大数 m 的整数分割 n 的方式的数量"""
				# 整数 0 只有一种分割方式
				# 负整数 n 无法分割，即 0 种方式
				# 任何大于 0 的正整数 n 使用 0 或更小的部分进行分割的方式数量为 0
        if n == 0:
            return 1
        elif n < 0:
            return 0
        elif m == 0:
            return 0
        else:
            return count_partitions(n-m, m) + count_partitions(n, m-1)

>>> count_partitions(6, 4)
9
```



# Ch2

type(a)：整数（`int`）、浮点数（`float`）、复数（`complex`），[原始数据类型](http://getpython3.com/diveintopython3/native-datatypes.html)

整数字面量 literals 会求解为 `int` 类型

```python
>>> 1/3 == 0.333333333333333312345  # 请注意浮点数近似
True
```

```python
pair = [10, 20]
x, y = pair
>>> x
10
>>> y
20
```



有理数

```python
>>> def rational(n, d):
        return [n, d]

>>> def numer(x):
        return x[0]

>>> def denom(x):
        return x[1]
  
>>> def add_rationals(x, y):
        nx, dx = numer(x), denom(x)
        ny, dy = numer(y), denom(y)
        return rational(nx * dy + ny * dx, dx * dy)

>>> def mul_rationals(x, y):
        return rational(numer(x) * numer(y), denom(x) * denom(y))

>>> def print_rational(x):
        print(numer(x), '/', denom(x))

>>> def rationals_are_equal(x, y):
        return numer(x) * denom(y) == numer(y) * denom(x)

>>> half = rational(1, 2)
>>> print_rational(half)
1 / 2
>>> third = rational(1, 3)
>>> print_rational(mul_rationals(half, third))
1 / 6
>>> print_rational(add_rationals(third, third))
6 / 9

# 自动约分
>>> from fractions import gcd
>>> def rational(n, d):
        g = gcd(n, d)
        return (n//g, d//g)
    
# 实现二元列表
>>> def pair(x, y):
        """Return a function that represents a pair."""
        def get(index):
            if index == 0:
                return x
            elif index == 1:
                return y
        return get
>>> def select(p, i):
        """Return the element at index i of pair p."""
        return p(i)
```

抽象屏障（abstraction barrier）：高级的函数应该使用比它低的级别中尽量高的级别

![image-20231202142607846](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312021426974.png)

```python
>>> def square_rational(x):
        return mul_rational(x, x)
  
# 直接引用分子和分母会违反一个抽象屏障
>>> def square_rational_violating_once(x):
        return rational(numer(x) * numer(x), denom(x) * denom(x))

# 假设有理数会表示为双元素列表将违反两个抽象屏障
>>> def square_rational_violating_twice(x):
        return [x[0] * x[0], x[1] * x[1]]
```



List

```python
# list乘以整数表示重复
>>> [2, 7] + [1, 8, 2, 8] * 2
[2, 7, 1, 8, 2, 8, 1, 8, 2, 8]

>>> pairs = [[10, 20], [30, 40]]
>>> pairs[1]
[30, 40]

# s是一个list
for elem in s:
  
# 序列解包 (Sequence unpacking)
>>> pairs = [[1, 2], [2, 2], [2, 3], [4, 4]]
# 找到第一个和第二个元素相同的对，x和y自动绑定到第一个和第二个元素
>>> for x, y in pairs:
        if x == y:
            same_count = same_count + 1
>>> same_count
2

>>> list(range(5, 8))
[5, 6, 7]
>>> list(range(4))
[0, 1, 2, 3]

# 列表推导式 (List Comprehensions)
# <map expression> for <name> in <sequence expression> if <filter expression>
# 返回列表<map expression>
>>> odds = [1, 3, 5, 7, 9]
>>> [x+1 for x in odds]
[2, 4, 6, 8, 10]
>>> [x for x in odds if 25 % x == 0]
[1, 5]

# 城市三年的人口数
data_dict = {
    'New York': [8200000, 8300000, 8100000],
    'Los Angeles': [4000000, 4050000, 4020000],
    'Chicago': [2700000, 2750000, 2680000]
}
population_list = [population for city in sorted(data_dict.keys()) for population in data_dict[city]]
# 按城市名排序
population_list = [
    2700000, 2750000, 2680000,  # Chicago
    4000000, 4050000, 4020000,  # Los Angeles
    8200000, 8300000, 8100000   # New York
]

# perfect number是等于其约数之和的正整数
# 列出 n 的除数
>>> def divisors(n):
        return [1] + [x for x in range(2, n) if n % x == 0]
>>> divisors(12)
[1, 2, 3, 4, 6]
# 寻找完美数
>>> [n for n in range(1, 1000) if sum(divisors(n)) == n]
[6, 28, 496]

# 在给定面积的情况下找到具有整数边长的矩形的最小周长
# 面积 = 高乘宽
>>> def width(area, height):
        assert area % height == 0
        return area // height
# 求周长
>>> def perimeter(width, height):
        return 2 * width + 2 * height
>>> def minimum_perimeter(area):
        heights = divisors(area)
        perimeters = [perimeter(width(area, h), h) for h in heights]
        return min(perimeters)

>>> area = 80
>>> minimum_perimeter(area)
36
>>> [minimum_perimeter(n) for n in range(1, 10)]
[4, 6, 8, 8, 12, 10, 16, 12, 12]

# Higher-Order Functions
>>> def apply_to_all(map_fn, s):
        return [map_fn(x) for x in s]
>>> def keep_if(filter_fn, s):
        return [x for x in s if filter_fn(x)]
  
# 许多形式的聚合都可以被表示为：将双参数函数重复应用到 reduced 值
>>> def reduce(reduce_fn, s, initial):
        reduced = initial
        for x in s:
            reduced = reduce_fn(reduced, x)
        return reduced
# 将序列的所有元素相乘
>>> reduce(mul, [2, 4, 8], 1)
64

# 寻找perfect number
>>> def divisors_of(n):
        divides_n = lambda x: n % x == 0
        return [1] + keep_if(divides_n, range(2, n))
>>> divisors_of(12)
[1, 2, 3, 4, 6]

>>> from operator import add
>>> def sum_of_divisors(n):
        return reduce(add, divisors_of(n), 0)
>>> def perfect(n):
        return sum_of_divisors(n) == n
>>> keep_if(perfect, range(1, 1000))
[1, 6, 28, 496]

# 约定俗成的名字 (Conventional Names)。apply_to_all 名称是 map，keep_if 名称是 filter
# Python 中，内置的 map 和 filter 是这些不返回列表的函数的归纳。这些函数在第 4 章中讨论
# 上面的定义等效于调用内置 map 和 filter 函数的结果被应用于 list 构造函数
>>> apply_to_all = lambda map_fn, s: list(map(map_fn, s))
>>> keep_if = lambda filter_fn, s: list(filter(filter_fn, s))

# reduce 函数内置于 Python 标准库的 functools 模块中，initial 参数是可选的
>>> from functools import reduce
>>> from operator import mul
>>> def product(s):
        return reduce(mul, s)
>>> product([1, 2, 3, 4, 5])
120

# in 和 not in
# Slicing 见Xiaxiaojun ch2
```



string

```python
# *2 表示重复

# 字符串强制 (String Coercion) 
>>> str(2) + ' is an element of ' + str(digits)
'2 is an element of [1, 8, 2, 8]'
```



tree = root + branch；leaf（没有分支的树），子树，node（子树的根）

闭包 closure 指一个函数及其环境（lexical environment）的组合。闭包属性 (closure property) 是指在闭包中，函数可以访问和操作外部作用域中的变量。

![image-20231202165941064](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312021659162.png)

树的数据抽象由构造函数 `tree`、选择器函数 `label` 和 `branches` 组成

```python
>>> def tree(root_label, branches=[]):
        for branch in branches:
            assert is_tree(branch), '分支必须是树'
        return [root_label] + list(branches)
>>> def label(tree):
        return tree[0]
>>> def branches(tree):
        return tree[1:]
  
>>> def is_tree(tree):
        if type(tree) != list or len(tree) < 1:
            return False
        for branch in branches(tree):
            if not is_tree(branch):
                return False
        return True
      
>>> def is_leaf(tree):
        return not branches(tree)
  
>>> t = tree(3, [tree(1), tree(2, [tree(1), tree(1)])])
>>> t
[3, [1], [2, [1], [1]]]
>>> label(t)
3
>>> branches(t)
[[1], [2, [1], [1]]]
>>> label(branches(t)[1])
2
>>> is_leaf(t)
False
>>> is_leaf(branches(t)[0])
True

# 树递归
# 定义 The nth Fibonacci tree 是指以第 n 个斐波那契数为根标签的树
>>> def fib_tree(n):
        if n == 0 or n == 1:
            return tree(n)
        else:
            left, right = fib_tree(n-2), fib_tree(n-1)
            fib_n = label(left) + label(right)
            return tree(fib_n, [left, right])
>>> fib_tree(5)
[5, [2, [1], [1, [0], [1]]], [3, [1, [0], [1]], [2, [1], [1, [0], [1]]]]]

# 计算树的叶子数
>>> def count_leaves(tree):
      if is_leaf(tree):
          return 1
      else:
          branch_counts = [count_leaves(b) for b in branches(tree)]
          return sum(branch_counts)
>>> count_leaves(fib_tree(5))
8

# 分割树：将 n 分割为不超过 m 的若干正整数之和，二叉树
>>> def partition_tree(n, m):
        """返回一个分割树，将 n 分割为不超过 m 的若干正整数之和。"""
        if n == 0:
            return tree(True)
        elif n < 0 or m == 0:
            return tree(False)
        else:
            left = partition_tree(n-m, m)
            right = partition_tree(n, m-1)
            return tree(m, [left, right])

>>> partition_tree(2, 2)
[2, [True], [1, [1, [True], [False]], [False]]]

>>> def print_parts(tree, partition=[]):
        if is_leaf(tree):
            if label(tree):
                print(' + '.join(partition))
        else:
            left, right = branches(tree)
            m = str(label(tree))
            print_parts(left, partition + [m])
            print_parts(right, partition)

>>> print_parts(partition_tree(6, 4))
4 + 2
4 + 1 + 1
3 + 3
3 + 2 + 1
3 + 1 + 1 + 1
2 + 2 + 2
2 + 2 + 1 + 1
2 + 1 + 1 + 1 + 1
1 + 1 + 1 + 1 + 1 + 1

# 转换为二叉树 binarization，分支数量限制
>>> def right_binarize(tree):
        """返回一个右分叉的二元树"""
        if is_leaf(tree):
            return tree
        if len(tree) > 2:
            tree = [tree[0], tree[1:]]
        return [right_binarize(b) for b in tree]
>>> right_binarize([1, 2, 3, 4, 5, 6, 7])
[1, [2, [3, [4, [5, [6, 7]]]]]]
```



链表 (linked list) 

![image-20231202173747888](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312021737042.png)

```python
>>> empty = 'empty'
>>> def is_link(s):
        """ s 是一个链表，如果它是空的或一个 (first, rest) 对。"""
        return s == empty or (len(s) == 2 and is_link(s[1]))
>>> def link(first, rest):
        """用 first 元素和 rest 元素构建一个链表。"""
        assert is_link(rest), "rest 必须是一个链表。"
        return [first, rest]
>>> def first(s):
        """返回链表 s 的 first 元素。"""
        assert is_link(s), "first only applies to linked lists."
        assert s != empty, "empty linked list has no first element."
        return s[0]
>>> def rest(s):
        """返回链表 s 的 rest 元素"""
        assert is_link(s), "rest only applies to linked lists."
        assert s != empty, "empty linked list has no rest."
        return s[1]
      
>>> four = link(1, link(2, link(3, link(4, empty))))
>>> first(four)
1
>>> rest(four)
[2, [3, [4, 'empty']]]

# 迭代实现
>>> def len_link(s):
        """返回链表 s 的长度"""
        length = 0
        while s != empty:
            s, length = rest(s), length + 1
        return length
>>> def getitem_link(s, i):
        """返回链表 s 中索引为 i 的元素"""
        while i > 0:
            s, i = rest(s), i - 1
        return first(s)
      
# 递归实现
>>> def len_link_recursive(s):
        """返回链表 s 的长度."""
        if s == empty:
            return 0
        return 1 + len_link_recursive(rest(s))
>>> def getitem_link_recursive(s, i):
        """返回链表 s 中索引为 i 的元素"""
        if i == 0:
            return first(s)
        return getitem_link_recursive(rest(s), i - 1)
      
>>> def extend_link(s, t):
        """将 t 插入到链表 s 的每个元素之后"""
        assert is_link(s) and is_link(t)
        if s == empty:
            return t
        else:
            return link(first(s), extend_link(rest(s), t))
>>> extend_link(four, four)
[1, [2, [3, [4, [1, [2, [3, [4, 'empty']]]]]]]]

>>> def apply_to_all_link(f, s):
        """应用 f 到 s 的每个元素"""
        assert is_link(s)
        if s == empty:
            return s
        else:
            return link(f(first(s)), apply_to_all_link(f, rest(s)))
>>> apply_to_all_link(lambda x: x*x, four)
[1, [4, [9, [16, 'empty']]]]

>>> def keep_if_link(f, s):
        """返回 s 中 f(e) 为 True 的元素"""
        assert is_link(s)
        if s == empty:
            return s
        else:
            kept = keep_if_link(f, rest(s))
            if f(first(s)):
                return link(first(s), kept)
            else:
                return kept
>>> keep_if_link(lambda x: x%2 == 0, four)
[2, [4, 'empty']]

>>> def join_link(s, separator):
        """返回由分隔符分隔的 s 中的所有元素组成的字符串。"""
        if s == empty:
            return ""
        elif rest(s) == empty:
            return str(first(s))
        else:
            return str(first(s)) + separator + join_link(rest(s), separator)
>>> join_link(four, ", ")
'1, 2, 3, 4'

# 递归构造 (Recursive Construction)：链表在递增地构造序列时特别有用
# 将 n 分割为不超过 m 的若干正整数之和的方法数
>>> def partitions(n, m):
        """返回一个由 n 个分区组成的链表，每个分区的部分数最多为 m。每个分区表示为一个链表。
        """
        if n == 0:
            return link(empty, empty) # A list containing the empty partition
        elif n < 0 or m == 0:
            return empty
        else:
            using_m = partitions(n-m, m)
            with_m = apply_to_all_link(lambda s: link(m, s), using_m)
            without_m = partitions(n, m-1)
            return extend_link(with_m, without_m)
          
>>> def print_partitions(n, m):
        lists = partitions(n, m)
        strings = apply_to_all_link(lambda s: join_link(s, " + "), lists)
        print(join_link(strings, "\n"))
        
>>> print_partitions(6, 4)
4 + 2
4 + 1 + 1
3 + 3
3 + 2 + 1
3 + 1 + 1 + 1
2 + 2 + 2
2 + 2 + 1 + 1
2 + 1 + 1 + 1 + 1
1 + 1 + 1 + 1 + 1 + 1
```



函数执行操作，数据被操作。对象 (objects) 将数据值与行为结合起来，类class 和 实例 instance 和 属性attributes 和 方法methods

```python
>>> from datetime import date
>>> tues = date(2014, 5, 13)
>>> print(date(2014, 5, 19) - tues)
6 days, 0:00:00
>>> tues.year
2014
>>> tues.strftime('%A, %B %d')
'Tuesday, May 13'
```

数字 number、字符串 string、列表 list 和范围 range 也都是对象。他们代表值，也拥有这些值的行为，有属性和方法。Python 中的所有值都是对象

```python
>>> '1234'.isnumeric()
True
>>> 'rOBERT dE nIRO'.swapcase()
'Robert De Niro'
>>> 'eyes'.upper().endswith('YES')
True
```

**共享 (Sharing) 和身份 (Identity)** 

```python
>>> chinese = ['coin', 'string', 'myriad']
>>> suits = chinese                        
>>> suits.pop()             # 移除最后的元素，并返回该元素
'myriad'
>>> suits.remove('string')  # 移除第一个等于 string 的元素
>>> suits.append('cup')              # 在最后追加一个元素
>>> suits.extend(['sword', 'club'])  # 在最后添加列表中的所有元素
>>> suits[2] = 'spade'  # 替换一个元素
>>> suits[0:2] = ['heart', 'diamond']  # 替换一个 slice 切片
>>> suits
['heart', 'diamond', 'spade', 'club']

# 使用列表构造函数来复制列表
>>> nest = list(suits)  # 用与 suits 一样的元素新建一个列表

# 列表比较，is 和 is not。is 和 == 之间的区别：前者检查身份，而后者检查内容的相等性。
>>> suits
['heart', 'diamond', 'spade', 'club']
>>> nest[0] = suits     # 创建一个内嵌列表
>>> suits is nest[0]
True
>>> suits is ['heart', 'diamond', 'spade', 'club']
False
>>> suits == ['heart', 'diamond', 'spade', 'club']
True

# 列表推导式 (List comprehensions)
# 查找名称对应的字符
>>> from unicodedata import lookup
>>> [lookup('WHITE ' + s.upper() + ' SUIT') for s in suits]
['♡', '♢', '♤', '♧']
```



元组 (Tuples)，不可变

```python
>>> ("the", 1, ("and", "only"))
>>> ()    # 空
>>> (10,) # 单元素

>>> code = ("up", "up", "down", "down") + ("left", "right") * 2
>>> len(code)
8
>>> code[3]
'down'
>>> code.count("down")
2
>>> code.index("left")
4

# 虽然无法更改元组中的元素，但可以更改元组中包含的可变元素的值。
nest = (10, 20, [30, 40])
nest[2].pop()
nest = (10, 20, [30])

# 元组会在多重赋值中被隐式地使用。将两个值绑到两个名称时，会创建一个双元素元组，然后将其解包。
>>> 1, 2 + 3
(1, 5)
```



字典， key & value，每个键最多一个值，键不能是可变值。无序集合（Python 3.7 及以上的字典顺序会确保为插入顺序）

```python
>>> numerals = {'I': 1.0, 'V': 5, 'X': 10}
>>> numerals['I'] = 1
>>> numerals['L'] = 50
>>> numerals
{'I': 1, 'X': 10, 'L': 50, 'V': 5}
>>> sum(numerals.values())
66

>>> dict([(3, 9), (4, 16), (5, 25)])
{3: 9, 4: 16, 5: 25}

# get 的参数是键和默认值，返回键的值（如果键存在）或默认值。
>>> numerals.get('A', 0)
0
>>> numerals.get('V', 0)
5

# 推导式（会创建一个新的字典对象），键表达式和值表达式由冒号分隔
>>> {x: x*x for x in range(3,6)}
{3: 9, 4: 16, 5: 25}
```



列表和字典具有局部状态 (local state)：它们可以在程序执行的任何节点改变有特定内容的值。

函数也可以有局部状态。

```python
# withdraw是非纯函数，调用该函数返回一个值，而且还具有以某种方式改变函数的副作用，使得下一次使用相同参数的调用将返回不同的结果。这种副作用是对当前帧之外的“名称-值”绑定进行更改
>>> withdraw(25)
75
>>> withdraw(25)
50
>>> withdraw(60)
'Insufficient funds'
>>> withdraw(15)
35

# 为了构建 withdraw ，必须同时用初始帐户余额来构建它。函数 make_withdraw 是一个高阶函数，起始余额是它的参数，函数 withdrawal 是它的返回值。
>>> withdraw = make_withdraw(100)

#非局部语句 (nonlocal statement)。更改 balance 的绑定时，绑定关系都会在已经绑定 balance 的第一个帧中更改。
>>> def make_withdraw(balance):
        """返回一个每次调用都会减少余额的 withdraw 函数"""
        def withdraw(amount):
            nonlocal balance                 # 声明 balance 是非局部的
            if amount > balance:
                return 'Insufficient funds'
            balance = balance - amount       # 重新绑定
            return balance
        return withdraw
```

两个 withdraw 帧都具有相同的父级。局部定义的函数可以在局部帧之外查找名称，访问非局部名称不需要nonlocal语句，更改非局部名称需要nonlocal语句。Python 2 中不存在非局部语句。

![image-20231207104630585](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312071046686.png)

非局部赋值模式是具有高阶函数和词法作用域的编程语言的普遍特征。

删除nonlocal报错，Python 无法在非局部帧中查找名称的值，然后在局部帧中绑定相同的名称。

![image-20231215085127168](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312150851275.png)

多个 unlocal 绑定，每个 withdraw 实例都保持自己的 balance 状态

![image-20231208083013739](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312080830839.png)

警惕：两个名称都引用同一个函数

![image-20231208083427059](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312080834162.png)

纯函数是引用透明 (referentially transparent) 的，unlocal违反了引用透明



可变列表实现，消息传递 (Message passing)

不能使用 None 来空列表，因为两个空列表是不同的，但 None 始终是 None；使用empty表示

```python
>>> def mutable_link():
        """返回一个可变链表的函数"""
        contents = empty
        def dispatch(message, value=None):
            nonlocal contents
            if message == 'len':
                return len_link(contents)
            elif message == 'getitem':
                return getitem_link(contents, value)
            elif message == 'push_first':
                contents = link(value, contents)
            elif message == 'pop_first':
                f = first(contents)
                contents = rest(contents)
                return f
            elif message == 'str':
                return join_link(contents, ", ")
        return dispatch
      
# 构建链表，反序添加每个元素即可
>>> def to_mutable_link(source):
        """返回一个与原列表相同内容的函数列表"""
        s = mutable_link()
        for element in reversed(source):
            s('push_first', element)
        return s
    
>>> s = to_mutable_link(suits)
>>> type(s)
<class 'function'>
>>> print(s('str'))
heart, diamond, spade, club
>>> s('pop_first')
'heart'
>>> print(s('str'))
diamond, spade, club
```



字典实现，双元素列表

```python
>>> def dictionary():
        """返回一个字典的函数实现"""
        records = []
        def getitem(key):
            matches = [r for r in records if r[0] == key]
            if len(matches) == 1:
                key, value = matches[0]
                return value
        def setitem(key, value):
            nonlocal records
            non_matches = [r for r in records if r[0] != key]
            records = non_matches + [[key, value]]
        def dispatch(message, key=None, value=None):
            if message == 'getitem':
                return getitem(key)
            elif message == 'setitem':
                setitem(key, value)
        return dispatch
      
>>> d = dictionary()
>>> d('setitem', 3, 9)
>>> d('setitem', 4, 16)
>>> d('getitem', 3)
9
>>> d('getitem', 4)
16
```



调度字典（Dispatch Dictionaries）

`dispatch` 函数是实现抽象数据消息传递接口的通用方法。balance 是一个数字，而消息存款 deposit 和取款 withdraw 则绑定到函数。这些函数可以访问 dispatch 字典，通过将 balance 存储在 dispatch 字典中而不是直接存储在帐户帧中，避免在存款和取款中对非局部语句的需要。

```python
# 可变帐户是作为字典实现的。它有一个构造器 amount 和选择器 check_balance，以及存取资金的功能。
def account(initial_balance):
    def deposit(amount):
        dispatch['balance'] += amount
        return dispatch['balance']
    def withdraw(amount):
        if amount > dispatch['balance']:
            return 'Insufficient funds'
        dispatch['balance'] -= amount
        return dispatch['balance']
    dispatch = {'deposit':   deposit,
                'withdraw':  withdraw,
                'balance':   initial_balance}
    return dispatch

def withdraw(account, amount):
    return account['withdraw'](amount)

def deposit(account, amount):
    return account['deposit'](amount)

def check_balance(account):
    return account['balance']

a = account(20)
deposit(a, 5)
withdraw(a, 17)
check_balance(a)
```



约束传递 (Propagating Constraints)















# LAB&HW

求因子：for i in range(x//2, 0, -1):

return 输出含""， print不含

if 语句是按需进行求值的，而函数调用是在调用函数时立即求值的

```python
def count_cond(condition):
  	"""
    >>> count_factors = count_cond(lambda n, i : n % i == 0)
    >>> count_factors(2)   # 1, 2
    2
    >>> count_factors(4)   # 1, 2, 4
    3
    >>> count_factors(12)  # 1, 2, 3, 4, 6, 12
    6
    """
    def f(n):
        count = 0
        for i in range(1, n+1):
            # if condition:
            if condition(n, i):
                count += 1
        return count
    return f
```

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312021649615.png" alt="image-20231202164949465" style="zoom:110%;" />
