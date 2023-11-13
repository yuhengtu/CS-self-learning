# LEC1 INTRO

![image-20231101214804594](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311012148651.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311012150423.png" alt="image-20231101215009394" style="zoom:150%;" />

Amdahl's Law 并行的能力是有限的。假设一个程序由串行部分和并行部分组成，串行部分无法并行化，而并行部分可以并行执行。如果你优化了并行部分，使其运行更快，那么整个程序的性能提升将受到串行部分的限制。

Latency：完成一个任务要多少时间

Throughput：一定时间内能完成多少任务

redundancy：冗余计算，计算多个结果后vote

摩尔定律放缓，电池限制主频；现在是CA的时代，各种专用的架构，TPU



# LEC2 Number Rep

音乐采样，每秒44100次

ASCII码，8位，实则只用了7位；unicode（8/16/32位）包含emoji 

16进制 Hexadecimal

4 Bits = 1 Nibble

![image-20231105191154386](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311051911454.png)

Sign and Magnitude，第一位符号位，后面绝对值；2个0，电路复杂，仅用于signal processors；(C’s unsigned int, C18’s uintN_t)

One’s Complement，反码，有两个0（overlap），算术复杂；后被放弃

![image-20231105192912418](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311051929446.png)

把负数部分左移1个位置，得到补码，解决overlap；-15 ~ +15 变成 -16 ~ +15；正数在原、反、补都没动

Two’s Complement (C’s int, C18’s intN_t），取反加一，硬件简单

转10进制，首位乘负号

![image-20231105231919765](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311052319794.png)

求补码是1101的值的10进制数

![image-20231105232136040](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311052321068.png)

取反加一，相当于十进制数加负号

![image-20231105232328644](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311052323674.png)

看见0开头，直接得到正数值；看见1开头，取反加一，转10进制，加负号

![image-20231105232912011](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311052329057.png)

bias encoding：现有值域0-31的信号（unsigned），拉低到中心位置在0；bias = ![image-20231105234357195](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311052343217.png)，拉低到 -15 ~ +16；0对应的 5bit 就是bias值

![image-20231105234552701](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311052345752.png)

原码反码都不用；只使用unsigned，Two’s Complement，bias；对unsigned和Two’s Complement做运算，用的是同样的硬件；注意overflow问题

硬件存储和传输的单位都是*1000，其他一切（cache memory）都 *1024

![image-20231106002410743](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311060024776.png)

# LEC3 C

C在各种机器上一致，编译成不一致的汇编语言；C不安全（指针，内存泄漏），C不是强类型；Rust是安全的C；Go，在多核多线程上运行程序

compile & interpret

1. 编译器（Compiler）：
   - 编译器将源代码一次性翻译成机器代码或中间代码，通常在程序运行之前。
   - 翻译的结果是独立的可执行文件，不需要源代码来执行程序。
   - 编译过程较慢，但生成的可执行文件在执行时通常更快。
   - 编译器检查整个代码并提供详尽的错误信息，包括潜在的问题。
   - 典型的编译语言包括C、C++、Rust等。

2. 解释器（Interpreter）：
   - 解释器将源代码逐行翻译成机器代码或直接执行。
   - 翻译和执行是同时进行的，不生成独立的可执行文件。
   - 解释过程较快，但通常比编译后的程序慢。
   - 解释器在运行时逐行检查和执行代码，发现错误时可能会中断程序。
   - 典型的解释语言包括Python、JavaScript、Ruby等。

C compilers map C programs directly into architecture-specific machine code (string of 1s and 0s) ，不同架构不同（Mac&PC&Linux，processor、OS、library不同）；port code 在不同architecture间转换；不进行语法检查

Java C（Java compiler）converts to architecture-independent bytecode that may then be compiled by a just-in-time compiler (JIT) ；Java有compiler和interpreter

Python converts to a byte code at runtime. interpret language；python强在调用GPU；在python中复杂操作时call C program，cython

For C, generally a two part process of compiling；1: .c files to .o files（assembly） 2: then linking the .o files with libraries into executables

makefile：修改foo.c，无需编译bar.o，只需编译foo.c后relink，加速；因此鼓励分多文件；parallel compile：make -j；linker is sequential

![image-20231106082740882](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311060827953.png)

#开头（宏编写）是C pre-processor命令，抓取.h文件放入.i文件；#if 用于debug；#define 加括号，safe

![image-20231106084848449](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311060848524.png)

next = min(w, foo(z)); 令foo(z)内容为print helloworld并返回一个值，此时比较w和foo(z)时call第一次foo(z)；若foo(z)较小，输出结果时call第二次foo(z)，打印两次helloworld

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311060858554.png)

注意命名规范，gcc -o hello hello.c，由此给a.out起名为hello.c

![image-20231106090542419](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311060905450.png)

return 0；Unix/Linux中0是程序运行成功，其他是失败

此对比表格是C99之前的ANSI C，C99之后和Java一样

![image-20231106091320634](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311060913666.png)

C99的升级

![image-20231106095142676](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311060951730.png)

printf(“%ld\n", __STDC_VERSION__);  199901，201112L（C11），201710L（C18）；移除gets（very unsafe）

![image-20231106095308436](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311060953473.png)

int main (int argc, char *argv[])；

`argc`（参数计数）用于存储在命令行中传递给程序的参数的数量。这个参数包括程序自身的名称，所以`argc`的值至少为1（如果在命令行中没有提供任何参数，`argc`的值将是1）；

`argv`（参数值）是一个指向字符指针数组的指针。这个数组中的每个元素都是一个指向以空格分隔的参数的字符串的指针。`argv[0]`通常是程序的名称，`argv[1]`、`argv[2]`，依此类推，包含了传递给程序的其他参数。这些参数以C字符串的形式存储，以null终止（字符串末尾有一个空字符 '\0' 来表示字符串的结束）。

$ ./my_program arg1 arg2 arg3；在这种情况下，`argc`的值将是4（包括程序名称），而`argv`数组将包含以下内容：`argv[0]` 指向程序名称 "./my_program" 的字符串，`argv[1]` 指向字符串 "arg1"，`argv[2]` 指向字符串 "arg2"，`argv[3]` 指向字符串 "arg3"

![image-20231106101108697](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061011729.png)

C Syntax

用int时指定位数宽度bit，use intN_t and uintN_t；sizeof返回Byte；int是最高效的数据类型

![image-20231106101703655](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061017689.png)

enum color {RED, GREEN, BLUE};

![image-20231106102215037](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061022071.png)

别用goto

![image-20231107110203067](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071102136.png)

switch别漏break

![image-20231106102648920](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061026954.png)

while等判断语句括号内用int变量，别用float/double（0.0001,0.99999）

# LEC4

变量初始化指定为0，养成习惯（否则是未知数garbage）“Heisenbugs”（opposed to “Bohrbugs” which are repeatable）

![image-20231106103924304](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061039340.png)

pointer：int *p; p = &y; z = *p；& called the “address operator” ；\* called the “dereference operator” 

![image-20231106104504864](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061045898.png)

y = 3，3传入函数变成4，y仍是3

![image-20231106105152205](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061051241.png)![image-20231106105332515](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061053551.png)

野指针

![image-20231106105635190](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061056225.png)

pass pointer instead of copying all data 轻量化

Normally a pointer can only point to one type；void * (generic pointer)可以指向多种数据类型，慎用

pointers to functions： int (*fn) (void *, void *) = &foo ；fn is a function that accepts two void * pointers and returns an int and is initially pointing to the function foo；(*fn)(x, y) /foo(x, y)will then call the function

p1 = p2;  使二者的xy值相同

![image-20231106113653792](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061136882.png)

NULL pointers

![image-20231106114042186](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061140219.png)

Modern machines are “byte-addressable”（Hardware’s memory composed of 8-bit storage cells, each has a unique address）A C pointer is just abstracted memory address

word alignment 就算char只用了1Byte，也自动占用4Byte空间，十六进制地址结尾是048C（如 0x1004）；32位机中，1字=4字节，变量起始地址必定是4的倍数（64位机，1字=8字节）

![image-20231106114944451](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061149485.png)

array：ask for continuous block of memory

char *string and char string[] are nearly identical；An array variable is a “pointer” to the first element.

ar[0] is the same as *ar；ar[2] is the same as *(ar+2)

典型错误：char *foo() { char string[32]; ...; return string; }，返回一个指向 `string` 数组的指针，一旦函数执行完成，局部变量被销毁，指针会变得无效，因为它指向的内存空间已经被释放

代码风格

![image-20231106165230152](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061652218.png)

pass the array and its size，C不会检查也不知道数组长度；Segmentation faults（读写无权限的空间） and bus errors（对齐错误，十六进制地址末尾不是048C）

q指针仍未移动，用指针的指针，称为handle：**h ；一般不会超过两层

![image-20231106193510293](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061935393.png)

![image-20231106193605165](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061936198.png)

`map`函数：用于对可迭代对象（如列表、元组等）中的每个元素应用一个特定的函数，返回一个新的可迭代对象，其中包含了经过该函数处理的每个元素的结果。

![image-20231106194255363](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061942404.png)





# LEC5

动态内存分配（几乎不用于单变量）：C从不做初始化，malloc返回一个void*（garbage），需要typecast；

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311062303221.png" alt="image-20231106230345160" style="zoom:70%;" />

必须手动 free(ptr)；否则作为子程序容易内存泄漏；常见error：初始化未赋值时使用，free两次，free正在运作的malloc pointer（calling free() on something you didn’t get back from malloc）

![image-20231106230750307](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311062307342.png)

如果空间不够，malloc会返回pointer = 0，所以最好判断一下 if 0 == pointer（写成pointer=0会bug，写成0=pointer没事）

realloc，重新调整alloc的大小；原本空间10，realloc成20，把原本存的10个值移动到新的20空间内的前十个；check realloc是否成功：1.pointer ！= null  2.resize是否正确

![image-20231106232849346](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311062328402.png)

![image-20231106233334142](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311062333194.png)

a linked list of strings

`typedef struct Node *List;` 这行代码定义了一个新的类型 `List`，它是一个指向 `struct Node` 结构体的指针，实际上是用来表示链表的头节点。

![image-20231107104834986](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071048076.png)

`List list_add(List list, char *string)` 用于将一个字符串 `string` 添加到已经存在的链表中。函数接受两个参数：`list`：一个指向链表头的指针。`string`：一个指向字符串的指针。

分配了一个新的节点 `struct Node` 的内存；为新节点中的 `value` 成员分配内存；使用 `strcpy` 函数将字符串复制到新分配的内存中；将新节点的 `next` 指针指向现有的链表头（`list` 参数传入的链表头），以将新节点添加到链表的开头；返回新链表的头节点

strlen不含\0，strlen（abc）= 3，因此开空间strlen + 1

![image-20231107110712369](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071107412.png)

![image-20231107112348653](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071123715.png)

![image-20231107112432723](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071124764.png)

![image-20231107112455640](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071124673.png)

![image-20231107112516591](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071125623.png)

![image-20231107112628997](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071126036.png)

memory location

Structure declaration does not allocate memory；

三种 allocate memory：  local Variable declaration（int i; struct Node list; char *string; int ar[n];）；“Dynamic” allocation（ptr = (struct Node *) malloc (sizeof(struct Node) *n);）；global Variable declaration before main（int myGlobal; global变量别定义太多）

C has 3 pools of memory ；stack和数据结构中的stack类似，heap和数据结构中的heap完全不同

Static storage: global variable storage, basically permanent, entire program run 

The Stack: local variable storage, parameters, return address，free when return (location of “activation records” in Java or “stack frame” in C)  

The Heap (dynamic malloc storage): data lives until deallocated by programmer（Photoshop调用大量heap）

![image-20231107123302287](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071233369.png)

array很快但占空间，malloc较慢，但是mallo初始是null，不占空间，可以防止内存不足

深入stack：heap is not contiguous， stack must be contiguous；LIFO，Last In First Out

![image-20231107123920331](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071239378.png)

![image-20231107124438204](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071244237.png)

当return d，SP指针上移，但是最下一块的变量并未被erase，只是无法访问

![image-20231107124526930](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071245970.png)

深入heap：避免 external fragmention

![image-20231107125426866](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071254900.png)

Heap类似一个循环链表，每一块存内容长度和指向下一块的指针，当中夹杂着free fragment space

![image-20231107125905811](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071259870.png)

写数组只需要一个时钟周期，写malloc很慢，找空间，free时是否合并其他free空间

![image-20231107125959336](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071259387.png)

选怎样的free block 是一种 trade off；想要100个空间，找到一个104的，会出现长度为4的碎片，说不定继续找会有正好100 的？

![image-20231107130346598](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071303645.png)

常见error

Dangling reference 在给定指针指向的malloc value之前就使用指针；Memory leaks 没有free或free的太晚

![image-20231107131133504](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071311572.png)

注意数组边界，数组是0-99，把100设成了0

![image-20231107131445851](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071314894.png)

重要：不可Return Pointers into a memory stack space

Ptr()调用后销毁，留下了不可访问的y=3，stackAddr指向这个3，第一个printf函数使用了新的stack frame，篡改了原本3的位置的内容，变成garbage

![image-20231107132024284](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071320323.png)

free后不可再使用：如果未被覆盖，a还在内存中（不erase），可能work；如果被覆盖，crash

![](../Library/Application Support/typora-user-images/image-20231107132424187.png)

别用realloc；f和g指向同一个空间为10的部分，realloc为20后，如果空间足够，f和g仍在原地，不出错；如果空间不足，f被移动到其他block，g变成stale pointer（陈旧指针）

![image-20231107132731373](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071327407.png)

Freeing the Wrong Stuff

free list is a list of address，malloc申请也会有一个list，free和malloc必须是同一个位置

![image-20231107133144854](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071331895.png)

dont Double-Free

Lose the initial pointer! (Memory Leak)

![image-20231107135112708](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071351758.png)

解决办法：使用Valgrind check code

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071352048.png)

# LEC6

Nvidia  A100 引入了一种TF32的 floating point format，19-bit （之前有IEEE754，BFLOAT16），机器学习注重指数位数exponent，不注重significand；format这一块没有统一标准，trade off

![image-20231107140521001](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071405118.png)

“95% of the folks out there are completely clueless about floatingpoint.” – James Gosling, 1998-02-28









# LEC24 Caches

Binary prefix：硬盘storage/网络传输都是变单位*1000，但是其他所有包括cache和memory都是变单位 *1024，IEC将base2的单位改名xxbi避免混淆

![image-20231107154829868](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071548934.png)

kmgtpezy，记，Kissing Mentors Gives Testy Persistent Extremists Zealous Youthfulness.

2^34 = 16GB；2.5TB，ceiling = 4TB = 2 ^42，42bit

![image-20231107160849192](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071608279.png)

caches

一切都是为了处理：processor很快，但是memory很慢

![image-20231107162039804](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071620903.png)

cache，从茫茫图书馆，复制10本书摊开在桌子上，working set ，计算思维

重要：Cache is a **copy** of a subset of main memory，高层都是低层的subset copy

Most processors have separate caches for instructions and data. 

![image-20231107165015712](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071650787.png)

![image-20231107170030697](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071700755.png)

Caches work on the principles of temporal and spatial locality. 现在用了，很可能马上会再用（时间），很可能会用它的neighbor（空间）；因此存储刚用到过的内容，从遥远的地方取东西的时候顺便取其neighbor

![image-20231107191118615](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071911724.png)

# LEC25

Direct-Mapped Cache

传输单位（transfer between cache and memory）：block （memory all physically wired-in, no search）

蓝色block去cache中蓝色区域；mod4（余数）就是block位置，或者说是二进制的后两位所决定<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082018085.png" alt="image-20231108201849385" style="zoom:15%;" />

![image-20231108202018905](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082020931.png)

上图中如果load一个float 32位，会分布在不同的cache块上；为了避免之，cache一般至少1 word wide 

从右往左，从上往下；11101，111是7，左移2位乘4=28，+1=29

index选择cache中哪一行；offset选择block中哪一位；如何知道cache中的蓝色block来自memory中的哪个蓝色block？用tag，使用memory地址高位部分，把1E简化为3，14简化为2

![image-20231108204652340](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082046367.png)

![image-20231108205306948](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082053976.png)

TIO Dan，Uncle Dan

![image-20231108205401749](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082054771.png)

Index -> number of block；Offset -> Bites per block

![image-20231108210859786](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082108818.png)

example

![image-20231108221831443](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082218474.png)

![image-20231108221916689](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082219731.png)

![image-20231108222057944](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082220970.png)

计算机有无cache不影响program本身

在无cache的情况下，t1是指向memory地址1022的指针，地址1022处值99 ，load到t0中

![image-20231108222713910](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082227942.png)

有cache

![image-20231108222838213](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082228248.png)

从右往左，从上往下

![image-20231108223951806](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082239834.png)

覆盖原来cache的第一个位置

![image-20231108223858948](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082238984.png)

![image-20231108224058689](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311082240716.png)

memory stride：bit size左移一位；block size 下移一位；cache size 下移到下一个tag同一位置

cache terminology

hit，能从cache取到东西；miss，cache中无，要去memory取；要从memory取来覆写之前cache中存的内容

![image-20231109004431929](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090044007.png)

informal：cache的温度反映其是否满

![image-20231109004806184](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090048247.png)![image-20231109005258684](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090052743.png)

用valid bit代替erase的过程，0代表invalid，内容变成garbage；为0时必定miss，不可能hit

![image-20231109005526762](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090055819.png)

16KB = 2^14，14位；16K ÷ 16 = 1000 需要 10bit index（I）；剩下14 - 10 = 4 bit，offset；还要valid bit + tag(32位机，32-14 = 18)

![image-20231109005840237](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090058294.png)

# LEC26

Direct-Mapped Cache example；memory中value都是4Byte为单位

![image-20231109015456824](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090154885.png)![image-20231109015556284](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090155330.png)

读入第一个：1、由index找到位置；2、check valid bit，cache miss；3、读入tag，读入b及其附件的acd（都是4Byte的word），valid bit置1；

顺序 IVTO，index，valid，tag，offset

![image-20231109020121231](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090201270.png)![image-20231109020822261](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090208320.png)

读第二个，cache hit

![image-20231109021110526](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090211590.png)

读第三个，同1

![image-20231109021309452](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090213504.png)

读都四个，tag不匹配，cache miss，block replacement

![image-20231109021535930](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090215982.png)

![image-20231109021745947](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090217998.png)![image-20231109021814930](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311090218990.png)

硬件上并非按顺序进行，而是全部同时并行执行

![image-20231109102914098](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091029171.png)

对于同一地址不断更新的值：write-through，每次写入cache立刻也写入memory；write-back，cache更新完了再写入memory；因此需要加入dirty bit，指示最新的结果是在cache，不在memory（就算新结果和老结果一样）；memory stale -> dirty bit = 1 ；OS会定期把cache的新值回传memory

![image-20231109104604399](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091046435.png)

软件中采用cache思想：cache for recent phone call，cache for web 

问题：如果stride = cache size，正好每个循环都要block replacement

![image-20231109105008244](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091050276.png)

整个cache就是one row * cache size ，很糟糕 

![image-20231109110507443](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091105475.png)

average access time = miss penalty * miss rate

 ![image-20231109110906562](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091109595.png)

cache miss：1、because of cold cache；2、memory中两个蓝色的part在cache中竞争同一个位置（可解决）

![image-20231109111029893](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091110932.png)

![image-20231109111214825](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091112868.png)

Fully Associative Cache

可以进入任何一行，并非按颜色对应

![image-20231109112547109](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091125142.png)

![image-20231109112806563](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091128605.png)

![image-20231109112833716](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091128759.png)

第三种cache miss

![image-20231109113206838](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091132871.png)

分类方法（第四种miss是由于parallel，之后会讲）；三种miss分类很复杂，详见DIS9

![image-20231109113706096](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091137128.png)

direct mapped cache，下图未加入dirty bit

![image-20231109114308139](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091143172.png)

# LEC27

一种direct mapped 和 fully associative 的中间值：Set Associative Cache

一个set有很多block，index指向set，set内是fully associative的

![image-20231109135808759](../Library/Application Support/typora-user-images/image-20231109135808759.png)

![image-20231109140241487](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091402527.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091405482.png" alt="image-20231109140537456" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091406025.png" alt="image-20231109140612980" style="zoom:25%;" />

横着一排4个是一个set，one-hot 只有一个能通过

![image-20231109141205431](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091412481.png)

Block Replacement Policy

![image-20231109142245394](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091422436.png)

LRU，4-way有4! = 24种顺序，要5bit来存储；random经常表现最好；LRU是最常用的

![image-20231109142719352](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091427388.png)

example

 ![image-20231109144937395](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091449476.png)

![image-20231109144956856](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091449891.png)

cache要optimize的评价标准 AMAT，Average Memory Access Time，由此选择超参数

不论miss或not miss，都要付出 hit time，因此相当于*100%

![image-20231109174839325](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091748415.png)

多个级别的cache，L1是最小最贵的

![image-20231109175147010](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091751064.png)

![image-20231109175448634](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091754677.png)

每个core有自己的L1 L2 cache；所有core共享L3 cache，memory是RAM

AMAT的递归计算

![image-20231109175851795](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091758830.png)

example

![image-20231109175948062](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091759089.png)

![image-20231109180146507](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091801543.png)

old computer：L1在芯片上；L2在芯片上，使用SRAM

![image-20231109180511923](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091805959.png)

此处L2 miss rate是local miss rate，详见DIS9

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091805678.png" alt="image-20231109180524649" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091807088.png" alt="image-20231109180733050" style="zoom:20%;" />

actual single-core CPU

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091835750.png" alt="image-20231109183557710" style="zoom:200%;" />

![image-20231109183709761](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091837808.png)

![image-20231109183720034](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091837076.png)

有参考hardware cache思想的software cache

![image-20231109184047148](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311091840186.png)

![image-20231110001743299](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311100017366.png)



# LAB0

```
$ ssh cs61c-???@hive#.cs.berkeley.edu
# password: https://acropolis.cs.berkeley.edu/~account/webacct/
$ git config --global user.name "John Doe"
$ git config --global user.email johndoe@example.com
# Take a look at your repo’s current remotes and status
$ git remote -v
$ git status

$ exit

# If nothing happens for a long time: check your internet connection. Some network firewalls, including CalVisitor on campus, block SSH. Try another network (AirBears2 or eduroam if you’re on campus).
# Connection refused or other weird errors: the Hive machine you picked might be down. Try another one
# Reserved for cs61c staff: try another Hive machine :)
```

- `<ctrl> + a` will move the cursor to the beginning of the current line (helpful for fixing mistakes)
- `<ctrl> + e` will move the cursor to the end of the current line (also helpful for fixing mistakes)
- `<ctrl> + r` will let you search through your recently used commands
- List all hidden files (`ls -a`).
- `man echo | less`

```
$ scp <source> <destination>
# get individual files or entire folders from the Hive machines onto your local system, or vice versa. run on local system terminal
$ scp cs61c-???@hive3.cs.berkeley.edu:~/some_folder/example.txt ~/Downloads/
$ scp ~/Downloads/example.txt cs61c-???@hive7.cs.berkeley.edu:~/some_folder/
# To copy folders, -r
$ scp -r cs61c-???@hive7.cs.berkeley.edu:~/some_folder ~/Downloads/
```

![image-20231112170906323](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311121709430.png)

yy复制一整行

