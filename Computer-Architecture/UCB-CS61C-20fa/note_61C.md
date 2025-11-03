#   LEC1 INTRO

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221641832.png" alt="image-20240622164100654" style="zoom: 67%;" />

![image-20240622164250421](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221642470.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311012148651.png" alt="image-20231101214804594" style="zoom: 50%;" />

![image-20240622164348537](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221643575.png)

![image-20240622164447829](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221644872.png)

![image-20240622164532326](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221645368.png)

![image-20240622164553152](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221645189.png)

![image-20240622164615927](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221646967.png)

![image-20240622164636571](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221646610.png)

Amdahl's Law 并行的能力是有限的。假设一个程序由串行部分和并行部分组成，串行部分无法并行化，而并行部分可以并行执行。如果你优化了并行部分，使其运行更快，那么整个程序的性能提升将受到串行部分的限制。

Latency：完成一个任务要多少时间

Throughput：一定时间内能完成多少任务

![image-20240622164831345](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221648400.png)

redundancy：冗余计算，计算多个结果后vote

![image-20240622164915422](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221649459.png)

![image-20240622164954539](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221649612.png)

摩尔定律放缓，电池限制主频；现在是CA的时代，各种专用的架构，TPU



# LEC2 Number Rep

音乐采样，每秒44100次

ASCII码，8位，实则只用了7位；unicode（8/16/32位）包含emoji 

16进制 Hexadecimal

4 Bits = 1 Nibble

![image-20231105191154386](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311051911454.png)

Sign and Magnitude，第一位符号位，后面绝对值；2个0，电路复杂，仅用于signal processors；

(C’s unsigned int, C18’s uintN_t)

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

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311060024776.png" alt="image-20231106002410743" style="zoom:67%;" />





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

# LEC6 Floating Point

Nvidia  A100 引入了一种TF32的 floating point format，19-bit （之前有IEEE754，BFLOAT16），机器学习注重指数位数exponent，不注重significand；format这一块没有统一标准，trade off

![image-20231107140521001](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311071405118.png)

“95% of the folks out there are completely clueless about floatingpoint.” – James Gosling, 1998-02-28

![image-20231115175323430](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151753551.png)

![image-20231115175829174](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151758222.png)

![image-20231115180346323](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151803374.png)

以上是固定小数点；如何同时表示很大的和很小的数？-> 移动小数点 floating point

MSB 最高有效位（Most Significant Bit）

![image-20231115220811246](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152208350.png)

![image-20231115220928211](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152209250.png)

二进制中有效数字除非=0，其他时候必然是1.xxxx，个位必定是1，因此不存，只存小数点后的；每一位是2^-n

![image-20231115223453088](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152234185.png)

![image-20231115223652181](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152236232.png)

IEEE754，大多数电脑采用的；2个0

![image-20231115224107407](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161622648.png)

为了普通CPU也能比较浮点数，要求负的比正的小，因此采用bias notation，-127是0，+128是255，bias = 127 = (2^(N-1))-1

![image-20231115225413979](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152254039.png)

特殊数

![image-20231116011957601](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161622342.png)

exponent = 255，sig = 0是无穷；sig != 0，存放报错信息，即各种错误类型的NaN

最小数是2^-126^ 数字之间最小间距是2^-149^，0附近有一个巨大的gap，要把他们填充起来

![image-20231116005935461](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161622594.png)

这部分叫denormalized number，默认为significance*2^-126^，denormalized中步长为 2^-149^；随后进入normalized部分，一开始步长也是 2^-149^，当exponent *2，步长变为2^-148^；exponent+1，步长 *2；步长越来越大，才能走到很大的数去

![image-20231116010740345](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311160107406.png)

the bias；1之后的step = 2^-23^

![image-20231116012627789](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161622330.png)

永远是2^23^ = 800 million个间隔，起先step = 2^-149^ 置于0~ 2^-149^~2^-126^之间；之后进入normalized，800 million个间隔置于2^-126^和2^-125^之间；exponent = 127时，800 million个间隔，step = 2^-23^置于1~2之间；exponent = 128时，800 million个间隔，step = 2^-22^置于2~4之间

exponent = 150时表示2^23^，此时step =1

![image-20231116015005787](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161622557.png)

之后是infinity和NaN

![image-20231116015343588](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161622805.png)

![image-20231116015357578](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161622062.png)

换算，exponent得到的十进制值，是sig二进制数小数点移动的位数

![image-20231116015854543](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311160158616.png)

![image-20231116020322423](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311160203473.png)

加法不associative，1.5*10^38^是很大的数，step很大，存不下1

![image-20231116101606028](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161016329.png)

两个术词

![image-20231116101905231](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161019266.png)

![image-20231116102144522](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161021556.png)

如图的例子是十进制，实则是二进制；0101(000~111，中间值是100)，向上入到0110

![image-20231116102539176](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161025211.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161035412.png" alt="image-20231116103520379" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161036798.png" alt="image-20231116103612765" style="zoom:15%;" />

其他表示方法 double quad

64位机上的，分两个数在32位机上发送

![image-20231116104028284](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161040315.png)

bfloat 16，16bit，exponent位数和fp32一样

![image-20231116104108596](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161041627.png)

![image-20231116104324410](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161043441.png)

![image-20231116104422799](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161044828.png)

给exponent和sig可变的长度，添加一个u-bit指示是否精确

![image-20231116104713395](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161047429.png)

![image-20231116105122423](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311161051453.png)



# LEC7 RISC-V

![image-20231211132157002](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111321139.png)

![image-20231211132435021](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111324076.png)

Reduced Instruction Set Computing

x86 complex 但是生态完整，RISC 易于理解 但是old and no software，RISC-V 开源，易于理解，近年在工业/科研发展迅速，2010起源于61C教学 

本课程使用RB32

assembly的operand不是变量，而是registers，没有type

32 registers in RISC-V, each register is 32/64 bit wide(32/64 bit is called a word); 8 registers in X86

![image-20231211134812493](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111348586.png)



![image-20231211135604603](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111356700.png)

![image-20231211135715621](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111357665.png)

![image-20231211135836267](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111358331.png)

![image-20231211135950934](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111359974.png)

immediate 立即数 addi，没有subi

![image-20231211142451259](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111424347.png)

x0的值永远是0，不可改变

![image-20231211142734339](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111427387.png)

# LEC8

![image-20231212201033203](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312122010244.png)

来源：little endian指从小头剥鸡蛋，big endian反之；RISC-V及90%的处理器是little endian，表示word中byte的排布顺序，最低位在最低地址；不论big/little endian，byte中bit的排布顺序始终是最低位在最低地址

![image-20231213122703829](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312131227949.png)

1025 = 1024 + 1

![image-20231213123431002](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312131234040.png)

DRAM，DDM345，HBM123

![image-20231213130020604](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312131300636.png)

![image-20231213130030159](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312131300202.png)

![image-20231213130058338](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312131300365.png)

![image-20231213130217506](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312131302535.png)

offset = 12，3个integer每个4Byte，must be multiple of 4

![image-20231213201432496](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312132014606.png)

sw的数据流向右

![image-20231213201808576](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312132018623.png)

integer是32bit，和register的大小匹配；char和color channel是8bit，使用 lb 和 sb，offset(Byte)不用是4的倍数

Sign-extension，当操作有符号数时，把符号位x写满其他位置，为了preserve the sign

如果确定是unsigned number，如char/color intensity，用lbu，不做Sign-extension，无sbu

![image-20231213202517533](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312132025571.png)

offset 1，单位Byte

![image-20231213204257455](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312132042509.png)

addi可以被两条语句代替，但是去memory拿数据太慢了，因此保留addi

![image-20231213204509613](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312132045662.png)

branch；otherwise，执行下一行

![image-20231213205327543](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312132053571.png)

每条命令都是32bit，因此 j 相比 beq，可以去到更远的地址（不用存两个reg）

![image-20231213205516643](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312132055674.png)

汇编语言里用 bne 表示 if，更方便

![image-20231213210041920](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312132100958.png)

![image-20231213210203143](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312132102180.png)

记忆：有BLT(bacon, lettuce, tomato) sandwich，没有BGT sandwich

![image-20231213210824789](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312132108826.png)

Loop；复制x8到x9，好习惯，避免污染原数据；x13初始化，因为不能和立即数比较；loop的第一句写退出条件；下一个integer是+4

![image-20231213210849494](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312132108524.png)

# LEC9

![image-20231214004217961](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140042041.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140044208.png)

![image-20231214004516538](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140045573.png)

实现*12，<<2 + <<3 

![image-20231214005103199](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140051239.png)

空白处补入最高位（符号位）；-25 / 16 = -1.5625，得到-2；但是C语言一般朝0近似，因此得-1

![image-20231214005114128](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140051155.png)

.s 汇编程序，.o 机器码，lob.o 包含常用库函数；a.out live in memory，是RISCV instructions，每句32bit 

![image-20231214005856808](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140058845.png)

![image-20231214010047635](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140100693.png)

PC，program couter

![image-20231214010331039](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140103087.png)

32个register中，有一些有特殊用途，因此起了symbolic name

Pseudo-instructions 伪指令，常见汇编语句的简写语法；li load immediate；nop用于等待

![image-20231214010839813](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140108865.png)

 function

![image-20231214011912116](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140119175.png)

sum函数

![image-20231214012125414](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140121444.png)

![image-20231214012357793](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140123827.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140124068.png)

![image-20231214012601153](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140126192.png)

 使用 jal 和 ret

![image-20231214012756915](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140127947.png)

j 本质上是 jal，jr 本质上是 jalr，他们是伪指令

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312140132519.png)

# LEC10

![image-20231214155811645](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141558737.png)

RISCV 没有 pop 和 push

x0 是0， x1 是return address， x2 是stack pointer

![image-20231214160221280](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141602315.png)

入栈，sp--

![image-20231214161018070](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141610111.png)

leaf，called by others but not call others 

![image-20231214160151064](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141601104.png)

sw，数据流向右，store from register to memory；前三句话称为prologue，序幕；后四句称为epilogue；最后一句即ret

![image-20231214161649670](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141616711.png)

nested/recursive function call，一个函数调用另一个，return address x1和argument register会被覆盖，需要go to stack，stack(memory)效率低

register分为saved/temporary or volatile 

![image-20231216105257505](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161052620.png)

stack pointer, global/thread/frame pointer与compiler有关

![image-20231216105934573](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161059661.png)

注意saver，即出现nested call时，caller还是callee负责保存该变量（存到s/入栈）

![image-20231216110616848](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161106895.png)

memory allocation

local var 用stack存，因此stack包含了return address，一些要保存的register， local var

![image-20231216194843669](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161948728.png)

![image-20231216194911879](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161949912.png)

![image-20231216195119065](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161951121.png)

初始x -> a0, y -> a1, mult函数的两个参数分别在a0和a1，因此调用mult之后要重新lw y的值；mv所做的是复制而不是移动；黄色是push，红色是pop

![image-20231216195135411](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161951443.png)

![image-20231216200942648](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312162009706.png)

![image-20231216200958542](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312162009574.png)

之后将会按照format重新分组

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170119867.png" alt="image-20231217011914776" style="zoom:67%;" />



# LEC11

dive down another level of abstraction

![image-20231217160909207](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171609286.png)

ENIAC upeen 1946, solve triangle, women programmer

Von neumann's report was leaked

EDSAC Cambridge 1949, general computer, 35bit, two's complement

![image-20231217162959757](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171629806.png)

ISA, Instruction Set Architecture 指令集架构，x86(PC)，ARM(phone)，RISCV；向后兼容，8088指令仍能运行在今天的电脑上

![image-20231217163413443](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171634487.png)

为了兼容性，RV32, RV64, RV128也是32bit instructions

![image-20231217170409956](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171704028.png)

2^32很多，因此进行分区，每个区存不同类型的instructions；32个寄存器占用5bit；load和immediate有相同的格式；U是很长的立即数

![image-20231217170640599](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171706645.png)

R-format layout, instruction  destination register(rd), first source register(rs1), second source register(rs2)；register都是5bit

![image-20231217214035266](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312172140336.png)

R-format 的7位opcode都一样；funct3和funct7为了make processor work easier

![image-20231217214114460](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312172141503.png)

![image-20231217214732390](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312172147437.png)

slt 新指令， 如果rs1 < rs2，set rd = 1；sltu，unsigned；add和sub用同样的hardware，但sub需要sign extension，funct7的第二位表示是否需要sign extension

![image-20231217215009188](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312172150245.png)

I-format，有9个指令

12bit，-2048 ～ 2047

![image-20231218104542891](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181045979.png)

![image-20231218104619991](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181046064.png)

slt，如果rs1 < imm，set rd = 1；shift amount limit to 5 bit 32种

![image-20231218105230075](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181052127.png)

![image-20231218105930410](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181059494.png)

![image-20231218110453454](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181104508.png)

lh和lb做sign extension，lhu和lbu不做；byte 和 halfword -> place in the least significant position of register最低位；funct3中第一位用于标记sign/unsign

![image-20231218110620629](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181106684.png)

S-format

12bit imm；rs2中的值是要存到memory的值，rs1是memory基址，imm是offset 

![image-20231218113741401](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181137463.png)

![image-20231218113755375](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181137424.png)

![image-20231218114039680](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181140739.png)

# LEC12

B-format

![image-20231218144836701](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181448814.png)

因此使用PC- relative addressing，从当前程序位置 + offset(imm) 来寻址；单位是4Byte，即指令长度(实则是2Byte)

![image-20231218150058379](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181500443.png)

![image-20231218150630693](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181506755.png)

![image-20231218150649350](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181506406.png)

但是PC = PC + imm * 4 并不是RISCV的实施方法，对于便宜的小设备，为了让code memory 尽量小，使用16bit的compressed instructions set，RISCV为了支持它，因此使用PC = PC + imm * 2；小心出错

![image-20231218151449180](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181514245.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181517047.png" alt="image-20231218151720986" style="zoom:25%;" />

![image-20231218152126132](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181521195.png)

branch不要写成a0 

![image-20231218152352249](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181523313.png)

![image-20231218153152809](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181531875.png)

4种type，三种都涉及立即数，立即数放进32bit里必然要sign extension；I-type的20-31为立即数，进入32bit的低12位，最高位进行sign extension；S-type的7-11和25-30为立即数；easy for register to find location

![image-20231218153206547](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181532600.png)

offset = 16，是10000，去掉最低一位0

![image-20231219113908825](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312191139964.png)

最低四位一直是0011

![image-20231219114512728](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312191145847.png)



Long imm 20bit, U-format

PC relative 优点是 relative, position-independen，寻址范围 13bit，+-1024条instructions (符号位 32bit 最低位默认0)

![image-20231221172834297](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211728331.png)

far超出+-1024条instructions范围，使用j，jump有更大的跳转范围

![image-20231221172846681](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211728719.png)

![image-20231221172818140](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211728183.png)

第一步LUI赋值高5位，第二步addi赋值低三位

![image-20231221173051215](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211730257.png)

会出问题，addi会 sign extend 最高位

![image-20231221173259592](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211732655.png)

RISCV中没有addi unsigned，因此只能人为调整，DEADC；compiler使用伪指令解决这一问题，li自动包含上述步骤，使用li 

![image-20231221173529430](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211735476.png)

![  ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211739998.png)



J- format，和U- format同样是20bit imm，排布类似B-format

![image-20231221182704906](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211827003.png)

同理最低位默认0，一共21bit，可以寻址+-2^18条指令

![image-20231221183755502](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211837574.png)

j是jal x0的伪指令

![image-20231221184010793](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211840856.png)

jalr是I-format，imm被sign extend to 32bit 后加到rs中；没有默认的最低位

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211842093.png)

![image-20231221184607506](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211846552.png)



summary

![image-20231221185211909](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211852968.png)

![image-20231221185332051](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312211853118.png)





# LEC13 CALL: Compilation, Assembly, Linking, Loading

Interpret/translate

![image-20231222082402645](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220824737.png)

![image-20231222082433183](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220824227.png)

![image-20231222082515127](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220825177.png)

simulate, apple (change ISA) Back-compatible

![image-20231222083325875](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220833926.png)

Write interpreter (61A) is much easier than write compiler (164); interpreter兼容各种hardware, compiler translate 到特定hardware 

![image-20231222083423468](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220834541.png)

从执行文件executable反编译很难

![image-20231222083456522](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220834572.png)

CALL

compiler, .c -> .s

gcc 直接出.out，gcc -S拆解，loader is OS； compiler/hand translate .c to .s 

![image-20231222084208458](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220842513.png)

mv is copy 

![image-20231222084640952](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220846033.png)



Assembler .s -> .o

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220856386.png)

![image-20231222085930658](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220859711.png)

li: load imm; xor -1, -1是32bit全1, not相当于reverse all 32bit of t1然后存到t0; la有两种，static/PC- relative addressing

![image-20231222090642017](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220906078.png)

![image-20231222091202737](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220912796.png)

第一步：替换伪指令；第二步：记录label位置，以便后续跳转；第三步：遇到跳转时会知道跳转多少half-word，用这个值替换label

![image-20231222092159305](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220921363.png)

PIC相对寻址，数值到最后的.out文件中仍成立；对于la等要求静态（非PIC）32bit地址的，不知道到.out文件会变成多少

![image-20231222093029442](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220930502.png)

sin lives in math lib

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220935274.png)

有些静态（非PIC）32bit地址要有了.out才知道，fill in later (to do)-> relocation table 

![image-20231222093614540](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312220936594.png)

![image-20231222163147301](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221631411.png)



Linker .o->.out，compile和assembler可以仅作用于有改动的文件，然后再link所有.o文件

![image-20231222163530516](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221635579.png)

text是代码，info是两个table和debugging info，library text放在text最后

![image-20231222164123542](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221641604.png)

![image-20231222164218831](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221642892.png)

![image-20231222164303107](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221643165.png)

gp, global pointer point to the beginning of static area; conditional branch 是PC relative寻址，不用relocate 

![image-20231222164550493](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221645554.png)

address start at 0x10000 = 64k, 往上是text1 2 3, data1 2 3, 之后是heap; stack从顶向下；link knows 来自table 

![image-20231222164946983](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221649105.png)

symbol not found (找不到symbol), symbol unknown (找不到function), conflict symbol type (有两个同名函数)

![image-20231222165816666](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221658765.png)

Dynamically Linked Libraries是一种本地的二进制代码库，在程序运行时加载到内存中，在运行时做link，而不是在编译时静态链接到程序中

![image-20231222171840039](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221718150.png)

runtime 做  link有时间开销，会runtime error

![image-20231222172105201](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221721260.png)

link at machine code level

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221728393.png)



loader disk -> memory

![image-20231222174439539](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221744613.png)

CS162; argc和argv to stack

![image-20231222174541440](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221745503.png)



example helloworld

![image-20231222175546721](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221755786.png)

2 ^ 2 = 4Byte, word-aligned

![image-20231222175556835](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221755893.png)

.o中只有数字部分，其他都是ppt注释；红色数字是fill in later，之后linker做的事

![image-20231222180530885](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221805960.png)

![image-20231222180937135](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221809247.png)

sign extend问题 ，要lui 21而不是20

![image-20231222181131724](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221811840.png)



![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312221821740.png)



# LEC14 Synchronous Digital Systems (SDS)

clock: GHz

![image-20240622170213313](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221702485.png)

switch assert/close

![image-20240622173946329](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221739483.png)

![image-20240622174118957](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221741016.png)

Transistor (amplifier/switch, used as switch in info system)

![image-20240622174546913](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221745075.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221754601.png" alt="image-20240622175424449" style="zoom:25%;" />

![image-20240622192515795](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406221925983.png)

![image-20240623015930388](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406230159616.png)

NOT

![image-20240623020525447](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406230205562.png)

NAND

![image-20240623020804247](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406230208364.png)



Signals and Waveforms

clock

![image-20241020212443004](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202124027.png)

X0: lab (least significant bit), x3: msg (most significant bit), 头歪向左看

![image-20241020212521037](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202125144.png)

Circuit Delay

![image-20241020212639150](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202126211.png)

Combinational Logic/State Elements

![image-20241020212704049](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202127142.png)

when load goes high, register update the memorized value

![image-20241020213016419](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202130479.png)

![image-20241020213258054](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202132117.png)



# LEC15

![image-20241020213812341](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202138459.png)

suppose we do not have state elements, we want to build an accumulator

![image-20241020215614570](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202156698.png)

![image-20241020215645759](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202156844.png)

Reset = 1 -> register value = 0

![image-20241020215935606](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202159728.png)



Register details inside: Flip-flop

A n-bit register is n 1-bit flip-flop (parallel)

![image-20241020220433719](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202204847.png)

Trigger -> load

![image-20241020220415786](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202204969.png)

![image-20241020220757185](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202207283.png)



add delay & clock-to-q delay; reset has higher priority than load

![image-20241020221916388](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202219480.png)

first clock reset

Si那一行，X0 和 X0+X1之间的阴影部分是错误的add result, add 是CL, 即时的

between setup time and hold time, value should be right

![image-20241020222117795](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202221901.png)

setup time = stable time before next rising edge

![image-20241020222746288](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202227539.png)

![image-20241020224501141](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202245293.png)

![image-20241020224904204](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202249304.png)

![image-20241020225115565](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410202251680.png)



Finite State Machines (FSM)

![image-20241025234148241](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410252341561.png)

input/output; double circle means initial state  

![image-20241025234520558](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410252345611.png)

![image-20241025235515462](../Library/Application Support/typora-user-images/image-20241025235515462.png)

![image-20241025235545502](../Library/Application Support/typora-user-images/image-20241025235545502.png)

左半部分（CL+reg）就是已经讲的那一块，可以重复很多次，块之内/不同块之间可以有feedback

![image-20241026000416142](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410260004253.png)

![image-20241026000556811](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410260005867.png)

![image-20241026001100659](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410260011708.png)



# LEC16

一共2^16^种

![image-20241026001607348](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410260016433.png)

![image-20241026002323916](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410260023964.png)

2bit adder easy to build, how about 32bit? cannot use truth table

![image-20241026002444166](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410260024225.png)



Logic Gates

![image-20241026002715358](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410260027423.png)

treat xor treat

![image-20241026003538881](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410260035974.png)

![image-20241026143315313](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261433528.png)

![image-20241026145000096](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261450152.png)

PS = PS1 PS0, present state, NS next strate, FSM finite state machine

![image-20241026144944986](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261449149.png)

Boolean Algebra

![image-20241026145650439](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261456504.png)

![image-20241026150352042](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261503173.png)

Laws of Boolean Algebra

![image-20241026161246387](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261612915.png)

![image-20241026161306168](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261613227.png)

Canonical forms (Sum-of-products)

![image-20241026161804405](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261618479.png)

![image-20241027224849225](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410272248406.png)

we don't go from truth table to gate diagram

![image-20241026162805340](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261628404.png)



# LEC17

Data Multiplexor, S=0则选A, S=1则选B

![image-20241026163520568](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261635671.png)

![image-20241026163710431](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261637498.png)

![image-20241026164302401](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261643498.png)

![image-20241026164728332](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261647397.png)

ALU

![image-20241026170926045](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261709150.png)

![image-20241026171043816](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261710906.png)

Adder / Subtracter Design

![image-20241026171644664](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261716753.png)

LSB 无进位c

![image-20241026171935203](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261719286.png)

![image-20241026171945205](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261719257.png)

![image-20241026172411192](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261724286.png)

unsigned computation overflow: cn = 1

![image-20241026173602657](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261736759.png)

signed computation overflow

The reason why we loved 2's complement number is they are the same as unsignec except the overflow trigger

if unsigned, blue boxes overflow; if signed, red boxes overflow; c2 XOR c1 trigger the overflow. (For n bit, also just need to look at highest bit or sign bit)

![image-20241026175013806](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261750938.png)

Subtractor, SUB = 1 -> b取反加一，加一在c0

![image-20241026180231283](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261802359.png)

![image-20241026180714076](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202410261807197.png)



# LEC18 build a RISCV processor/CPU

![image-20231224015444940](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312240154076.png)

![image-20231224015741406](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312240157529.png)

processor 包含 control & datapath 

![image-20231224020139596](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312240201665.png)

datapath是肌肉，执行所有ISA中有的指令，可以为RISCV中每一条指令建立单独的datapath

![image-20231224020424893](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312240204973.png)

store state/data in CPU: PC, register, memory; IMEM instruction memory; DMEM data memory

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241906487.png)

monolithic 整体的; bulky 笨重的

![image-20231224191547359](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241915442.png)

Stage4 load/store；5步暂时先看做在同一个时钟周期

![image-20231224191615443](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241916512.png)

MUX 多路复用器（Multiplexer），PC + 4，或者由ALU计算出跳转数进行跳转

![image-20231224192429391](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241924473.png)

![image-20231224193342929](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241933007.png)

N/ N = 32，表示32 wires

![image-20231224193713485](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241937558.png)

Register file包含32个register，two-read single-write register file; output/read delay -> access time, write 也有 access time

![image-20231224193643589](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241936668.png)

![image-20231224193726423](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241937491.png)

32个register，每个32 bit，一共1024flip-flop 触发器；两个read rs1 rs2；一个write rd

![image-20231224195414479](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241954569.png)

![image-20231224195422669](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241954783.png)



datapath for R-type

![image-20231224201712809](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312242017032.png)

 use add as example

![image-20231224201833728](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312242018815.png)

Inst -> instruction; next clock tick 更新 PC和Reg[rd]; control logic enable write

![image-20231224202800444](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312242028532.png)

红框是delay，其他类似的小延迟也是delay；alu的结果到下一个时钟才被写入reg1

![image-20231224203523120](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312242035206.png)



sub 就是多一个control bit，ALUSel

![image-20231224204516873](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312242045960.png)

![image-20231224204538092](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312242045173.png)

![image-20231224204636060](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312242046154.png)



I-Format, datapath with imm

![image-20231224205013585](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312242050683.png)

加入一个MUX和一个control bit来区分R-type还是I-type

![image-20231225010241474](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312250102611.png)

Works for all other I-format arithmetic instructions (slti,sltiu,andi, ori,xori,slli,srli, srai) just by changing ALUSel

![image-20231225010429594](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312250104689.png)

![image-20231225013723419](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312250137514.png)



# LEC 19

load

R/I type只有四个phase，load有五个，多了data memory

![image-20231225205812172](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252058550.png)

WBsel，控制是R还是I type；MemRW = R

![image-20231225210148734](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252101777.png)

![image-20231225211144225](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252111274.png)



store

![image-20231225211254660](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252112699.png)

WBsel = 0/1都可以，之后会用来优化减少gate数目

![image-20231225211759722](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252117771.png)

决定是I还是S type只需要5 bit mux，请哦他都一样

![image-20231225212414327](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252124374.png)

![image-20231225213038982](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252130027.png)



B-Format for Branches；12-bit immediate imm[12:1]，最低位默认0

![image-20231225213625147](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252136197.png)

![image-20231225214156108](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252141153.png)

BrUn 表示signed/unsigned，BrEq 是否equal，BrLT 是否小于；PC+imm

![image-20231225214505558](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252145601.png)

![image-20231225215307078](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252153127.png)

其他ISA need a wide mux

![image-20231225215451319](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252154360.png)

RISCV编码中把imm[11:1]放在相同位置，因此只需要1bit mux来区分S/B type

![image-20231225220443945](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252204997.png)

![image-20231225220618974](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252206017.png)



JALR: jump and link register, 复用了I format, 没有默认最低位为0, 损失了寻址范围, 减少了指令

![image-20231226024036825](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260240876.png)

![image-20231226101916016](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261019122.png)



JAL, J format

![image-20231226102324139](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261023190.png)

只需要一个 J type imm

![image-20231226102707408](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261027462.png)



U type

![image-20231226102952833](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261029927.png)

![image-20231226103152096](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261031196.png)



# LEC20

component

CSRs 不在processor旁边，在32个register (general purpose integer register)之外；曾经是ISA的一部分，为了模块化被踢出；功能：count 已经执行的 # of cycles, or # of retired instruction, communication with core processors/peripheral, 打印机status: ready/waiting/done -> flag

![image-20231226160215597](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261602685.png)

类似I-type，20-31用来寻址csr，15-19是source reg/imm，12-14决定哪条指令；csrrw (read write): copy what is in csr to rd, 同时copy what is in rs1 to csr, 如果rd是x0则不执行第一步；csrrs/c (read set/clear flag): rs1是uimm

当15-19是source reg

![image-20231226162352727](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261623780.png)

当15-19是uimm，sign extend to 32bit 写入csr，高位始终补0

![image-20231226163514435](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261635514.png)

![image-20231226163908044](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261639099.png)

![image-20231226164058468](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261640540.png)



control

inst[31:0]那一行是IF；control bit 设置 及 reg 两行是decode；ALU和WB是EX；下一个时钟上升沿是WB，包括更新reg内容和下一个PC

![image-20231226171635384](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261716439.png)

4 phase

![image-20231226172734133](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261727193.png)

5 phase

![image-20231226172813129](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261728183.png)

![image-20231226185722640](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261857774.png)

![image-20231226185804728](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312261858781.png)

control logic

![image-20231230022855636](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312300229379.png)

有两种选择；前者是只能读的memory，仅在设置时可以写

![image-20231230023245345](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312300232394.png)

9 bit用来区别control logic

![image-20231230023515890](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312300235943.png)

第一种方法

![image-20231230023812153](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312300238197.png)

address decoder 出 one-hot

![image-20231230023825771](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312300238810.png)

第二种方法

![image-20231230024043245](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312300240289.png)

![image-20231230024058516](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312300240557.png)



# LEC21 pipelining

how to measure performance

![image-20231231013959649](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312310139713.png)

![image-20231231131739362](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311317449.png)

Latency：完成一个任务要多少时间

Throughput：一定时间内能完成多少任务

redundancy：冗余计算，计算多个结果后vote

time to execute a program such as picture display; datacenter的energy cost一年后就超过了hardware cost; power (功率)不是好的衡量标准，energy (能量)才是

![image-20231231131753111](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311317157.png)

“Iron Law” of Processor Performance

![image-20231231134232758](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311342833.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311346711.png" alt="image-20231231134602664" style="zoom:25%;" />

有些ISA中不是每条指令都耗时1cycle；Superscalar processors: execute >1 instructions per cycle

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311346972.png" alt="image-20231231134614899" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311352351.png" alt="image-20231231135240297" style="zoom:25%;" />

![image-20231231135724493](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311357547.png)



energy efficiency

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311403889.png" alt="image-20231231140350837" style="zoom:25%;" />

transistor is slower at lower voltage

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311409889.png" alt="image-20231231140925833" style="zoom:25%;" />

C is capacitance

![image-20231231141620893](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311416950.png)

![image-20231231142059896](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311420951.png)

![image-20231231142139977](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311421025.png)

![image-20231231142343772](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312311423822.png)



Pipelining -> increase compute throughput

理论上speed * 4，实际上speed * 2.3，因为有初始和最后的入和出；如果有1000个人，把两个步骤变成20min，总时间不会变化（仅初始和最后的入和出有优化）

![image-20231231235221320](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312312352421.png)![image-20231231235350229](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312312353295.png)



# LEC22

RR和RW时间较短；read 亮左半部分，write 亮右半部分

single-cycle CPU

![image-20240101165910320](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401011659366.png)

时间统一200ps + pipeline

![image-20240101165931255](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401011659319.png)

二者对比

![image-20240101170334530](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401011703573.png)

Datapath

![image-20240101204534136](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401012045166.png)

insert registers, Pipeline registers separate stages, hold data for each instruction in flight；前三个stage需要PC，第四个MA stage需要PC + 4，

![image-20240101204555175](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401012045209.png)

![image-20240102150322728](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021503821.png)



三种pipeline hazard 危险

![image-20240101171631458](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401011716501.png)

如ALU计算和PC+4同时使用ALU

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401011728869.png)

![image-20240101202720482](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401012027514.png)

![image-20240101203226505](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401012032534.png)

![image-20240101203437833](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401012034865.png)

![image-20240101203853953](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401012038992.png)



Data Hazard

read和write同时发生

![image-20240102152241090](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021522185.png)

reg R/W只要100ps，是一个cycle时间的一半，两者各占一半时间；这在复杂CPU中不一定成立，不做深入

![image-20240102152521168](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021525221.png)

另一个例子

![image-20240102152744071](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021527133.png)

delay 2 cycles

![image-20240102153045396](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021530451.png)

![image-20240102153332837](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021533923.png)

以直接传值（another mux + control bit），代替等待2个stage写入reg

![image-20240102153415942](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021534986.png)

![image-20240102153637491](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021536545.png)

control mux: if source reg == des reg != x0, then forward

![image-20240102153949605](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021539662.png)

![image-20240102155606962](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021556047.png)



# LEC23

Forwarding 可以解决data hazard，但是解决不了load data hazard

无法forward到一个周期前的地方（黄线），只能stall；但是蓝线和绿线可以通过forward解决 

![image-20240107151617995](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071516105.png)

![image-20240107152135750](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071521816.png)

Put 与load的内容无关的 instruction into load delay slot，编译器的新任务

![image-20240107152642024](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071526093.png)

compiler调整指令顺序，9cycle -> 7cycle

![image-20240107153000373](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071530429.png)



Control hazard: branch & jump

要ALU比较t0,t1是否相等之后才知道会不会跳转，即前两个周期不知道要不要继续执行指令，应该 stall 2 cycle

![image-20240107154418776](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071544867.png)

避免stall，在branch结果出来之后控制control bit，如果预测错误则取消执行一半的两条指令（刷新pipeline）

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071548786.png" alt="image-20240107154822685" style="zoom:25%;" />

![image-20240107154844257](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071548298.png)

用1bit记录上次跳转了没有，以此预测下一次是否会跳转；更复杂的predictor可以达到90%准确率，见CS152

![image-20240107155536027](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071555125.png)



superscalar processor

Clock rate被晶体管速度和发热限制；提升performance可以进一步细分stage数量/deep pipeline（虽然90%的processor都是5-stage）

![image-20240107160840600](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071608708.png)

CPI 每指令周期（Cycle Per Instruction），superscalar CPI<1了；issue 分配；多个execution unit pipeline，每个专门用于执行某一类型的指令；every cycle fetch&decode multiple instructions, issue to different execution unit pipeline；16 BIPS billion instructions per second， 

![image-20240107161519634](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071615739.png)

![image-20240107162359810](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071623853.png)

![image-20240107162941746](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071629791.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071630678.png" alt="image-20240107163058635" style="zoom:25%;" />

x86的指令长度不等 

![image-20240107163205402](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071632446.png)

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401071635980.png)



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

# 每次在服务区上先声明git name和email

# If nothing happens for a long time: check your internet connection. Some network firewalls, including CalVisitor on campus, block SSH. Try another network (AirBears2 or eduroam if you’re on campus).
# Connection refused or other weird errors: the Hive machine you picked might be down. Try another one
# Reserved for cs61c staff: try another Hive machine :)
```

- `<ctrl> + a` will move the cursor to the beginning of the current line (helpful for fixing mistakes)
- `<ctrl> + e` will move the cursor to the end of the current line (also helpful for fixing mistakes)
- `<ctrl> + r` will let you search through your recently used commands
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

yy复制一整行，vim复制全部 ggVGy

Which command walks a file hierarchy in search of a keyword?

grep -r "关键字" /路径；搜索指定路径下文件内容包含关键字的

Which command can help you find the difference between two files?

diff 文件1 文件2

vimdiff 文件1 文件2    可视化

diff 文件1 文件2 | less -R





# LAB1

```
$ gcc program.c
# compiles program.c into an executable file named a.out
$ ./a.out

$ gcc -o program program.c
$ ./program
# -o specify the name of the executable file that gcc creates.
```

install gcc 见lab网页

宏定义的工作方式是，在编译 C 文件之前，所有定义宏常量的名称都会被替换为它们所指的值。

GDB stands for “GNU De-Bugger. [GDB reference card](https://inst.eecs.berkeley.edu/~cs61c/sp21/resources-pdfs/gdb5-refcard.pdf)

```
$ gcc -g -o hello hello.c
# This causes gcc to store information in the executable program for gdb to make sense of it. Now start our debugger, (c)gdb:
$ cgdb hello
$ gdb hello

# (c)gdb may not install on (updated) macOS machines. use lldb
gcc -g -o hello hello.c
lldb hello
```

[lldb guide](https://lldb.llvm.org/lldb-gdb.html)

1. setting a breakpoint at main: (lldb) breakpoint set --name main
2. using gdb’s run command: (lldb) run; (lldb) continue 直到下一个断点
3. using gdb’s single-step command: (lldb) next

(lldb) quit

**printf.c: No such file or directory.** You probably stepped into a printf function! If you keep stepping, you’ll feel like you’re going nowhere! CGDB is complaining because you don’t have the actual file where printf is defined. This is pretty annoying. To free yourself from this black hole, use the command finish to run the program until the current frame returns (in this case, until printf is finished). And **NEXT** time, use next to skip over the line which used printf.

In cgdb, you can press `ESC` to go to the code window (top) and `i` to return to the command window (bottom) — similar to vim. The bottom command window is where you’ll enter your gdb commands.

1. While you’re in a gdb session, how do you **set the arguments** that will be passed to the program when it’s run?

   (gdb/lldb) run arg1 arg2

2. How do you **create a breakpoint**?

   (gdb) break main; (gdb) break file.c:15; (gdb) break *0x0804839a

   (lldb) breakpoint set --name main; (lldb) breakpoint set --file file.c --line 15; (lldb) breakpoint set --address 0x0804839a

3. How do you **execute the next line of C code** in the program after stopping at a breakpoint?

   (gdb) next  将执行当前源代码行上的下一条语句，而不会进入函数调用

   (gdb) step  会进入函数调用

   (lldb) next

   (lldb) thread step-over

4. If the next line of code is a function call, you’ll execute the whole function call at once if you use your answer to #3. (If not, consider a different command for #3!) How do you tell GDB that you **want to debug the code inside the function** (i.e. step into the function) instead? (If you changed your answer to #3, then that answer is most likely now applicable here.)

   (gdb) step  会进入函数调用

   (lldb) thread step-over

5. How do you **continue the program after stopping** at a breakpoint?

   (gdb) info breakpoints

   (gdb) continue

   (lldb) breakpoint list

   (lldb) continue

6. How can you **print the value of a variable** (or even an expression like 1+2) in gdb?

   (gdb/lldb) print variable_name

   (gdb/lldb) print 1+2

7. How do you configure gdb so it **displays the value of a variable after every step**?

   (gdb) watch variable_name

   (gdb) display

   (lldb) watch set variable variable_name

   (lldb) watch enable

8. How do you **show a list of all variables and their values** in the current function?

   (gdb) info locals

   (lldb) frame variable

   (lldb) fr v

9. How do you **quit** out of gdb?

   quit

```
# cgdb调试有输入的程序，使用：
(cgdb) run < input.txt
  
(lldb) settings set target.input-path ?.txt
(lldb) process launch
  
(lldb) process launch -i ?.txt
```

调试器可以捕捉"bohrbugs"。相反， "heisenbugs"（至少在 C 语言中）通常是由于内存管理不当造成的，use Valgrind模拟 CPU 并跟踪内存访问，会很慢但可以发现heisenbugs

```
valgrind ./segfault_ex
# 使用WSL，macos不兼容
```

Why didn’t the `no_segfault_ex` program segfault?

循环会从 `j = 0` 到 `j = 19` ，超出了数组 `a` 的界限，没有导致段错误的原因是访问的内存区域仍在程序的地址空间中

Why does the `no_segfault_ex` produce inconsistent outputs?

内存对齐问题。`total` 的类型是 `unsigned`，而 `printf` 语句中的格式说明符是 `%d`有符号整数。导致 `printf` 语句输出不一致的结果，因为它试图解释 `total` 的二进制表示。

用于检测单链表是否有环的算法，[Floyd's Cycle Detection Algorithm](https://en.wikipedia.org/wiki/Cycle_detection#Floyd.27s_Tortoise_and_Hare)，龟兔赛跑算法；两只乌龟可以找到循环起点

检查指针 `h` 是否为NULL: if (h == NULL);if (!h)

用不到 ->value
