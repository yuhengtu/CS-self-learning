# INTRO

hw released on Tue and due the next week on Th 8:59 pm

Late policy:  lose 5% for every hour that they are late. drop 2

- Homeworks (every week) - 20%
- Midterm I - 25%
- Midterm II - 25%
- Final - 30%



Hey! I took 161 as an undergrad and I would say 171 is definitely a great followup to take. 161 covers a large survey of computer security topics and scratches the surface of cryptographic primitives such as MACs, ciphers, hashes, etc. Only a few weeks is spent on cryptography, though. 171 will cover these topics with more formalism and depth, including some fundamental mathematical and theoretical underpinnings. There are also several topics that 161 did not cover, like ZK, secret sharing, group theory basics. However, the main difference is the theoretical flavor and greater depth. If you enjoyed 161's cryptography segment, I would say this is a good class to take.



MA116 covers cryptography from a mathematical perspective while CS171 covers it from a complexity-theoretic perspective. Specifically, CS171 focuses on modern cryptography, which is distinguished from classical cryptography, by its emphasis on definitions, precise assumptions and rigorous proofs of security. In contrast, MA116 (based on what I have read about the course online) focuses on the use of number theory for cryptography; which CS171 does not delve much into. So, while there is overlap in terms of applications of interest, there is very little overlap in terms of the technical content.



In more detail, besides the special topics, proposed CS171’s core course content can be broken into three parts: (i) private-key encryption and message authentication codes: with a focus on definitions and provably secure constructions from simpler assumptions, (ii) practical and theoretical constructions of symmetric-key primitives, and (iii) Public-key encryption and digital signatures: with a focus on definitions and provably secure constructions from number-theoretic assumptions. The main overlap of CS171 with MA116 is in terms of the number-theory background covered in the third part of the course, which CS171 will covers very briefly (in one class). Of course, students interested in learning about the number-theoretic aspects will benefit from additionally taking MA116. Finally, there is also some overlap in the constructions covered in the context of public-key encryption and digital signatures (both of which will be covered in the proposed CS171 in three classes). Here again, unlike MA116, the treatment of these schemes in CS 171 is from a complexity perspective.



# LEC1 Private-key, shift/substitution/vigenère 

![image-20240121231159441](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401221511567.png)

![image-20240121231249759](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401221512799.png)

encryption tool: https://www.privacytools.io/secure-file-encryption https://cryptomator.org/open-source/

kerckhofff's principle: The cipher method is not secret, Only the key is kept secret

The shift cipher

![image-20240121232214242](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401221522355.png)

![image-20240121232230221](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401221522273.png)

Any secure encryption scheme must have a sufficiently large key space.

The mono-alphabetic substitution cipher

![image-20240121232818097](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401221528126.png)

![image-20240121233103253](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401221531348.png)

not secure because of frequency analysis attack

![image-20240121233340625](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401221533652.png)

Need to smooth out the frequency distribution of characters 

Vigenère Cipher (multiple shift cipher) 关键字被重复地与明文相结合，t为关键字长度

![image-20240121234248831](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401221542913.png)

c(j), c(j+t), c(j+2t) 都使用同样的位移, 随后可以进行 frequency analysis 

![image-20240121235643507](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401221556637.png)

![image-20240122000805498](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401221608575.png)

![image-20240122001147360](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401221611407.png)

Ad hoc fixes are likely to break 临时修补通常容易出问题



# LEC2 three perfectly secure definition

![image-20240129152150520](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300721647.png)

![image-20240129152228932](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300722953.png)

Shift cipher is insecure even for messages of length two

0.6 -> 1, 0.4 -> 0, not secure

![image-20240129153753039](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300737107.png)

secure definition

![image-20240129153500833](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300735903.png)

![image-20240129153913516](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300739553.png)

![image-20240129154853769](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300748806.png)

![image-20240129163427796](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300834887.png)

![image-20240129155532612](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300755679.png)

attacker A come to play game, 与challenger C交流, 最终输出0/1表示A失败/成功

PrivKeav (privacy against an eavesdropping attacker)

![image-20240129160526658](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300805720.png)



# LEC3 OTP, computational security, negligible

OTP: 一种 perfectly secure and simple algorithm

l是m的固定bit长度， 逐位相加不进位，加两次恢复原本m，k是任意的

![image-20240129162222567](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300822653.png)

太占内存；如果同一个k用在不同message就不再secure，因此cant reuse；

![image-20240129163836328](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300838359.png)

![image-20240129164157456](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300841544.png)

we cannot make |M|>|K|

M(c): for c, decrypt with all possible k to get a set of message; 可能有overlap，因此$|M(c)|\leq\mathcal{K}$，K < M

One-time pad K = M, optimal

![image-20240129164642632](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401300846682.png)

为了实用性 efficiency，relax some security

![image-20240207145923270](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071459367.png)

![image-20240207150349850](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071503958.png)

![image-20240207150759310](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071507359.png)

honest party run efficiently -> Enc Dec Gen run polynomial time in n (in this class, polynomial is efficient)

n衡量了honest party run time, attacker run time &  attacker succeed prob

n变大，security变大；n增加一点 (honest party的Enc Dec Gen算法慢一点)，t 增加很多，$\epsilon$ 减少很多

PPT -> Probabilistic **polynomial** time 能够在多项式时间内使用概率算法解决的问题集合

**restrict attack to be polynomial time** -> we win with neg prob

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071510504.png)

integer -> 闭区间[0,1]; negligible定义中的p(n)是polynomial function

$2^{-\alpha n}$ 拉低了poly(n)，成为negligible (n增加一点，honest party增加了poly(n)的复杂度，negligible func降低了exp(n)复杂度)

![image-20240207155739369](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071557443.png)

for large enough n -> $\color{red}{\exists N\in Z^+\forall n>N}$

![image-20240207180658110](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071806241.png)

![image-20240207182853056](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071828142.png)

前两个是，第三个不是

![image-20240207182915307](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071829349.png)

![image-20240207183323331](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071833375.png)

![image-20240207184331607](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071843699.png)

1^n^ 表示写n个1，并非次方

引入security parameter n，n决定了key的长度，可以encrypt任意多的message

![  ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071853321.png)

m0, m1是两个长度相同bit串；无法辨别两个相同长度的bit串，但是可以辨别几个不同长度的bit串

![image-20240207191029045](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071910140.png)

![image-20240207191430535](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402071914597.png)



# LEC4 PRG->computational security, PRG-OTP (fixed length; save bits), seed usage

encryption scheme that supports fixed length of message

Pseudorandom Generators / stream cipher

Randomness

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402072138848.png)

pseudorandomness: should be indistinguishable from uniform, is a property of a distribution

G is a deterministic function, l(n) 生成的bit串的长度; G生成的key space size <= 2^n^ (overlap -> <)，是2^l(n)^空间的子集

$x\leftarrow U_{\ell(n)}$ 表示从 l(n) 空间中任取；$s\leftarrow U_{n}$ 表示从 n 空间中任取，再expand到l(n)长度 (实则 key space size <= 2^n^)；Attacker cant distinguish

![image-20240207230051381](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402072300447.png)

b=0 选pseudorandom; b=1 选random

![image-20240210161843859](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402101618005.png)

G和n (fixed length) 都是公开的，只有k是保密的

![image-20240210163149263](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402101631345.png)

通常assume G is secure，然后证明在G secure的基础上encryption scheme secure

eavesdropper 窃听者

![image-20240210164850906](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402101648988.png)

X is assumption, $\pi$ is encryption scheme based on X

proof by reduction: 证非B→非A

![image-20240210170740570](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402101707677.png)

Assume there exists a PPT A that breaks $\pi$, then we construct PPT B that breaks X

3: if A succeed with non-negligible probability, then B continue to succeed with non-negligible probability 

![image-20240211040910423](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402110409515.png)

if attacker break the encryption scheme, we will construct attacker B who break PRG/G

![image-20240211041523158](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402110415245.png)

$\epsilon(n)$ is non-negligible function, 注意大于等于号 表示break

![image-20240211152253049](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402111522124.png)

![image-20240211154703864](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402111547929.png)

B guess correct -> pseudorandom or random

One-time pad is perfect secure

![image-20240211155617653](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402111556753.png)

OTP: one time pad; key节省空间

![image-20240211170612395](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402111706472.png)

||表示concatenation 级联，s is n bit；G已经secure，H进行进一步处理后是否还secure? No

seed只能用在G中，不能用在其他地方

H对于不同的G方案不一定secure，构造一个使得H insecure的G特例

证明：如果用把![image-20240211171909261](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402111719286.png)(PRG)用在H中，H就不再secure

![image-20240211171210969](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402111712033.png)

![image-20240211172200264](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402111722307.png)

first n/2 bit the same, so they cancel out; 前n/2位永远是0 -> 判断是pseudorandom

![image-20240211180816252](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402111808343.png)



# LEC5 mult msg security, CPA, PRF->CPA-Security, PRF-OTP

Security for multiple messages

i个message，每个抽等长度的m0, m1

m_b,i 作为一整个list用来加密

![image-20240211184139620](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402111841708.png)

chosen plaintext attack;  CPA是一个stronger definition (minimum security)，意味着满足CPA-security的也满足mult security

(·)表示任何信息; attacker 可以事先任意选择query询问(如直接问m0的c是什么)，最后再尝试m0 m1游戏模式看可否成功

attacker仍旧是PPT (polynomial)

Keep encrypting one message multiple times and still get different c, deterministic 就是不行的 

![image-20240211184211749](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402111842786.png)

绿色箭头表示query可以重复进行

![image-20240321195742720](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211957787.png)

![image-20240211222225080](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112222140.png)

CPA is the strong clean notion

stateful意思是给每个message加一个序号i，不同序号用不同加密，因此不会有重复的c；stateless是其反义 (但实际中不是这样实现的)

![image-20240211222629088](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112226131.png)

Constructing CPA-Secure Encryption: Pseudorandom Functions (a building block)

random 的是function的distribution

![image-20240212001253960](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402120012051.png)

输入直接按顺序排好固定，函数不可以一个输入有多个输出。一个函数：有2^n^个输出，每个输出有n  bit，想像成2^n^行，每行有n bit；之后每改变矩阵的任意一个元素就是一个新的函数，因此一共有$2^{n\cdot2^n}$ in Func_n

![image-20240212002005504](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402120020569.png)

![image-20240212003604759](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402120036849.png)

Keyed Functions; efficiently computable: polynomial time

![image-20240212150308638](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121503753.png)

F_k是一个黑盒函数，只给输出，不给得到输出的过程

PRFs in general do not have to be length-preserving. The definition can be slightly modified to work with arbitrary length inputs and outputs (as long as they're not too large that the function cannot be computed in polynomial time).

We do not have functions that are known to be PRFs unconditionally, however. What we *can* show is that a function F* is a PRF given that *F* is a PRF. We don't have any that aren't based on hardness assumptions. IIRC, we can show that if PRFs exist, then P!=*NP* 

![image-20240212151133132](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121511174.png)

![image-20240212151307021](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121513063.png)

![image-20240212151436022](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121514065.png)

基于PRF构建CPA security

r是随机n bit string， F_k(r)用来one-time pad

k决定function(一个table)，r是输入(table 哪一行)

![image-20240212151717627](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121517672.png)

boring proof

![image-20240212154615406](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121546496.png)

已知F_k情况下能破解

![image-20240212154703162](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121547247.png)

把F_k抽象出来，只做一些形式上的变化，本质不变

![image-20240212160636339](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121606393.png)

CPA2 world中，使用 真随机f 代替 伪随机F_k，分两种情况

1: $\delta$ = non-neg，说明可以区分PRF和f；CA: challenger adversary combination

we prove that there exists a (polynomial-time) adversary that distinguishes the PRF from a truly random function (assuming that there is an adversary that breaks CPA security). However, the problem stated that the PRF is secure, so in fact there does not exist a (polynomial-time) adversary that can distinguish the PRF from a truly random function. That's the contradiction. The way to resolve the contradiction is to conclude that our assumption (that there is an adversary that breaks CPA security) is wrong. In other words, we conclude that there is no adversary that breaks CPA security.

![image-20240212161506985](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121615029.png)

2: $\delta$ = neg，说明CPA2也是non-neg

x* = xi 的概率是q(n)/2^n^，q(n)是query的数量

![image-20240212161702462](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121617504.png)

boring proof end

PRF-OTP的key长度是两倍msg长度，很占空间

![image-20240212170014357](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121700457.png)



You can think of oracle as a black box that an algorithm can query and get responses from (in this case, the distinguisher has access to this function that is either a PRF or truly random function without knowing the key - and can get evaluations on multiple inputs from it)

The few paragraphs above and below Def 3.25 in the Katz-Lindell book are useful.



# LEC6 stream-cipher; block-cipher/PRP; ECB, CBC, OFB, CTR; CCA; hybrid proof: CPA strong

![image-20240212185103968](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121851013.png)

![image-20240212185156950](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121851991.png)

optional initialization vector (IV) 不一定用得到；GetBits algorithm 可以输出任意长度

看作flexible PRG

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121855261.png)

s表示seed，即key；st表示state

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121857868.png)

PRF: 可以query; weak-PRF: 发送一个长度为n的指定bitstring x，收到一个x, F_k(x)

key is hidden, IV is available to attacker

![image-20240212190951690](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121909751.png)

Stateful

![image-20240212191402358](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121914432.png)

permutation 排列；bijective 双射: no two input have the same output；并且permutation可以逆向计算 (function 和 permutation的区别所在)

注意感叹号 factorial

![image-20240212192122206](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121921253.png)

key的长度n，key的空间2^n^

![image-20240212192208325](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121922392.png)

![image-20240214112522617](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402141125900.png)

strong PRF: have access to inverse

分不清黑盒机器是F_k & F_k^-1^ 还是f & f ^-1^

![image-20240212194556747](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121945828.png)

Block Cipher modes of operation: to achieve shorter ciphertext (past scheme len(c) = 2len(m))

ECB: not a randomized scheme

![image-20240212231235219](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402122312315.png)

![image-20240212231255997](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402122312042.png)

set c0 = IV; sequencial processing -> no parallel, low efficiency

IV′ refers to the "next value" of the counter; In the second run, ![image-20240213014629402](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130146506.png)

![image-20240212231945584](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402122319698.png)

使用之前最后一个c作为下一段msg的IV；c2 is initialization vector; IV1 c1 c2 are available to attacker, he can pick m3 to break

Choose m3 as follows, so c3 = c2 XOR IV1 XOR c2 XOR m1 =  IV1 XOR m1 = c1, thus distinguish m1

![image-20240212234725442](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402122347549.png)

F_k可以提前运作

![image-20240213004130460](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130041566.png)

不用sequencial；ctr counter是uniformly chosen；encrypt 同一批msg用同一个ctr，下一批msg换新的随机ctr

注意，多次query之下，每次的ctr都不能overlap，因此2^n^很大，如n = 256，来确保几乎不发生overlap

书上有CPA-secure证明

![image-20240213195830305](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402131958433.png)

![image-20240213004218678](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130042730.png)



CCA Security, chosen cipher text; 可以问c解码是什么，唯一不能问c*

![image-20240213013622833](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130136960.png)

![image-20240213013653559](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130136604.png)

c' is flip last bit of c *, 把c‘ 给challenger让它返回m', flip last bit of m' then is m *, r  和 k 全程都保持不变

![image-20240213013708021](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130137067.png)



hybrid arguments: CPA-Security => Mult-Security; contrapositive: if we break Mult-Security, then we break CPA-Security

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402131647565.png" alt="image-20240213164702454" style="zoom: 50%;" />

![image-20240213021944982](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130219093.png)

右下角只是换一种定义方式，目前还在定义阶段；j = 0 两种定义相同；j > 0 is hybrid case

![image-20240213021958985](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130219085.png)

j = t 最右侧情况，attacker选的msg都一样，因此他自己也无法区分，结果必定是1/2

j 增大，attacker goes from doing well to not doing well 

![image-20240213022513118](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130225236.png)

![image-20240213023315207](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130233380.png)

A distinguish i and i-1, construct B break CPA security; ci是分界线，一个是m一个是0；< i 的都是0，> i 的都是m，只有紫色的mi and 0^i^ 我们不知道返回的c是哪个的加密，证明B可以tell between mi and 0^i^

![image-20240213025348638](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130253765.png)

![image-20240213032122330](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130321405.png)



LEC5做法 (未讲)

![image-20240212170234489](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402130201611.png)

![image-20240212184750834](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121847940.png)

![image-20240212184807418](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402121848454.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291451331.png" alt="image-20240229145131239" style="zoom:50%;" />



# LEC7 stream cipher: LFSR, non-linear without bias, trivium, RC4; block cipher: confusion-diffusion, SPNs, Feistel networks

PRF vs PRG vs PRP

![image-20240229020432340](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290204385.png)



![image-20240228153509587](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281535671.png)

![image-20240228153521779](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281535818.png)

![image-20240228154102489](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281541544.png)

![image-20240228154115140](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281541185.png)

buikding object

$\Sigma c_js_j$ (这里异或等价于求和mod2 ) 算出新的s3，原序列右移一位 

![image-20240228154517158](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281545205.png)

S3‘ = s2 XOR s0; initial state: 0100 (s3, s2, s1, s0)

![image-20240228155048682](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281550716.png)

maximum length LFSR: cycle through 2^n^-1个不同non-zero state (不然会陷在zero-state出不来；出现same state就是下一个循环)

![image-20240228155457636](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281554690.png)

如果知道coeff，前四个bit就leak了s3, s2, s1, s0，之后是s2 XOR s0 

如果不知道coeff，前n个bit leak s3, s2, s1, s0，后n个bit leak coeff (solve linear equation)

![image-20240228160532896](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281605933.png)

Non-linear的一些小技巧

![image-20240228161623708](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281616779.png)

biased -> 与门有3/4概率得到0，1/4概率得到1; biased -> broken

![image-20240228161906664](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281619718.png)

![image-20240228162140663](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281621703.png)

![image-20240228162308270](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281623320.png)

![image-20240228162421077](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281624103.png)

一个正式scheme，至今未被破解

![image-20240228162459196](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281624228.png)

初始 80bit key, 80bit IV, 3bit 1, 其他都是0

key is hidden, IV is known to attacker; 下图用80bit key生成无限长伪随机序列

![image-20240228165315262](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281653387.png)

之前的一些被破解的scheme，not LFSR-based

![image-20240228171915061](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281719153.png)

j是一个随机位置；addition is mod 256

![image-20240228172143039](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281721078.png)

加上IV (RC4 was not designed for IV)

![image-20240228182340851](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281823955.png)

S0是uniform，S1 会有slightly > 50% property 出0

y1不知道是多少，但是y2一定是0

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281824545.png)

![image-20240228190306733](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281903857.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281903206.png" alt="image-20240228190336168" style="zoom:25%;" />

https://nordvpn.com/zh/blog/block-cipher-vs-stream-cipher/

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290211019.png" alt="image-20240229021158977" style="zoom:67%;" />



Block Ciphers

input length l 是IV, output a l bit block

For stream cipher: IV是uniform，输出才是uniform

For block cipher: 不论input如何，output都uniform (stronger)；input output 一一对应

![image-20240228233747984](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402282337065.png)

How many bits do we expect to change? half; Which bits do we expect to change? uniform

![image-20240228235328370](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402282355073.png)

create a little confusion then spread (diffusion) it

![image-20240228235838842](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402282358891.png)



Design Paradigms

- Substitution-permutation networks (SPNs) 
- Feistel networks

Substitution-permutation networks (Substitution add confusion,  permutation add diffusion)

8Byte-64bit, 分成8个8bit，$2^{64}!$  -> $2^{8}*8!$  已经减少了很多；但不是pseudorandom因为change1bit input只有前8bit output变化，并且key还是太多

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290007289.png)

diffusion

除了key k其他都public，mix执行一次是可逆的，因此通常执行多次

![image-20240229001911475](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290019540.png)

Avalanche effect 雪崩效应: 1bit input 变化，8bit变化，每轮diffusion导致更多bit变化

![image-20240229002812379](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290028422.png)

使用fixed permutation (S1~S8)代替random permutation；S1~S8 are public, only key private; (假设run三轮, 第1轮: S are carefully designed, 1bit change in input lead to at least 2bit change in output, permutation put 2 changed bit in different block; 第2轮: substitution+permutation again, 4 changed bit; 第3轮: 8 changed bit ... -> PRF output 雪崩效应，循环多次后，input改变1位，output好像是另一个随机数)

![image-20240229003420341](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290034404.png)

只有64bit sub-key是private；和key异或&Sbox是confusion，permutation是diffusion

![image-20240229134718564](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291347635.png)

![image-20240229013805533](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290138630.png)

![image-20240229014418725](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290144767.png)

![image-20240229014804554](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290148595.png)

![image-20240229015127747](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290151785.png)

反向也有雪崩效应 -> 双向 -> strong PRP

![image-20240229015332832](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290153869.png)

One-round, 已知xy，可知k

![image-20240229015652592](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290156646.png)

多一个subkey k2 mix at output (图上没画)

given k1，可以计算出k2，因此遍历k1即可 (k2和k1一一对应: 知道k1可以算出z'，和y异或就是k2 )

![image-20240229022029038](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290220078.png)

一种更快的attack, 把k1 k2分成八份仍旧使用同样的attack

![image-20240229021452158](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402290225436.png)



# LEC8 Feistel Networks, DES

Feistel Networks

A Feistel network is used to construct a pseudorandom permutation F given a pseudorandom function f that is not necessarily a permutation

have non-invertible components, but overall invertible

f_k is non-invertible component making confusion

L1, R1, key are enough to recover L0, R0

![image-20240229150949760](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291509838.png)

2 round: 如果L0 L0‘ 有1bit不同，则L2 L2‘ 有1bit不同；3round及以上，如果f是PRF且non-invertible，则secure

![image-20240229151708273](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291517329.png)

只有暴力遍历可以破解，但是暴力破解只需要一天；在此基础上搭建triple DES来增强security (outdated)

mangler function是one-round non-invertible SPN

每一轮的key不同，但都是由56bit key得来的 (每轮抽取一些)

![image-20240229152658155](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291526199.png)

32-bit input encode 到 48-bit intermediate (类似纠错码) (expansion)；Sbox are non-invertible (squeeze back to 32bit)；使得雪崩效应发生的更快 (32-bit expand to 48-bit, go to more Sbox)；key: 24bit 从 28 bit subsample

![image-20240229153756647](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291537728.png)

![image-20240229165915492](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291659617.png)

1轮是不安全的，可以recover 48bit key，不知道56bit key

f: input: 32bit R0, 48bit key; output: 32bit

同之前一样，整体遍历是2^48^；逐个box，每个box是6bit -> 4bit，2^6^ -> 2^4^，4 to 1

![image-20240301165447500](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403011654588.png)

![image-20240229170035834](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291700903.png)

和1round一样

![image-20240229180403786](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291804884.png)

更快的攻击方法需要大量样例，不现实；因此暴力遍历最优 

![image-20240229180602265](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291806366.png)

遍历可以破解，需要update key

Modify DES, people dont believe; if sth work, dont change it

![image-20240229180858909](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291808992.png)

16 round -> 32 round; 前16和后16用different key 

![image-20240229181532602](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402291815675.png)

2倍长度key，security并未增强

![image-20240301020752073](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403010207159.png)

建立k1&z表格，k2&z表格，按照z值合成一张表(k1 k2一一对应)，遍历该表即可

![image-20240301021144302](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403010211352.png)

no attack yet

F2使用逆的原因： let k1=k2=k3，即是F_k(x)， 兼容old system(只用single DES和single key)

block size仅仅64bit，2^64^不够用，因此出现了AES，被广泛使用

![image-20240301021448944](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403010214989.png)

![image-20240301022956782](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403010229830.png)



# LEC9 fixed/arbitrary length MAC, CBC-MAC (prefix-free)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131524933.png)

still private key, attacker change the c, integrity problem

![image-20240313153023618](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131530648.png)

传输过程中未出错Vrfy输出1 

![image-20240313153041986](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131530021.png)

Unforgeability 不可为造性；attacker想要生成valid (m,t) pair；A has oracle access to Mac_k, M is the list of query A tried；A then choose m* and t*, tag is n (security parameter ) bit long

![image-20240313153649797](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131536867.png)

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131556117.png)

Strong: 可以使用之前query过的message和新的tag组成pair进行最终测试，或者用之前query得到过的tag和新的message进行最终测试，either m or t should be different 

![image-20240313155921929](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131559964.png)

MAC实例

PRF: key ✖️ input -> output, is deterministic

![image-20240313160432911](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131604955.png)

左PRF，右random fn; proof use hybird, 略

![image-20240313161021502](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131610534.png)

fixed length Mac -> arbitrary length Mac' (很容易出错，crypto is brittle); key are the same

1. 

![image-20240313163433799](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131634879.png)

attack: query m1, m2; 最终测试m2, m1

![image-20240313164011073](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131640107.png)

2. 

![image-20240313163642926](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131636975.png)

attack: query m1,m2; m1' m2', 最终测试m1, m2'

![image-20240313164913098](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131649145.png)

3. 

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131655747.png)

Attack: query m1||m2, drop m2, 最终测试m1

4. secure方法:

but r induce randomness; Vrfy should also run fn r; MAC' 返回 r||t1...td; l = d

![image-20240313170437020](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131704065.png)

红框是Mac‘ 的work方法，返回Tag = (r,t1,t2...td)，sent to red Adeversary A

B是A的subroutine，B首先要answer A的query list M；B向绿色Mac询问![image-20240313191458387](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131914434.png)，得到ti，由此Mac‘有了ti和r，可以生成Tag返回给A；Use (m *, t *) and output one of the block as (m, t)；绿色m,t要求不在B的query list中

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131909584.png" alt="image-20240313190934488" style="zoom:67%;" />

绿色Mac is fixed length, adversary B sample r, B生成下图中的 m1～md, send to challenger 绿色Mac, 收到t1~td, 再组装(r,t1,t2...td);  for each query A makes (with same r), B makes multiple query

![image-20240313182013605](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403131820713.png)

绿色m,t要求不在B的query list中 (接下来会证明如果A能找到(m *, t *)不在A的query list中，B也能找到(m, t)不在B的query list中)

A最终测试了t*；case1: r *与之前B sample to answer A query的ri都不同，那么下图t1 *就是一个forgery；case3: block with different m就是一个forgery

![image-20240313205231692](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403132052776.png)

4ln是t1...tn，省略了r的长度

![image-20240313204206072](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403132042184.png)

proof end

make size smaller

![image-20240313210529423](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403132105531.png)

attack, 2 query

![image-20240313210849875](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403132108925.png)

![image-20240313211811376](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403132118501.png)

if only allow query/forgery of fixed length, can't make forgery, is secure

F_k is PRF -> CBC_k is PRF

Not fixed length but prefix-free -> CBC-MAC is ef-cma secure 

(fixed length is prefix-free, prefix should be shorter; 如果没有prefix-free property就可以执行上述相同attack)

![image-20240313211844109](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403132118183.png)

然而现实生活中的message绝大多是prefix-free的，但是为了避免极少情况，我们可以获得prefix-free property

若两m长度相同，满足；若两m长度不同，开头加上长度获得了prefix-free property

![image-20240313213710491](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403132137618.png)

最后一步使用一个不同的key，代价是key length double；之前的attack问m1得到F_k'(m1)；allow not prefix-free

![image-20240313213941017](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403132139066.png)



# LEC10 Authenticated Encryption: confidentiality (CCA) + integrity (UE); Encrypt then Authenticate

A成功如果它output c*是一个没见过的m的valid密文；垂直符号是error symbol

![image-20240314012817871](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140128012.png)

![image-20240314013527485](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140135625.png)

PRF-based OTP, it is not unforgeable, 随机选一个2n长的就是某个m的valid密文 (虽然不知道是哪个m的密文 )

![image-20240314013559036](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140135091.png)

Authenticated Encryption: both integrity and confidentiality 

![image-20240314014401793](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140144852.png)



![image-20240314014636220](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140146292.png)

unforgeable but not CCA secure: 修改一个Authenticated Encryption scheme，使得Enc时在最前面加一个随机bit b，Dec时忽略b

attacker query 1||c, 最终测试用0||c -> break CCA

![image-20240314015213303](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140152416.png)

CCA secure but not unforgeable:

F_k是PRP, m长度n/2, r是长度n/2的随机比特串 -> c covers all n bit string, 任选一个n bit string就是一个valid c

![image-20240314015603686](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140156795.png)

Gen都是生成n位k，暂时省略；(begin with CPA secure, 之后会讲加上Mac就得到CCA secure)

Dec&Vrfy过程

- vrfy t, if valid, Dec c
- Dec first, vrfy tag
- vrfy first, if valid, Dec c

![image-20240314021102565](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140211752.png)

1. Encrypt and Authenticate

unforgeable but not CCA secure

这里假设Enc本身是CPA，Mac本身是unforgeable；tag do not need to hide anything about the message，如使得tag = m||t仍旧可以满足unforgeable要求 

![image-20240314030648195](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140306314.png)

2. Authenticate then Encrypt

unforgeable but not CCA secure: Enc时在首位加random bit, 用m query oracle 返回c, 再flip c的首位得到c'做最终测试

![image-20240314031324444](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140313524.png)

3. Encrypt then Authenticate

发送 (c, t)；strong unforgeable -> CCA secure，避免new tag on same message (可以用来做最终测试)

![image-20240314032450488](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140324589.png)

K_E 不可以 = K_M; 最后一行，c leak message m 

![image-20240314034252855](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140342949.png)

authenticated encryption在实际应用中无法抵御的三种attack (application 层面)

1. 框框表示unknown黑盒，reorder the (c,t) pairs -> 把消息变个顺序发送 

![image-20240314035628927](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140356001.png)

2. 重复m
3. 把A发给B的内容drop掉，改成由B发给A，A收到自己发给B的message (B did not intend to)

counters表示message计数器，direction bit表示A -> B or B -> A; 实际应用中可以放在associated data里面，即不用保证confidentiality但是要保证integrity的data， 提高一些efficiency

![image-20240314040919439](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140409491.png)

![image-20240314035224354](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140405457.png)



# LEC11 keyed/unkeyed Hash; birthday attack; fixed length -> arbitrary length; Hash-MAC; OWF

Cryptographic Hash Functions

not rely on secrecy of the key, key is public 

![image-20240314041416927](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140414028.png)

*表示arbitrary long；cryptographic hash functions严格不允许collision, even when attacker know the key；collision definitely exist, but should be hard to find

![image-20240314041504450](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140415500.png)

l(n) is poly;  Data structure不用cryptographic hash function的原因: output space too big, 如2^256^bit, 难以存储 (速度其实还挺快)

通常关注$\ell^{\prime}(n)>\ell(n)$，因为小于的话一般根本没有collision

![image-20240314042309688](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140423781.png)

如果是unkeyed hash, s is fixed and given to attacker, PPT A 可以在poly time破解, keyed就不行

![image-20240314042948715](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140429842.png)

![image-20240315012028375](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403150120497.png)

![image-20240315012040055](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403191514871.png)

但实际生活中unkeyed hash也从来没被发现过collision

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140455721.png)

attack: $\ell^{\prime}>\ell$，只要尝试$2^{\ell}$个input就必定能找到collision

![image-20240314045832204](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403140458287.png)

![image-20240314134118150](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141341267.png)

q表示query数量，需要用较少的query找到collision；生日问题365必碰撞，根号365有1/2概率碰撞；这里$2^{\ell}$必碰撞，开根号$2^{\ell/2}$有1/2概率碰撞

![image-20240314134944841](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141349981.png)

more devastating attacks 可能要求更大的security para n

![image-20240314140532938](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141405080.png)

fixed length -> arbitrary length

Attack: M(长度不够最后自动补0)和M||0两个碰撞 

![image-20240314140851877](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141408991.png)

to fix: 在最后加上exact length of message (不是block数量) -> secure

![image-20240314142355055](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141423250.png)

Proof: 如果arbitrary length H被破解，可以通过构造破解fixed length h

![image-20240314150911309](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141509448.png)

唯一能确保是最后红框输出的地方output hash是相同的，因此做backward tracking，往前直到找到output hash相同，input不同的点

![image-20240314151503246](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141515327.png)

use hash to build MAC

![image-20240314152225043](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141522100.png)

m经过Hash fn后变短；hash then Mac；k secret, s do not need to be secret

![image-20240314152421797](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141524858.png)

Proof: assume attacker break MAC and output (m *, t *)

Case1: m*和之前query中的某mi hash collision, break Hash

Case2: break fixed-length MAC

![image-20240314154104020](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141541166.png)

[相比于CBC-MAC的一些好处：例如有一个GB movie，已经生成一个tag，对movie修改一些，如果是Hash-Mac可以从修改处开始继续生成新tag，但是如果是CBC-MAC则需要从头开始]

Hash application

- 维护一个virus文件hash database。，新文件与之对照判断是不是病毒
- Deduplication 重复数据删除 同一张大照片在各种文件夹 (云中心存储不同人的两个同样文件)

![image-20240314155113757](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141551833.png)

但是还不知道怎么construct PRF

![image-20240314155731800](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141557879.png)

理论上有完美的PRF，但是效率低，因此实际中仍旧使用更快的block cipher (not yet but may be broken someday)

![image-20240314160829035](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141608142.png)

![image-20240314161029014](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141610086.png)

arbitrary length -> arbitrary length, 但是实际是fixed particular length，没有随机性 

OWF: 可以正向快速计算，不可逆向慢速计算，基本是all symmetric crypto object的基础

![image-20240314161255674](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141612766.png)

- easy to compute: 比如M_f是fn f的truth table

- hard to invert: x是randomly picked; set of $f^{-1}(f(x))$是因为同一个y可能有多个逆，attacker拿到f(x)，找到任何一个valid inverse 就算成功; 1^n^是security parameter (typically security parameter is fixed at the time to sample the key, 这里没有key因此加上1^n^)

  ![image-20240314162219250](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141622366.png)

![image-20240314161553248](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141615321.png)

![image-20240314163102021](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141631101.png)

![image-20240314163254354](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141632479.png)

Candidate construction of One-Way Functions

- prime 素数，两个素数相乘，分解factor很困难
- SS: subset sum, J is a subset, 输出的前半部分𝑥1,… 𝑥n是可逆但不可省略的，如过省略就可以如下图破解

![image-20240314165751854](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141657938.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403141635099.png)



# LEC12 OWF, Hardness-Concentrate bit, OWP -> PRG -> PRF

![image-20240317130004221](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403171300379.png)

证明判断OWF

![image-20240317160341803](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403171603830.png)

g(x) simply drop f's last bit; $y=0^{n/2}$发生的概率极低

对于g(x)，adversary对于(n/2, n/2-1)bit输出，output(n/2, 0^{n/2})就是inverse

![image-20240317160325650](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403171603681.png)

![image-20240317161002259](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403171610307.png)

![image-20240317215558220](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403172155361.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180252498.png" alt="image-20240318025221346" style="zoom:50%;" />

f OWP -> g OWP: 前半部分f(x)保证了一对一，后半部分y也保证一对一

OWF可以leak some input bit, 但是PRF要求输出完全random -> figure out what info about input is hidden

![image-20240317161746685](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403171617743.png)

one bit hidden to prevent being inverted -> Hardness Concentration: Concentrate hardness in 1 bit

fn that output 1 bit is called predicate

以下都使用OWP，given f(x) -> x is unique -> hc(x) is unique/fixed 

attacker is given f(x), 先不说invert全部, 就让你 to guess 1bit hc(x) -> hc(x) is seemingly independent of f(x)

![image-20240317171336940](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403171713060.png)

![image-20240321201220169](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403212012254.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180110809.png" alt="image-20240318011034698" style="zoom:50%;" />

typo: or -> of

![image-20240318141718904](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181417014.png)is parity bit 奇偶校验位

$f(x_{-1})$表示先丢弃first bit 再输入f函数; f is OWF: n-1bit -> n-1bit; f is OWF -> g is OWF

g(x)在泄漏了hc(x)的情况下仍旧是一个OWF，在hardness- concentrate游戏中attacker is given g(x)

![image-20240318141920713](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181419786.png)

![image-20240317213823118](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403172138231.png)

every single bit is not hard to find (hard-concentrate), 但是合在一起是一个OWF g 

(x,i)是input, drop i^th^ bit；多1/n的概率破解单个bit

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403172226917.png)

f is OWP -> g is OWP, x and r are both randomly chosen

let attacker guess what hc(x,r) is, claim: if attacker can predict single bit hc(x,r), he can invert entire g(x,r)

![image-20240317232453726](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403172324806.png)

e1 = 100..0; e2 = 0100...0; hc(x,ei) = i^th^ bit of x

![image-20240318000014015](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180000137.png)

Pr = 1/2 + non-neg, 可以证明Pr > 3/4时以下模式work

？？？？？

![image-20240318000807265](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180008401.png)

f(x)*ei = xi; 第四步，两个![image-20240318004315185](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180043245.png) cancel out

![image-20240318003846467](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180038684.png)



OWP -> PRG with one more bit (from Hardness Concentration)

[also can OWF -> PRG/PRF but complicated, not covered in this class]

PRG(x,r), (x,r) is seed, secure

Hybrid proof; U 表示 uniform bit

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180118164.png)

![image-20240318020509141](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180205236.png)

PRG with multiple more bit

![image-20240318013121278](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180131375.png)

![image-20240318013228140](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180132274.png)

r可以用同一个 

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181439205.png" alt="image-20240318143939106" style="zoom:25%;" />

proof

already prove PRG1 secure, now prove PRG2 secure 

![image-20240318013727501](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180137604.png)

对于H0和H1，给attacker 𝑦1 … 𝑦𝑛𝑦𝑛+1，让他区分是来自PRG1还是U_n+1

![image-20240318014948524](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180149711.png)



Getting PRFs (High Level)

PSF的key come from PSG的seed, both cannot leak

![image-20240318015025576](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180150628.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180153199.png)

![image-20240318015553552](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180155609.png)

![image-20240318015653271](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403180156352.png)

Proof hint: hybrid G->Uniform



# LEC13 number theory (cyclic Group), DLOG, CRHF , CDH/DDH, key exchange, public key crypto: PKE (Public-Key Encryption) + digital signature

1. set up key
2. 每个对象单独一个key，大量存储空间，有集体泄漏风险，频繁更新
3. set up key with person never met before

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181505936.png)

KDC maintain private key with every user kA, kB; generate fresh key k which is Alice-Bob specific key

Problem: KDC can decrypt everybody's message

？？？？？

![image-20240318151003445](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181510519.png)



Public-Key Cryptography

'·' is abstract operation; G is sth like N/Z

![image-20240318153039909](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181530987.png)

整数&加法运算

![image-20240318153502706](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181535792.png)

![image-20240318153653854](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181536899.png)

size is finite

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181600037.png" alt="image-20240318160046945" style="zoom:50%;" />

for finite group; order of group = # of elements

![image-20240318160235204](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181602255.png)

![image-20240318161408332](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181614395.png)

![image-20240318161456954](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181614026.png)

only prove for the abelian case, but is true in general

g · gi, i $\in$ [0,n-1] must be n-1 different element in set, thus $\prod g · gi = \prod gi$

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181619574.png" alt="image-20240318161931498" style="zoom:30%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181629054.png" alt="image-20240318162937952" style="zoom:30%;" />

$\left\langle g\right\rangle $ 表示对g取1~ m次方形成的新set，可能会有重复，取1~ i次方就够了, i is order of g

Proof: assume i is first element that g^i^ = 1, but i dont divide  m -> m = ai + b

![image-20240318162958060](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181629157.png)

$\left\langle g\right\rangle $ 表示对g取1-m次方，要求形成set G；order是素数，不存在i divide m

Zp is {0,1,2...n-1}; $Z_{p}^{*}$ remove 0, 包含数字1～p-1 -> cyclic group with p-1 element, not prime order

p q are prime, $Z_{p}^{*}$ order = 2 * q, 因此$Z_{p}^{*}$会有order 1 or 2 or q or 2q的元素 (cyclic group的定义只要存在一个order2q的元素即可)

g is an element of $Z_{p}^{*}$ with order q, and $\left\langle g\right\rangle $ is cyclic group with q element, prime order (crypto desire prime order cyclic group)

![image-20240318165554602](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181655747.png)

In practice, p q are very long, 1000 bit, there are endless p q pairs

![image-20240319002251058](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190022180.png)

![image-20240319002627641](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190026745.png)

DLOG is the hardest one to solve, CDH is the second-hardest, and DDH is the easiest. 

in these group DLOG problem is hard

![image-20240318180130081](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181801225.png)

DLOG can construct OWF, attacker cannot recover x

![image-20240318181548766](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181815938.png)

dot_operation(g^x^, h^r^); 输出相较输入进行了压缩

![image-20240318182438456](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181824584.png)

proof: 如果破解CRHF，即找到冲突数据(𝑥, 𝑟), (𝑥’, 𝑟’)，可以利用之破解DLOG

![image-20240318183104298](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181831387.png)



The Diffie-Hellman Problems

1. computational variant

given g^x^ and g^y^, cannot compute g^xy^

如果可以破解DLOG，可以根据g^y^知道y，计算(g^x^)^y^即可破解Diffie-Hellman (Diffie-Hellman is easier than DLOG, but still very hard)

2. decisional variant

Harder than computational variant

![image-20240318183410279](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181834350.png)

![image-20240318183931657](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181839760.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181840819.png)

construct PRG, attacker cannot distinguish between  following two 

![image-20240318184319192](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181843323.png)

other group using more complicated operation -> Elliptic Curve Groups: each element is on Elliptic Curve

![image-20240318184729327](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181847405.png)

realize public key crypto

Alice and Bob talk on public internet channel

![image-20240318190245980](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181902057.png)

b=0, 给attacker key; b=1, 给attacker random string, attacker 区分不了 

![image-20240318190418089](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403181904263.png)

![image-20240318191018127](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403191337100.png)

Public-Key Encryption similar to Enc (confidentiality); Digital Signatures similar to Mac (integrity )

![image-20240319000038399](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190000550.png)

Alice has public key and secret key, Bob download public key from website

authenticated channel 确保Bob拿到的是对的pkA (attacker给一个假的attacker自己的public key，Bob用假的public key encrypt之后attacker可以decrypt，然后attacker再用Alice的public key reencrypt之后发给Alice，Alice也可以正常收到)

![image-20240319000132538](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190001613.png)

Alice cannot recover b, he just calculate g^ab^

![image-20240319001043080](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190010149.png)



# LEC14 PKE: EAV=CPA/CCA, ElGamal encryption, hybrid encryption

![image-20240319004724534](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190047601.png)

垂直符号表示error msg

![image-20240319005115475](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190051552.png)

![image-20240319005605473](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190056635.png)

EAV-security and CPA Security are the same in public mode

[in private mode, EAV can be deterministic, CPA must be randomized]

![image-20240319005846887](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190058952.png)

m_1~m_n分别Enc仍旧CPA-secure

![image-20240319010728763](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190107909.png)

Malleability 延展性

![image-20240319011113319](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190111381.png)

Enc oracle省略了因为默认所有人都可以；也正是因为所有人都可以generate legitimate cipher text，Dec oracle更加有用了，CCA security因此更难达到

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190119930.png)



Construction of PKE

G, g, q are fixed once and for all; m会被映射到group element

c2 = m g^xr^; c1^sk^ = g^XR^; output m

![image-20240319014904593](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190149708.png)

proof: adversary break ElGamal Encryption, then he can break DDH 

![image-20240319020551740](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190205862.png)

Enc之后c的长度是m的两倍

![image-20240319020635134](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190206199.png)

用publice key得到shared key之后进行private key scheme，更加efficient

public/private 都用到了 -> hybrid encryption

![image-20240319020938520](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190209632.png)

sample fresh key k, encrypt key k with pk (pk作为Enc的key，k作为Enc的message)

Enc‘ is private encryption scheme

c = (encapsulated key, ciphertext)

receiver decrypt encapsulated封装 key to get k (public scheme), and use k to decrypt ciphertext (private scheme)

![image-20240319022020380](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403190220505.png)



# LEC15

![image-20240321170300557](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211703750.png)

如果每个msg各自用一个encapsulated key，$\pi'$只需要EAV secure就够了(can be deterministic, no need randomized)；如果每个msg用相同的encapsulated key，$\pi'$需要CPA secure

![image-20240321171014639](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211710753.png)

Hybrid proof

H0和H1: 如果A可以区分H0和H1，那么就可以构造B break security of public $\pi$

![image-20240321172757862](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211730814.png)

H1和H2: 如果A可以区分H1和H2，那么就可以构造B break security of private $\pi'$

H2是纯随机的，因此必然是1/2

(hybrid中$\pi$和$\pi'$都要用到，才能证明$\pi$和$\pi'$都必须secure)

![image-20240321171727750](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211717848.png)

问题：可不可以把H1和H2内容变化的顺序调换？H1变绿色，H2变红色 -> 不可以 略

CCA security proof 略



KEM: another PKE (not hybrid encryption)

ElGamal for hybrid encryption 

与其sample a key and change it into group element，不如sample a group element and change it into key (H是hash函数)

![image-20240321173620522](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211736631.png)

Enc algo input is only public-key, no message, so attacker cannot query message

k=Decaps(c, sk)

？？？？？？？/

![image-20240321174008043](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211740113.png)

KEM中attacker只能看见pk  c c‘，k是中间过程

![image-20240321191628227](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211916323.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211922852.png)

k = H(g^xr^) 用于private encrypt

![image-20240321192520885](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211925927.png)

实际中

![image-20240321193511421](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211935487.png)

![image-20240321195241534](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211952588.png)

another valid cipher text encrypting m, ask decrypt oracle

![image-20240321195117104](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211951159.png)

可以get到一个new valid cipher text (但是不知道m·m'是什么)

需要要c和c‘用the same key

![image-20240321195521785](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403211955868.png)














