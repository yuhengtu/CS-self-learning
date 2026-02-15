# Intro

drop 2 lowest homework, give you **full credit on a homework if you score above 90%**. If you score below 90%, we will divide your score by 0.9, e.g. a score of 50% becomes 55.5%

[homework guidelines](https://cs170.org/resources/homework-guidelines/) 

- Midterm 1: Monday, 2/26/2024, 7:00 PM - 9:00 PM

- Midterm 2: Tuesday, 4/2/2024, 7:00 PM - 9:00 PM

- Final: Friday, 5/10/2024, 7:00 PM - 10:00 PM (Group 20)

- Homeworks: 15%

- Exams: 85%
  - Midterm 1: 25%

  - Midterm 2: 25%

  - Final: 35%

    


Oftentimes, a problem will ask you to “give/design/devise/formulate/create/etc. an algorithm” or to “show how” to solve some computational task. In this case, write your solution in the 3-part algorithm format:

1. Algorithm description
   - This can come in terms of pseudocode, or a description in English. It must be unambiguous, as short as possible (but no shorter), and precise.
   - Your pseudocode does not need to be executable. You should use notation such as “add *X* to set *S*” or “for each edge in graph G”. Remember you are writing your pseudocode to be read by a human, not a computer.
   - See DPV for examples of pseudocode.
2. Proof of correctness
   - Give a formal proof (as in CS 70) of the correctness of your algorithm. Intuitive arguments are not enough.
   - Again, see DPV for examples of proofs of correctness.
3. **Runtime analysis**. You should use big-O notation for your algorithm’s runtime, and justify this runtime with a runtime analysis. This may involve a recurrence relation, or simply counting the complexity and number of operations your algorithm performs.



natural naive/brute-force algorithm or apply [suboptimal known algorithm] without any modification or thought”) or exponential-time algorithms usually get no credit. We often award partial credit to algorithms that are slower than our official solutions.



cite theorems and algorithms from the textbook or lecture



\(\Theta(\cdot)\)  better than \(\mathcal{O}\), but  \(\mathcal{O}\)  is always correct



**CS 270**

This is the most direct extension of CS 170. There's a lot of overlap in terms of problems (e.g. shortest paths, max flow, linear programs, hashing, etc.), except you learn about various data structures and techniques to improve runtime! For example when I took it, we learned how to speed up the max flow algorithm from 𝑂(∣𝑉∣∣𝐸∣2)*O*(∣*V*∣∣*E*∣2) to 𝑂(∣𝑉∣∣𝐸∣log⁡∣𝑉∣)*O*(∣*V*∣∣*E*∣log∣*V*∣) (and also how to speed it up to almost linear time). You also learn a new way to analyze runtime (using price functions), and several new data structures. Highly recommend taking this!

**CS 172**

This is also a good extension if you enjoyed learning about P/NP and complexity classes. It's a theory-heavy class where you learn about automata theory (basically formalization of a "problem"), turing machines, and more applications of complexity. If you're interested in any complexity theory research, this is the class to take!

**CS 176**

This is a great class to take if you're interested in algorithmic applications in biology, dynamic programming, and/or machine learning. Although the course focuses on biological applications, a lot of concepts you learn (hidden markov models, more advanced DP string matching algorithms, and stochastic processes) are super applicable to other areas! I've taken this class and recommend it; it's a bit difficult but super rewarding!

**CS 174** 

This is a great class to take if you liked EECS 126 on top of CS 170 (if EECS 126 and CS 170 had a baby it would be CS 174). The class does a big review on basic probability theory and a bunch of concentration inequalities (e.g. Markov's inequality). It also covers a bunch of randomized graph algorithms, parallel computing, hashing, a bit of cryptography, and some other stuff I forgot about LOL.

**EECS 127**

This is a cool class to take if you liked learning about linear programs, simplex, and duality. The course introduces the notion of convex programs, of which LPs are a subset. For instance, another example of a convex program is a quadratic program (QP), where the constraints are linear and the objective is quadratic in the variables. The class also rigorously defines duality (which we handwave in this class lol), and proves when exactly duality holds for various convex programs. You also learn about the ellipsoid method and interior point method, which in this class we say runs in polynomial time but never delve into it. I personally really liked this class, and recommend anyone interested in optimization or machine learning to take it!

**Miscellaneous**

If you either REALLY love algorithms or are interested/engaged in theory research, I recommend taking a graduate special topics class (i.e. any CS 294-). They are mainly discussion/seminar based so lectures are very chill and grading for grad classes are always extremely lenient. More importantly, you get to dive deep into a specific part of CS theory with a professor that is super cracked in their field (Berkeley is #1 for theoretical CS!). 



# LEC1 Big-O Notation

只mark了first node

![image-20240119215845865](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401192158963.png)

Fibonacci

![image-20240126134704946](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401270547971.png)

![image-20240119221843809](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401192218834.png)

n * 2 -> time * 4 由于数字很长

![image-20240119221658139](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401192216165.png)

![image-20240126134649604](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401270546698.png)

polynomial: O(n^2^): n * 2 -> time * 4

exponential: n * 2 -> time ** 2

![image-20240119222923293](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401192229316.png)

O: no faster than

![image-20240119223722001](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401192237024.png)

![image-20240119223835798](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401192239317.png)

ADD: proved can't do better

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401192241004.png)

MULTIPLICATION: can do better

![image-20240119224304645](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401192243666.png)

normal multiply -> n * 2 -> time * 4

python multiply -> n * 2 -> time * 3

![image-20240126155001647](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401270750729.png)



# LEC2 int/matrix multi, Master Theorem

![image-20240127110158097](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280301140.png)



乘法竖式中加法个数，从右到左，1 -> ... -> n -> n -> ... -> 1，前一半是k，后一半是2n - k

![image-20240127111028212](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280310288.png)

![image-20240127111653888](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280316973.png)

树一共log2(n)层，最底层leaves有n^2^个

![image-20240127113017098](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280330213.png)

![image-20240127121327295](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280413335.png)



faster algorithm

![image-20240127114927341](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280349434.png)

![image-20240127115220474](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280352517.png)

![image-20240127115638638](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280356692.png)

![image-20240127121544741](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280415778.png)



Master's theorem

![image-20240127121904004](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280419050.png)

![image-20240127122256819](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280422871.png)

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280426790.png)



matrix multiplication

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280433903.png)

![image-20240127123812898](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280438971.png)

![image-20240127124323255](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280443288.png)

![image-20240127125445242](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401280454298.png)



# LEC3 sort & selection

merge sort

![image-20240202161524952](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402030815037.png)

num of list -> 1,2,4,8... 共logn次

![image-20240202162936094](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402030829239.png)



comparison sort algorithm lowerbound

![image-20240202184530562](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031045652.png)

![image-20240202184552558](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031045579.png)

![image-20240202190354199](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031103269.png)

![image-20240202184635953](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031046991.png)

![image-20240202191218225](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031112298.png)



find median/find nth element

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031112119.png)

中括号向下取整

![image-20240202192447979](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031124105.png)

worst VS average

![image-20240202193846238](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031138276.png)

有1/2概率得到head，1表示投硬币

E[X] = 1/2×(1+E(X)) +1/2×(1): 有1/2几率继续，有1/2几率成功

![image-20240202200017257](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031200357.png)

worst case O(n^2^)

![image-20240202193726627](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031137705.png)

average case: 有1/2的几率pick中间蓝紫色区域，那么必定至少淘汰掉1/4的序列长度

![image-20240202201128495](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031211602.png)

Expected time to get a middle element

![image-20240203012410628](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031724730.png)



quick sort (Randomized)

![image-20240202202054955](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031220009.png)

Master Theorem doesn’t apply... use the point of view of an element. 每个元素比较的次数 = group size decrease to 1 所需的次数

如果partition element取到middle half，group size become <= 3n/4；有50%的几率取到middle half，类似抛硬币，因此平均而言，递归两次，group size become <= 3n/4

G: geometric distribution

![image-20240202202935323](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031229441.png)



# LEC4 polynomial multi & FFT

![image-20240203123150513](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040431585.png)

最多2d个multi，multi是O(d^2^)

![image-20240203124045397](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040440481.png)

Can we do better? Use different representation 特殊点表示

![image-20240203130937208](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040509320.png)

![image-20240203130953814](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040509857.png)

拉格朗日插值：第一项都是一条在x1处 = y1，在x2 x3处 = 0的二次曲线；这样三条曲线叠加

证明unique solution: assume 两条曲线都能拟合所有点，

![image-20240203131147363](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040511401.png)

Because of minus these pairs turn into roots

![image-20240203161058383](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031611347.png)

![image-20240203161116062](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031611086.png)

Error tolerance

![image-20240203134242237](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040542314.png)

polynomial -> point-value: O(n^2^ )

![image-20240203134654930](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040546020.png)

divide and conque: still O(n^2^)

![image-20240203140140141](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040601215.png)

![image-20240203140201859](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040602881.png)

Choose +- points

每个点one multi one add -> O(n)

![image-20240203141924798](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040619837.png)

但实则只有第一层递归能做，第二层就不行了

![image-20240203143451899](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040634988.png)

如果要每一层都成立，必须满足集合的元素数量每平方一次就减少一半，最后减少到1

![image-20240203143639521](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040636567.png)

polar notation ![image-20240203144220963](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040642007.png)

![image-20240203144750853](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040647894.png)![image-20240203144800152](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040648187.png)

对角线相连的两个蓝点平方相同，都是红点；框出来的是n/2次方根

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040653166.png)



FFT

![image-20240203164806884](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031648012.png)

![image-20240203165016515](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031650544.png)

![image-20240205013107706](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402050131813.png)

![image-20240203165223554](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031652582.png)

![image-20240205014323230](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402050143273.png)



# LEC5 

last lecture: evaluation; this lecture: interpolation

![image-20240204001016488](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040010568.png)

Interpolation: 两边同左乘逆矩阵，O(n^2^)

![image-20240204001435542](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040014573.png)

先把特殊点换成FFT选的点

![image-20240204002806506](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040028538.png)

evaluation相当于change basis，黑色 -> 蓝色；因此Interpolation只要转回来

![image-20240204004344782](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402040043852.png)

证明求逆公式

![image-20240204135827740](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041358849.png)

![image-20240205014510028](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402050145071.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402050146786.png)

计算inverse

![image-20240204140912480](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041409556.png)

![image-20240204140943433](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041409461.png)

![image-20240204141309571](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041413622.png)

![image-20240204141855460](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041418521.png)

![image-20240204142941157](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041429195.png)

![image-20240204143035220](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041430246.png)

左右的二进制数是对称的

![image-20240204144500089](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041445144.png)

row number 从上往下 0-7

红线在每个level变化1/2/4 row number

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041432445.png)

row number change

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041456965.png)

![image-20240204145726628](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041457663.png)

only using "divide and conquer" take O(n^{log_{2}3});

but when using FFT, evaluation and interpolation take O(nlogn) and point-value multiplication take O(n), so the whole algorithm take O(nlogn), better than just "divide and conquer".

![image-20240204150637935](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402041506984.png)

公式->蝶形图/矩形图？？



# LEC6 Graph, DFS, DAG, Topological Sort

Matrix Representation: symmetric square matrix

Adjacency List: for sparse matrix

d is length of one list; V is row of the matrix

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402101849197.png)

迷宫；We will use array (visited) for chalk and stack for thread

![image-20240210194335092](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402101943198.png)

遍历所有节点，形成了红线tree，蓝线未走过 back edges

![image-20240210195140670](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402101951796.png)

证明过程: 假设u not found, so w not found; z is the last vertex that is explored

![image-20240210201940232](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102019332.png)

Running Time, Don’t use recurrence

![image-20240210202954848](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102029929.png)

遍历whole graph (包括未连接的部分)：unvisited -> call explore；可以输出 forest

![image-20240210205133686](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102051792.png)

connected component number，统计forest中tree的数量 

![image-20240210210437065](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102104170.png)

记录stack时间

![image-20240210212803679](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102128793.png)

![image-20240210212835136](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102128180.png)

![image-20240210212857662](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102128712.png)

![image-20240210213014529](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102130576.png)

![image-20240210213127291](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102131320.png)

Directed graphs

![image-20240210213742310](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102137397.png)

int 表示 interval

![image-20240210213757375](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102137413.png)

找有向图是否有循环 For each edge (u,v) remove, check if v is connected to u O(|E|(|E|+|V|))

better algorithm: find back edge O(|V|+|E|))

![image-20240210214446493](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102144598.png)

![image-20240210214646626](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402102146698.png)

无循环有向图 Directed Acyclic Graph (No back edges)

DAG -> No cycles! Can tell in linear time!

Topological Sort: For G = (V,E), find ordering of all vertices where each edge goes from earlier vertex to later in acyclic graph

按照pop序号逆序

![image-20240211012944601](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402110129707.png)

![image-20240211024253526](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402110242621.png)

维护一个pop list，最后逆序即可

![image-20240211024435020](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402110244058.png)



# LEC7 SCC

![image-20240211030823207](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402110308322.png)

![image-20240211030844310](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402110308340.png)

![image-20240211031218139](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402110312182.png)

DAG, Topological Sort

![image-20240211031402876](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402110314924.png)

u,v strongly connected: 有路从u->v，也有路从v->u

![image-20240211031752348](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402110317390.png) 

DAG -> 无循环

![image-20240211031917347](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402110319387.png)

![image-20240211201436946](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112014107.png)

to find SCC of Graph G; sink-SCC是指某个SCC不指向其他SCC

![image-20240211202606185](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112026296.png)

![image-20240211202627907](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112026947.png)

![image-20240211202649593](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112026663.png)

flip all edges, strongly connected component stay the same, sink <-> source, thus finding sink-SCC

component元素最高post作为component的post，降序排列 (以component为单位做 Topological Sort)

![image-20240211202810408](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112028463.png)

boring proof

![image-20240211205453303](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112054442.png)

![image-20240211205539911](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112055963.png)

算法逻辑

![image-20240211205722191](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112057257.png)

![image-20240211205758687](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112057724.png)

最终算法

![image-20240211211058433](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112110485.png)

![image-20240211211107427](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112111500.png)

example (BFS写错了，应该是DFS)

![image-20240211205824931](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112058988.png)

按照DAEFCB在原graph中DFS，D是一个SCC，ABC是一个SCC，EF是一个SCC

![image-20240211210802311](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112108354.png)

![image-20240211205938770](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112059812.png)

去掉了source，剩下的第一个仍旧是剩余所有的source

![image-20240211211639748](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402112116829.png)



example2

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402162346216.png" alt="image-20240216234625127" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402162347253.png" alt="image-20240216234744219" style="zoom:25%;" />

(BFS打错了 应该是DFS)

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402162349999.png)





# LEC8 BFS, Dijkstra

Def: dist(u, v) is length of shortest path between u and v

![image-20240217001312329](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402170013367.png)

BFS只可用于 unit length

![image-20240217013432176](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402170134254.png)

u是任意一点，s是source，使用queue

BFS running time: O(|V|+|E|)  (turn queue into stack -> DFS)

![image-20240217065900690](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402170659759.png)

example 队列右入左出

![image-20240217065928185](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402170659226.png)



Dijkstra; k -> known, u -> unknown

![image-20240217073705821](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402170737891.png)

![image-20240217073920908](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402170739948.png)

![image-20240217074621648](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402170746692.png)

![image-20240217074655464](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402170746508.png)

优先队列实现

![image-20240217151942567](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402171519643.png)

decrease-key类似于最小堆中重新排序，小的元素上移

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402171556201.png" alt="image-20240217155643177" style="zoom:50%;" />

Dijkstra ![image-20240219213008099](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402192130204.png)

![image-20240217155628319](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402171556384.png)

![image-20240217162955471](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402171630932.png)



# LEC9 Bellman-Ford, Greedy(Schedule, Huffman Code)

Dijkstra does not work with negative path

d表示距离，dist表示最近距离；property2: 如果u是v前一点且u是最短路，则更新后v是最短路

![image-20240218234005310](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402182340464.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190241261.png" alt="image-20240219024144165" style="zoom:50%;" />

![image-20240219025754620](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190257776.png)

![image-20240219025901301](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190259337.png)

![image-20240219025911967](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190259003.png)

存在negative cycle时无法找到最小边

![image-20240219035548517](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190355566.png)

boring proof

![image-20240219025947446](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190259495.png)

![image-20240219030002195](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190300232.png)



SSSP -> single source shortest path

![image-20240219032028517](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190320621.png)

Boring proof

![image-20240219032044693](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190320731.png)

![image-20240219032056731](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190320773.png)



greedy algorithm

![image-20240219033548015](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190335064.png)

1. scheduling

表格中是举出某种strategy不work的反例

![image-20240219033609768](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190336805.png)

proof技巧：exchange proof 假设存在一个最优解，然后通过与贪心算法得到的解进行交换，证明交换后得到的解仍然是最优解，从而推导出贪心算法得到的解也是最优解？？？？？？？？？？？？

![image-20240219160222839](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402191602949.png)

![image-20240219033852492](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190338535.png)

![image-20240219033914539](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190339578.png)

2. 压缩问题 (huffman code)，一共四个字母，每个2bit编码

![image-20240219040737146](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190407193.png)

Code2会出错，前缀不可相同

![image-20240219040900887](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190409928.png)

code3 -> binary tree; codeword on leaves <-> prefix-free

![image-20240219041309530](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402190413604.png)



# LEC10 Greedy (MST, Kruskal, Prim)

![image-20240224135824455](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241358540.png)

继续上一课

![image-20240224141304078](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241413154.png)

smallest freq -> longest codeword/largest depth

when you encounter the letter A' means you either have A or B, A' is added to list (literally or)

![image-20240224142541432](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241425475.png)

$f_A$和$f_B$在高度为k的位置，$f_A+f_B$在高度为k-1的位置，二者的cost之差是$f_A*k+f_B*k-(f_A+f_B)*(k-1)=f_A+f_B$

![image-20240224142611588](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241426628.png)

![image-20240224143010376](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241430409.png)

cost = 130 = 图中所有标注数字之和

![image-20240226013006427](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402260130464.png)

使用priority queue (binary heap)

deletemin删除最小的，k序号从n+1到2n-1 

![image-20240224143440260](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241434322.png)

boring proof

Claim1: 最小的f必定在树最下方

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241458661.png" alt="image-20240224145818530" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241458029.png" alt="image-20240224145840005" style="zoom:25%;" />



3. MST

Kruskal $O(|E|\log|V|)$

edge 上的值可以是weight/cost，取决于实际应用场景

删除没必要的边，得到MST (MST都是无向图)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402260129577.png" alt="image-20240226012915487" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241508109.png" alt="image-20240224150824082" style="zoom:42%;" />

3 claim

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241509667.png" alt="image-20240224150915636" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241509797.png" alt="image-20240224150931771" style="zoom:20%;" />

算法

![image-20240224151025975](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241510002.png)

boring proof

assume some set of edges X is part of optimal solutionMST, X是图中黑边

红边/蓝边使两块X连通成为spanning tree

cut property的定义

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241514026.png" alt="image-20240224151421936" style="zoom:40%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241518812.png" alt="image-20240224151857770" style="zoom:43%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241519792.png" alt="image-20240224151911769" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241521168.png" alt="image-20240224152118135" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241521604.png" alt="image-20240224152140574" style="zoom:20%;" />



Implementation

cc -> connected component

Disjoint Set 并查集 不相交集合

$m \leq n^2$ 因此 O(mlogm) -> O(mlogn)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241543371.png" alt="image-20240224154325286" style="zoom:50%;" />

![image-20240224160222434](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241602567.png)



另一种使用cut property的算法

![image-20240224160706282](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241607354.png)

Prim algorithm $O(|E|\log|V|)$

从某一点开始，选择现在选中的vertex相连的edge最小的vertex加入到MST中，并选中新vertex，类似dijkstra

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241719676.png" alt="image-20240224171934586" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241720965.png" alt="image-20240224172005928" style="zoom:25%;" />

![image-20240224161752245](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241617287.png)

![image-20240224162022075](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241620120.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402241622461.png" alt="image-20240224162200418" style="zoom:50%;" />

![image-20240228150559032](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402281505136.png)

# LEC11 Greedy (Union Find, Horn Formula)

find函数相当于不断往上找，直到找到根节点

union函数，如果属于同一集合，return；小树成为大树的儿子；如果两树rank不等，合并后rank不用调整；如果两树rank相等，合并后rank+1

![image-20240225043821899](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402250438006.png)

![image-20240225044649980](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402250446072.png)

两树rank不等，合并后rank不变；两树rank相等，合并后rank+1 -> 只有两个等高树union才能得到rank+1；rank=1至少2个node，递归2^k-1^+2^k-1^=2^k^

run time O(rank) = O(logn)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402250500037.png" alt="image-20240225050013955" style="zoom:45%;" />



4.  Horn Formula

to be happy you need ... is a SAT <img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402250511477.png" alt="image-20240225051107406" style="zoom:25%;" /> , 任何SAT都可以表示成[clause(条款，一些Literal相或)]相与的形式，即CNF

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402250517364.png" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402250517184.png" alt="image-20240225051736152" style="zoom:30%;" />

判断每个literal的True/False使得F True (to be happy)

2-SAT, x False -> y must true; y False -> x  must true; 具体如何reduce to SCC见hw

![image-20240225052645366](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402250526408.png)

horn set最多一个positive literal -> 两种可能：

- 全negative (pure neg)
- 仅一个positive (implication)

![image-20240304000227168](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040002268.png)

![image-20240304000317605](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040003645.png)

优先满足implication，再去看pure neg

![image-20240304000414049](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040004094.png)

以下例子，四个implication

如果没有最后一个![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040027237.png)，那么全False就能满足前三个clause了；![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040028306.png)表示x必然是True；W is true (写错了 )

![image-20240303144933127](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403031449158.png)

![image-20240303144950500](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403031449558.png)

Boring proof

![image-20240226031936633](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402260319670.png)

![image-20240226031945970](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402260319000.png)

O(|F|^2)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402260320022.png" alt="image-20240226032012989" style="zoom:50%;" />

转化为图算法 O(|F|)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402260320582.png" alt="image-20240226032052549" style="zoom:50%;" />

总结

![image-20240226032107891](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402260321920.png)



# LEC12 DP (longest path, longest increasing sequence, edit distance)

![image-20240304213628578](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403042136684.png)

Dynamic Programming: an algorithm design principle

1. longest path (many maximum/minimum problem can be transfered into this)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040203221.png" alt="image-20240304020331087" style="zoom:50%;" />

Step1: write recursive relation

![image-20240304020956267](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040209307.png)

![image-20240304021206325](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040212382.png)

DAG: 只单向前进

![image-20240304021513974](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040215015.png)

Example: 最长的到节点5的路径 (斐波那契数列，exponential时间)

![image-20240304021906191](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040219265.png)

存储程序子call

写出一共9个subproblem，每个节点各是一个；写出依赖关系；写出计算顺序

![image-20240304022026797](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040220837.png)

![image-20240304022331010](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040223070.png)

拓展到general problem

![image-20240304022401803](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040224853.png)

every vertex is a step, maximum looks all edges

![image-20240304023008691](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040230742.png)

总结

![image-20240316150551019](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161506503.png)



2. longest increasing sequence

![image-20240304023254615](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040232679.png)

总结

![image-20240316172049439](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161720513.png)



3. edit distance

SUNNY -> SNOWY: 法1 substitute中间三个字母；法2 delete U, N -> O,  insert W

![image-20240304023840721](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040241450.png)

- come from: keep/insert/substitute
- go to: keep/delete/substitute

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040312803.png" alt="image-20240304031204640" style="zoom:50%;" />

![image-20240304032108546](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040321649.png)

![image-20240316150925678](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161509721.png)

![image-20240304033735720](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040337855.png)

edit distance between prefix of the original word

![image-20240304034059892](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040340986.png)

如果最后一步分别是keep/insert/delete，上一步分别是什么

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403041458088.png" alt="image-20240304145844055" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403041502753.png" alt="image-20240304150200716" style="zoom:50%;" />

generally

第三个是keep/substitute，diff是0/1

![image-20240304150151663](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403041501790.png)

![image-20240316151017590](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161510641.png)



# LEC13,14 DP (knapsack w/wo replacement, SSSC, all pairs shortest path, TSP)

按矩阵中蓝色箭头的方向进行是一种可行的order (只是可行方法之一)

![image-20240316151341031](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161513102.png)

按行顺序space可以优化为O(n)，只需要存储正在工作的那一行和上一行 (按列顺序space是O(m))

![image-20240316151645515](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161516580.png)

parallel -> 时间复杂度也可以降到linear (见下方总结图)

store all the edits, 以下表格中的数字都是cost数值，cost只能增加不能减少，因此右下角的3只能由keep得到；如果想要知道all the edits method，需要O(mn)空间

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161529080.png)

总结

![image-20240316172219465](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161722512.png)

![image-20240316172231280](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161722311.png)

DP remarks

![image-20240316154410488](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161544621.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161556379.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310026049.png" alt="image-20240331002645984" style="zoom:80%;" />



4. knapsack

 在不超过weight limit情况下达到最大value；repetition重复，replacement放回；36/38 typo -> 46/48

![image-20240316160222979](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161602080.png)

(1) with replacement

![image-20240316162748262](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161627381.png)

w is a integer number, store w takes log(w) bits, runtime is exponential in log(w), really slow and can't optimize

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161712857.png" alt="image-20240316171216753" style="zoom:50%;" />

![image-20240316162807006](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161628062.png)

总结

![image-20240316173340218](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161733307.png)

![image-20240316174021090](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161740218.png)

(2) without replacement

![image-20240316175453945](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161754085.png)

 if后面typo: v -> w

![image-20240316175511623](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161755659.png)

O(n+W): W指只要存储正在工作的两行，n是要存储input vector v和w

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161815687.png)

better implementation

![image-20240331002844636](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310028706.png)



5. SSSC

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161857277.png" alt="image-20240316185710205" style="zoom:25%;" />

subproblem: 使用k edge的shortest path

![image-20240316183514017](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161835074.png)

algorithm第二行typo: v -> source

类似Bellman-Ford (Bellman如果选的顺序好的话会比这个算法快，Bellman最差的情况(跑满n-1轮)和这个算法时间一致)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161925657.png" alt="image-20240316192508587" style="zoom:67%;" />这句话指的是所有以v结束的edge

runtime: 一般|E|>>|V|

![image-20240316185922039](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161859131.png)



6. all pairs shortest path (not single source)

从vertex i到vertex j，允许经过中间节点1, 节点2,...,节点k

![image-20240316194954913](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161949955.png)

Case2 k 分开的两段都只经过了中间节点1, 节点2,...,节点k-1

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161950787.png" alt="image-20240316195006749" style="zoom:33%;" />



7.  TSP

sterling formula ![image-20240316203238449](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403162032521.png)

![image-20240316195709292](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403161957371.png)



# LEC15 DP (independent set, chain matrix multiplication)

![image-20240317003715533](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403170037592.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403170218328.png" alt="image-20240317021840196" style="zoom:50%;" />

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403170219488.png)



8. independent set

![image-20240329225930396](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403292259583.png)

![image-20240329230336903](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403292303025.png)

tree -> delete root -> get bunch of trees

Case1: root $\notin$ independent set 

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403302124229.png" alt="image-20240330212447059" style="zoom:60%;" />

Case2 -> grandchildren

![image-20240330213027342](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403302130435.png)

generally

![image-20240330213353599](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403302133649.png)

topo sort O(n), 初始化children O(n), 初始化grandchildren <img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403302150554.png" alt="image-20240330215037451" style="zoom:25%;" />, <img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403302152951.png" alt="image-20240330215200861" style="zoom:25%;" />

![image-20240330213631789](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403302136882.png)

runtime: O(n)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403302153438.png" alt="image-20240330215357393" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403302154032.png" alt="image-20240330215407004" style="zoom:25%;" />



9. chain matrix multiplication

first consider multiply 2 matrix

![image-20240331000530900](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310005021.png)

![image-20240331000544744](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310005787.png)

binary tree representation

![image-20240331000618867](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310006914.png)

must keep original matrix order

![image-20240331000735993](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310007053.png)

parentheses 括号

![image-20240331001052761](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310010870.png)

![image-20240331001400957](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310014994.png)

i = 1,...,n-1; j = 2,...,n  

![image-20240331001414134](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310014163.png)

![image-20240331002343493](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310023636.png)



# LEC16 LP, canonical form, simplex

linear programming

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310045370.png" alt="image-20240331004548271" style="zoom: 30%;" />

![image-20240331004728341](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310047461.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310055632.png" alt="image-20240331005523552" style="zoom:50%;" />表示(class, room) is an edge 

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310048346.png" alt="image-20240331004849231" style="zoom:20%;" />

convex feasible region

![image-20240331010133012](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310101129.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310110207.png" alt="image-20240331011051074" style="zoom: 25%;" />

bouded&non-empty -> optimum must be at vertex 

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310124478.png" alt="image-20240331012405361" style="zoom:20%;" />

dual program gives magic number

one magic number for each constraint -> add to give optimal objective

![image-20240331012437441](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310124484.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310124783.png" alt="image-20240331012449741" style="zoom:25%;" />

General LP

![image-20240331013216249](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310132357.png)

canonical form

![image-20240331013240076](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310132146.png)

![image-20240331013321118](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310133180.png)

(1) $-I$是对角线上全-1；![image-20240331130441600](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311304716.png)上半部分是Ax<=b, 下半部分是-x<=0

(2) x $\in$ R

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310133260.png" alt="image-20240331013346200" style="zoom:15%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310134249.png" alt="image-20240331013404210" style="zoom:22%;" />

canonical form总结

![image-20240331131238787](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311312902.png)

n variable

![image-20240331013628842](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310136969.png)

三个hyperplane (constraint)相交得到vertex

![image-20240331014051030](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310140151.png)

n constraint 相交得到的vertex在feasible region中

![image-20240331014533953](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310145073.png)

Brute force traversal

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310146115.png" alt="image-20240331014602058" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310146933.png" alt="image-20240331014635869" style="zoom:20%;" />

simplex

![image-20240331015717654](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310157795.png)

![image-20240331015729364](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310157409.png)

work in practice but doesn't work in theory

smooth analysis: real world data has noise, when add noise to pure data, with very high prob worst case does't happen

![image-20240331015839123](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310158227.png)

![image-20240331131651799](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311316919.png)

ellipsoid work in theory but doesn't work in practice

interior point method (Berkeley) work in theory and work in practice

but both slower than simplex (change one constraint时gaussian elimination可以从O(n^3^)降到O(n^2^))

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310253505.png" alt="image-20240331025340376" style="zoom:25%;" />

standard form

![image-20240331030357277](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310303397.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310304167.png" alt="image-20240331030414127" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403310304724.png" alt="image-20240331030430676" style="zoom:25%;" />



# LEC17 Max Flow Min Cut

Max Flow: a particular LP which is easier than general LP

左图只用了两条路线，每条flow = 1，总flow = 2；右图用四条路线，每条flow = 1/2，总flow = 2

![image-20240331132354245](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311323343.png)

![image-20240331133221750](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311332873.png)

![image-20240331133334701](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311333775.png)

try greedy first

加一条back edge, weight = fe -> be able to undo mistakes

![image-20240331141458680](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311414830.png)

![image-20240331142032116](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311420257.png)

![image-20240331142043041](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311420127.png)

saturated 饱和的

![image-20240331203309787](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312033919.png)

Max flow = 8

![image-20240331150503445](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311505539.png)

验证

![image-20240331203456041](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312034098.png)

notation: augmenting path

![image-20240331145647248](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311456397.png)

Proof: flow = 8 is optimal

Size of flow <= all capacity of (L,R) cut (from L to R ) -> max flow = min cut

in the example, min cut from s to t is 8

![image-20240331151242242](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311512333.png)

![image-20240331151543481](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311515580.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311531910.png" alt="image-20240331153151753" style="zoom:25%;" />

![image-20240331153216710](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311532771.png)

不可以有在S和L之间来回横跳的edge

![image-20240331153304697](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311533774.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403311533348.png" alt="image-20240331153353220" style="zoom:60%;" />



# LEC18 Bipartite Perfect Matching, Zero-Sum Game, Min-Max Theorem

Bipartite Perfect Matching

give student dorm room, 每个student有一些可能的dorm room set (用edge表示，如有的学生怕吵只能去安静的房间)

红edge是最终选定的perfect matching，黑edge是没用上的

![image-20240331210440642](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312104775.png)

integer flow -> 不会出现1/2+1/2的情况 

one problem reduce to another problem

![image-20240331210904295](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312109419.png)

![image-20240331211304832](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312113999.png)



Max flow <= min cut is actually a very general property in LP, called LP duality

注意，三组magic number只有一组给出了optimal solution

![image-20240331211723669](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312117823.png)

dual LP

![image-20240331220248444](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312202606.png)

general form

![image-20240331220513588](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312205807.png)

weak duality: LP opt <= dual opt

![image-20240331221134302](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312211449.png)

Strong duality: LP opt = dual opt

![image-20240331224111870](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312241983.png)

more general duality: 有一些constraint是![image-20240331224146417](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312241475.png)形式 (如flow in = flow out)，他们可以转换成![image-20240331224217957](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312242021.png)但是那样整体会很乱，因此这里不做转换；Ax = b -> dual -> A^T^y = c, 不要求 yi >= 0 / xj >= 0  (等号不会有不等号乘负数变号的问题)

![image-20240331224125351](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312241416.png)



zero sum game (Berkeley) game theory +CS

I get more, you get less, sum = 0

矩阵里的数值是payoff for the first player/row player (剪刀石头布)

row player play rock, col player play paper -> row palyer -1, col player +1

![image-20240331233108339](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312331536.png)

![image-20240331233224293](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403312332401.png)

1. row player goes first, col player give response 使得自己利益最大化

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404010015479.png" alt="image-20240401001509365" style="zoom:25%;" />

![image-20240401002013445](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404010020542.png)

 (矩阵里的数值是payoff for the first player: row player)

如果col player选col1, row player get 3p1-2p2; 如果col player选col2, row player get -p1+p2

col player will choose the lower one of the two to minimize what row player get

pure strategy: q1,q2 = 0,1/1,0

![image-20240401002055296](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404010020378.png)

row player want to maximize what he get 

maximize min{a,b} -> max x, x<=a, x<=b

![image-20240401002504204](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404010025277.png)

2. Col player goes first

(矩阵里的数值是payoff for the row player)

如果row player选row1, row player get 3q1-q2; 如果row player选row2, row player get -2q1+q2

row player will choose the higher one of the two

col player want to minimize what row player get

![image-20240401002919559](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404010029651.png)

![image-20240401003829831](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404010038893.png)

summary: player其实不想做first player，想晚开始 (for pure strategy, first player先出石头, second player必定出布)

如果是bounded -> strong duality -> =

![image-20240401005651129](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404010056243.png)

just general form of above content

![image-20240401010310636](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404010103759.png)

![image-20240401010321106](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404010103155.png)



# LEC19 P/NP, Reduction

![image-20240506143731292](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405061437414.png)

computational complexity

![image-20240414181514604](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141815776.png)

example

![image-20240414182108284](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141821374.png)

![image-20240414182500289](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141825427.png)

![image-20240414182811482](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141828569.png)

there are problem that are not NP and not P

- the halting problem: given input program, judge whether it has infinite loop
- the counting problem: given a graph, tell the number of the 3-coloring
- game winning strategy: can white win chess no matter how black move?



Factoring: find 2 factor of 1 big number, crypto based on this

![image-20240414183753545](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141837601.png)

RSA: public-key

![image-20240414184228222](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141842395.png)

![image-20240414184454697](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141844777.png)

Min-TSP -> not NP (TSP is not poly time)

![image-20240414185206344](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141852455.png)

Budget-TSP is NP (is a easier variation, can solve Min-TSP -> can solve Budget-TSP)

![image-20240414185423134](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141854211.png)

![image-20240414185710089](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141857140.png)

![image-20240414190825146](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141908255.png)



To compare complexity -> reduction

matching reduce to maxflow, maxflow is at least as hard as matching

![image-20240415150932783](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151509853.png)

![image-20240415164705684](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151647771.png)

Example

Rudrata cycle: hard, nobody has solved

![image-20240415151546790](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151515886.png)

add |V| dummy vertices and no edge (useless for find cycle )

![image-20240415160018734](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151600858.png)



# LEC20 Reduction, NP-complete map

![image-20240415180802378](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151808558.png)

linear programming is poly time, easy; but integer programming is hard, no poly algorithm yet

![image-20240415181846607](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151818743.png)

Green part is intuition

![image-20240415182656420](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151826547.png)

![image-20240415183400134](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151834246.png)

proof: 证明了IND-Set can reduce to LP; not the opposite (the opposite need to show an F: LP -> IND-Set)

![image-20240415184409410](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151844516.png)

![image-20240415184756988](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151847059.png)

![image-20240415185022498](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151850557.png)

In practice, usually no package for weird graph problem like IND-Set, but we can reduce it to integer program and use package



NP-complete: harder than every other problem in NP (as least as hard as)

-> every problem in NP reduce to them

![image-20240415191009116](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151910204.png)

![image-20240506175625357](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405061756485.png)

![image-20240415191955088](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151919193.png)



given any problem you can verify the sol, they can reduce to NP-complete problem (Berkeley)

-> Factoring/RSA can correspond to some graph, you can solve 3-color on the graph, then you can solve factoring/RSA

today there are tens of thousands NP-complete problem in all domains, they don't have a alg unless this there are breakthrough

every NP-complete problem is reducible to all other NP-complete problem, all same complexity

![image-20240415192204734](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151922829.png)

![image-20240415192602873](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151926950.png)

![image-20240415193313073](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151933142.png)

![image-20240506151850879](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405061518983.png)

![image-20240506144523618](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405061519333.png)

3-SAT: a clause has 3 literals

![image-20240415194506924](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151945020.png)

n boolen variable, m clauses

![image-20240415195238800](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151952881.png)

K是IND-Set的bound

![image-20240415195531504](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151955555.png)

![image-20240415195811515](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404151958569.png)

Proof forward:

![image-20240415224900791](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404152249864.png)

3个相或为真 -> 至少1个为真；每个三角仅选一个真 -> independent set (左下角三角形的w非也是真，但是不能选中，否则independent set就不满足，每个三角选exactly one)

![image-20240415224817256](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404152248596.png)

Proof backward:

m triangle & has independent set of size m -> must pick one node in every triangle

![image-20240415225834681](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404152258785.png)

![image-20240415225939205](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404152259306.png)

prove satisfy all constraints

![image-20240415230235979](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404152302086.png)

Notice: there is no one to one corresponding solution between 3SAT & IND-Set solution, just being able to solve one can enable us to solve another



# LEC21 NP-complete map, approximation algo

![image-20240428154137869](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281541035.png)

directed rudrata cycle

![image-20240428154148095](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281541129.png)

两步证明，第二步选择合适的NP-complete problem可以简化证明

![image-20240428154350980](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281543026.png)

review 3-SAT

![image-20240428154600441](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281546494.png)

![image-20240428154627821](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281546878.png)

intuition: x is boolen variable -> find sth in graph has 2 state

There are only 2 way to traverse the graph below: left to right & right to left 

(only these 2 way to traverse, constraint: exactly once)

one boolen variable -> one such graph

![image-20240428155138408](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281551505.png)

n variable; 加上start/end node，加上绿色edge相连

下图的graph一共有2^n^种rudrata cycle

![image-20240428155545157](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281555248.png)

想要到达vertex C，x那一行必须从左向右

xyz任一行到达C即可 -> OR

![image-20240428161234674](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281612785.png)

***

Next: all of NP -> circuit SAT

![image-20240428162020071](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281620152.png)

![image-20240428162122448](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281621494.png)

![image-20240428162232825](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281622917.png)

Factorization -> circuit SAT

NP can verify -> all computation can be done on circuits

![image-20240428162339734](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281623771.png)

Verify algo

![image-20240428163716555](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281637661.png)

verify algo对应的circuit

![image-20240428164054620](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281640658.png)

将N hard code into the circuit, try p & q; if output 1 -> find sol, thus is a circuit SAT problem

[we are not solving factorization, just reduce it to circuit SAT]

![image-20240428164114311](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281641344.png)

***

skip, read the book

![image-20240428165415672](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281654773.png)

![image-20240428165430204](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281654249.png)

![image-20240428165438525](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281654554.png)

***

approximation algo

NP-complete -> we do not have poly time algo for it

but we still have to cope with it

- 放宽限制，如rudrata允许经过每个点2次/n次
- 现实生活中输入量级很小，直接bruteforce
- greedy
- find a approximation to the optimal sol (here)

![image-20240428165453724](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281654772.png)

Def: pick one vertex -> all the edges connected to the vertex are covered

- NP-hard: donot have to be NP, the set of problems that any NP problem can reduce to
- NP-Complete: have to be NP, the set of problems that any NP problem can reduce to

[NP-C is a subset of NP Hard]

![image-20240428170748394](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281707507.png)

maximal (not maximum, do not need to be clever) matching: keep adding edges until add one more will cause vertex overlap (purple edge)

![image-20240428172346515](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281723631.png)

把所有edge涉及的vertex作为output

![image-20240428173312035](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281733153.png)

$\#$ output vertex = 2 * ($\#$ maximal matching edge)

![image-20240428173324826](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281733863.png)



# LEC22 approximation algo

2 more approximation problem

![image-20240428184507565](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281845709.png)

min vertex cover -> LP

黄色部分的intuition在LP中实则无法做到

![image-20240428185421387](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281854480.png)

LP output can be fraction (we don't use integer LP because no efficient algo)

![image-20240428185944247](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281859347.png)

![image-20240428190553770](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281905832.png)

LHS: left hand side

![image-20240428192856859](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281928009.png)

![image-20240428193850581](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281938641.png)

![image-20240428193952194](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281939263.png)

![image-20240428194505909](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281945044.png)

***

Metric TSP: metric指满足下方蓝字assumption (不满足metric就没有approximation algo)

visit exactly once

![image-20240429003506053](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404290035215.png)

![image-20240429003552185](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404290035247.png)

![image-20240429004054750](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404290040807.png)

approximation algo

step3的不等式用到了assumption2

![image-20240429005348995](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404290053068.png)

this is 2-approximation; there exist 1.49-approximation which is STOA



# LEC23 Gradient Descent: Point Pursuit game, solve LP with exp constraint

Point Pursuit game

![image-20240430132802676](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301328881.png)

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301328475.png)

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301328931.png)

![image-20240430142521944](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301425046.png)

![image-20240430161743208](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301617394.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301622247.png" alt="image-20240430162259194" style="zoom:50%;" />

![image-20240430162731584](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301627636.png)

![image-20240502025807976](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405020258241.png)

w is normal vector

![image-20240502030429112](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405020304170.png)

Normalize the normal vector

![image-20240502030337933](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405020303000.png)

![image-20240505002106520](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050021661.png)

***

Application: solve LP with exponential constraint

constraint写成了inner product的形式

没有objective fn, objective fn can be reduced to constraint

[比如max 2x+3y, 可以写成2x+3y>=b, b = 100, 200..., do binary search for objective value]

![image-20240505002404957](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050024124.png)

if Bob output a result satisfying all constraint, Alice lose; else, Alice output a unsatisfied constraint as hyperplane

[Alice donot know feasible region/sol, she just check output of Bob among constraints]

![image-20240505005505058](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050055155.png)

这里objective fn is reduced to constraint, so reach feasible region is win

if after D^2^/$\epsilon^2$, game still go on, there is no feasible region

[in actual LP, we somehow know the upper bound distance D, just donot know whether it is empty]

![image-20240505005950139](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050059202.png)

example

![image-20240505010922118](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050109204.png)

$\#$ constraint is exponential in n, cannot directly guess the feasible region, but can solve with GD

![image-20240505011536922](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050115063.png)

![image-20240505011700005](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050117070.png)



# LEC24 Arbrosence (exp constraint), convex set (infinite constraint)

formally (constraint are normalized)

![image-20240505014214306](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050142406.png)

simplex is not necessarily poly

GD for LP is pseudo poly time, is not necessarily poly, good in practice

![image-20240505014429667](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050144722.png)

go directly towards the violated constraint line

![image-20240505015425528](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050154689.png)

$\epsilon$-approximate means <ai,p^(t)^> $\leq$ bi - $\epsilon$ [in pt pursuit game, Alice come uo with seperating line with tolerance $\epsilon$]

![image-20240505015609252](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050156313.png)

LP-solving algo only need to check and find a violated constraint

![image-20240505021709149](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405050217310.png)

***

Arbrosence def: broadcast msg

![image-20240508213516101](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082135163.png)

Min-cost Arborescence

![image-20240508213459836](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082134977.png)

wt -> weight

Typo: $\sum_{e\in E}w_ex_e\leq B$

At least one edge leave S to $\overline{S}$ -> Arborescence can reach anywhere

![image-20240508214439201](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082144323.png)

这些constraint不可或缺，不然会出现下图反例

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082143981.png" alt="image-20240508214322865" style="zoom:25%;" />

the verifier only need to run the seperation oracle using min-cut algo (on r and any other vertex as t)

![image-20240508214609462](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082146533.png)

![image-20240509033717025](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405090337219.png)

***

LP with infinite constraint

H is halfspace, A convex set is an intersection of a (possibly infinite) family of halfspace

![image-20240508221309078](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082213240.png)

1. non-convex example

![image-20240508221430406](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082217408.png)![image-20240508221950619](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082219691.png)

2. Halfplane

![image-20240508221617942](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082216031.png)

- One halfspcae

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082217818.png" alt="image-20240508221704753" style="zoom:33%;" />

- intersection of 4 halfspcae

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082220559.png" alt="image-20240508222022437" style="zoom:25%;" />

- intersection of infinite halfspcae

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082220051.png" alt="image-20240508222050957" style="zoom:25%;" />

given one pt outside of S, there will be a line (hyperline) seperating that pt from S

![image-20240508222316801](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082223878.png)

![image-20240508222731132](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082227249.png)

![image-20240508222839123](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082228200.png)

convex fn

![image-20240508223102297](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082231412.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082232166.png" alt="image-20240508223237054" style="zoom:15%;" />

绿箭头就是gradient/normal vector

![image-20240508223308400](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082233498.png)

![image-20240508223552941](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405082235032.png)



# LEC25 

both are valid proof that can be verified in finite time, proof 1 is better than proof 2 because it take less time to verify

![image-20240518181910095](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405181819896.png)

proof definition: prover send a string, verifier check it in poly time and decide to accept/reject

![image-20240518182523506](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405181825581.png)

prover can prove 3-colorable by just sending the sol to verifier; but it is difficult to prove "not 3-colorable"

Smart & powerful guy send the proof, verifier who is limited & not smart can easily check

![image-20240518183315170](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405181833233.png)

***

math guy dont use interactive proof, but it is common in cs

![image-20240518183449663](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405181834716.png)

verifier can be randomized

![image-20240518183658433](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405181836492.png)

![image-20240518185348151](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405181853262.png)

application of interactive proof

1. ZKP

commitment & ZKP for 3-coloring 略

2. prove "G is not 3-colorable" or other complement of NP
3. can white always win in chess? (completely solve a game) (poly space verifible statement)

[prove there is a sol to 3-coloring can just send sol; prove not 3-colorable can try all possible sol and say no one satisfy; prove white can always win in chess is the most difficult (try all possible white move in every step)]

***

typically we need to read every bit of the proof to verify it; check the correctness of the proof by reading only 3 bit

![image-20240518194036549](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405181941135.png)

![image-20240518194439197](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405181944265.png)

***

2 prover that cannot communicate with each other

for statement have exp long proof, verify in poly time 

![image-20240518194917614](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405181949681.png)

***

you can prove things you cannot even compute

![image-20240518195026470](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405181950550.png)

***

Halting problem

![image-20240518195209138](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405181952198.png)





# HW2

Question 6 

GF(5): 包含元素0、1、2、3 和 4的有限域，3 + 3 = 1 (因为 3 + 3 = 6，取模5等于1)，3 * 3 = 4 (因为 3 * 3 = 9，取模5等于4)

unity = 1；4次单位根（4th roots of unity）：x^4^ = 1

![image-20240205004426122](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402050044183.png)

![image-20240205004905195](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402050049258.png)

-1次方：模逆元（modular inverse）：对于给定的元素a，存在一个元素b，使得它们的乘积模5等于1

![image-20240205174455899](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402051744961.png)

在GF(5)中，+4和-1是等价的

![image-20240205173103041](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402051732906.png)
