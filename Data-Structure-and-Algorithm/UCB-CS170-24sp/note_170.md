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



# LEC10 Greedy(MST, Kruskal, Prim)

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

# LEC11 Greedy(Union Find, Horn Formula)

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



# LEC12

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

2. longest increasing sequence

![image-20240304023254615](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040232679.png)

3. edit distance

SUNNY -> SNOWY: 法1 substitute中间三个字母；法2 delete U, N -> O,  insert W

![image-20240304023840721](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040241450.png)

- come from: keep/insert/substitute
- go to: keep/delete/substitute

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040312803.png" alt="image-20240304031204640" style="zoom:50%;" />

![image-20240304032108546](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040321649.png)

![image-20240304033735720](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040337855.png)

edit distance between prefix of the original word

![image-20240304034059892](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040340986.png)

如果最后一步分别是keep/insert/delete，上一步分别是什么

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403041458088.png" alt="image-20240304145844055" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403041502753.png" alt="image-20240304150200716" style="zoom:50%;" />

generally

第三个是keep/substitute，diff是0/1

![image-20240304150151663](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403041501790.png)



