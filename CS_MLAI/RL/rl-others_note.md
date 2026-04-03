# [NTU-DL-22sp](https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php) LEC12

![image-20251105221258212](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532942.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532793.png" alt="image-20251105221412621" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511052215041.png" alt="image-20251105221531996" style="zoom:25%;" />

早期的policy并不是network而是一个look up table

use prob for each action to do sampling (70% do left), this is for exploration

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511052219909.png" style="zoom:25%;" />

![image-20251105231304264](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532216.png)

terminology：游戏开始到结束称为一个episode，得到整场游戏的total reward

单个行动得到reward，整场游戏得到return=total reward

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511052223704.png" alt="image-20251105222340658" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511052224014.png" alt="image-20251105222443968" style="zoom:20%;" />

terminology: trajectory / $\tau$ is a sequence of s and a

reward depend on both action and observation, e.g., only when action is fire can the model get reward

看起来像RNN，但是不一样在于输出a是sample得到的，有随机性；env and reward are also random

![image-20251105222743319](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532166.png)

policy gradient

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511052239611.png" alt="image-20251105223944531" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511052240880.png" alt="image-20251105224002819" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511052241482.png" alt="image-20251105224136392" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532088.png" alt="image-20251105224219108" style="zoom:25%;" />

不正确的版本，急功近利的actor没有长远眼光

![image-20251105224617984](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532310.png)![image-20251105224935329](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532911.png)

cumulated reward：a1的reward取决于之后所有发生的事情的reward

问题：如果游戏很长，rN和r1好像没什么关系 -> 引入discounter factor

![image-20251105225331037](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511052253125.png) -> ![image-20251105225640547](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511052256629.png)

normalize the reward value, e.g., -b (baseline value)

![image-20251105225813317](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532717.png)

A is reward; collecting data is expensive and time-consuming

![image-20251105230106193](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532363.png)![image-20251105230335374](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532060.png)

off policy can save the data collecion cost in for loop；off policy 要意识到自己不是和环境interact的那个人

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511052307323.png" alt="image-20251105230719271" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511052310275.png" alt="image-20251105231047212" style="zoom:30%;" />



Actor Critic

G1‘ is discounted cumulated reward; 得到G1‘本来要玩完游戏，value fn用来没玩完游戏就预测G1‘，$V^\theta$的$\theta$表示value fn在观察参数为$\theta$的actor，同样的s，不同的$\theta$应该得到不同的value fn输出

![image-20251106002229570](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532466.png)

训练critic

![image-20251106003043944](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532269.png)

另一种方法，只需要一个t的数据就可以训练，训练$V^\theta(s_t) - \gamma V^\theta(s_{t+1})$

![image-20251106003212432](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532934.png)

2种方法在同样数据训练结果可能不一样，例子当中MC结果是0，但是TD结果是3/4（TD式子中的r是ra）

![image-20251106003631377](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532145.png)

set normalization term b to be critic

![image-20251106004252804](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511060042870.png)![image-20251106004325569](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532426.png)

value fn output is like an expectation value, 图中G是cumulative reward (没有discount，discount之后notation是G'，现实中是用discount，这里只是demonstration需要), average G is V

![image-20251106004447847](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511060044894.png)

常用的Advantage actor-critic reward

rt is the actual reward after at; At解释为采取at这个行动得到的平均reward和不采取at得到的平均reward的差值，加上immediate reward rt

![image-20251106005945341](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511060059450.png)

可以共用参数

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532213.png" alt="image-20251106011522592" style="zoom:25%;" />

DQN

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532081.png" alt="image-20251106011613294" style="zoom:25%;" />



Sparse reward，如下围棋一局结束才有reward；如机械臂拧螺丝，随机初始化之后乱动，无法拧紧螺丝获得reward

reward shaping就是设计其他细小的reward，避免只看结果

![image-20251106143420806](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532922.png)

打设计游戏，活着就扣分，强迫agent去杀敌人

![image-20251106144315269](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061443396.png)

Curiosity based reward，不给sparse reward（如通关游戏加分），只说让机器探索到有意义的新东西就给reward，即可通过部分关卡

![image-20251106144656545](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532331.png)

No reward；人定的reward会被hack

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532887.png" alt="image-20251106145128041" style="zoom:25%;" />

记录人类（expert）和环境互动作为示范

![image-20251106145354960](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532819.png)

可能没见过失败情况；可能无法区分人类的个人特质（无需学习）和普遍行为（需要学习）

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061455819.png" alt="image-20251106145551769" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061457777.png" alt="image-20251106145751717" style="zoom:25%;" />

inverse RL: 用expert行为学reward fn

![image-20251106150001540](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532384.png)

假设老师的行为可以取得最高reward（注意，不是完全模仿老师行为） ；similar to GAN

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532935.png" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061503913.png" alt="image-20251106150342863" style="zoom:25%;" />

robot IRL

![image-20251106151542418](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061515517.png)![image-20251106151534905](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061515009.png)



math of policy gradient

![image-20251106152135879](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532933.png)

Total reward is random, we treat it as random variable and maximize the expected value

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061523915.png" alt="image-20251106152341842" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061815171.png" alt="image-20251106181512065" style="zoom:25%;" />

$R(\tau)$ does not depend on $\theta$

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061816022.png" alt="image-20251106181638915" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532507.png" alt="image-20251106181911722" style="zoom:25%;" />

$p(r_1, s_2|s_1,a_1)$ depends on the environment

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071532591.png" alt="image-20251106182125745" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061822138.png" alt="image-20251106182203042" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061825161.png" alt="image-20251106182503065" style="zoom:20%;" />

除p起到normalization作用；假设4个trajectory看见了同一个observation s，几率大的action b reward小，不normalize会倾向action b

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061829786.png" alt="image-20251106182934722" style="zoom:25%;" />![image-20251106183403572](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071533089.png)



PPO

importance sampling: we cannot sample from distri p, we can only sample from distri q

![image-20251106184458674](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071533067.png)

p和q不能差太多；需要sample很多次，sample到negative的绿色点才行

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061849835.png" alt="image-20251106184948777" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061854038.png" alt="image-20251106185452980" style="zoom:25%;" />

训练$\theta$，$\theta'$只负责和环境互动

![image-20251106185934181](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071533032.png)

A is advantage, suppose $p_\theta(s_t) = p_{\theta'}(s_t)$; J is objective fn

![image-20251106190517447](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071533101.png)

为了让$\theta$和$\theta'$不要差太多，PPO add a KL divergence as constraint (TRPO set a separate constraint)；KL divergence算的是input同样一个state，output的action背后的prob distribution的差异

![image-20251106191021033](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061910109.png)

![image-20251106191637625](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061916746.png)

PPO2：选两项中小的进行优化（min(第一项，第二项)）

第二项的clip项是蓝线，第一项的![image-20251106192123183](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071533468.png)是绿线，取最小是红线，再乘![image-20251106192217244](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511061922321.png)

目标是让$p_\theta$和$p_{\theta'}$不要差太大

- 如果A>0，reward是好的，希望$p_\theta$越大越好，但是和$ p_{\theta'}$差距不能大过$1+\epsilon$

![image-20251106192322962](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071533210.png)



# [DPO](https://www.youtube.com/watch?v=k2pD3k1485A&t=68s)

PPO optimizes a continuous scalar reward, DPO uses pairwise comparisons

DPO is to turn reward into prob

$\pi_{ref}$ is the SFTed model

![image-20251106225505958](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511062255067.png)

To deal with negative reward, use exp, which turns into sigmoid

![image-20251106230017265](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511062300431.png)

![image-20251106230125977](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071533194.png)

![image-20251106230613830](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071533305.png)

![image-20251106230834211](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602071533769.png)

