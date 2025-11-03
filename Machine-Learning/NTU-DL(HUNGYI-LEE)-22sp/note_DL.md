 

# 入门

structured learning 生成式ai

y是机率，用cross entropy

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132007384.png" alt="image-20231013200713355" style="zoom: 25%;" />

error surface，暖色系loss大，冷色系loss小

error surface维度由网络参数决定，有几个w，b就是几维

local minimum不是梯度下降的主要痛点

观察到播放人数以7天为一个周期，更改模，这类模型叫linear model

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132015884.png" alt="image-20231013201551850" style="zoom: 25%;" />

模型的复杂度缺陷叫model bias

piecewise linear函数（线段组成）都可以由多个蓝色的折线函数相加得到

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132022359.png" alt="image-20231013202228317" style="zoom:25%;" />

蓝色的折线函数即是sigmoid

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132024876.png" alt="image-20231013202419827" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132029475.png" alt="image-20231013202941410" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132026019.png" alt="image-20231013202608982" style="zoom:25%;" />

就是mlp

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132027244.png" alt="image-20231013202759208" style="zoom: 25%;" />

所有w，b等统称为$\theta$，$\theta$求导($\theta$ = $\theta$0)就是g，$\theta$1 = $\theta$0 - lr * g

用batch时，一个epoch有很多update

hard sigmoid实际上是两个relu叠加，因此relu通常需要相比于sigmoid两倍多的神经元

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132043195.png" alt="image-20231013204316159" style="zoom:25%;" />

为什么是竖着不是横着，为什么是deep不是fat？

lec 1-5 监督学习

lec 7 self-supervised 先pre-train，又叫foundation model（喂大量未标注资料），典型是Bert；然后投入到下游任务downstream task

lec6 生成式模型，搜索大量的x和y，他们是unpaired，未配对，以此训练（Facebook 语音识别，非监督比监督效果好）

lec12 RL，不知道怎么标注label的时候，只能判断好坏（人类也说不准怎么下棋最好）

lec8 异常检测，让机器对没见过的样本回答我不知道（如宝可梦网络见到真实动物照片）

lec9 explainable AI 判断结果同时说出为什么

lec10 model attack 原本一个测试样本被分类为cat，在图片中加一点肉眼不可见的杂绪，机器就不认识了

lec11 domain adaption  用黑白手写数字训练的模型，在彩色测试样本上正确率暴跌

lec13 模型压缩 跑在算力有限的小设备上

lec14 lifelong learning 机器不断学习统治人类

lec15 meta learning 学习如何学习，面对任务自己发明演算法，基于meta learning才能实现 few-shot learning（在少量标注资料上学习）

DL历史 1958没有激活函数，论文被拒魔咒，RBM理论深厚但没用，石头汤

![image-20231014004000828](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310140040909.png)

input layer 没有神经元；output layer 有神经元 分类则包含softmax

在语音识别和图像识别deep learning很强，因为人类在这类问题中提取feature的能力很弱，不如神经网络自己提取

在NLP，深度学习的优势并不夸张，因为人类NLP的能力很强

自动找出network structure的方法有了但还并不普及

universality theorem 一层hidden layer的神经网络可以表示任何function

back propagation 利用chain rule高效计算梯度



训练样本 iid（independently and identically distributed）；在子集loss比全集低很正常

![image-20231016102736929](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161027962.png)

能找到all训练集就找all，找不到才用test集

因此想要模型泛化能力强，要满足训练集和全集分布类似

![image-20231016105703083](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161057117.png)

那么分到不好的训练集的概率有多大？![image-20231016121133558](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161211604.png)

搜数据很难，找N很难，但是用复杂度降低的H，又是自欺欺人

![image-20231016111639015](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161116047.png)

如何达到两个small？使用深度学习

![image-20231016113910198](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161139236.png)

以上推导见手写笔记“generalization analysis”

![image-20231212200006223](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312122000269.png)

 由于累加 P往往大于1，所以没什么用 ，上界往往比实际大得多

![image-20231016112116749](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161121788.png)

模型越复杂，需要的训练资料越多

![image-20231016112231132](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161122169.png)

![image-20231016113314653](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161133706.png)

为什么用了validation set还是overfit？如三个模型选一个，Hval = 3，P很小；但是待选择的模型太多了，还是很容易overfit

![image-20231016140434091](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161404176.png)

如何达到两个small？H小，其中是少量精英模型，并且他们都能使得loss降低

![image-20231016141131440](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161411483.png)

why deep？实验中deep 比 fat效果更好，deep可以使用更少的参数得到更好的效果，参数少意味着overfit的概率更低，需要的训练样本更少

类比数电，2层数字电路可以表示任何function ，2层需要2^n个逻辑门，但层数多会简洁很多

固定神经网络架构

![image-20231016152313340](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161523377.png)

1层，2层，3层时，线段条数成指数增长

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161518938.png" alt="image-20231016151852748" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161520330.png" alt="image-20231016152024287" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161520872.png" alt="image-20231016152051807" style="zoom:25%;" />

每层2个神经元，k层，就可以模拟2\^k线段的函数，即2k个神经元，用fat网络则要2\^个神经元，因此deep的H更小，不容易overfit，且需要更少的训练数据 

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161524244.png)

就算function并不复杂，deep也有优势，fat网络需要指数倍多的参数来达到和deep一样的效果

![image-20231016153008815](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310161530857.png)



# 调参

![image-20231007140543191](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071405219.png)

trainloss大：1，使模型更复杂，增加feature或layer和neuron ；2，优化问题 （见下方）

判断是1还是2，尝试不同复杂度的model

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071147816.png" alt="image-20231007114720757" style="zoom:33%;" />

先train比较浅层或者非ml模型（如svm），他们一般不会有优化问题

![image-20231007115021227](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071150256.png)

overfit： 最有效的是增加训练数据集；data augmentation（处理后的图片要合理）；减少模型复杂度（共用参数CNN，减少feature，earlystopping，regularization，dropout）

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071202255.png" alt="image-20231007120210221" style="zoom: 33%;" />

在public testset上结果经常超越人类，在陌生的private testset上往往就遭，使用验证集选model，这样publictest就相当于privatetest了，不要由publictest的结果再回去调整model，不然验证集就失去了意义，把publictest当作privatetest

![image-20231007121409303](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071214339.png)

更好的validation方法，切成N份

![image-20231007140524057](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071405092.png)

mismatch 训练和测试分布不同，训练再多都没用

遇到优化问题：说local minimum很没水准，说卡在critical point；判断local minimum还是saddle point

![image-20231007141403894](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071414924.png)

误差函数在整体写不出来表达式，在某一点附近写得出来

![image-20231007141752327](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071417359.png)

在critical point一阶导为0

![image-20231007141927038](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071419072.png)

通过特征值判断三种类型

![image-20231007142331236](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071423267.png)

举例，左上右下高，左下右上低

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071426423.png" alt="image-20231007142600391" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071429588.png" alt="image-20231007142914548" style="zoom: 33%;" />

如果是saddle point问题不大，g=0但是可以看H矩阵（二阶导）， 沿着负的特征值对 应的特征向量方向变化即可

![image-20231007143749454](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071437494.png)

举例，但是实际情况不会用这个方法逃离saddle point，因为计算量过大

![image-20231007143923139](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071439169.png)

三体3君士坦丁堡魔法师迪奥伦纳，三维空间封闭的东西在高维空间是有路可走的

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071443701.png" alt="image-20231007144347663" style="zoom:25%;" />

现代的model的error surface都是极高维度，几乎没有local minimum

![image-20231007144734113](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071447150.png)

处理卡在平坦点的办法：batch和momentum

batch：shuffle，每个epoch开始时重新分batch，1个epoch指把所有batch都训练一遍。batch可以帮助模型达到更好的结果，小的batchsize可以解决卡在平坦处的问题，也可以在test获得更好的结果（考虑平行计算，batch并不会花更多时间）

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071501084.png" alt="image-20231007150152049" style="zoom:25%;" />

平坦的minimum是好的minimum，有更好的泛化结果

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071510843.png" alt="image-20231007151013804" style="zoom: 25%;" />

结合大小batchsize的优势

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071512532.png" alt="image-20231007151240495" style="zoom:25%;" />

momentum：考虑小球动量，不卡在平坦点

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071522920.png" alt="image-20231007152216877" style="zoom:25%;" />

综合考虑过去计算出的所有gradient

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071518733.png" alt=" " style="zoom: 25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071520098.png" alt="image-20231007152023057" style="zoom:25%;" />

自动调整学习率，有时loss很平稳了，但是梯度值还在跳动，这时候也不叫卡在平坦处。这是大多数情况loss卡住的，其实很难训练到critical point

![image-20231007152757802](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071527868.png)

学习率10^-2，振荡；10-7，小步下降但是到不了local minimum（x点），因为下面太平坦了；gradient decent是解决复杂error surface的唯一方法，主要问题在于学习率自适应变化

![image-20231007153542317](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071535348.png)

又称Adagrad，陡峭调小，平坦调大

![image-20231008081727400](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080817462.png)

传奇方法RMSprop，没有论文，要citeHinton的coursera，alpha是新的超参数，要调；设置alpha小，新的g更重要，相比Adagrad，调整的更灵敏 

![image-20231008082247829](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080822864.png)

![image-20231008082509790](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080825824.png)

最常用的optimization的方法Adam

![image-20231008082852869](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080828981.png)

到平坦的地方步子变大会暴走，但是很快会修正回来

![image-20231008084028126](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080840170.png)

解决暴走问题，两种Learning Rate Scheduling

![image-20231008084428908](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080844944.png)![image-20231008085502898](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080855931.png)

最终版本

![image-20231008085639217](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080856249.png)

更多optimization相关

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080856384.png)

接下来，分类问题中使用cross entropy函数直接夷平error surface

![image-20231008085913697](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080859732.png)







# Normalization

![image-20231115163842367](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151638524.png)

![image-20231115164029627](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151640682.png)

网络中间的输出也要做normalization，sigmoid在激励函数之前做normalization，其他放在激励函数之前之后无差别

原本z1改变，只有a1改变，做归一化后z1改变，所有a都改变，相当于一个大网络，由于GPU无法load进所有数据，所以一般只在一个batch内做，即batch normalization。因此使用batch normalization时不能设过小的batch 

![image-20231115164741179](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151647236.png)

有人怀疑让所有网络输出平均为1可能影响网络性能，因此加入 $\gamma$ 和 $\beta$ 两个vector，初始化为1和0，是可学习参数 

![image-20231115165210813](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151652867.png)

以上是训练部分，测试部分进来一笔资料就要预测出结果，没有batch的概念；pytorch会自记录平均的 $\mu$ 和 $\sigma$ 值用于测试，超参数p默认0.1

![image-20231115165720741](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151657825.png)

CNN做batch normalization，看原文献

![image-20231115165853301](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151658351.png)

原作者认为BN有用是因为 internal covariate shift，被1805.11604驳论

![image-20231115170204849](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151702902.png)

该文章理论+实验证明了BN有用，但是其他方法也可以做到夷平error surface，甚至效果更好；serendipitous 偶然的

![image-20231115170419391](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151704461.png)

![image-20231115170537516](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151705587.png)









# Classification

长时间版本的分类

![image-20231008090414835](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080904875.png)

回归+softmax得到0-1即分类，softmax的输入叫logit；分两类时sigmoid和softmax一致；分类的loss用cross entropy；pytorch找不到softmax函数，因为softmax和corss entropy绑定，使用cross entropy loss时自动在model里加入softmax函数 

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080912728.png)

MSE在loss很大的地方很平坦，很容易卡住，crossentropy有类似夷平error surface的效果

![image-20231008091617387](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310080916428.png)







# beyond Adam

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191455807.png" alt="image-20231019145455939" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191455350.png" alt="image-20231019145525308" style="zoom:25%;" />

momentum 记录了从0-t 的$\theta$的梯度

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191458125.png" alt="image-20231019145852061" style="zoom:25%;" />

online/offline-learning，前者每次训练只喂一组（x，y）给模型，后者每次训练把所有（x，y）给模型

SGD<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191508531.png" alt="image-20231019150808490" style="zoom:25%;" />SGDM<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191508978.png" alt="image-20231019150829947" style="zoom:25%;" />

Adagrad<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191511540.png" alt="image-20231019151109502" style="zoom:25%;" />RMSprop<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191512067.png" alt="image-20231019151250036" style="zoom:25%;" />

Adam = SGDM + RMSprop

![image-20231019151610459](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191516508.png)

一般模型都用SGDM或Adam训练，Adam在valid set上表现一般，generalization能力一般

![image-20231019152529494](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191525543.png)

![image-20231019152815631](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191528672.png)

Adam会被小的gradient牵着鼻子走，遇到好的大的gradient却因为之前的积累只能走一个有上限的步数 

![image-20231019153525294](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191535335.png)![image-20231019153632139](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310191536179.png)

AMSgrad，不好，处理了大的lr，但是这其实是回头路，RMSprop比adagrad优化的就是害怕一开始步子太大一直震荡

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310200814516.png" alt="image-20231020081459429" style="zoom:25%;" />

Adam的lr要不很小要么很大，很极端，Adabound，不好，设了一个上下限，完全不adaptive，rude，式子完全人为定义，不优雅，偏工程 

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310200821286.png" alt=" " style="zoom:25%;" />

改进SGDM的固定lr，都不是漂亮的方法

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310200824618.png" alt=" " style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310200825524.png" alt="image-20231020082545485" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310200827991.png" alt="image-20231020082711953" style="zoom:25%;" />

Adam也需要warm-up

![image-20231020084745107](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310200847197.png)

![image-20231020171228181](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310201712262.png)

![image-20231020171439930](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310201714966.png)

每往前走k步，回头看看方向对不对，和MAML中的reptile很像

![image-20231020171939110](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310201719157.png)

![image-20231020172046526](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310201720576.png)

更多关于momentum，如图往左走，是一个高坡，因此提出一个预见未来走势的算法NAG 

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310201725491.png)

![image-20231020172904040](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310201729077.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310201729256.png" alt="image-20231020172937223" style="zoom: 33%;" />

![image-20231020173016507](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310201730543.png)

之后再补





# Bayesian NN

联合概率因子化，不然最大似然推理推不下去，因为搜寻空间太大



从另一个角度看线性回归

统计机器学习是最小二乘估计

最大似然推理：已有一些采样值，他们独立同分布，采样自一个高斯曲线。高斯曲线由参数$\mu$和$\sigma$定义。问题是已知采样值，如何确定他们来自于什么样的高斯曲线，即分布的$\mu$和$\sigma$是多少。因此要最大化这些采样值出现的概率，连乘他们得到联合概率，为了防止越乘越小下溢到0，使用ln转换为累加

![image-20231107214346517](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311072240009.png)

这种依赖于条件分布的假设在机器学习中叫判别式模型，假设所有数据遵循同一个条件分布且数据点相互独立

![image-20231107214649133](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311072146163.png)

![image-20231107214725273](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311072240767.png)

![image-20231107214749525](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311072240008.png)

剔除与模型无关的噪声参数![image-20231107214818527](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311072148552.png)

因此最小二乘估计 是 极大似然估计 在 高斯分布 下的特殊形式

如果是伯努利分布，极大似然估计的结果就是逻辑回归

如果是多项分布，结果就是softmax回归



熵，交叉熵，KL散度

信息量 I(x) 与概率P成反比；取log，当独立事件概率相乘，独立信息量相加

![image-20231108004135395](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311080041486.png)

香农熵（不同于物理学的熵），针对一个概率分布，描述概率分布的不确定性；"PDF" 代表的是 "概率密度函数"（Probability Density Function）；uniform均匀，condensed聚拢

![image-20231108004942866](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311080049999.png)

交叉熵：对gt的估计（gt 0.5和0.5未知情况下，给出一个估计概率值0.2和0.8）；p按gt计算因为这是真实事件按p发生，信息量用q计算

![image-20231108005945845](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311080059985.png)

KL散度：定量衡量两个分布的区别 = 交叉熵 - 熵；机器学习经常要最小化KL散度，但是在损失函数中都是最小化交叉熵，二者等价，优化参数$\theta$时，D对$\theta$求导，展开，H(p)对$\theta$求导是恒定值

![image-20231108010451128](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311080104265.png)

另一个角度理解KL散度，基于gt p采样一个随机序列，分别求序列在p和q下的概率，二者比较；取log，求1/N平均，正面朝上次数为Nh，反面朝上次数为Nt

![image-20231108011055122](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311080110241.png)



贝叶斯网络，并不预测hard value，而是预测一个分布

![image-20231108002058619](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311080020731.png)

参数为 mean vector $\mu$ 和 covariance matrix $\sigma$ ；惩罚项是KL散度项，要尽量小，防止偏离正态分布，防止过拟合；在损失函数中KL项变加号，因为要minimize

![image-20231108011659061](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311080116192.png)



# pytorch

https://pytorch.org/docs/stable/
torch.nn -> neural network
torch.optim -> optimization algorithms
torch.utils.data -> dataset, dataloader

GPU和autograd

torchaudio：speech/audio processing

torchtext：natural language processing

torchvision：computer vision

skorch：scikit-learn + pyTorch



自定义dataset，读如要写三个函数

```python
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset): 
    def __init__(self, file): #Read data & preprocess
        self.data = ... 
    def __getitem__(self, index): #Returns one sample at a time
        return self.data[index] 
    def __len__(self): #Returns the size of the dataset
        return len(self.data)
```

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132135165.png" alt="image-20231013213506097" style="zoom:33%;" />

torch.utils.data.Dataset 读入；torch.utils.data.DataLoader 打包

训练shuffle 测试不

```python
dataset = MyDataset(file) 
dataloader = DataLoader(dataset, batch_size=5, shuffle=False)  
```

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230902225606273.png" alt="image-20230902225606273" style="zoom:67%;" />

![image-20231013213724439](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132137490.png)

shape（）

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230902225659204.png" alt="image-20230902225659204" style="zoom:67%;" />

transpose（0，1）转置，0维和1维互换

squeeze，unsqueeze，reshape

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230902231512483.png" alt="image-20230902231512483" style="zoom:67%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230902231604812.png" alt="image-20230902231604812" style="zoom:67%;" />

cat

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230902231709374.png" style="zoom:67%;" />

不同datatype会报错

.shape; .dtype

![image-20230903001257303](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230903001257303.png)

 ![image-20230903001853814](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230903001853814.png)![image-20230903001956167](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230903001956167.png)

nn.Sigmoid()  ; nn.ReLU()  

```python
import torch.nn as nn
class MyModel(nn.Module):
	def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32), 
            nn.Sigmoid(),
            nn.Linear(32, 1) #10->32->1 三层
		)
    def forward(self, x):
        return self.net(x)
#二者完全等价 
import torch.nn as nn
class MyModel(nn.Module):
	def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 32)
        self.layer2 = nn.Sigmoid(),
        self.layer3 = nn.Linear(32,1)
        
	def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
```



torch.nn.MSELoss ；  torch.nn.CrossEntropyLoss  

Mean Squared Error (for regression tasks)：criterion = nn.MSELoss()

Cross Entropy (for classification tasks)：criterion = nn.CrossEntropyLoss()

● loss = criterion(model_output, expected_value)



torch.optim  

torch.optim.SGD(model.parameters(), lr, momentum = 0)  

optimizer = torch.optim.SGD(model.parameters(), lr, momentum = 0)  

For every batch of data:

1. Call optimizer.zero_grad() to reset gradients of model parameters.

2. Call loss.backward() to backpropagate gradients of prediction loss.

3. Call optimizer.step() to adjust model parameters.



训练代码

```python
dataset = MyDataset(file) 
tr_set = DataLoader(dataset, 16, shuffle=True) 
model = MyModel().to(device) 
criterion = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), 0.1)

for epoch in range(n_epochs):
    model.train() #训练模式
    # tqdm
    for x, y in tr_set:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
```

验证 Validation Loop  

```python
model.eval() #测试模式
total_loss = 0
for x, y in dv_set:
	x, y = x.to(device), y.to(device)
	with torch.no_grad():#关闭梯度下降
        pred = model(x)
        loss = criterion(pred, y)
    total_loss += loss.cpu().item() * len(x)
    avg_loss = total_loss / len(dv_set.dataset)

```

测试

```python
model.eval()
preds = []
for x in tt_set:
	x = x.to(device)
	with torch.no_grad():
        pred = model(x)
        preds.append(pred.cpu())
```

model.eval()

Changes behaviour of some model layers, such as dropout and batch normalization.

with torch.no_grad()

Prevents calculations from being added into gradient computation graph. Usually used to prevent accidental training on validation/testing data.



Save

torch.save(model.state_dict(), path)

Load

ckpt = torch.load(path) 

model.load_state_dict(ckpt)



常见报错

*以前是parameters，以后是keyword argument，前者直接写数字，后者要写out = 数字；default 默认![image-20231013214507731](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132145768.png)

to("cuda:0")

OOM 减少batchsize，甚至设为1，不推荐![image-20231013215205022](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132152061.png)

cross entropy 的label参数要变成long type![image-20231013215325927](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132153962.png)



配环境

cuda，平行计算平台，使得软件使用GPU

![image-20231013215937325](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310132159359.png)

其他pipenv，docker，Nvidia docker见ppt







# Colab & Kaggle & JudgeBoi 

Colab

打开internet，选GPU

![image-20231009224500111](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310092245602.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310092245707.png" alt="image-20231009224551676" style="zoom:50%;" />

模块内前加！或%为shell命令，！后的命令执行完立刻关闭shell，cd命令用%

! gdown--id '1sUrlx-GhJ80vIGZVGEgFUSDYfwV50YW' --output pikachu.png; 从drive指令下载output为某名字

![image-20231009225141063](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310092252526.png)

关闭页面后file不会保存，及时下载

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310092254520.png" alt="image-20231009225431482" style="zoom:50%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310092255842.png" alt="image-20231009225518808" style="zoom:50%;" />

也可以用google drive存取文件

![image-20231009225637011](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310092256044.png)![image-20231009225704106](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310092257153.png)

写一个javascript防止熄屏

![image-20231009230129229](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310092301258.png)

If you would like to use colab, DO NOT store data in your drive and load from there, the input/output is very slow. (store at ./ instead) ； If you mount your google drive in colab : G-suite google drive now has a storage limit. Since models and data can be large, keep an eye on your used space to prevent your account being suspended. 



Kaggle

public测试集，ddl时公布private测试集

Kaggle GPU : 16G NVIDIA TESLA P100
○ https://www.kaggle.com/docs/efficient-gpu-usage
● Faster data IO 相比谷歌colab
● Easier checkpoint reusing 会存下资料
● Limited to 30+ hrs/week depending on usage.
● Limited to 12hrs/run 

debug阶段不要用gpu，跑时用；output限制20G，不要每个epoch存model

new dataset 把model参数存成一个dataset 下次再 import继续训练：

1. Run your code in background
2. Find the output model and save it as a dataset
3. Import your dataset into the notebook via “Add data”
4. Modify your code to load your checkpoint
5. Run your code in background
6. Find the output data “./submission.csv” and upload it to the submission page

输出submission.csv，下载上传

Time resets every Saturday at 08:00, Taipei Time.
=> Run any code with GPU on kaggle today (3/4) to get (possible) extra time next week. 

You can go over the time limit moderately. Kaggle will not interrupt your code in the background if it is already running. If your time limit is
reached, you cannot run any code with GPU.
=> 時間快用完的時候在背景跑一隻程式，等於多12小時runtime
=> 時間快用完的時候在背景跑兩隻程式，等於多24小時runtime

You can run two codes in the background
● If you are satisfied with your code, utilize this to run multiple random seeds/multiple train validation split/multiple model structures, so you can ensemble



 JudgeBoi

HW5，经过授权才能submit

![image-20231121165612378](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211656424.png)

![image-20231121165656962](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211656023.png)

![image-20231121165924545](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211659590.png)











# CNN

一般假设image大小一致，不一致的需要rescale再输入，100 * 100 * 3

判断影像通常由小块特征得出结果（眼，嘴，爪子）

receptive field 感受野 (红色立方体) -> 1个神经元只管3*3的RGB向量(共3 * 3 * 3)

[receptive field可重叠，可以一个receptive field使用两个神经元，可大小不同，可只考虑某些channel（RGB之一），可以是其他形状]

![image-20230823175242462](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823175242462.png)

经典安排方式如下图，一个receptive field常对应多个神经元（如64，即64个filter）；stride步长，超出范围空白处补值padding

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823175919798.png" alt="image-20230823175919798" style="zoom:67%;" />

鸟嘴可以在不同位置，那么每个感受野都要配备一个鸟嘴识别神经元 -> 权重共享

同一个神经元对于不同的感受野共用参数，每个神经元是一个filter，扫过不同的感受野

![image-20230823190825446](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823190825446.png)![image-20230823190944558](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823190944558.png)

-> first hidden layer: 3 * 3 * 64channel

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405051757148.png)

感受野 + 参数共享 = convolutional layer

[模型bias大（bias小容易overfit），bias大可能不适合影像外其他任务，需要思考是否有以上上两种特性]

另一个CNN讲解方法

![image-20230823191602343](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823191602343.png)

filter里的数值即weight，通过梯度下降得到，得到后可用来detect图片某一块pattern；如下filter侦测对角线为1的区域（得到值最大）

6 * 6 image -> filter -> 4*4 feature map, 64个filter -> 64 feature map

![image-20230823192027963](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823192027963.png)

第一层layer64个filter得到64张feature map，可以看作新的4*4图片，有64个channel

第二层layer就必须是深度为64的filter，深度必须与channel一致

![image-20230823192716552](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823192716552.png)

第二层3 * 3的filter已经涉猎了5 * 5的原图范围，因此可以侦测到大范围的pattern

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823192902363.png" alt="image-20230823192902363" style="zoom: 50%;" />

filter是有bias (wx+b的b) 的，不要忽略；一个filter扫过一张图片就是参数共享，就是卷积

![image-20230823193125205](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823193125205.png)

简化3：pooling池化，图片压缩不影响图片识别；这是一个固定过程，不随学习改变，类似激活函数，以max pooling为例，就是2*2的区域中留下最大的数字以压缩图片，还有mean pooling等；通常2次conv1次pooling循环进行

[为了侦测细小特征，近年常不使用pooling]

最终把conv和pool层的输出flatten（拉直成向量），送到传统mlp

![image-20230823194007361](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823194007361.png)

阿法狗是基于CNN的，是一个19*19（棋盘大小）的分类问题，是由于棋盘类似图片；每个位置被设计成48个channel（围棋中专业的特征，如是否跳吃等）

![image-20230823194356306](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823194356306.png)

压缩棋盘不合适，所以没有使用pooling

![image-20230823194501573](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823194501573.png)

无影像特征的问题不适用CNN；CNN也应用于语音识别和NLP，但是感受野设计不一样，考虑具体专业背景

![image-20230823194756341](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230823194756341.png)

经过训练的CNN不能识别缩放和旋转的图片，所以必须通过丰富数据集让他学习过缩放和旋转的样本才行；spatial transformer可以识别缩放和旋转的图片

spatial transformer layer（NN） 用来对input image/feature map 做旋转缩放

把卷积核旋转 平移，通过一个NN决定卷积核如何变化

![image-20231021220524733](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310212205865.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310212209138.png" alt="image-20231021220935070" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310212209273.png" alt="image-20231021220910238" style="zoom:25%;" />

![image-20231021220856050](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310212208126.png)

若算出小数坐标则四舍五入<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310212221209.png" alt="image-20231021222144126" style="zoom:25%;" />

 maxpooling并不是在每个地方都可微，但是可以用梯度下降解；上述的NN不能用梯度下降解，黄色和红色矩阵改变，输出因为四舍五入可能不变，微分=0，因此要用interpolation插值，这样就可以用梯度下降解

![image-20231021222923537](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310212229598.png)

![image-20231021223131297](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310212231344.png)

由此，对input做旋转缩放平移，CNN看见的input是不动的；transformer放大并且 截出了图片中主体的部分

![image-20231021224115746](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310212241786.png)

![image-20231021224058461](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310212240512.png)

resnet

![image-20240418005133508](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404180051660.png)



# Self-attention

每次输入sequence长度不一样 -> word embedembedding给每个词汇一个向量，按照语义分区

![image-20231023210410292](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310240957019.png)

语音上每25ms信号对应的向量叫frame，每两个frame隔10ms

![image-20231023212141421](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310240957657.png)

Graph的每个vertex是一个向量

case1：输入几个向量，输出几个label，POStagging 词性标注

![image-20231023212632502](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310240957006.png)

case2：输出1个label

case3：seq2seq语音识别为文字，翻译 

***

case1：输入几个向量，输出几个label

saw有时是名词有时是动词，函数只能映射到一个输出，解决方法：take in前后相邻内容

![image-20231023213311256](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310240958340.png)

但是句子长短不一，如何确定window的大小？

![image-20231023213814662](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310240958458.png)

![image-20231023213827441](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310240958938.png)

如何计算两个vector间权重？常用dot-product法（transformer所用）

![image-20231024100406594](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241004656.png)

最后可以用relu代替softmax

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241007118.png" alt="image-20231024100706081" style="zoom: 25%;" />

![image-20231024100817224](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241008281.png)

矩阵运算角度，I矩阵的四个column分别是a1a2a3a4，A’是attention matrix；需要学的参数，三个W矩阵；整个layer的input和output是I和O矩阵

![image-20231024102033662](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241020727.png)

![image-20231024102255552](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241022582.png)

![image-20231024102638731](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241026773.png)

更常用的是multi head self-attention，head数量是超参数，就是生成n组qkv，每组各自做以上运算出bi1和bi2

![image-20231024103522050](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241035104.png)

***

目前self- attention没考虑到位置的资讯，positional vector 上标i表示位置序号，position encoding待研究，也可以根据资料学出来，在原始论文attention is all you need中通过sincos function手动产生positional vector，就为了不同位置不同feature

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241039089.png" alt="image-20231024103959048" style="zoom:25%;" />

![image-20231024104208999](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241042041.png)

transformer Bert（NLP）都用了self-attention

语音识别中因为向量太多，运算量太大，使用truncated self-attention，只看一个小的范围，不看一整句话

把图片看作vector set 三维向量，W*H个

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241048619.png" alt="image-20231024104829577" style="zoom:25%;" />

![image-20231024104920577](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241049612.png)

CNN只考虑感受野内信息，感受野人工划定；self- attention考虑整张图片的信息，学习出比较重要的部分，类似学习出一个感受野；数据量小用CNN，大用self- attention；经过精心设计，self- attention可以变成CNN；理论推导见以下论文

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241053163.png" alt="image-20231024105336120" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241056178.png" alt="image-20231024105616120" style="zoom:25%;" />

HW4中strong baseline 使用Conformer，CNN + self-attention 

RNN已经基本被self-attention 淘汰，RNN只考虑了左边已输入的部分，self-attention 考虑了整句话 ，也有双向RNN 

![image-20231024110415622](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241104686.png)

RNN和self- attention的关系

![image-20231024110313861](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241103891.png)

graph self-attention，仅计算相连的node的attention，其他置0，就是GNN的一种

self-attention有很多变形，主要问题是计算量大；横轴speed，纵轴performance

![image-20231024111451119](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310241114184.png)



# RNN & LSTM & GRU

slot filling

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310221723533.png" alt="image-20231022172318459" style="zoom:25%;" />

label词汇编码方法

![image-20231030215048476](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302150601.png)

为了区分不同语境下的Taipei，引入记忆模块

![image-20231030215905076](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302159164.png)

三个network是同样的，只是记忆模块内容不同，但是RNN是和顺序强相关的

![image-20231030220116629](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302201750.png)![image-20231030220518283](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302205380.png)

双向RNN

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302206712.png)

LSTM（long short-tern memory 比较长的short-tern memory，RNN只记得上一个，LSTM多记得一些，但还是short-tern）

input/output gate打开时才能传数据，什么时候打开关闭是由网络学习的；forget gate决定什么时候忘掉数据

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302215444.png" alt="image-20231030221553355" style="zoom:80%;" />

cell中原本是c，经过一轮变成c'；f用sigmoid，表示门开的大小；forget gate关闭是遗忘

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302224452.png" alt="image-20231030222440330" style="zoom: 50%;" />

具体案例，蓝色表示memory；绿色是bias，假设网络权重已学到，x2x3的作用是由已学习到的权重达成的

![image-20231030223142635](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302231757.png)

<img src="C:/Users/tyhhh/AppData/Roaming/Typora/typora-user-images/image-20231030223729661.png" alt="image-20231030223509683" style="zoom: 50%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302238638.png" alt="image-20231030223808549" style="zoom:50%;" />

LSTM参数量是一般网络的四倍；不同颜色的线代表不同权重，两个输入x1x2，乘以四个不同权重，得到1个LSTM cell的四个输入；LSTM cell有很多个

![image-20231030232528530](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302325609.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302324109.png" alt="image-20231030232431000" style="zoom: 67%;" />

数 -> 向量形式，z的dim = LSTM cell的个数； zi，zo，zf同理

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302332909.png" alt="image-20231030233224830" style="zoom:50%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302335253.png" alt="image-20231030233554169" style="zoom:50%;" />]

以上只是LSTM的简化版本；还会参考RNN，有一个h向量；还会加上c，称为peephole

![image-20231030233746630](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302337760.png)![image-20231030233837913](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302338069.png)

LSTM还会叠很多层很多神经元，RNN常指代LSTM

![image-20231030235106009](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302351159.png)

Keras中 GRU是LSTM简化版本，2个gate，表现不差；simpleRNN是最开始讲的RNN

![image-20231030234035681](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310302340731.png)

时间序列下加速反向传播的算法BPTT（backpropagation through time）；训练非常困难，经常震荡，因为 error face要不很平坦要不很陡峭；必须使用clipping，给gradient设置一个上限thereshold

w是正向记忆的权重，x=1是输入

![image-20231031105308473](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310311053509.png)

梯度消失（Gradient Vanishing）： 梯度在反向传播过程中逐渐变得非常小（层数深的RNN，CNN），权重更新几乎不会影响网络的参数，导致网络无法学习有效的特征表示

梯度爆炸（Gradient Explosion）： 梯度在反向传播过程中逐渐变得非常大，网络参数不断增加（RNN），最终导致数值不稳定和数值溢出。梯度爆炸通常发生在网络的某些层次具有非常高的权重或梯度放大的激活函数

LSTM可以解决梯度消失（初代LSTM没有forget gate，记得以前所有东西），但解决不了梯度爆炸；LSTM的error surface没有平坦部分，只有陡峭，所以lr要设置的很小；LSTM训练时forget gate的bias要尽量小，使之少忘记一些

GRU，2个gate（少一个gate少1/3参数），LSTM过拟合可以用GRU；input gate打开，forget gate forget一些东西，旧的不去新的不来

其他解决梯度消失问题的方法；单位矩阵初始化参数+relu效果比较好；随机初始化参数+sigmoid效果比较好 

![image-20231031112533667](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310311125774.png)

RNN还可以 sequence to vector，如sentiment analysis，key term extraction

![image-20231031171929201](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310311719281.png)

RNN seq to seq（output短）；trimming去重；为了识别叠词，使用CTC，train时穷举所有null的位置，一起当作正确label训练；谷歌用CTC语音辨识

![image-20231031172330364](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310311723395.png)

![image-20231031172259329](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310311722359.png)

![image-20231031172511981](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310311725022.png)

 RNN seq to seq（无限制）；用于语音辨识不如CTC，用于翻译很好

![image-20231101014158076](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010141243.png)

输入中文语音，直接输出英文单词，适合方言

![image-20231101014439087](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010144200.png)

代替structure learning，输入语句输出语法结构树，把树状图描述成序列

![image-20231101014708045](C:/Users/tyhhh/AppData/Roaming/Typora/typora-user-images/image-20231101014708045.png)

考虑词汇顺序

![image-20231101014902327](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010149466.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010151858.png" alt="image-20231101015111690" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010151505.png" alt="image-20231101015136336" style="zoom:25%;" />

不同长度语音 -> 相同长度向量

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010152061.png" alt="image-20231101015240904" style="zoom:25%;" />![image-20231101015417140](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010154292.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010155541.png" alt="image-20231101015559473" style="zoom: 50%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010156123.png" alt="image-20231101015646973" style="zoom:25%;" />

f变成n，变化方向相同

![image-20231101015753816](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010157902.png)

RNN chatbot

![image-20231101015938100](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010159218.png)

attention-based

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010202437.png" alt="image-20231101020213314" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010202017.png" alt="image-20231101020228879" style="zoom:25%;" />

![image-20231101020425202](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010204363.png)

![image-20231101020627475](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010206623.png)

![image-20231101020713863](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010207007.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010207771.png" alt="image-20231101020750592" style="zoom:25%;" />

![image-20231101020950175](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010209281.png)

先RNN再structured learning，各自发挥特长

![image-20231101021544488](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010215638.png)![image-20231101021718047](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010217195.png)

![image-20231101021812263](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010218363.png)![image-20231101021840070](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010218171.png)

GAN和structured learning，GAN就是train structured learning的一种方法

![image-20231101022038662](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010220793.png)![image-20231101022107852](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010221927.png)

![image-20231101022153665](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311010221870.png)





# GNN

Deep Graph Library 库

处理graph，即节点+边，node+edge，这类结构的input和output

图的classification 或 generation

或者处理单个样本的feature的同时考虑样本之间的关系

不是每个node都有label

![image-20231101235951443](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311012359524.png)

常用GAT和GCN

![image-20231102000106415](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020001540.png)

GNN的衡量标准

![image-20231102000322170](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020003232.png)

采纳周围相连结点一起更新，得到下一层的内容；最后把所有node readout成统一特征

![image-20231102000810187](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020008279.png)

NN4G

![image-20231102001225139](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020012235.png)![image-20231102001257837](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020012928.png)

DCNN，第一层处理距离为1的node，第二层仅处理距离为2的node；k层视野距离为k；每一层的参数形成矩阵H

![image-20231102001910825](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020019913.png)![image-20231102002102158](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020021218.png)

DGC，区别仅仅在于把每一层H加起来

![image-20231102002216530](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020022600.png)

其他

![image-20231102002610880](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020026978.png)![image-20231102002710449](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020027540.png)

![image-20231102002734082](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020027144.png)

GAT，对邻居做attention，学习重要性权重e

![image-20231102002804029](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020028109.png)![image-20231102002944871](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020029938.png)

GIN 理论证明，第k层的结点v更新时，邻居要用sum，不用mean/max pooling；公式中epsilon可以置0/学习；用mlp而非one-layer

![image-20231102003118791](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020031844.png)

![image-20231102003510788](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020035864.png)

时域卷积，频域相乘

![image-20231102101959815](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311021019895.png)

推导见ppt？

Adjacency matrix 邻接矩阵







# Transformer

seq2seq，关键在于机器自己决定输出长度 

speech translation，没有文字的语言

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151715361.png" alt="image-20231115171530265" style="zoom:25%;" />

语音合成：输入中文文字，输出台语声音 

chatbot

NLP的绝大多数课题都可以理解为 question answering；但是客制化专用模型往往比seq2seq效果好

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151723142.png" alt="image-20231115172334054" style="zoom:25%;" />

树状结构也可以表示成seq，1412.7449 Grammar as a Foreign Language

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151727609.png" alt="image-20231115172738554" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151732499.png" alt="image-20231115173206444" style="zoom:15%;" /><img src="../Library/Application Support/typora-user-images/image-20231115173225825.png" alt="image-20231115173225825" style="zoom:15%;" />

***

seq2seq模型架构

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152056855.png" alt="image-20231115173357425" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152056459.png" alt="image-20231115205606653" style="zoom:80%;" />

encoder输入输出长度一致

![image-20231115173518533](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151735590.png)

![image-20231115173903421](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151739484.png)

transformer中用的比上图更复杂一点，加入了norm和residual

![image-20231115173929432](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151739518.png)

光用self-attention没有位置信息，再加入positional encoding

![image-20231115174152736](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151741796.png)

其他设计方法

![image-20231115174252539](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311151742607.png)

***

最常用autoregressive decoder；读入encoder输出的过程先省略，之后再讲；以语音识别为例

BEGIN -> begin of sentence BOS，one-hot；size：vocabulary size，一共有多少个汉字字符/词汇/字母/subword，每个字符对应一个概率数值；可能一步错步步错；end作为句子的结束，作业中与BEGIN使用了同一个符号

![image-20231115202746557](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152027603.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152018653.png" alt="image-20231115201858611" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152029859.png" alt="image-20231115202922778" style="zoom:15%;" />![image-20231115202002389](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152020456.png)

遮住中间部分，二者大致相同

![image-20231115202145604](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152021661.png)

上图中的masked指只考虑左边的

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152023664.png" alt="image-20231115202342622" style="zoom:15%;" />

原本self attention的b2

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405210209548.png" style="zoom:20%;" />

masked self attention的b2

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152025174.png" alt="image-20231115202517080" style="zoom:20%;" />

NAT，non-autoregressive model，一次性产生整个句子；决定句子长度，learn一个classifier输出句子长度，或给300个begin，等他输出end（句子上限300） ；语音合成里常用NAT， 让系统讲快一点，把classifier的输出➗2，慢✖️2

![image-20231115203921249](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152039293.png)

***

接下来讲encoder输出如何被decoder读入；蓝圈是k，v，绿圈是q

![image-20231115204057895](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152040956.png)

![image-20231115204414592](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152044673.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152044385.png" alt="image-20231115204450324" style="zoom:25%;" />

cross attention在transformer之前就有了，及各种cross attention方式的研究

![image-20231115204638433](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152046516.png)

![image-20231115205932605](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152059651.png)

以上是推理

训练过程：decoder的输入是GT（推理时decoder的输入是上一次的输出）

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152104519.png" alt="image-20231115210443466" style="zoom:25%;" />![image-20231115210731382](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152107471.png)

训练技巧

chatbot或做summary，输出可以从输入中复制一些东西（专有名词）；这种能力叫[pointer network](https://www.youtube.com/watch?v=VdOyqNQ9aww)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152111968.png" alt="image-20231115211142922" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152112986.png" alt="image-20231115211218934" style="zoom:20%;" />

![image-20231115211422943](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152114025.png)

语音合成TTS（Text-to-Speech）中有时会出现短的词汇发音错误，因为ML是黑盒子；guided attention避免机器漏字，强迫机器看所有输入，适合语音辨识，语音合成，要求attention必须从左向右

![image-20231115212053236](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152120288.png)

 假设世界上只有两个字符，每次选概率最高的是greedy decoding

![image-20231115212411975](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152124059.png)

Beam search有时候有用，有时候没有用；如果任务答案非常明确，如语音识别，适合用Beam search；如果任务是创造性的，如TTS、写故事，不适合用Beam search，要给decoder多一些随机性；TTS中甚至推理时主动给decoder加noise（否则出来的声音像机关枪）

![image-20231115212617346](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152126401.png)

作业中评价机制是机器推理出的句子和GT做比较得出bleu score，因此validation set不用cross entropy而用bleu score，选bleu score最高的模型（bleu score 不能微分）

遇到optimization无法解决的问题，用RL硬train一发 ，loss func当作激励函数，decoder当作agent；但很难实现

![image-20231115213250881](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152132974.png)

exposure bias，训练时看gt，推理时看自己上一个时刻的输出，可能一步错步步错 ；解决方法：训练时给机器看一些错误的输入，即schedule sampling；schedule sampling在transformer出现前就有，会伤害到transformer的平行化能力，后来有了专用于transformer的schedule sampling

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152136520.png" alt="image-20231115213623466" style="zoom:25%;" />![image-20231115213709873](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311152137925.png)









# GAN

Generative Adversarial Network

输入包含噪声z，从一个 简单 分布中随机抽取；输出y是一个机率的分布；用于同一输入不同输出；用于需要创造力的任务

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311182336917.png" alt="image-20231118233615763" style="zoom:25%;" />

GAN动物园

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311182344825.png" alt="image-20231118234447507" style="zoom:25%;" />

unconditional GAN即删去了x；generator和discriminator架构可自行设计

low dim = 50/100，使用除gaussian distribution之外的其他distribution差异不大

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311182346594.png" alt="image-20231118234628401" style="zoom:25%;" />

discriminator，输入一张图片，输出一个分数，越像二次元人脸分数越高

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311182348201.png" alt="image-20231118234833993" style="zoom:25%;" />

对抗，共同进化

先固定G，更新D，使得D可以区分gt 和 G初始生成的杂絮，可以抽象成分类或回归问题

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311182358376.png" alt="image-20231118235858099" style="zoom:25%;" />

固定D，训练G，把GD接成一个巨大网络，中间连接处的那一层相当于图片的维度，更新G的参数使输出分数越高越好

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311190003282.png" alt="image-20231119000332117" style="zoom:25%;" />

theory，KL散度和JS散度算不出来；用采样即可

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311190011491.png" alt="image-20231119001156318" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311190014180.png" alt="image-20231119001433935" style="zoom:25%;" />













# VAE

e从高斯分布采样得到；exp($\sigma$) 是学到的方差，决定noise的分布， 做exp保证>0

![image-20231123104604445](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231046528.png)

让机器自己学，肯定是 exp($\sigma$) = 0 loss最小；损失函数中 蓝-红=绿，使得 exp($\sigma$) = 1 loss最小

![image-20231123110031688](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231100732.png) 

autoencoder和VAE的区别

![image-20231123104858302](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231048349.png)

宝可梦处对应的probability最大

![image-20231123110244250](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231102316.png)

每个gaussian有自己的weight，从P(m)曲线决定选哪个gaussian（第m个gaussian），再从该单一gaussian sample data

![image-20231123111004596](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231110649.png)

 实际中z是非常高维，z上的某一点对应到一个gaussian function，有无穷多个这样的映射

![image-20231123112216832](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231122882.png)

x是一个image样本，log相加变相乘，最大似然；另一个NN由x推出z的分布，即q(z|x)

![image-20231123112523623](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231125667.png)

式子<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231128815.png" alt="image-20231123112849750" style="zoom:25%;" />对任何分布都成立，logP(x)与z无关，提出，分布积分为1

![image-20231123113243878](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231132927.png)

P(z)已知；P(x|z)和q(z|x)是两个神经网络； 找P(x|z)，使得L(likelihood = logP(x) = Lb + KL)最大  ->  找P(x|z)和q(z|x)使得Lb(lowerbound)最大

如果只提升P(x|z)，Lb上升，L可能下降

 q并不改变L的长度，只改变其中Lb和KL各自长度，max q可以max Lb使之约等于L（同时KL=0，即q(z|x)和P(z|x)一样）；之后再提升P(x|z)，L必定上升

![image-20231123114040448](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231140522.png)

![image-20231123115210312](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231152365.png)

损失函数中，minimize regularization项m就是调整q的参数，minimizeKL，使得q接近高斯分布P(z)

另一项可以看作logP(x|z)根据q的期望，即从q网络输出的$\mu$' 和$\sigma$'组成的高斯分布，sample data z，使得logP(x|z)越大越好

![image-20231123120530543](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231205599.png)

conditional VAE，指定数字的style

![image-20231123120707831](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231207877.png)

 VAE的问题，没有创造力，之后才有GAN

![image-20231123120917152](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311231209202.png)









# SSL

![image-20231125142626070](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311251426179.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311251428085.png" alt=" " style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311251429622.png" alt="image-20231125142921519" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311251430696.png" alt="image-20231125143022619" style="zoom:15%;" />

SSL属于unsupervised

![image-20231125143344483](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311251433548.png)

BERT，随机决定要盖住哪些字，随机决定要用两种遮盖方法的哪一种（遮黑/用另一个随机token替换）

<img src="../Library/Application Support/typora-user-images/image-20231125144316559.png" alt="image-20231125144316559" style="zoom:25%;" />











# Autoencoder

类似cycleGAN 不需要label self-supervised pre-train reconstruction 中间叫embedding 

![image-20231001203421539](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310012034581.png)

RBM 古早深度学习技术 2012废除 hinton 深度学习之父

denoising autoencoder 网络多了去噪功能 类似于Bert

![image-20231001204813183](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310012048225.png)

BERT decoder不一定要是linear

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310012051584.png)

autoencoder除了可以降维， 还有feature disentanglement，从embedding提取不同信息

![image-20231001205909206](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310012059255.png)

应用：voice conversion 区分语音的内容和声音特征的神经元

![image-20231007105620592](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071056003.png)

discrete latent representation，探索各种形式的embedding，第三种onehot->unsupervised classification

![image-20231007105920624](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071059663.png)

VQVAE使embedding是一系列codebook，对于语音问题，VQVAE最后学出的codebook就是类似音标

![image-20231007110258649](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071102694.png)

使embedding为文字，是看不懂的，使用discriminator判断embedding是否是人类文字，使之可读，从而生成summary

![image-20231007110746901](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071107947.png)

用tree structure做embedding

 ![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071111767.png)

Decoder也可以当作Generator用，即VAE（variational autoencoder）的原理

encoder也可以用作失真压缩![image-20231007111333968](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071113017.png)

作业：应用于anomoly detection 异常检测系统，难点在于只有正常的数据集，没有或很少有异常的数据集，因此不能做二元分类

![image-20231007111556608](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071115647.png)

异常检测的应用

![image-20231007111917751](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071119797.png)

使用autoencoder，输入输出差异大则是不能重建，则是异常情况

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310071121425.png)



# Adversarial Attack

应付来自人类的恶意

![image-20240410175527207](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404101755996.png)

non-targeted: 网络输出任何东西除了猫都算攻击成功；targeted: 输出海星算攻击成功

![image-20240410175910643](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404101759702.png)

加肉眼可见的噪声网络可以输出错的不离谱的结果

![image-20240410180347396](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404101803453.png)

0.64是post prob，加的噪声小到肉眼看不出来，反而输出错的离谱的结果

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404101801577.png" alt="image-20240410180135485" style="zoom:25%;" />

1. white-box attack 知道模型参数

L是$y$和$\widehat{\boldsymbol{y}}$的差距(cross-entropy)取负号，差距越大越好

![image-20240410180913297](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404101809347.png)

L-∞适合近人类肉眼，4-pixel example: 上图四个颜色各变化一小点，下图仅右下角绿色变化了一大点，两者L2-norm一样

![image-20240410181258061](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404101812115.png)

d(x^0^, x^t^)；fix指拉回到正方形边缘

![image-20240410194158948](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404101941091.png)

FGSM 只update一次；learning rate $\eta = \epsilon$，二维情况下必然落到正方形四个顶点之一

![image-20240410194553785](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404101945931.png)

上述方法iterate

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404101948764.png" alt="image-20240410194815683" style="zoom:25%;" />

2. black-box attack 不知道参数

用同样的data自己train一个network，攻击自己的network2

如果不知道training data，对于network输入data输出predict，形成paired data用来训练proxy network

![image-20240410195143943](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404101951001.png)

black-box attack non-targeted成功率很高 (表格1的对角线是white-box attack)；targeted成功率不高

表二第一行: -Resnet152表示使用对其他四个网络都攻击成功的data来攻击Resnet152 (非对角线是white-box attack)

![image-20240410195726191](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404101957250.png)

一个可能的解释：深蓝色是小丑鱼类别；图中横轴是攻击成功的方向，纵轴是攻击失败的随机方向；可见各种network攻击成功的方向都很类似；攻击容易成功，问题来自于data而不是model (linear network/SVM也很容易攻击成功)

![image-20240410200312867](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102003921.png)

attack越小越好，可以小到1pixel的程度

![t](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102006922.png)

universal attack 不用针对图片客制化

![image-20240410200712559](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102007628.png)

detect 合成声音的network也很容易被攻击；NLP结尾加上红色语段，机器回答相同

![image-20240410201038652](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102010732.png)

真实世界中的attack

从各个角度看某个带上神奇眼镜的人，都可以欺骗人脸识别系统

真实世界可以多角度观察物体，可能只有一个角度能骗过network其他不会；微小的变化摄像头抓不到 (解析度不够)；生产眼镜时避免真实世界颜色和电脑上颜色的区别

![image-20240410201656508](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102016590.png)

自动驾驶路牌

![image-20240410201753696](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102017769.png)

特斯拉自动驾驶看见被攻击的35会当成85然后加速

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102018372.png" alt="image-20240410201812287" style="zoom:25%;" />

输入4方块噪声图，模型输出tiger shark；输入10方块噪声图，模型输出ostrich

![image-20240410211701040](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102117194.png)

以上都是测试阶段攻击 (加噪声等骗过模型)

backdoor (训练阶段攻击)

training data中加入看起来正常的图片正常label但是adversarial的data (有人上传恶意图片到人脸识别资料集)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102131229.png" alt="image-20240410213122174" style="zoom:25%;" />



防御: 被动or主动

被动防御: 加一个filter (如smoothing就有很好的效果)

![image-20240410213326785](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102133840.png)

smoothing可能有副作用

![image-20240410213437632](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102134680.png)

其他方法：压缩再解压缩；让generator产生一张和输入尽量相似的图片 (会删除恶意噪声)

![image-20240410213945128](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102139213.png)

被动防御的弱点：一旦别人知道你用smoothing防御，别人也先用smoothing就仍旧可以产生attack signal

因此可以给被动防御加上随机性 (随机resize，再贴在随机灰色幕布上)

问题：别人可能知道你的随机防御方法并且找到universal attack signal 

![image-20240410214308449](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102143507.png)

主动防御: adversarial training 训练一个不容易被攻破的模型

训练完成后攻击自己，针对攻击data再进行训练；typo: $\hat{y}^{y}$ -> $\hat{y}^{N}$，类似于data augmentation (就算没人攻击也可以使模型robust避免overfit)；问题：computing expensive (adversarial training for free可以解决); adversarial training过程中未使用新算法，attacker可能使用新算法攻击

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404102151027.png" alt="image-20240410215145949" style="zoom:60%;" />

















# Federated Learning

超参数；参数：即权重

并行计算：CPU、GPU time；钟表时间；GPU time/GPU number = wall clock time

least squares regression最小二乘回归，即min cost function

慢在求梯度，用2个GPU，训练样本对半分给两个GPU；data parallelism 数据并发，每个节点只有一部分数据，用这些数据计算梯度

![image-20230821225956210](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053877.png)

要规模大，有通信问题，节点内共享内存，节点间通信

![image-20230821230346064](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053767.png)

方法1：一个节点用作server

![image-20230821230409072](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053814.png)

方法2：所有节点用来计算

![image-20230821230450554](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053369.png)

以下算法只用了parameter parallelism和message passing

![image-20231029171715952](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053375.png)

model parallelism 每个节点有全部数据和部分模型参数

并行计算和分布式计算，大多情况下混用

![image-20231029172030636](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053042.png)

算法1：同步mapreduce

synchronous + message passing + client-server

谷歌mapreduce未开源（bulk syncharonous 每一轮全部算完才进行下一轮），两个模仿者中spark更好(另一个是hadoop)。适合数据处理，用来ml并不高效

![image-20230821230855931](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053409.png)

![image-20240627133941416](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406271339526.png)

broadcast（server传出），map（并行计算），reduce（回传server）；server分三种：sum（求和），mean（平均），collect（收集，得到很多矩阵）

![image-20230821231433295](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310281332309.png)

cost：worker计算（map）；通信（broadcast和reduce）；同步 

通信时间=通信复杂度/带宽+网络延迟

加速比，减少通信代价，实际m个节点的加速比<m

![image-20230821231623709](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053600.png)![image-20230821231818247](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053007.png)

同步cost：等最慢的worker， 另一部分时间损耗在同步（一个节点挂掉straggler，要等他重启，struggler effect) 规模越大挂的几率越大

![image-20230821231932399](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821231932399.png)

算法2：异步parameter server

asynchronous + message passing + client-server，实际并行计算用的

异步梯度下降，收敛更慢，但节省通信cost，所以实际更快，用parameter server架构实现，主要区别是异步，推荐用Ray开源系统（和spark出自同一个Berkeley实验室，比spark好上手很多）

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310291658119.png" alt="image-20231029165812994" style="zoom:25%;" />

异步梯度下降，经典论文hogwild，证明异步GD可以收敛

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053867.png" alt="image-20231029170153355" style="zoom:25%;" />

异步要求worker整体比较稳定，才能收敛，联邦学习就出现不稳定的问题，异步在联邦学习不适用

![image-20231029170435601](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053015.png)

算法3: 去中心化，p to p，点对点，peer to peer

synchronous + message passing + peer to peer（大多是同步，也有异步）

![image-20231029170809238](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053674.png)

![image-20231029170930079](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310291709123.png)

证明Decentralized GD 收敛

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053110.png)

![image-20231029171106503](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053575.png)

联邦学习，从分布式ml找idea 

计算主要在worker，server端的计算量很小；另有两次通信

![image-20230821232812851](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053121.png)

4：数据并非独立同分布（很多减少通信次数的算法不再适用）；5：节点负载不均衡，有人不拍照片有人一天100张，权重设立不能按照片也不能按用户

![image-20230821233439732](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053510.png)

方向1：降低通信次数，通常提升计算量（更好的梯度下降方向传给server）

fedavg算法：重复ab两步1-5个epoch；在 server更新w时取平均

![image-20230821234140070](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821234140070.png)![image-20230821234210692](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053633.png)

fedavg算法减少通信提升计算量（并不完美，不适合联邦学习，数据不是独立同分布。）

![image-20230821234409527](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053900.png)![image-20230821234418160](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053983.png)

方向2：隐私保护

梯度和权重反推数据，可知人种，性别等；分类器对用户分类

噪声加少没用，加多影响学习 

方向3：鲁棒性，拜占庭错误（故障节点叛变，修改标签降低模型准确率，拖慢训练，开后门等）和恶意攻击

用测试集测一下梯度准确率，排除错误梯度；检测数据分布；用中位数代替加权平均整合梯度；三者都不好，因为用户数据本身就可能不符合模型，且不是独立同分布的 ；目前攻击容易防御难 ，差分隐私可以保护但是收敛结果变差



Tensorflow实现并行计算

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053660.png" alt="image-20231029172340862" style="zoom:25%;" />

mirrorstrategy 类似 同步mapreduce，用SGD（batch），适用一台机器多个GPU 

```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
# For example, my server has 1 CPU and 4 GPUs:
#• /device:CPU:0
#• /device:GPU:0
#• /device:GPU:1
#• /device:GPU:2
#• /device:GPU:3

from tensorflow import distribute
# mirrorstrategy（指定用哪些GPU，不指定默认全用）
strategy = distribute.MirroredStrategy()

m = strategy.num_replicas_in_sync
print('Number of devices: {}'.format(m))
# Number of devices: 4
```

用CNN训练minst

导入数据，SGD别用太大batchsize，影响测试集准确度 

```python
import tensorflow as tf
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

import tensorflow_datasets as tfds
datasets, info = tfds.load(name='mnist',
                          with_info=True,
                          as_supervised=True)
mnist_train = datasets['train'].map(scale).cache()
mnist_test = datasets['test'].map(scale)

# Buffersize，对于大数据集，每次存放在内存中对数据数量
# 每个GPU 128个样本，4个GPU
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 128
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * m
data_train = mnist_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
data_test = mnist_test.batch(BATCH_SIZE)
```

神经网络结构及训练测试，加一行with strategy.scope():

```
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
with strategy.scope():
    model = keras.Sequential()
    model.add(Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
with strategy.scope():
    model.compile(loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.RMSprop(learning_rate=1E-3),
    metrics=['accuracy'])

model.fit(data_train, epochs=10)

eval_loss, eval_acc = model.evaluate(data_test)
```

mirrorstrategy的技术原理：ring all-reduce

reduce，只有server知道reduce结果；all-reduce，所有节点都知道reduce的结果

![image-20231029174446837](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053359.png)

![image-20231029174512773](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053222.png)

all-reduce有三种方法，ring-all-reduce最好，需要同步，即等每个GPU计算出梯度；低效方法：转一圈累加，再转一圈传播g的和

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310291750496.png" alt="image-20231029175000454" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130053771.png" alt="image-20231029175204467" style="zoom:25%;" />

 高效方法，快m倍；4个GPU，每个向量切成4（m）份，不论有几块GPU通信时间都一样

![image-20231029180022238](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130054980.png)

第1轮通信![image-20231029175431719](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130054148.png)

第2轮通信：发送a0和a1的加和![image-20231029175512686](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130054702.png)

第3轮通信![image-20231029175712054](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310291757089.png)

第4轮通信，广播![image-20231029175813442](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130054308.png)

第5轮通信，第6轮通信![image-20231029175857914](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311130054720.png)







# Quantum Machine Learning

![image-20231102005042671](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020050743.png)

learning：学以致用；

HHL算法（Harrow-Hassidim-Lloyd算法）是一种量子算法，旨在解决线性方程组Ax=b，其中A是一个大型稀疏矩阵。利用量子叠加和量子并行性，将线性方程组的解编码到一个量子态中，然后使用量子计算来找到这个编码的解。HHL算法在一些情况下具有指数级的加速度。HHL算法并不适用于所有线性方程组，而且它需要一些特殊的条件和前提知识，如A矩阵的特性以及与其特征值相关的信息。此外，由于量子计算硬件的限制，HHL算法在实际应用中仍然面临一些挑战，但它代表了量子计算在优化和线性代数领域的一个重要潜在应用。

![image-20231102005352460](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311020053567.png)

问题，批判，就算有quantum computer，转换为quantum state的过程也无法实现或很复杂，HHL算法做不到加速，读出同样复杂

![image-20231111223912203](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112239295.png)

两种QNN，后者是主流

![image-20231111224637388](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112246444.png)

寻找应用、好处

![image-20231111224803297](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112248387.png)

circuit太大，gradient会vanish

![image-20231111225103851](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112251938.png)

定义quantum ML

横轴，machine device 是classical还是quantum；纵轴，data是classical还是quantum

![image-20231111225651671](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112256744.png)

传统ML

![image-20231111225955161](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112259240.png)

进一步分类

没有quantum专用的ML algorithm，仅仅是通过HHL加速传统算法如SVM

![image-20231111230600328](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112306405.png)

第二行，Lloyd在做；第三行的learning mechanism本质上还是classical

![image-20231111231129905](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112311989.png)



![image-20231111231145143](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112311189.png)

![image-20231111231351835](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112313908.png)

![image-20231111231458536](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112314607.png)

1、basis encoding

![image-20231111231610259](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112316325.png)

M是data数，N是feature数；加速到O(log N)的算法无法implement

![image-20231111231856429](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112318498.png)

![image-20231111232153418](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112321491.png)

2、amplitude encoding 常用

![image-20231111232351909](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112323976.png)

加速了聚类，但是读出的时候又很耗时，抵消了

![image-20231111232723388](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112327460.png)

![image-20231111232954346](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112329429.png)

HHL对矩阵condition number有限制，同样需要amplitude encoding，有amplitude encoding的问题

![image-20231111233425570](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112334657.png)

可不可以去掉amplitude encoding？有了rotation encoding

![image-20231111234046578](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112340667.png)

![image-20231111234215214](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112342307.png)

![image-20231111234353191](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112343254.png)



![image-20231111234402342](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112344383.png)

![image-20231111235133917](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311112351993.png)

听不懂了



# LLM

gpt实则两三个token组成一个中文文字

llm在做文字接龙

![image-20240605024926314](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406050249473.png)

![image-20240605164241940](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051642076.png)

![image-20240605160746067](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051607181.png)

***

第一阶段 pre-train：SSL (masked) 学习网上爬下来的乱七八糟资料

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406050252525.png" alt="image-20240605025241463" style="zoom:25%;" />

![image-20240605150921605](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051509657.png)

GPT3很难控制，prompt格式很重要，你问他问题他只会乱接龙；没有教他回答问题，只是教他文字接龙

****

第二阶段：人类整理问题+答案，supervised learning

人类标注资料有限，因此关键是使用第一阶段的参数作为初始参数

![image-20240605152440543](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051524662.png)

保持第一阶段的参数固定，新加一些参数xyz以供优化，更快

![image-20240605153142815](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051531955.png)

![image-20240605153822904](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051538065.png)

![image-20240605154359001](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051543149.png)

![image-20240605154558096](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051545170.png)

finetune这一步不需要大量资料，几万笔即可，重点是质量高，less is more

openai知道真实用户会问什么问题

![image-20240605155942184](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051559306.png)

GPT逆向工程

![image-20240605160115152](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051601239.png)

但是没有pretrain参数 -> llama开源了

![image-20240605160402929](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051604065.png)

![image-20240605160517493](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051605564.png)

***

第三阶段：RLHF (PPO)

人类比较轻松，只需要判断两个模型输出哪个比较好即可，人类回馈有偏见

训练一个reward model来模拟人类的判断

横轴是向reward model学习的次数，实线是人类的喜好程度，虚线是reward model的喜好程度

![image-20240605162707562](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051627680.png)

![image-20240605162823411](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051628494.png)

![image-20240605163547234](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051635381.png)

RLAIF，可以让模型自己对自己做RLAIF，模型可能没能力输出好答案，但有能力判断答案好坏

![image-20240605163751663](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051637764.png)

训练一个safety reward model，一个helpfulness reward model

![image-20240605163903897](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051639972.png)

***

Speculative Decoding: 加快语言模型输出时间

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051702260.png" alt="image-20240605170221139" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051702752.png" alt="image-20240605170246682" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051703517.png" alt="image-20240605170323403" style="zoom:25%;" />

![image-20240605170402333](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051704458.png)



# LoRA & QLoRA

![image-20240613165149324](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406131651475.png)

r = 1-64

alpha: 0-1, amount of change added to original model weights

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406131656388.png" alt="image-20240613165654285" style="zoom:25%;" />

![image-20240613172819064](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406131728206.png)



# LLM safety

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051706556.png" alt="image-20240605170615421" style="zoom:25%;" />



<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061606890.png" alt="image-20240606160643692" style="zoom:25%;" />

gemini有谷歌查询验证功能

![image-20240606161517151](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061615237.png)

***

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061614062.png" alt="image-20240606161452911" style="zoom:25%;" />

![image-20240606161715500](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061617641.png)

让LLM排序一模一样的履历，仅名字不同，排一千次

![image-20240606162010757](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061620864.png)

![image-20240606162254276](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061622427.png)

![image-20240606162403294](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061624371.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061625733.png" alt="image-20240606162549604" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061629166.png" alt="image-20240606162956023" style="zoom:25%;" />

***

目前只能训练出比较准确的针对单一语言模型的分类器

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061632854.png" alt="image-20240606163221729" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061634296.png" alt="image-20240606163441221" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061635472.png" alt="image-20240606163523397" style="zoom:25%;" />

如果只是润稿的话检测器不会输出这么高的概率

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061636021.png" style="zoom:25%;" />

水印

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061638340.png" alt="image-20240606163855267" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061639218.png" alt="image-20240606163916143" style="zoom:25%;" />

***

诈骗LLM prompt hacking

![image-20240606164424934](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061644063.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061645831.png" alt="image-20240606164514744" style="zoom:25%;" />

对gpt4o使用注音符号

![image-20240606164648411](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061646488.png)

![image-20240606164732170](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061647253.png)

![image-20240606164817120](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061648198.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061650900.png" alt="image-20240606165028744" style="zoom:25%;" />

重复单词，突然吐出个人信息，10%准确率

![image-20240606165149186](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061651265.png)

Prompt Injection

![image-20240606165603128](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061656211.png)

![image-20240606165528407](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061655504.png)



# LLM Audio

![image-20240625235013928](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406252350141.png)

![image-20240625235030575](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406252350635.png)

声音16kHz，一秒采样16000次，接龙太慢；使用speech unit

![image-20240625235527130](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406252355211.png)

![image-20240625235635287](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406252356362.png)

文字 + speech unit

![image-20240625235842860](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406252358924.png)

2个人同时跟model说话

![image-20240626000015930](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260000077.png)

棒读指说话非常平淡，模型大了会学出语气

![image-20240626000316442](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260003567.png)

全部用语音pretrain资料太少

![image-20240626000543953](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260005071.png)

教已经pretrain的语言模型学一门新的语言

![image-20240626001243522](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260012641.png)

![image-20240626001423722](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260014796.png)

模型讲话都是同一个人的声音；finetune往往不需要太多资料，只要少量高质量

也可以用语音转换技术转换成同一个人的声音

![image-20240626001832834](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260018898.png)

语音模型相比语言模型，需要猜什么时候开始接话；人可能会打断语音模型，也有可以在和语音模型合唱

第二个频道记录模型自己说过什么，模型输入两个频道的内容，接龙第二个频道的内容

![image-20240626002429395](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260024508.png)

![image-20240626002502628](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260025707.png)

![image-20240626002702265](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260027342.png)



https://www.youtube.com/watch?v=2vu6u5CrMYQ

llama vocabulary size 32K

audio token vocabulary 1024

![image-20240630153652966](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406301536155.png)







# HW3

mixup augmentation https://arxiv.org/pdf/1710.09412.pdf 一张图两个label 

 <img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230921001209434.png" alt="image-20230921001209434" style="zoom: 67%;" />

Coding :
● In your torch.utils.Dataset, __getitem__()needs to return an image that is the linear combination of two images.
● In your torch.utils.Dataset, __getitem__() needs to return a label that is a vector, to assign probabilities to each class.
● You need to explicitly code out the math formula of the cross entropy loss, as CrossEntropyLoss does not support multiple labels.

Test Time Augmentation: 在模型对输入图像进行预测时，对输入图像进行多次变换或增强，并将多次预测结果进行汇总，通常采用平均值或投票的方式，最终得出一个更可靠的预测结果。

Coding : You need to fill in train_tfm, change the augmentation method for test_dataset, and modify prediction code to gain this effect. Usually, test_tfm will produce images that are more identifiable, so you can assign a larger weight to test_tfm results for better performance.

![image-20230921002505836](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230921002505836.png)

Cross-validation:每次用部分数据训练多个model(从而使用所有数据训练，不损失validation数据)，Coding : You need to merge the current train and validation paths, and resample form those to form new train and validation sets.

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230921131019449.png" alt="image-20230921131019449" style="zoom:150%;" />

2种Ensemble
● Average of logits or probability : 要保存完整的output，不会平手
● Voting : Easier to implement, need to break ties，要解决平手的问题
● Coding : basic math operations with numpy or torch

Residual Connection is widely used in CNNs such as https://arxiv.org/abs/1512.03385. 输出和输入相加，避免深度网络中梯度消失

![image-20231102145015766](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311021450856.png)

# HW4

![image-20231114172747179](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311141727289.png)

![image-20231106150732752](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061507835.png)

![image-20231114172833386](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311141728427.png)

![image-20231106161231941](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061612032.png)

 ![image-20231106161635005](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061616053.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311061616584.png" alt="image-20231106161645552" style="zoom: 25%;" />

# HW5

![image-20231121162917134](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211629824.png)

![image-20231121163235940](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211632047.png)

![image-20231121163333881](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211633929.png)

![image-20231121163842908](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211638956.png)

![image-20231121163859136](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211638185.png)

![image-20231121163918447](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211639487.png)

![image-20231121164158309](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211641380.png)

![image-20231121164410214](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211644284.png)

![image-20231121164428991](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211644053.png)

![image-20231121164502242](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211645299.png)

这篇paper visualize了RNN的error surface，seq2seq模型都很常用

![image-20231121164604130](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211646203.png)

gradient clipping：把每个参数的gradient收集起来变成一个vector，计算p-norm

![image-20231121165004215](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211650269.png)

![image-20231121165201306](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211652348.png)![image-20231121165211396](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211652457.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311211652236.png)



