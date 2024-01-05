# 入门

![image-20231212195628102](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312121956183.png)

## 一元 linear regression

$\theta:=\theta-\alpha\frac{\partial}{\partial\theta_0}Loss\\$     simultaneously update w and b

步子大小由 $\alpha$ 和 导数大小 共同决定

$\alpha$太大可能错过局部低谷甚至发散（步子随着导数增大而增大）

![截屏2023-08-07 21.56.07](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/截屏2023-08-07 21.56.07.png)

线性回归+平方误差函数，仅有一个最小值，学习率选的小必定收敛

## 多元linear regression

除了梯度下降，还有正规方程法 normal equation，仅仅适用于线性回归，无需迭代，慢，不如梯度下降

有些库在线性回归求解w和b时用正规方程法



Normal Equation 正规方程法

Intuition: $\frac{\partial}{\partial\theta_1}J(\theta)=\frac{\partial}{\partial\theta_2}J(\theta)=\dots=\frac{\partial}{\partial\theta_j}J(\theta)=0\\$

Solution: $\theta=(X^TX)^{-1}X^Ty$

Example: $m$ training examples, $n=4$ features.

$ \theta=\begin{bmatrix}\theta_0\\\theta_1\\\theta_2\\\theta_3\\\theta_4\end{bmatrix}$ $X=\begin{bmatrix}1&x_1^{(1)}&x_2^{(1)}&x_3^{(1)}&x_4^{(1)}\\1&x_1^{(2)}&x_2^{(2)}&x_3^{(2)}&x_4^{(2)}\\1&x_1^{(3)}&x_2^{(3)}&x_3^{(3)}&x_4^{(3)}\\\vdots&\vdots&\vdots&\vdots&\vdots\\1&x_1^{(m)}&x_2^{(m)}&x_3^{(m)}&x_4^{(m)}\end{bmatrix}$ $Y=\begin{bmatrix}y^{(1)}\\y^{(2)}\\y^{(3)}\\\vdots\\y^{(m)}\end{bmatrix}$

Note that, $x_1^{(3)}$ denotes the 1st attribute of the 3rd training example.

| Gradient Descent                    | Normal Equation                                              |
| ----------------------------------- | ------------------------------------------------------------ |
| Need to choose $\alpha$.            | No need to choose $\alpha$.                                  |
| Need many iterations to converge.   | No need to iterate.                                          |
| Still works well when $n$ is large. | Need to compute $(X^TX)^{-1}$ with complexity $O(n^3)$.<br>Slow if $n$ is large (over 100000). |

## Logistic Regression  分类 

就是linear regression最后加sigmoid，输出分类为类别1的概率

Sigmoid/Logistic Function；z大于0，经过sigmoid是0.5-1，z小于0，经过sigmoid是0-0.5

决策边界：z是一条直线，即w1x1+w2x2+b=0，在其上z大于0，在其下z小于0 

![image-20231028155407833](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310281554899.png)

相当于用回归模型学习分界线，多元一次则分界线是直线（线性），高次是曲线

$h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}\\$ ![截屏2023-08-08 17.54.48](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/截屏2023-08-08 17.54.48.png)

  $\begin{aligned}&y=1\\\Leftrightarrow\ &P(y=1|x;\theta)=h_\theta(x)=g(\theta^Tx)\ge0.5\\\Leftrightarrow \ &\theta^Tx\ge0\end{aligned}$

**<u>Logistic Regression 不能用平方cost function</u>**，会变成非凸问题，有很多局部最小

![截屏2023-08-08 19.11.14](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/截屏2023-08-08 19.11.14.png)

用Minus Log Cost，Have a unique global minimum point. (convex)，基于最大似然估计

![image-20230808192141307](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230808192141307.png)

![image-20230808192115902](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230808192115902.png)

Loss func：$\begin{aligned}J(\theta)&=\frac{1}{m}\sum_{i=1}^m\text{Cost}(h_\theta(x^{(i)}),y^{(i)})\\&=-\frac{1}{m}\sum_{i=1}^m\bigg[y^{(i)}\text{log}(h_\theta(x^{(i)}))+(1-y^{(i)})\text{log}(1-h_\theta(x^{(i)}))\bigg]\end{aligned}$

### <img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/截屏2023-08-08 19.33.49.png" alt="截屏2023-08-08 19.33.49" style="zoom:25%;" />

化简后最终形式和线性回归相同，除了f表示的内容不同

## 调参

Feature Scaling:   $-1\le{x_i}\le1$ 

1. $x_i:=\frac{x_i}{\text{max}(|x_i|)}\\$

2. Mean Normalization: Make features have approximately $0$ mean.  

   $x_i:=\frac{x_i-\mu_i}{\text{max}(x_i)-\text{min}(x_i)}\\$

3. Z-score normalization

   $x_i:=\frac{x_i-\mu_i}{\sigma_i}\\$

polynomial regression多项式回归，引入x的高次方（非线性）或用根号，此时归一化显得更加重要

![截屏2023-08-08 15.49.36](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310281654468.png)



Choosing $\alpha$ : $\dots0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1\dots$

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202310281547743.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230808153941726.png" alt="image-20230808153941726" style="zoom:20%;" />



feature engineering 基于已有参数增加新的参数，注重物理可解释性（已有长宽，加一个feature面积）



过拟合 high variance，overfit；欠拟合 high bias，underfit

generalization正则化，解决过拟合



### Advanced Optimization

| Examples                             | Advantages                                                   |
| ------------------------------------ | ------------------------------------------------------------ |
| Conjugate Gradient<br>BFGS<br>L-BFGS | No need to pick $\alpha$. <br>Often faster than gradient descent. |

### 3.4 - Solving the Problem of Overfitting

| Underfit  |  Well-fit  |    Overfit    |
| :-------: | :--------: | :-----------: |
| High Bias | Just Right | High Variance |

There are two main options to address the issue of overfitting:

1) Reduce the number of features减少变量；或降低次数

- Manually select which features to keep.
- Use a model selection algorithm.之后会讲

2) Regularization正则化or惩罚项: 最常用，减小系数而非一删了之，只对w正则化，不对b正则化 （有的人也对b正则化，对结果没什么影响）![截屏2023-08-08 19.59.30](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/截屏2023-08-08 19.59.30.png)

- Keep all the features, but reduce the magnitude of parameters $\theta_j$.
- Regularization works well when we have a lot of slightly useful features.
- 常常如图，由于不知道什么特征重要，而对所有w都一起惩罚，➗2m为了使得惩罚系数随着m变化，使得选择一个lambda更加容易且通用，由此lambda类似alpha是一个定值![截屏2023-08-08 20.14.46](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/截屏2023-08-08 20.14.46.png)

3. 增加样本，样本足够多足够密集不会过拟合

#### Cost Function

Add a penalty term to eliminate the parameters. ($\theta_0$ not included)

$J(\theta)=\frac{1}{m}\sum_{i=1}^m\text{Cost}(h_\theta(x^{(i)}),y^{(i)})+\lambda\sum_{j=1}^n\theta_j^2\\$

Regularization Parameter $\lambda$ too large: Cause underfitting.

![image-20230808202524780](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230808202524780.png)

![截屏2023-08-08 20.33.19](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/截屏2023-08-08 20.33.19.png)

![image-20230808202611745](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230808202611745.png)

每次略微缩小一点w，乘0.9998

![image-20230808202858040](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230808202858040.png)

#### Regularized Linear Regression

| Gradient Descent                                             | Normal Equation                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $\begin{aligned}\theta_0&:=\theta_0-\alpha\frac{\partial}{\partial\theta_0}J(\theta)\\\theta_j&:=\theta_j-\alpha\bigg[\frac{\partial}{\partial\theta_j}J(\theta)+\frac{\lambda}{m}\theta_j\bigg]\\&:=\big(1-\frac{\lambda}{m}\big)\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)\space,j\in\{1,2,\dots,n\}\end{aligned}$ | $\theta=(X^TX+\lambda L)^{-1}X^Ty$<br>where $L=\begin{bmatrix}0&0&0&\cdots&0\\0&1&0&\cdots&0\\0&0&1&\cdots&0\\\vdots&\vdots&\vdots&\ddots&\vdots\\0&0&0&\cdots&1\end{bmatrix}$ |

#### Regularized Logistic Regression

$\theta_0:=\theta_0-\alpha\frac{\partial}{\partial\theta_0}J(\theta)\\$

$\begin{aligned}\theta_j&:=\theta_j-\alpha\bigg[\frac{\partial}{\partial\theta_j}J(\theta)+\frac{\lambda}{m}\theta_j\bigg]\\&:=\big(1-\frac{\lambda}{m}\big)\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)\space,j\in\{1,2,...n\}\end{aligned}$

![image-20230808203857090](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230808203857090.png)

仍旧是形式与线性回归模型一模一样，除了f的定义式





## 4 - Neural Networks: Representation 升级版的逻辑回归

mlp： multilayer perception

input是第0层，隐藏➕output为层数

一种经典的网络结构，隐藏层神经元数量随层数增加而减少

tensorflow： 最普通的叫dense层

numpy和tensorflow矩阵表示不一致问题

![image-20230809160529981](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230809160529981.png)

第三行是一维数组，非二维矩阵，tensorflow用前两行的表示法，计算更快

tensor张量类型，可理解为矩阵，用于加速计算，打印tensor类型如图![image-20230809161128994](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230809161128994.png)

已有tensor类型变量a，用a.numpy()转回numpy类型![image-20230809161342231](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230809161342231.png)

![image-20230809162844779](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230809162844779.png)

### 4.1 - Model Representation

| First Layer | Intermediate Layer | Last Layer   |
| ----------- | ------------------ | ------------ |
| Input Layer | Hidden Layer       | Output Layer |

$g(x)$ : activation function

$a_i^{(j)}$ : "activation" of unit $i$ in layer $j$

$\Theta^{(j)}$ : matrix of weights controlling function mapping from layer $j$ to layer $j+1$

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/4-example_network.png" alt="example_network" style="zoom:50%;" />

In the example network above, the data flow looks like:

$[x_1,x_2,x_3]\rightarrow\big[a_1^{(2)},a_2^{(2)},a_3^{(2)}\big]\rightarrow h_\theta(x)\\$

The values for each of the "activation" nodes is obtained as follows:

$a_1^{(2)}=g\big(\Theta_{10}^{(1)}x_0+\Theta_{11}^{(1)}x_1+\Theta_{12}^{(1)}x_2+\Theta_{13}^{(1)}x_3\big)\\$

$a_2^{(2)}=g\big(\Theta_{20}^{(1)}x_0+\Theta_{21}^{(1)}x_1+\Theta_{22}^{(1)}x_2+\Theta_{23}^{(1)}x_3\big)\\$

$a_3^{(2)}=g\big(\Theta_{30}^{(1)}x_0+\Theta_{31}^{(1)}x_1+\Theta_{32}^{(1)}x_2+\Theta_{33}^{(1)}x_3\big)\\$

$h_\theta(x)=a_1^{(3)}=g\big(\Theta_{10}^{(2)}a_0^{(2)}+\Theta_{11}^{(2)}a_1^{(2)}+\Theta_{12}^{(2)}a_2^{(2)}+\Theta_{13}^{(2)}a_3^{(2)}\big)\\$

Where the addition $\Theta^{(j)}$ is the "bias", whose corresponding inputs $x_0,a_0^{(2)}=1$.

The matrix representation of the above computations is like:

| Inputs                                                       | Weights                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $a^{(1)}=x=\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}\overset{\text{add bias}}{\Longrightarrow}\begin{bmatrix}x_0\\x_1\\x_2\\x_3\end{bmatrix}$ | $\Theta^{(1)}=\begin{bmatrix}\Theta_{10}^{(1)}&\Theta_{11}^{(1)}&\Theta_{12}^{(1)}&\Theta_{13}^{(1)}\\\Theta_{20}^{(1)}&\Theta_{21}^{(1)}&\Theta_{22}^{(1)}&\Theta_{23}^{(1)}\\\Theta_{30}^{(1)}&\Theta_{31}^{(1)}&\Theta_{32}^{(1)}&\Theta_{33}^{(1)}\end{bmatrix}$ |
| $a^{(2)}=g\big(z^{(2)}\big)=g\big(\Theta^{(1)}a^{(1)}\big)=\begin{bmatrix}a_1^{(2)}\\a_2^{(2)}\\a_3^{(2)}\end{bmatrix}\overset{\text{add bias}}{\Longrightarrow}\begin{bmatrix}a_0^{(2)}\\a_1^{(2)}\\a_2^{(2)}\\a_3^{(2)}\end{bmatrix}$ | $\Theta^{(2)}=\begin{bmatrix}\Theta_{10}^{(2)}&\Theta_{11}^{(2)}&\Theta_{12}^{(2)}&\Theta_{13}^{(1)}\end{bmatrix}$ |
| $a^{(3)}=g\big(z^{(3)}\big)=g\big(\Theta^{(2)}a^{(2)}\big)=\begin{bmatrix}a_1^{(3)}\end{bmatrix}$ | none                                                         |

### bin4.2 - Multiclass Classification

One-vs-all for neural networks: Define the set of resulting classes like:

$y^{(i)}=\begin{bmatrix}1\\0\\0\\0\end{bmatrix},\begin{bmatrix}0\\1\\0\\0\end{bmatrix},\begin{bmatrix}0\\0\\1\\0\end{bmatrix},\begin{bmatrix}0\\0\\0\\1\end{bmatrix}$

AI分为ANI和AGI：narrow和general 

## 5 - Neural Networks: Learning cost函数：binary cross entropy loss function

### 5.1 - Cost Function 

Take the multiclass classification problem as example:

![image-20230810162925999](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230810162925999.png)

**Logistic Regression:** $J(\theta)=-\frac{1}{m}\sum_{i=1}^m\bigg[y^{(i)}\text{log}(h_\theta(x^{(i)}))+(1-y^{(i)})\text{log}(1-h_\theta(x^{(i)}))\bigg]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2\\$

**Neural Network:** $J(\theta)=-\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K\bigg[y^{(i)}\text{log}(h_\theta(x^{(i)})_k)+(1-y^{(i)})\text{log}(1-h_\theta(x^{(i)})_k)\bigg]+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_l+1}\big(\Theta_{j,i}^{(l)}\big)^2\\$

$L$ : total number of layers in the network 

$s_l$ : number of units (not counting bias unit) in layer $l$

$K$ : number of output units/classes

**The first part:** Sum up all the costs for each output class ($K$ in total).

**The second part:** Sum up the squared values of all parameters except bias.

选激活函数，输出层：二分类问题用sigmoid；回归问题且结果可正可负，用线性函数（y=x）；回归问题且结果为正（房价），用relu

hidden层：用relu（用sigmoid梯度下降很慢）

其他：softmax，tanh，leakyrelu，switch 少数情况下表现更好

不用激活函数（y=x），神经网络就相当于线性回归模型；隐藏层用y=x，输出层用sigmoid，相当于逻辑回归模型



以上是二分类问题，接下来是多分类问题，用softmax

![image-20230810171106497](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230810171106497.png)

由于计算机的舍入误差，如下代码更加精确

二分类（舍入误差影响不大）![image-20230810173706597](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230810173706597.png)![image-20230810174122470](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230810174122470.png)

multi classification 舍入误差影响很大![image-20230810174008804](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230810174008804.png)![image-20230810174035247](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230810174035247.png) 

multilabel classification![image-20230810174325314](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230810174325314.png)![image-20230810174446935](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230810174446935.png)



adam算法，比梯度下降好，自动增大或减小alpha的值，且每个b，w的alpha不同。代码设置初始默认值![image-20230810183454068](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230810183454068.png)

除了dense层，还有卷积层，dense层本质上是其中的每个神经元都可以读入每个像素；卷积层中的每个神经元只可读入一部分像素（如3*3），加速，减少过拟合，同理隐藏层只可读入部分神经元；二分类问题可以hidden卷积层，输出dense sigmoid层

### 5.2 - Backpropagation

[反向传播算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/40761721)

[Backpropagation Algorithm | Coursera](https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm)

N节点P参数，一个个求导要N*P

backprop是一个用来计算导数的方法，只要N+P，加速计算，随后使用梯度下降或adam算法



#### **Problem Setting**

Our object is to calculate $\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}\\$ for all $\Theta_{i,j}^{(l)}$ .

$l$ : index of the layer

$i$ : index of the neuron in layer $l$

$j$ : index of the neuron in layer $l+1$

$\Theta_{i,j}^{(l)}$ : weight from the $i$ th neuron in layer $l$ to the $j$ th neuron in layer $l+1$

#### Preliminary Deduction

It is quite hard to calculate $\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}\\$ directly.

By applying the chain rule of derivatives, we get $\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}=\frac{\partial J(\Theta)}{\partial z_j^{(l+1)}}\frac{\partial z_j^{(l+1)}}{\partial\Theta_{i,j}^{(l)}}\\$ .

It is easy to deduce that $\frac{\partial z_j^{(l+1)}}{\partial\Theta_{i,j}^{(l)}}=\frac{\partial\bigg(\sum_{k=0}^{N(l)}\Theta_{k,j}^{(l)}a_k^{(l)}\bigg)}{\partial\Theta_{i,j}^{(l)}}=a_i^{(l)}\\$ . 

So we define "error value" $\delta_j^{(l+1)}=\frac{\partial J(\Theta)}{\partial z_j^{(l+1)}}\\$ , and then we have $\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}=\delta_j^{(l+1)}a_i^{(l)}\\$ .

Since we already got $a_i^{(l)}$, our object is to calculate $\delta_j^{(l+1)}$ for all $j,l$ .

#### Algorithm Summary

1. Perform forward propagation to compute $a^{(l)}$ for $l=1,2,3,...,L$ .

2. Compute $\delta_j^{(L)}=a^{(L)}-y^{(t)}$ .

3. Compute $\delta_j^{(L-1)},\delta_j^{(L-2)},\dots,\delta_j^{(2)}$ , using $\delta^{(l)}=\big(\Theta^{(l)}\big)^T\delta^{(l+1)}\times g'\big(z^{(l)}\big)$ .

4. Compute $\Delta^{(l)}_{i,j}:=\delta_i^{(l+1)}a_j^{(l)}$ , or with vectorization, $\Delta^{(l)}:=\delta^{(l+1)}\big(a^{(l)}\big)^T$ .

5. If we didn't apply regularization, then $D_{i,j}^{(l)}=\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}=\frac{1}{m}\Delta^{(l)}_{i,j}\\$ .

   Otherwise, We have $D_{i,j}^{(l)}=\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}=\begin{cases}\frac{1}{m}\big(\Delta^{(l)}_{i,j}+\lambda\Theta_{i,j}^{(l)}\big)\space,j\ne0\\\frac{1}{m}\Delta^{(l)}_{i,j}\space,j=0\end{cases}$ .

### 5.3 - Random Initialization

Initialize each $\Theta_{i,j}^{(l)}$ to a same number: Bad! All the units will compute the same thing, giving a highly redundant representation.

Initialize each $\Theta_{i,j}^{(l)}$ to a random value in $[-\epsilon,\epsilon]$ : Good! This breaks the symmetry and helps our network learn something useful.

### 5.4 - Training a Neural Network

1. Randomly initialize the weights.
2. Implement forward propagation to get $h_\Theta(x^{(i)})$ for any $x^{(i)}$ .
3. Implement the cost function.
4. Implement backpropagation to compute partial derivatives.
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.



## 6 - Advice for Applying Machine Learning

### 6.1 - Evaluating a Learning Algorithm

Once we have done some trouble shooting for errors in our predictions by: 

- Getting more training examples.
- Trying smaller sets of features.
- Trying additional features.
- Trying polynomial features.
- Increasing or decreasing $\lambda$ .

 One way to break down our dataset into the three sets is:

- Training set: 60%.
- Cross validation set: 20%.    下标cv![image-20230811181030398](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811181030398.png)
- Test set: 20%.

We can move on to evaluate our new hypothesis:

1. Learn $\Theta$ and minimize $J_{train}(\Theta)$ .
2. Find the best model according to $J_{cv}(\Theta)$ .
3. Compute the test set error $J_{test}(\Theta)$ .

![image-20230811175836220](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811175836220.png)

![image-20230811175932689](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811175932689.png)

对于高次回归问题，一个个尝试次数，看Jtest选最佳次数，这样不好![image-20230811180843090](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811180843090.png)引入cv集，专门用20%的数据用于选择最佳次数d，使得在测试时是全新的模型。更广泛的，cv集合可以用于选择神经网络的隐藏层和神经元数量![image-20230811181144028](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811181144028.png)![image-20230811182118058](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811182118058.png)有时侯即欠拟合又过拟合（情况3)![image-20230811182559222](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811182559222.png)

### 6.2 - Bias vs. Variance

#### Checking Model Complexity

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/6-parameter.png" style="zoom:50%;" />

|                         | Training Set Error | Cross Validation Set Error |
| ----------------------- | ------------------ | -------------------------- |
| Underfit (High Bias)    | High               | High                       |
| Overfit (High Variance) | Low                | High                       |

#### Regularization  以上方法同样可以用来选择惩罚项lambda（已选定次数d）

![image-20230811182950010](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811182950010.png) 

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/6-lambda.png" style="zoom:100%;" />

|                                     | Training Set Cost | Cross Validation Set Cost |
| ----------------------------------- | ----------------- | ------------------------- |
| $\lambda$ too large (High Bias)     | High              | High                      |
| $\lambda$ too small (High Variance) | Low               | High                      |

选取基准误差值，如语音识别以人类分辨能力为准![image-20230811183908288](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811183908288.png)

baseline和jtrain的差距决定是否欠拟合，jtrain和jcv的差距决定是否过拟合 ![image-20230811184301957](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811184301957.png) 

#### Learning Curves

欠拟合曲线如下，样本更多，Jtrain会增大，二者后续趋于平缓，又称plateau，add样本无益处，本质是模型能力有限![image-20230811185347474](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811185347474.png)

过拟合，add样本有益处![image-20230811185506867](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811185506867.png) 

总结：过拟合，增加数据集或简化模型（加lambda，减d）；欠拟合，复杂化模型（而非减少样本量 ）![image-20230811185950029](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811185950029.png) 

神经网络，使得无需做bias variance tradeoff，对于中 小数据集，更大的神经网络总可以降低J train，如果Jcv高即过拟合，可以增加数据量；只要做好正则化，神经网络越大越好，除了算的慢![image-20230811190916878](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811190916878.png)![image-20230811191018817](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811191018817.png)![image-20230811191238730](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230811191238730.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/6-learning_curve1.png"  style="zoom:75%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/6-learning_curve2.png"  style="zoom:75%;" />

**Experiencing High Bias:**

Model underfits the training set and cross validation set, getting more training data will not help much.

**Experiencing High Variance:**

Model overfits the training set, getting more training data is likely to help.

### 6.3 - Deciding What to Do Next

- **Getting more training examples:** Fixes high variance

- **Trying smaller sets of features:** Fixes high variance

- **Adding features:** Fixes high bias

- **Adding polynomial features:** Fixes high bias

- **Decreasing $\lambda$:** Fixes high bias

- **Increasing $\lambda$:** Fixes high variance.

垃圾邮件分类器，垃圾邮件常常用0代替o等手段逃过filter

### 6.4 - Error Analysis ( Quite Practical and Useful !!!!!!!!!! ) 前提是人能判断 

对于结果错误的样本，进行手动分类。拼写错误算法很复杂，但只能解决很少一部分垃圾邮件，研究意义不大；但是有些错误量大的类别，则收集更多数据或提取更多特征。若错误样本量大，随机抽取100个进行手动分类![image-20230812173601114](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812173601114.png)

- Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.
- Plot learning curves to decide if more data, more features, etc. are likely to help.
- Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.
- Make sure the quick implementation incorporated a single real number evaluation metric.

ADD DATA增加更多数据，cv中把图片缩放，改对比度，旋转，镜像再加入训练样本很实用，或随机扭曲；类似方法也适用于语音识别，但加随机噪音无意义，结合实际，想想测试集会不会出现这种？ ![image-20230812175039639](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812175039639.png)![image-20230812175134646](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812175134646.png)

OCR，识别出图中文字区域，训练数据可用计算机随机生成（随机字体随机背景），仅仅适用于cv![image-20230812175559385](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812175559385.png)

对于没有那么多数据的情况，使用迁移学习transfer learning

法1:要训练mnist，先训练一个网络分类1000个东西（任意其他领域），然后把输出换成10个，w5b5要换新的随机数，然后用梯度下降orADAM仅仅改变w5b5，在数据集很少时适用；法2:在后续训练中改变所有w和b，在数据集较多时适用；1000class叫supervised pretraining，10class叫fine tuning，有很多开源的pretraining网络![image-20230812201211978](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812201211978.png)

维护机器学习系统；随着用户增加升级服务器，注重隐私，与时俱进更新数据集，如新出现的流行语；MLOps，指部署维护组![image-20230812203725715](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812203725715.png)

注重道德准则，生成假视频（奥巴马讲话），散布煽动性言论，虚假评论，诈骗，通过diverse team brainstorm想可能出错的地方，参考行业准则，在发布前测试是否有种族歧视，准备缓解计划（退回之前版本），为汽车事故准备缓解计划![image-20230812204044096](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812204044096.png)

### 6.5 - Handling Skewed Data 倾斜的数据集，积极和消极标签占比差距很大，如罕见疾病侦测

且衡量模型的方法，测试误差的计算方法要变（print y=0有99.5%准确率但是是垃圾算法）用混淆矩阵![image-20230812212058806](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812212058806.png)

如果这个病需要昂贵侵入性治疗，不治也没什么大不了，提高门槛，升precision降recall；不治就可能要死，降低门槛![image-20230812212855954](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812212855954.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812213043484.png" alt="image-20230812213043484" style="zoom:50%;" />

用F1分数自动做 precision recall tradeoff，得到一个评估分数从而更好判断模型好坏，P和R有一个低模型就没用，F1是在 更注意偏低的那个分数的前提下计算平均值，即调和平均值![image-20230812213606207](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812213606207.png)![image-20230812213626978](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230812213626978.png)

To better handle skewed data, we can use precision and recall to evaluate the model.

$\text{precision}=\frac{\text{true positive}}{\text{true positive + false positive}}\\$

$\text{recall}=\frac{\text{true positive}}{\text{true positive + false negative}}\\$

We can further give a trade off between precision and recall using what is called F1 score.

$\text{F1 score}=2\frac{\text{precision}\times\text{recall}}{\text{precision + recall}}\\$



神经网络和决策树是两个最常用的算法，决策树算法用于选择work的最好的tree![image-20230813210312702](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230813210312702.png)

选择decision node：maximize purity，能用DNA分类就不用外貌

决定树的depth（何时停止分裂）：小树不容易过拟合

entropy用于Measure inpurity<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230813212026672.png" alt="image-20230813212026672" style="zoom:50%;" />

二分类的情况下，为了一个公式从cat和not cat两方面衡量entropy，log用于压缩函数<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230813212414572.png" alt="image-20230813212414572" style="zoom:50%;" />

有些库里有Gini creteria 类似用于衡量impurity

decision node在reduce impurity的时候被采用，即熵减，即信息增益information gain；判断不再split的标准建设就是信息增益不大就停止，防止过拟合![image-20230813213157706](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230813213157706.png)![image-20230813213620741](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230813213620741.png)

递归，小决策树到大决策树；用cv集决定决策树的规模或用信息增益的threshold

对于一个feature有大于2个可能取值，用one-hot encoding<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230814154433091.png" alt="image-20230814154433091" style="zoom:50%;" />

对于连续取值的feature，选信息增益最高的分割线；Choose the 9 mid-points between the 10 examples as possible splits, and find the split that gives the highest information gain.<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230814155541386.png" alt="image-20230814155541386" style="zoom:50%;" />

回归树，以预测体重为例，尽量减少子集的方差，尽量增大方差减少的值；同理递归![image-20230814160439475](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230814160439475.png)

建立很多个树，test样本分别输入，以这些树的结果为投票得出最终结果![image-20230814161931911](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230814161931911.png)

用放回抽样产生更多tree：从训练样本放回抽样抽10个样本（样本可重复），然后训练出一个树，循环B次，得到bagged decision tree或随机森林算法。B的选择，大了不会过拟合，但太大时（100，1000 ）性能不会进一步提升。这些树的根节点或者根节点附近的节点通常是一样的，为了减少雷同增加准确性，对于特征也取一个子集供树选择。N个特征，常取k个

更好的算法 boost decision tree，XGboost 最常用（加了正则化防过拟合 ），适合机器学习竞赛；deliberate practice 刻意训练薄弱部分；即对于完整的训练集做一个结果测试（用已有的森林模型 ），找出结果错误的样本，在训练下一个树随机选择样本的时候更高几率选择错误的样本。XGboost原理复杂，用开源的就行![image-20230814191905009](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230814191905009.png)

最好用的就是dl和XGboost；XGboost适合可以写在表格中的数据，训练很快![image-20230814192346968](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230814192346968.png)

## 7 - Support Vector Machines 2022的课被淘汰了 没讲

分类问题，用直线或平面分开，最大化距离<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230814162252774.png" alt="image-20230814162252774" style="zoom:50%;" />

核技巧，通过非线性变换把原本无法分类的数据变换到高维，在高维中分类<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230814162349208.png" alt="image-20230814162349208" style="zoom:50%;" />

即x->fx，区分两类数据在数学上即计算，即使变换很复杂但核函数计算很简单，用gamma参控制分类边界的平滑程度<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230814162928483.png" alt="image-20230814162928483" style="zoom:50%;" />

###  - SVM Hypothesis

Just like liner regression, we gain the cost function of SVM:

$J(\theta)=C\sum_{i=1}^{m}\big[y^{(i)}\text{cost}_1(\Theta^Tx^{(i)})+(1-y^{(i)})\text{cost}_0(\Theta^Tx^{(i)})\big]+\frac{1}{2}\sum_{j=1}^{n}\theta_j^2\\$

The cost function for each label looks like:

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/7-cost.png" style="zoom:50%;" />

So our hypothesis is:

$h_\theta(x)=\begin{cases}1\space,\Theta^Tx\ge1\\0\space,\Theta^Tx\le-1\end{cases}$

Given a trained weight $\Theta$ , we want each example to be classified correctly, so we have:

$\sum_{i=1}^{m}\big[y^{(i)}\text{cost}_1(\Theta^Tx^{(i)})+(1-y^{(i)})\text{cost}_0(\Theta^Tx^{(i)})\big]=0\\$

$\begin{aligned}\text{s.t.}\quad&\Theta^Tx^{(i)}\ge1\quad\space\space\space \text{if}\quad y^{(i)}=1\\&\Theta^Tx^{(i)}\le-1\quad \text{if}\quad y^{(i)}=0\end{aligned}$

So our final objective is to calculate the following equation:

$\min_\theta\space \frac{1}{2}\sum_{j=1}^{n}\theta_j^2\\$

$\begin{aligned}\text{s.t.}\quad&\Theta^Tx^{(i)}\ge1\quad\space\space\space \text{if}\quad y^{(i)}=1\\&\Theta^Tx^{(i)}\le-1\quad \text{if}\quad y^{(i)}=0\end{aligned}$

**Note: SVM is sensitive to noise.**

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/7-noise.png" style="zoom:50%;" />

### 7.2 - SVM Mathematical Inducement

Take $\Theta=\begin{bmatrix}\theta_1\\\theta_2\end{bmatrix}$ as an example: (regard $\theta_0$ as 0)

#### The Cost Function Part

$\begin{aligned}\min_\theta\space\frac{1}{2}\sum_{j=1}^{2}\theta_j^2&=\frac{1}{2}\big(\theta_1^2+\theta_2^2\big)\\&=\frac{1}{2}\bigg(\sqrt{\theta_1^2+\theta_2^2}\bigg)^2\\&=\frac{1}{2}\Theta^T\Theta\\&=\frac{1}{2}\big\|\Theta\big\|^2\end{aligned}$

#### The Condition Part

$\begin{aligned}\text{s.t.}\quad&\Theta^Tx^{(i)}\ge1\quad\space\space\space \text{if}\quad y^{(i)}=1\\&\Theta^Tx^{(i)}\le-1\quad \text{if}\quad y^{(i)}=0\end{aligned}$

Essentially $\Theta^Tx^{(i)}$ is the dot product of $\Theta$ and $x^{(i)}$ , it looks like:

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/7-dot_product.png" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/7-dot_product2.png" style="zoom:50%;" />

Let $p^{(i)}$ be the projection of $x^{(i)}$ to $\Theta$ , then we get:

$\begin{aligned}\text{s.t.}\quad&\big\|\Theta\big\|\cdot p^{(i)}\ge1\quad\space\space\space \text{if}\quad y^{(i)}=1\\&\big\|\Theta\big\|\cdot p^{(i)}\le-1\quad \text{if}\quad y^{(i)}=0\end{aligned}$

#### Summary

$\min_\theta\space\frac{1}{2}\big\|\Theta\big\|^2\\$

$\begin{aligned}\text{s.t.}\quad&\big\|\Theta\big\|\cdot p^{(i)}\ge1\quad\space\space\space \text{if}\quad y^{(i)}=1\\&\big\|\Theta\big\|\cdot p^{(i)}\le-1\quad \text{if}\quad y^{(i)}=0\end{aligned}$

To minimize $\frac{1}{2}\big\|\Theta\big\|^2\\$ , we need to maximize $\big|p^{(i)}\big|$ , which denotes the margin between the two classes.

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/7-margin.png" style="zoom:50%;" />

### 7.3 - Kernels

For some non-linear classification problems, we can remap $x^{(i)}$ to a new feature $f^{(i)}$ using kernel function.

#### Procedure

**Given $x^{(1)},x^{(2)},\dots,x^{(m)}$ , we can apply kernel function using following steps:**

1. Choose each training example $x^{(i)}$ as the landmark $l^{(i)}$ .

   We have $l^{(i)}=x^{(i)}$ for all $i$ .

   Finally we acquire $m$ landmarks $l^{(1)},l^{(2)},\dots,l^{(m)}$ in total.

2. For each training example $x^{(i)}$ , using all $m$ landmarks to calculate new features $f^{(i)}$ .

   The new feature $f^{(i)}$ have $m$ dimensions, and $f_j^{(i)}=f(x^{(i)},l^{(j)})\\$ .

   Finally we acquire $m$ features $f^{(1)},f^{(2)},\dots,f^{(m)}$ in total.

3. Using $f^{(1)},f^{(2)},\dots,f^{(m)}$ to train the SVM model.

   The new training objective is $\min_\theta\space C\sum_{i=1}^{m}\big[y^{(i)}\text{cost}_1(\Theta^Tf^{(i)})+(1-y^{(i)})\text{cost}_0(\Theta^Tf^{(i)})\big]+\frac{1}{2}\sum_{j=1}^{n}\theta_j^2\\$ . ( $n=m$ )

$f^{(i)}$ : $m$ dimensions.

#### Kernel Selection

Linear Kernel: Same as no kernel.

Gaussian Kernel: $f(x,l)=\exp(-\frac{\|x-l\|^2}{2\sigma^2})\\$ .

Polynomial Kernel: $f(x,l)=(x^Tl+b)^a$ .

And so on...

#### Parameter Selection

|       | $\sigma^2$ (Gaussian Kernel)                                 | $C\ (=\frac{1}{\lambda})\\$ (Cost Function) |
| ----- | ------------------------------------------------------------ | ------------------------------------------- |
| Large | Features $f^{(i)}$ vary more smoothly. <br>**High bias.**    | Small $\lambda$ . <br>**High variance.**    |
| Small | Features $f^{(i)}$ vary less smoothly. <br/>**High variance.** | Large $\lambda$ . <br/>**High bias.**       |

#### SVM vs Logistic Regression vs Neural Network

Suppose we have $m$ training examples. Each example contains $n$ features.

|                                       | SVM                                         | Logistic Regression               | Neural Network                       |
| ------------------------------------- | ------------------------------------------- | --------------------------------- | ------------------------------------ |
| $n>m$ <br>(e.g. $n=100,m=10$ )        | **Linear Kernel.** <br>(avoid overfitting)  | Work fine.                        | Always work well.                    |
| $n<m$ <br>(e.g. $n=100,m=1000$ )      | **Gaussian Kernel.** <br>(or other kernels) | Work fine.<br>(SVM may be better) | Always work well.<br>(SVM is faster) |
| $n\ll m$ <br>(e.g. $n=100,m=100000$ ) | **Linear Kernel.**<br>(reduce time cost)    | Work fine.                        | Always work well.                    |



## 8 - Unsupervised Learning

### 8.1 - Clustering with K-means 基因分类

#### Algorithm Steps

分两类：先随机选两中心点（实际中一般选择在随机两个样本点上），计算样本到两中心点间的距离，到红点近点标红色，到蓝点近点标蓝色，然后把红中心点移到红点集的中心；如此循环；分多类的情况下，如果有个中心点一个小弟都没有，则删除这一类 ，或重新随机初始化中心点<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230815172334409.png" alt="image-20230815172334409" style="zoom:50%;" />

Assuming that we have $m$ training examples. We want to divide these examples into $K<m$ clusters.

1. Randomly pick $K$ training examples $x_1,x_2,\dots,x_K$ .

   Initialize $K$ cluster centroids $\mu_1,\mu_2,\dots,\mu_K$ using picked examples.

   We have $\mu_i=x_i$ .

2. Loop:

   For all $x^{(i)}$ , choose the nearest centroid as its cluster, denotes $c^{(i)}$ .

   For all clusters, update $\mu_k$ with the mean value of the points assigned to it.

3. Stop when there is no change in each cluster.

#### Optimization Objective 本质上是minimize distortion function（即cost function ）必收敛

Minimize the average distance of all examples to their corresponding centroids.

$J(C,\Mu)=\frac{1}{m}\sum_{i=1}^m\|x^{(i)}-\mu_{c^{(i)}}\|\\$<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230815180203607.png" alt="image-20230815180203607" style="zoom:50%;" />

最终分类结果经常取决于初始化的中心点位置，跑多次选一个J最小的，通常随机初始化50-1000组初始值 <img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230815181752881.png" alt="image-20230815181752881" style="zoom:50%;" />

#### Choosing the Number of Clusters $K$

Elbow Method:几乎不用， 把K各个取值跑一遍，K越大J必定越小，所以取合适值而非最大 <img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230815182412197.png" alt="image-20230815182412197" style="zoom:33%;" />

Plot $J(C,\Mu)$ while increasing $K$ . Choose the "elbow point" as the answer,

More Importantly:结合实际，如更多尺寸更好但是成本高，进行一个tradeoff<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230815182655188.png" alt="image-20230815182655188" style="zoom:50%;" />

Evaluate based on a metric for how well it performs for later purpose.

### 8.2 - Principal Component Analysis (PCA)

#### Algorithm Steps

Assuming that we have $m$ training examples. Each example is a $n$ dimensional vector.

We want to compose each example into a $k$ dimensional vector. ( $k<n$ ) 

1. Do feature scaling or mean normalization.

2. Compute "covariance matrix" $\Sigma$ :

   $\Sigma=\frac{1}{m}\sum_{i=1}^nx^{(i)}\big(x^{(i)}\big)^T\\$ ( $n\times n$ )

3. Compute "eigenvector matrix" $U$ and "eigenvalue matrix" $V$ of $\Sigma$ :

   $U=\begin{bmatrix}u^{(1)}&u^{(2)}&\cdots&u^{(n)}\end{bmatrix}$ ( $n\times n$ )

   $V=\begin{bmatrix}v^{(1)}&v^{(2)}&\cdots&v^{(n)}\end{bmatrix}$ ( $1\times n$ )

   $u^{(i)}$ is the $i$ th eigenvector of $\Sigma$ . ( $n\times 1$ )

   $v^{(i)}$ is the $i$ th eigenvalue of $\Sigma$ .

4. Select the largest $k$ eigenvalues from $V$ , concatenate the corresponding $k$ eigenvectors together as a new matrix $U'$ .

   $U'=\begin{bmatrix}u'^{(1)}&u'^{(2)}&\cdots&u'^{(k)}\end{bmatrix}$ ( $n\times k$ )

5. Compute new features matrix $Z$ .

   $Z=XU'=\begin{bmatrix}z^{(1)}\\z^{(2)}\\\vdots\\z^{(n)}\end{bmatrix}$ ( $m\times k$ )

#### Optimization Objective

Minimize the average distance of all examples to the hyperplane.

$\min_U\space\frac{1}{m}\sum_{i=1}^md\big(x^{(i)},U\big)\\$

#### Choosing the Number of Principal Components $K$

We can reconstruct approximately our $n$ dimensional examples back with some error:

$X_{re}=Z(U')^T=\begin{bmatrix}x_{re}^{(1)}\\x_{re}^{(2)}\\\vdots\\x_{re}^{(n)}\end{bmatrix}$ ( $m\times n$ )

We can then compute the average information loss:

$L=\frac{1}{m}\sum_{i=1}^m\|x^{(i)}-x_{re}^{(i)}\|^2\\$

So the loss rate is:

$r=\frac{\frac{1}{m}\sum_{i=1}^m\|x^{(i)}-x_{re}^{(i)}\|^2}{\frac{1}{m}\sum_{i=1}^m\|x^{(i)}\|^2\\}=\frac{\sum_{i=1}^m\|x^{(i)}-x_{re}^{(i)}\|^2}{\sum_{i=1}^m\|x^{(i)}\|^2\\}\\$

We can choose the smallest value of $k$ that satisfies:

$r<0.01$ (99% of variance retained)

#### Optimization Objective (recap)

With the reconstructed examples, our training objective can also be described as:

$\min_U\space\frac{1}{m}\sum_{i=1}^m\|x^{(i)}-x_{re}^{(i)}\|^2\\$

#### Applications

1. Speed up computation by composing features.
2. Vasualization.

**Notice:** It is a bad way to use PCA to preventing overfitting. ( This might work, but why not use regularization instead? )



## 9 - Anomaly Detection异常检测，打字过快，交易过多，进行robot验证；检测飞机零件是否出问题

### 9.1 - Density Estimation

高斯分布 正态分布<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816150656836.png" alt="image-20230816150656836" style="zoom:50%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816150709322.png" alt="image-20230816150709322" style="zoom:50%;" />![image-20230816150916142](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816150916142.png)面积均为1 

一个feature的情况 ；n个feature![image-20230816162524915](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816162524915.png)![image-20230816163042786](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816163042786.png)

#### real-number evaluation 评价异常检测系统![image-20230816163955973](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816163955973.png)

用cv集衡量$epsilon$或增减feature![image-20230816164436079](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816164436079.png)

Course2week3 ![image-20230816180442372](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816180442372.png)

以上加了labely，则对比：异常检测（非监督）vs监督学习![image-20230816181349111](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816181349111.png)

异常检测：新型诈骗，发现新的制造故障，新型黑客手段 ；监督学习：手机盖损坏（常有的错误）![image-20230816181603763](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816181603763.png)

谨慎选择feature对于异常检测尤其重要，把非高斯分布的feature高斯化![image-20230816182607542](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816182607542.png)![image-20230816182827736](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816182827736.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816182922128.png" alt="image-20230816182922128" style="zoom:50%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816183028527.png" alt="image-20230816183028527" style="zoom:50%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816183135911.png" alt="image-20230816183135911" style="zoom:50%;" />

log的时候避免x中有0，➕0.001即可；有自动的高斯化数据算法，但手动尝试没什么不好的

已有的feature检测不出异常，尝试找到新的feature来共同鉴定异常![image-20230816184522357](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816184522357.png)![image-20230816184919022](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816184919022.png)

#### Problem Motivation

We have a **very skewed dataset** $\big\{x^{(1)},x^{(2)},\dots,x^{(m)}\big\}$ , in which the number of negative examples is much larger than that of positive ones (e.g. $m_0=10000,m_1=20$ ). Our objective is to detect the **anomaly examples** (positive ones).

One possible way is to use **supervised learning algorithms** to build a classification model. But we have too little positive examples that our model can't fit all possible "types" of anomaly examples. So future anomaly examples may looking nothing like the previous ones we used for training. **As a result, our model using supervised learning algorithm may behave quite bad.**

To handle extreme datasets like this, we need to use another method called **"density estimation"**.

#### Algorithm Formulation

Suppose we have $m$ training examples (all negative). Each example contains $n$ features.

1. Assume that our training examples follow a specific distribution $D\big(\Theta\big)$ , we have:

   $x^{(i)}\sim D\big(\Theta\big)$ for $i=1,2,\dots,m$

2. We can then estimate parameters $\Theta$ and fit this distribution.

3. For a new example $x^{new}$ , we can calculate the possibility that $x^{new}$ follows the distribution $D$ :

   $p^{new}=P\big(x_1^{new},x_2^{new},\dots,x_n^{new};\Theta\big)$

4. If $p^{new}<\epsilon$ , then we think that $x^{new}$ is a anomaly point.

#### Distribution Selection

**Single Variate Gaussian Distribution:**

Each feature follows a single variate gaussian distribution. (all features are independent of each other)

$x_k\sim N\big(\mu_k,\sigma_k^2\big)$ for $k=1,2,\dots,n$

$p^{new}=\prod_{k=1}^n P_k\big(x_k^{new};\mu_k,\sigma_k^2\big)\\$

**Multivariate Gaussian Distribution:**

All features together follow a $n$ variate gaussian distribution.

$x_1,x_2,\dots,x_n\sim N\big(\mu_1,\mu_2,\dots,\mu_n,\sigma_1^2,\sigma_2^2,\dots,\sigma_n^2\big)$

$p^{new}=P\big(x_1^{new},x_2^{new},\dots,x_n^{new};\mu_1,\mu_2,\dots,\mu_n,\sigma_1^2,\sigma_2^2,\dots,\sigma_n^2)\\$

#### Dataset Division

|                   | Training Set | Cross Validation Set | Test Set |
| ----------------- | ------------ | -------------------- | -------- |
| Negative Examples | 60%          | 20%                  | 20%      |
| Positive Examples | 0%           | 50%                  | 50%      |

### 9.2 - Recommender System

collaborative filtering，应用场景：You run an online bookstore and collect the ratings of many users. You want to use this to identify what books are "similar" to each other (i.e., if a user likes a certain book, what are other books that they might also like?)

r表示是否评分了，y表示具体几分 ![image-20230816185545477](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816185545477.png)

w，b为已经由训练得到的，若知道一些风格feature![image-20230816190329956](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816190329956.png)

对于单个user的cost函数，橙色为正则化项，m去掉不影响优化结果 ![image-20230816190705865](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816190705865.png)![image-20230816190911352](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816190911352.png)

若不知道一个电影的风格feature，w，b为已经由训练得到的，由用户评分推出电影风格![image-20230816191914164](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816191914164.png)![image-20230816192255559](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816192255559.png)

J合二为一，用梯度下降![image-20230816192547743](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816192547743.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816192805783.png" alt="image-20230816192805783" style="zoom:50%;" />

以上是一到五星评价法，以下是binary评价法（点赞与否）![image-20230816193559415](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816193559415.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816193652242.png" alt="image-20230816193652242" style="zoom:50%;" />![image-20230816193827744](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230816193827744.png)

利用tensorflow  autodiff（collaborative filter和dense layer等层类型不适配）![image-20230821161416580](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821161416580.png)

用Adam算法![image-20230821161656994](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821161656994.png) 

#### Content Based

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/9-movie.png" style="zoom:50%;" />

**Suppose we have $n_m$ movies and $n_u$ users.**

$r(i,j)$ : equals 1 if user $j$ has rated movie $i$ (else 0)

$m^{(j)}$ : number of movies rated by user $j$

$m'^{(i)}$ : number of users rated movie $i$

$x^{(i)}$ : feature vector of movie $i$

$\theta^{(j)}$ : parameter vector of user $j$

$y^{(i,j)}$ : rating by user $j$ on movie $i$

**We can train a linear regression model for every user.**

For movie $i$, user $j$ , our predicted rating is ${\theta^{(j)}}^Tx^{(i)}$ .

The cost function for user $j$ is:

$J\big(\theta^{(j)}\big)=\frac{1}{2m^{(j)}}\sum_{i:r(i,j)=1}^{m^{(j)}}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2m^{(j)}}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\\$

Combine all $J\big(\theta^{(j)}\big)$ together, the global cost function is:

$J\big(\theta^{(1)},\theta^{(2)},\dots,\theta^{(n_u)}\big)=\frac{1}{2n_m}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}^{m^{(j)}}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2n_m}\sum_{j=1}^{n_u}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\\$

#### Collaborative Filtering

Given $x^{(1)},x^{(2)},\dots,x^{(n_m)}$ , we can estimate $\theta^{(1)},\theta^{(2)},\dots,\theta^{(n_u)}$ by minimizing:

$J\big(\theta^{(1)},\theta^{(2)},\dots,\theta^{(n_u)}\big)=\frac{1}{2n_m}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}^{m^{(j)}}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2n_m}\sum_{j=1}^{n_u}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\\$

Similarly, given $\theta^{(1)},\theta^{(2)},\dots,\theta^{(n_u)}$ , we can also estimate $x^{(1)},x^{(2)},\dots,x^{(n_m)}$ by minimizing:

$J\big(x^{(1)},x^{(2)},\dots,x^{(n_m)}\big)=\frac{1}{2n_u}\sum_{i=1}^{n_m}\sum_{j:r(i,j)=1}^{m'^{(i)}}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2n_u}\sum_{i=1}^{n_m}\sum_{k=1}^n\big(x_k^{(i)}\big)^2\\$

Notice that both function have the same objective with the regularization term removed:

$\min\space\frac{1}{2}\sum_{(i,j):r(i,j)=1}^{(n_m,n_u)}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2\\$

So we can combine these two cost functions together:

$J\big(x^{(1)},\dots,x^{(n_m)},\theta^{(1)},\dots,\theta^{(n_u)}\big)=\frac{1}{2}\sum_{(i,j):r(i,j)=1}^{(n_m,n_u)}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n\big(x_k^{(i)}\big)^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\\$

We can randomly initialize $x^{(1)},\dots,x^{(n_m)},\theta^{(1)},\dots,\theta^{(n_u)}$ and use gradient descent algorithm to estimate the parameters.

#### Mean Normalization 归一化，加速训练；未评分的默认为平均分而不是0

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/9-movie2.png" style="zoom:50%;" />

![image-20230821155751198](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821155751198.png)

For a person who hasn't rated any movie, the objective turns out to be:

$\begin{aligned}J\big(\theta^{(j)}\big)&=\frac{1}{2m^{(j)}}\sum_{i:r(i,j)=1}^{m^{(j)}}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2m^{(j)}}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\\&=\frac{\lambda}{2m^{(j)}}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\end{aligned}$

So the estimates parameter will be like $\theta_1^{(j)}=\theta_2^{(j)}=\dots=\theta_n^{(j)}=0$ .

It is unreasonable to predict that a person will rate every movie 0 score.

**We can fix this problem by applying mean normalization for each movie:**

$\overline{y^{(i)}}=\frac{1}{m'^{(i)}}\sum_{j:r(i,j)=1}^{n_u}y^{(i,j)}\\$ (mean rating of movie $i$)

$y^{(i,j)}:=y^{(i,j)}-\overline{y^{(i)}}\\$

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/9-movie3.png" style="zoom:50%;" />

Then our predicted rating "0" will be a neutral score.

推荐类似items![image-20230821165208267](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821165208267.png)![image-20230821170231438](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821170231438.png)

缺点，新用户或新电影会误判，且信息利用不充分![image-20230821165509288](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821165509288.png)

更好的content-based filtering![image-20230821170548752](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821170548752.png)

性别：男，女，未知，因此one-hot；size，用户1500，电影50![image-20230821170957629](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821170957629.png)

由x计算出v向量，用户和电影的v向量必须等长；可以理解为：电影的v向量代表每种风格的含量，用户的代表对每种风格的喜爱程度![image-20230821171307082](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821171307082.png)

![image-20230821172717025](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821172717025.png)

y为已由一定用户和电影训练出的评分值，一个J同时训练两个神经网络![image-20230821172914976](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821172914976.png)

![image-20230821173254131](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230821173254131.png)

样本量大时候计算量太大，需要一些技巧，分两步：retrival中的电影不一定要用户喜欢，挑一些可能的电影做内积ranking减少计算量；retrival的电影多则推荐质量升高，耗时变长![image-20230822101825561](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230822101825561.png)![image-20230822102005885](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230822102005885.png)![image-20230822102222810](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230822102222810.png)

道德问题，recommendsystem通常基于利益最大化 或  最大化用户参与量![image-20230822102859920](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230822102859920.png)![image-20230822103040122](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230822103040122.png)

L2正则化 ![image-20230822103551909](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230822103551909.png)



强化学习（具有反馈机制 reward function）小错误-1，大错误-1000；自动控制直升机，机器狗![image-20230824131904889](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824131904889.png)

state，action，reward，next state；terminal state![image-20230824133409719](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824133409719.png)

discount factor奖励因子=0.9，0.99，0.999，也有负数；在金融问题中解释为利率或金钱时间价值![image-20230824134027167](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824134027167.png)

在此取0.5，前两张图数字是从某个state出发最终得到的reward![image-20230824135232373](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824135232373.png)

要学出一个policy，行动的方法 <img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824135542436.png" alt="image-20230824135542436" style="zoom:50%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824135626314.png" alt="image-20230824135626314" style="zoom:50%;" />

MDP马尔可夫过程，future depends on current state![image-20230824140216430](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824140216430.png)

State action value function or Qfunction or Q*functio 已知最佳policy，计算Q值

Q（2，->）就是先往右一格再往左三格![image-20230824190856793](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824190856793.png)![image-20230824191121945](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824191121945.png)

贝尔曼方程，reward已计算![image-20230824192130085](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202309281124009.png)Terminal state中简化为Q=R(s)，R(s)又称为immediate reward![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824192407866.png)

![image-20230824192744160](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824192744160.png)

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202309281131404.png)

Random (stochastic) environment，让车往左，车有90%概率往左，10%往右（misstep probablity）。做1000次实验，取平均return，maximize平均return  ![image-20230824193320147](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824193320147.png)![image-20230824193508037](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824193508037.png)

目前是在少量离散的state做选择，接下来使用连续state，且s变成向量，三个位置三个速度<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824195435816.png" alt="image-20230824195435816" style="zoom:50%;" />

state是8维向量，action是左右下不动四维向量<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824200440567.png" alt="image-20230824200440567" style="zoom: 25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824200654375.png" alt="image-20230824200654375" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824200805457.png" alt="image-20230824200805457" style="zoom:25%;" />

深度强化学习；神经网络输入current state和action 输出 Q值；onehot编码；用训练后的网络输出四种action的Q值，选最大的。![image-20230824201202228](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824201202228.png)

训练数据库，input x12维，output Q（y ）。随机取s和a，在月球模拟器得到R和s‘，以tuple存储 ；s和a组成x向量，R和s‘用来计算y（贝尔曼方程）；Q函数就是神经网络模型，先随机初始化网络参数，随机初始化Q，Q是最好选择下的return。![image-20230824201937783](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/image-20230824201937783.png)

（replay buffer，仅保留最近10000个样本） ；DQNorDQ algorithm，deep Q-network![image-20230824202732143](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202309281102111.png)

更高效实用的架构，直接输出四种action的Q![image-20230928140406197](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202309281404972.png)

在学习过程中选择action， epsilon-gredy policy or exploration step剥削步骤；常取epsilon=0.05，greedy95% explore5![image-20230928142141557](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202309281421142.png)

也有算法逐渐降低epsilon的值

监督学习，learning rate变小，时间变长，别的影响不大；强化学习，超参数（如epsilon）设置不太好，时间会长几十倍，调参很难

进一步改进：mini-batching和soft update

mini-batching 加速监督学习/RL：当数据太多，如100million，每个iteration选1000个数据进行训练，而不是全部

soft update：防止mini-batch训练有一个不幸的数据集使得Q函数向反收敛方向前进；原本Q中W=Wnew，B=Bnew，现在改成，可以更好地收敛<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202309281440666.png" alt="image-20230928144056477" style="zoom:25%;" />

强化学习被炒作了，其实应用价值有限，体系不成熟；现实世界不会给一个reward分数![image-20230928144616283](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202309281446341.png)

![image-20230928145749336](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202309281457398.png)

![image-20230928150345535](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202309281503598.png)

robotics换个演示环境就不行了，不能generalize；图片有固定格式，机器人数据集不同硬件格式不同意，不具有共享性，未标准化 

meta-learning 训练学习的方法，拥有以前的经验，在新环境中通过摸索和经验进行学习，双层优化 

meta reinforcement learning 用于学生作业评分

## 10 - Large Scale Machine Learning

### 10.1 - Stochastic Gradient Descent

| (Each Iteration)   | (Batch) Gradient Descent                                     | Stochastic Gradient Descent                                | Mini-batch Gradient Descent                                  |
| ------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------ |
| number of examples | $m$                                                          | $1$                                                        | $b<m$                                                        |
| cost function      | $J(\theta)=\frac{1}{m}\sum_{i=1}^m\text{Cost}\big(h_\theta(x^{(i)}),y^{(i)}\big)$ | $J(\theta)=\text{Cost}\big(h_\theta(x^{(i)}),y^{(i)}\big)$ | $J(\theta)=\frac{1}{b}\sum_{i=1}^b\text{Cost}\big(h_\theta(x^{(i)}),y^{(i)}\big)$ |

| (Each Epoch)         | (Batch) Gradient Descent | Stochastic Gradient Descent | Mini-batch Gradient Descent |
| -------------------- | ------------------------ | --------------------------- | --------------------------- |
| number of examples   | $m$                      | $m'<m$                      | $b<m$                       |
| number of iterations | $1$                      | $m'$                        | $1$                         |
| randomly shuffle     | No                       | Yes                         | Yes                         |

Choosing $m'$ : An example is $m'=0.1m$

### 10.2 - Online Learning

For some online learning problem, we have a stream of data that changes every time.

We can then train our model **only based on the latest data** and **discard the used data**.

Then our model can **fit the change of data stream** and get the latest features.

### 10.3 - Map Reduce

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/10-map_reduce.png" style="zoom:50%;" />



## 11 - Problem Description and Pipeline

### Real Life Example: Photo OCR

#### **Sliding Window Algorithm**

1. Repeatedly move a window in a fixed step to detect different parts of a picture.

2. If the content in a window part is likely to be "text", mark it as "text parts".
3. For all continuous "text parts", use a smaller window to do character segmentation.
4. For all characters, do character recognition.

#### Getting More Data

**Artificial Data:**

We can generate data autonomously using font libraries in the computer.

In this way, we have theoretically unlimited training data.

**Crowd Source:**

Hiring the crowd to label data.

**Special Attentions:**

1. Make sure we have a low-bias classifier, otherwise we should increase features in our model instead.

2. How much time it will save us if using artificial data rather than collecting real life data.



