# INTRO

Ed discussion, public/private with code (TA and prof); email, only prof

DIS attendance optional, no HW drop, no curve

超1分钟记为1天，共5天

kaggle一天只能交两次

<10 lines code shared, million lines code check cheating, 61b抓了100个cheating

Handwritten / latex

今年夏天左侧出了python版本，但是reading list章节序号是R版本的

![image-20240121000322951](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401211603054.png)

HW2 数学基础

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401311305144.png)

# Math4ml





# LEC1 Intro

jaggies 锯齿状 

![image-20231126151114032](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311261511127.png)

![image-20231126151304958](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202311261513981.png)

![](https://cdn.jsisdelivr.net/gh/yuhengtu/typora_images@master/img/202311261522342.png)

hyperplane decision boundary

outlier 异常值 

validation: select hyperparameter/algorithm

test is optional

找出最好的hyperparameter，之后用train和valid data一起trian一个最终版本

validation error usually > training error



# LEC2 Linear classifiers, centroid method, Perceptron

n: sample/observations

d: features/predictors

sample point / a feature vector / independent variables

decision function/predictor function/discriminant function

sample $x\in\mathbb{R}^d$, (decision boundary/isosurface for the isovalue 0) usually ![image-20240130200624287](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401311206447.png)

linear classifier: The decision boundary is a line/plane

Euclidean inner product/dot product

Euclidean norm/Euclidean length of vector: ![image-20240130202700761](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401311227850.png)

![image-20240130202736272](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401311227307.png) 



![image-20240130220244731](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401311402816.png) 

set H -> hyperplane (A line in 2D, a plane in 3D)

- it has dimension d-1 and it cuts the d-dimensional space into two halves
- it’s flat and it’s infinite

Let x, y be 2 points on H. Then w · (y - x) = 0. (-$\alpha$ - (-$\alpha$) = 0)

so w is a normal vector 法向量 of H, perpendicular, orthogonal

$\alpha$ = 0 情况，H pass origin，w决定了hyperplane的方向 (垂直于w)，$\alpha$ 改变 -> hyperplane 平移

x 和 w 都是二维

![image-20240130220949940](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401311409015.png)

If w is a unit vector, then f(x) = w · x + $\alpha$ is the signed distance from x to H. I.e., positive on w’s side of H; negative on other side

If w is not unit vector, 化成 unit vec w:  $f(x)=\frac{w}{||w||}\cdot x+\frac{\alpha}{||w||}$ which is signed distance

the length of *w* is sqrt(2) (not a unit vector), and the signed distance is 1/sqrt(2)

![image-20240215231604440](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402152316513.png)

Moreover, the distance from H to the origin is $\alpha$ (x = 0, f(x) = $\alpha$). Hence $\alpha$ = 0 if and only if H passes through origin.

if w is not unit vector, the distance from H to origin is alpha/|w|

The training points are linearly separable if there exists a hyperplane that correctly classifies all the training points.



Simple Classifier

Centroid method: compute mean µC of all training points in class C and mean µX of all points NOT in C

$f(x)=\quad(\mu_{\mathbb{C}}-\mu_{\mathbb{X}})\cdot x + \alpha$，为了过µC和µX中点，求得$\alpha$如下，w决定decision boundary的方向，$\alpha$决定平移位置

![image-20240131133919862](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402010539943.png)

![image-20240131134441808](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402010544893.png)

there are circumstances where this method works well, like when all your positive examples come from one Gaussian distribution, and all your negative examples come from another.



Perceptron Algorithm (Frank Rosenblatt, 1957): Slow, but correct for linearly separable points

numerical optimization algorithm: gradient descent

loss function; risk function = objective function = cost function

![image-20240203014550198](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031745266.png) 

![image-20240203014618527](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031746559.png) 

![image-20240203014749500](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031747541.png) 

Plot of risk R(w). Every point in the dark green flat spot is a minimum. We’ll look at this more next lecture.

![image-20240203014849220](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402031748255.png)



# LEC3 GD, SGD->Perceptron, hard SVM

Perceptron Algorithm

Our original problem was to find a separating hyperplane in one space, which is x-space. 

But we’ve transformed this into a problem of finding an optimal point in a different space, which is w-space. 

transformations: a geometric structure in one space becomes a point in another space

Dual; **w-space的hyperplane可以看作constraint**

So a hyperplane transforms to a point that represents its normal vector, and a sample point transforms to the hyperplane whose normal vector is the sample point.

In this algorithm, the transformations happen to be symmetric. That won’t always be true for the decision boundaries

![image-20240215010442484](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150104586.png)

If we want to enforce inequality x · w >= 0, that means 

- in x-space, x should be on the same side of {z : w · z = 0} as w 
- in w-space, w ” ” ” ” ” ” ” ” ” ” ” ” ” ”” ” ” ” ” ” ” ” ” {z : x · z = 0} as x

红叉和蓝C是两类training point；C在x-space是vector，在w-space变成垂直于vector C的hyperplane (横轴)，且范围是hyperplane下方由于w要和c在same side； X同理，由于X是另一类，所以w应该在X对应的hyperplane的另一侧；三者综合得到红色阴影区域

在最低处任取一点w，映射回左侧就是decision boundary，指向c侧

![image-20240215012927457](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150129520.png)

 the risk function these three sample points create (risk function should have 1/n, which is omitted here)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150148780.png)

optimization algorithm: gradient descent

Given a starting point w, find gradient of R with respect to w; this is the direction of steepest ascent (最陡峭的上升). Take a step in the opposite direction ![image-20240215021333712](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150213739.png)

z = -yi xi; ![image-20240215021234648](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150212691.png) 倒三角有个下标w

![image-20240215014426666](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150144702.png)

Slow! Each step takes O(nd) time (训练样本很多)

![image-20240215021656179](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150216220.png)

stochastic gradient descent / perceptron algorithm

each step, pick one misclassified Xi; do gradient descent on loss fn L(Xi · w, yi)

Each step takes O(d) time. Not counting the time to search for a misclassified Xi

![image-20240215023402704](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150234749.png)

stochastic gradient descent does not work for every problem that gradient descent works for. The perceptron risk function happens to have special properties that guarantee that stochastic gradient descent will always succeed.



when hyperplane doesn’t pass through origin, add one fictitious dimension, 增加一个feature x3  which is always 1

hyperplane through origin in 3D can cut the 2D plane x3=1 at non-origin place

Decision fn ![image-20240215023939069](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150239102.png)

Now we have sample points in R^d+1^, all lying on same hyperplane x_d+1 = 1

Run perceptron algorithm in (d + 1) dimensional space.

perceptron algorithm is an “online algorithm,” which means that if new training points come in while the algorithm is already running, you can just throw them into the mix and keep looping

Perceptron Convergence Theorem: If data is linearly separable, perceptron algorithm will find a linear classifier that classifies all data correctly in at most $O(r^2/\gamma^2)$ iterations, where r = max|Xi| is the longest feature of data; gamma is the “maximum margin.”

https://classes.engr.oregonstate.edu/eecs/fall2023/ai534-400/unit2/convergence.html

https://arxiv.org/abs/2301.11235

perceptron algorithm (stochastic gradient descent) still slow

You can get rid of the step size $\epsilon$ by using a decent modern “line search” algorithm, you can find a better decision boundary much more quickly by quadratic programming



MAXIMUM MARGIN CLASSIFIERS / hard-margin SVM

The margin is the distance from the decision boundary to the nearest training point

the constraints ![image-20240215025650218](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150256247.png)

slab指虚线到虚线，2/|w|；margin指实线到虚线，1/|w|

![image-20240215025629364](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150256402.png)

the margin is min(signed distance) = ![image-20240215034205052](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150342088.png)

To maximize the margin, minimize |w|. Optimization problem is a quadratic program in d + 1 dimensions and n constraints.

![image-20240215034311506](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150343532.png)

If the points are linearly separable, it has one unique solution; otherwise, it has no solution

(A reason we use |w|^2^ as an objective function, instead of |w|, is that |w| is not smooth at w = 0, whereas |w|^2^ is smooth everywhere. This makes optimization easier)

At the optimal solution, the margin is exactly 1/|w|

what these constraints look like in weight space: （cross section 横截面）

![image-20240215040412518](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150404585.png)



# LEC4 Soft SVM, fictitious dim->nonlinear classifier, 4 example

hard-margin SVM works only with linearly separable point sets, and sensitive to outliers

Soft-Margin SVM

Allow some points to violate the margin, with slack variables

Constraint: ![image-20240216141802638](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402161418698.png) ![image-20240216141815568](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402161418601.png)

don’t violate the margin -> $\xi_{i}=0$ ; violates the margin -> $\xi_{i}>0$

![image-20240216142142864](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402161421902.png)

margin definition: 1/|w| (no longer the distance from the decision boundary to the nearest training point)

a quadratic program in d + n + 1 dimensions and 2n constraints

![image-20240216143403734](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402161434767.png)

C > 0 is a scalar regularization hyperparameterl; c -> ∞, become hard margin SVM

nonlinear decision boundaries

![image-20240216143813778](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402161438805.png)

C变大，margin变窄；wide margin更好generalize；越接近hard margin求解越慢

![image-20240216144637016](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402161446047.png)



linear classifier -> non-linear classifier: Make nonlinear features that lift points into a higher-dimensional space

High-d linear classifier -> low-d nonlinear classifier

Example1: The parabolic lifting map

![image-20240216152126133](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402161521169.png)

added one new feature, |x|^2^,  Find a linear classifier in $\mathbf{\Phi}$-space -> a sphere classifier in x-space

Theorem: $\Phi(X_1),\ldots,\Phi(X_n)$ are linearly separable iff X1, ..., Xn are separable by a hypersphere. 

![image-20240216152625457](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402161526509.png)

R^d^ 是指左图，一个球面分开c和x

x是一个dim-d列向量，在最下方一列加上|x|^2^；c是一个dim-d行向量，在最右边加上1

在左图球面内部 -> 增加一维1，再乘$\mathbf{\Phi}$函数，得到(d+1)-dim的点在抛物线下方

![image-20240216160112761](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402161601822.png)

A hyperplane is essentially a hypersphere with infinite radius. So hypersphere decision boundaries can do everything hyperplane decision boundaries can do. With parabolic lifting map, if you pick a hyperplane in $\mathbf{\Phi}$-space that is vertical, you get a hyperplane in x-space. (a vertical plane in 3D $\mathbf{\Phi}$-space -> a linear line in 2D space)

![image-20240221181339082](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211813126.png)

![image-20240221181357667](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211813694.png)



Example 2: conic section圆锥曲面: Ellipsoid 椭球体/hyperboloid 双曲面/paraboloid 抛物面 decision boundaries

cylinder 柱体/double cones 双锥体/hyperplane (最左侧)

SVM中w没有最后一个元素1，常数项$\alpha$在外面

![image-20240220234518654](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402202345076.png)

![image-20240220234628599](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402202346627.png)

Isosurface defined by this equation is called a quadric 二次曲面. 

A linear decision boundary in $\mathbf{\Phi}$-space imposes a quadric decision boundary in x-space.

x1x2, x2x3, x3x1三项导致了O(d^2^)的计算复杂度，但可以达到更好的分类效果 (扭曲平面)

If extra features x1x2, x2x3, x3x1 make the classifier overfit or make it too slow, 可以删除他们, 这样就是O(d).Ddecision boundaries can be axis-aligned ellipsoids and axis-aligned hyperboloids, but they can’t be rotated in arbitrary ways.

$\Phi(x):\mathbb{R}^d\to\mathbb{R}^{(d^2+3d)/2}$ ？？？？？？？？

/2是因为矩阵是对称的，只用计算一半 

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402210022577.png)

[For perceptron or regression, $\Phi(x):\mathbb{R}^d\to\mathbb{R}^{(d^2+3d)/2+1}$] 



Example 3: Decision fn is degree-p polynomial

![image-20240221002626504](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402210026539.png)

we’re really blowing up the number of features! If you have 100 features per sample point and you want to use degree-4 decision functions, then each lifted feature vector has a length of roughly 4 million, and your learning algorithm will take approximately forever to run. later in the semester we will learn clever trick that allows us to work with these huge feature vectors very quickly, without ever computing them. It’s called “the kernel trick.” 

Hard-margin SVMs with degree 1/2/5 decision functions; margin get wider as the degree increases, might get a more robust decision boundary that generalizes better, might overfit as well; the data might become linearly separable when you lift them to a high enough degree, even if the original data are not linearly separable

d=5 可以想象成把所有点投射到5维空间，5维空间中一个linear平面和两个margin分开了他们，再把分界面投射回2维得到三条曲线 

![image-20240221002952589](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402210029623.png)

横轴是次数p；hyperparameter optimized by validation

![image-20240221003519715](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402210035761.png)

[If you’re using both polynomial features and a soft-margin SVM, you have two hyperparameters: the degree and the regularization hyperparameter C. Generally, the optimal C will be different for every polynomial degree, so when you change the degree, you should run validation again to find the best C for that degree.]



So far are only polynomial features. features can get much more complicated than polynomials, eg: designing application-specific features

example 4: edge detection

algorithm for approximating grayscale/color gradients in image

tap filter, sobel filter, oriented Gaussian derivetive filter

![image-20240221004823132](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402210048192.png)



# LEC5 4-level abstraction, 3 optimization: unconstrained, linear/quadratic program

four levels of ML abstraction

change one level -> change all the levels underneath it

given an application, turn it into an optimization problem

![image-20240221135915824](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211359929.png)



三种 OPTIMIZATION PROBLEMS

1. Unconstrained

Goal: Find w that minimizes (or maximizes) a continuous objective fn f(w)

f is smooth if its gradient is continuous too

global/local minimum; neural networks has lots of local minima, stochastic gradient descent can find one good enough

convex 直观: A function is convex if for every x, y $\boldsymbol{\in}\mathbb{R}^d$, the line segment connecting (x, f(x)) to (y, f(y)) does not go below f(·).

![image-20240221141922932](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211419017.png)

Formally: for every x,y $\in\mathbb{R}^d$ (x y 是两个横坐标) and $\beta\in[0,1],f(x+\beta(y-x))\leq f(x)+\beta(f(y)-f(x))$

![image-20240221142802646](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211428693.png)

perceptron risk fn is convex and nonsmooth

sum convex functions -> convex function

A [continuous] convex function [on a closed, convex domain] has either 

- no minimum (goes to -∞)
- just one local minimum
- a connected set global minima (如perceptron的pizza slice)

最后一张图是从下往上走的

![image-20240221143715651](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211437705.png)

- Fails/diverges if $\epsilon$ too large. Slow if $\epsilon$ too small.
- often optimized by trial & error [for slow learners like neural networks]
- check whether a step of gradient descent increases the function value; if so, reduce the step size
- adaptive learning rate / learning rate schedule become even more important when you do stochastic gradient descent or when you optimize non-convex, very twisty objective functions (neural networks)



Left: $\begin{aligned}f(w)=2w_1^2+w_2^2\end{aligned}$ 

Center: ill-conditioning of the Hessian (high ellipticity椭圆度 of contours轮廓) $\begin{aligned}{f(w)}=10w_1^2+w_2^2\end{aligned}$ , same $\epsilon$ diverge (too large)

Right: reduce $\epsilon$, converge but approach minimum slowly (too small)

![image-20240221145752774](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211457844.png)

gradient descent gets closer and closer forever, but never exactly reaches the true minimum. We call this behavior “convergence.” The last question of HW2 will give you some understanding of why convergence happens under the right conditions.？？？

ill-conditioning of the Hessian, means no learning rate is good in all directions (the learning rate that’s good for one direction might be terrible in another direction). The Hessian matrix is ill-conditioned if its largest eigenvalue is much larger than its smallest eigenvalue. adaptive learning rate algorithms choose different learning rates in different directions -> Adam and RMSprop



2. Linear Program

Linear objective fn + linear inequality constraints

Goal: Find w that maximizes (or minimizes) c · w, subject to Aw $\leq$ b

where A is n*d matrix, b $\boldsymbol{\in}\mathbb{R}^d$, expressing n linear constraints: $A_i\cdot w\leq b_i,\quad i\in[1,n]$

![image-20240221152726233](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211527313.png)

F: feasible region 又称 (convex) polytope (polyhedron)

data are linearly separable iff the feasible region is not empty set, also true for maximum margin classifier (quadratic program)

convex point set: for every p, q $\in$ P, the line segment with endpoints p, q lies entirely in P

The optimum achieves equality for some constraints, called the active constraints. In SVM, active constraints correspond to the training points that touch or violate the slab, aka support vectors.

Sometimes >1 optimal point. (c pointed straight up, top horizontal edge is optimal point set, optimal point set is always convex)

EVERY feasible point (w, $\alpha$) gives a linear classifier.  Find w, $\alpha$ that satisfies yi(w · Xi + $\alpha$) $\geq$ 1 for all i $\in$ [1, n] -> This problem can be cast as a slightly different linear program that uses an objective function to make all the inequalities be satisfied

most famous linear programming algorithm is the simplex algorithm, walks along edges of the feasible region, traveling from vertex to vertex (increase c) until it finds an optimum.

[Linear programming is very different from unconstrained optimization; it has a much more combinatorial flavor. If you knew which constraints would be the active constraints once you found the solution, it would be easy; the hard part is figuring out which constraints should be the active ones. There are exponentially many possibilities, so you can’t afford to try them all. So linear programming algorithms tend to have a very discrete, computer science feeling to them, like graph algorithms, whereas unconstrained optimization algorithms tend to have a continuous, numerical mathematics feeling.]



3. Quadratic Program

Quadratic, convex objective fn + linear inequality constraints.

Goal: Find w that minimizes $\begin{aligned}f(w)=w^\top Qw+c^\top w\end{aligned}$, subject to Aw $\leq$ b

where Q is a symmetric, positive semidefinite matrix. 

![image-20240221193038166](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211930269.png)

![image-20240221193046318](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211930385.png)

If Q is positive definite, only one local minimum -> global minimum

If Q is indefinite, then f is not convex, the minimum is not always unique, and quadratic programming is NP-hard

If Q is positive semidefinite, then f is convex and quadratic programming is tractable, but there may be infinitely many solutions

Example: Find maximum margin classifier

SVM, objective function w1^2^+w2^2^  (Right: SVM -> reminde there is also an $\alpha$-axis)

![image-20240221193421189](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211934245.png)

加上constraint，上图一个active constraint，下图两个active constraint

![image-20240221194654208](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211946267.png)

[Algs for quadratic programming: – Simplex-like [commonly used for general-purpose quadratic programs, but not as good for SVMs as the following two algorithms that specifically exploit properties of SVMs] – Sequential minimal optimization (SMO, used in LIBSVM, “SVC” in scikit) – Coordinate descent (used in LIBLINEAR, “LinearSVC” in scikit) Numerical optimization: EECS 127]



# LEC6 Decision theory/Risk Minimization, 3 classifier: Generative/Discriminative/find boundary

DECISION THEORY aka Risk Minimization

probabilistic classifier

eg1. 只知道身高，预测性别，男女身高是两个分布，只能中间划一条分界线，右边预测男，左边预测女，有>50%准确率

![image-20240221233342494](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402212333579.png)

eg2. Suppose 10% of population has cancer, 90% doesn’t. P(X|Y):

![image-20240221233631054](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402212336115.png)

a people is farmer

![image-20240221233855606](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402212338656.png)

a farmer has cancer

![image-20240221234110248](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402212341278.png)

we’re assuming that we want to maximize the chance of a correct prediction. If you’re developing a cheap screening test for cancer, you’d rather have more false positives and fewer false negatives. When there’s an asymmetry between the awfulness of false positives and false negatives, we can quantify that with a loss function.

predicts z, true class is y

asymmetrical loss fn

![image-20240221234811680](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402212348736.png)

symmetrical loss -> the same for false positives and false negatives, 0-1 loss function

![image-20240221234945786](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402212349827.png)

very asymmetrical loss -> spam detection



Let r : R^d^ -> ±1 be a decision rule, aka classifier: a fn map a feature vector x to 1 (“in class”) or 1 (“not in class”).

**risk -> expected loss; The risk for r is the expected loss over all values of x, y**

r is a function, X = x -> Y有30%可能是1 有70%可能是-1 (label并不唯一)

![image-20240222002225165](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402220022286.png)

The Bayes decision rule, aka Bayes classifier is the fn r^*^ that minimizes functional R(r)

L(1, 1) = L(1, 1) = 0

![image-20240222003519411](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402220035456.png)

1. When $L(-1,1)P(Y=1|X=x) > L(1,-1)P(Y=-1|X=x)$, r =1

   $L(r(x),1)P(Y=1|X=x)+L(r(x),-1)P(Y=-1|X=x)$ 

   = $0+L(1,-1)P(Y=-1|X=x)$ 

2. When otherwise, r = -1, = $L(-1,1)P(Y=1|X=x)$ + 0

Thus R(r) reach minimum



When L is symmetrical, L(-1,1) = L(1,-1) -> **pick the class with the biggest posterior probability**

In cancer example, r *(miner) = 1, r *(farmer) = 1, and r *(other) = -1.

The Bayes risk, aka **optimal** risk, is the risk of final Bayes classifier: R(r*)

$R(r^*)=0.1(5\times0.3)+0.9(1\times0.01+1\times0.1)=0.249$ (No decision rule gives a lower risk)

Deriving/using r* is called risk minimization



if we know all these probabilities, we can construct an ideal probabilistic classifier. But in real applications, we rarely know these probabilities; the best we can do is use statistical methods to estimate them

Continuous Distributions

Suppose X has a continuous probability density fn (PDF)

![image-20240222005620609](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402220056724.png)

X is daily calorie intake, Y=1 -> cancer, Y=-1 -> no cancer, 给一个calorie intake预测cancer

Firstly, 0-1 loss function (no prior probabilities)

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402220056021.png)

0-1 loss, scale by prior probabilities: blue multiply 0.1, red multiply 0.9

(如果是asymmetrical loss曲线还要scale，分别乘以L(-1,1), L(1,-1))

两边同/f(x)就是posterior prob；或两条曲的f换成P，再同时/P(X = x)就是posterior prob，可以省略这个分母

![image-20240222010629120](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402220106210.png)

risk: summation -> integral

![image-20240222122009208](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221220287.png)

Bayes risk is the area under minimum of functions above

L(1, 1) = L(1, 1) = 0 -> ![image-20240222122544571](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221225647.png)

If L is 0-1 loss, R(r) = P(r(x) is wrong), the Bayes optimal decision boundary is: (posterior prob = 0.5, isovalue等值线)

分界线 *P*(*Y*=1∣*X*=*x*)=*P*(*Y*=−1∣*X*=*x*), *P*(*Y*=1∣*X*=*x*)+*P*(*Y*=1∣*X*=*x*)=1 -> *P*(*Y*=1∣*X*=*x*)=0.5

![image-20240222124147784](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221241856.png)

![image-20240222124440550](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221244581.png)

accuracy of the prob is most important near the decision boundary, 别的地方有点误差不要紧

[The Bayesian approach is a particularly convenient way to generate multi-class classifiers, because you can simply choose whichever class has the greatest posterior probability.]

LEC7: how to build these PDFs



3 WAYS TO BUILD CLASSIFIERS

(1) Generative models (e.g., LDA)

- Assume sample points come from probability distributions, different for each class
- Guess form of distributions (normal)
- For each class C, fit distribution parameters to class C points, giving $f_{X\mid Y=\mathbb{C}}(x)$
- For each C, estimate P(Y = C) 
- Bayes’ Theorem gives posterior prob P(Y|X) 
- If 0-1 loss, pick class C that maximizes P(Y = C|X = x) [posterior probability] equivalently, maximizes $f_{X\mid Y=\mathbf{C}}(x)P(Y=\mathbf{C})$

(2) Discriminative models (e.g., logistic regression)

- Model posterior prob P(Y|X) directly

(3) Find decision boundary (e.g., SVM)

- Model r(x) directly (no posterior prob)

Advantage of (1 & 2): P(Y|X) tells you probability your guess is wrong [This is something SVMs don’t do.] 

Advantage of (1): you can diagnose outliers: f(x) is very small

Disadvantages of (1): often hard to estimate distributions accurately; real distributions rarely match standard ones

A generative model is a full probabilistic model of all variables, whereas a discriminative model provides a model only for the target variables that we want to predict

In practice, generative models are most popular when you have phenomena that are well approximated by the normal distribution or another “nice” distribution. Generative methods also tend to be more stable than other methods when the number of training points is small or when there are a lot of outliers



# LEC7 GDA: QDA/LDA, MLE

GAUSSIAN DISCRIMINANT ANALYSIS

Fundamental assumption: each class has a normal distribution [a Gaussian]

![image-20240222164250724](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221642840.png)

Use scalar variance instead of covariance matrix for simplicity -> that’s why isocontours are circles and not ellipses

- isotropic normal distribution -> isocontours are circles
- anisotropic normal distribution -> isocontours are ellipses

For each class C, suppose we know mean µC and variance $\sigma_{\mathbb{C}}^2$, 代入上式得到$f_{X|Y=\mathbb{C}}(x)$; also know prior $\begin{aligned}\pi_{\mathbf{C}}=P(Y=\mathbf{C})\end{aligned}$

x and µ are 2D vectors, 三维图中底下两轴是x1, x2，高度轴是y

![image-20240222175908812](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221759904.png)

Given x, Bayes decision rule r*(x) predicts class C that maximizes $f_{X\mid Y=\mathbf{C}}(x)\pi_\mathbb{C}$

ln(w) is monotonically increasing for w > 0, so it is equivalent to maximize (Qc见上右图，两者B.d.b相同)

![image-20240222181425237](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221814299.png)

[In a 2-class problem, you can incorporate an asymmetrical loss function by +ln(L(not C,C)).  In a multi-class problem, asymmetric loss is more diffcult to account for, the penalty might depend on both the wrong guess and the true class.]



Quadratic Discriminant Analysis (QDA)

Suppose only 2 classes C, D. Then the Bayes classifier is: (Picks biggest posterior prob)

![image-20240222182047346](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221820424.png)

Decision fn is $Q_C(x)-Q_D(x)$ (quadratic); Bayes decision boundary is  $Q_C(x)-Q_D(x)=0$

- In 1D, B.d.b. may have 1 or 2 points  [Solutions to a quadratic equation]
- In d-D, B.d.b. is a quadric  [In 2D, that’s a conic section (an ellipse in figure above)]

estimate the probability that your prediction is correct

recover posterior prob in 2-class case, use Bayes

![image-20240222182832115](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221828160.png)

![image-20240222182909260](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221829298.png)

Multi-class QDA: (黑点是高斯分布的中心最高点)

![image-20240222184534077](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221845122.png)

[it’s a special kind of Voronoi diagram called a multiplicatively, additively weighted Voronoi diagram.]



Linear Discriminant Analysis (LDA)

linear decision boundaries (straight line / hyperplane), less likely to overfit than QDA

Fundamental assumption: all the Gaussians have same variance $\sigma^2$ -> quadratic terms cancel out

![image-20240222185437942](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221854985.png)

Now it’s a linear classifier! 

- decision boundary is w · x + $\sigma$ = 0
- posterior is $P(Y=\mathbb{C}|X=x)=s(w\cdot x+\alpha)$

[for $s(w\cdot x+\alpha)$, the effect of w and $\alpha$ is to rotate/scale/translate the logistic fn in x-space/feature space]

Two Gaussians (red) and the logistic fn $s(w\cdot x+\alpha)$ (black). logistic function is the right Gaussian divided by sum of the Gaussians

logistic fn $s(w\cdot x+\alpha)$ 的意义是样本属于C类的概率

![image-20240310203948949](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403102039988.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403102039297.png" alt="image-20240310203935255" style="zoom:20%;" />

the logistic function only has curvature in one direction and just flat in all other direction (even in 100D)

Special case: if $\pi_C$ = $\pi_D$ = 0.5 -> decision boundary: ![image-20240222191815691](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221918765.png), This is the centroid method

[centroid method is a good classifier when isotropic normal distribution, same variance, equal priors]

Multi-class LDA: choose C that maximizes linear discriminant fn: ![image-20240222192152195](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221921239.png) (分几类就有几个linear discriminant fn)

![image-20240222192300945](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221923985.png)

![image-20240222192530499](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402221925538.png)

[a classical Voronoi diagram if the priors $\pi_C$ are equal. All the Gaussians have the same width]

通常并不是通过same/diff variance来选择QDA/LDA，而是通过数据量，数据量少QDA，数据量多防overfit选LDA



For QDA/LDA, you must know mean & variance & priors

given data, using MAXIMUM LIKELIHOOD ESTIMATION to get mean & variance & priors

1. estimating priors

Eg. flip biased coins, Heads with prob p; tails with prob 1-p ($p\neq50\%$) Binomial distribution

10 flips, 8 heads, 2 tails. What is the most likely value of p?

\# of heads is $X\sim\mathcal{B}(n,p)$, binomial distribution:

![image-20240222230208369](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402222302515.png)

Prob of x = 8 heads in n = 10 flips is: $P[X=8]=45p^8\left(1-p\right)^2\overset{\mathrm{def}}{\operatorname*{=}}L(p)$ (likelihood fn)

Written as a fn of distribution parameter p, this prob is the likelihood fn L(p)

Maximum likelihood estimation (MLE): picking the paras that maximize likelihood fn L (is one method of density estimation: estimating a PDF [probability density function] from data)

![image-20240222230916115](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402222309197.png)

Solve by finding critical point of L:

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402222309372.png" alt="image-20240222230940337" style="zoom:50%;" />

![image-20240222231110461](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402222311510.png)

p=0.8 is the estimated prior prob

-> training set has n points, with x in class C, estimated prior for class C is $\hat{\pi}_{\mathbf{C}}=x/n$ (帽子表示estimated, not true)



2. mean & variance

Given sample points X1, X2,..., Xn, find best-fit Gaussian -> Likelihood of a Gaussian/normal distribution

Likelihood: $\mathcal{L}(\mu,\sigma;X_1,\ldots,X_n)=f(X_1)f(X_2)\cdots f(X_n)$

maximizing log likelihood: ($\mu$ 是向量因此用梯度符号，$\sigma$ 是标量)

![image-20240222232550113](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402222325225.png)

We don't know $\mu$ exactly, so substitute $\hat{\mu}$ for $\mu$ to compute $\hat{\sigma}$

-> use sample mean & variance of pts in class C to estimate mean & variance of Gaussian for class C



For QDA: 

- estimate conditional mean $\hat{\mu}$ & conditional variance $\hat{\sigma_C^2}$ of each class C separately
- estimate the priors: ![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402222339651.png)

For LDA (assume all class share variance -> pooled within-class variance):

- same means & priors 

- one variance for all classes: ![image-20240222234007723](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402222340815.png) 

  (第二个sum表示all training pts in class C，每个class的点分别用每个class的mean计算 -> pooled within-class variance，别用global mean 计算 variance)



# LEC8 linalg review: eigendecomp, quadratic form, anisotropic gaussian->covariance matrix

EIGENVECTORS

Given square matrix A, if Av = $\lambda$v for some vector v $\neq$ 0, scalar $\lambda$

then v is an eigenvector of A; $\lambda$ is the eigenvalue of A associated with v

It means that v is a magical vector, after being multiplied by A, still points in the same direction/opposite direction

上图乘一个A变长两倍，Only the direction matters, not the length

![image-20240223122011959](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402231220029.png)

Theorem: if v is eigenvector of A with eigenvalue $\lambda$, then v is eigenvector of A^k^ with eigenvalue $\lambda^k$ [k is positive integer]

![image-20240223123056242](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402231230344.png)

Theorem: if A is invertible, then v is eigenvector of A^-1^ with eigenvalue 1/$\lambda$

![image-20240223123006818](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402231230865.png)

->

- When you invert a matrix, the eigenvectors don’t change, but the eigenvalues get inverted
- When you square a matrix, the eigenvectors don’t change, but the eigenvalues get squared  

Spectral Theorem: every real, **symmetric** n*n matrix has real eigenvalues and n eigenvectors that are mutually orthogonal

![image-20240223124117544](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402231241604.png)

If two eigenvectors have the same eigenvalue (蓝色$\lambda$ =2), then every linear combination of those eigenvectors is also an eigenvector with same eigenvalue (绿色$\lambda$ =2), span a plane. just arbitrarily pick two orthogonal vectors (红色) to do eigendecomposition, use them as a basis for R^n^

![image-20240223124714169](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402231247235.png)

-> 因此n mutually orthogonal eigenvectors 如果有两个eigenvalue一样，那么一整个平面的方向都是eigenvector with that eigenvalue

[set of eigenvalues is always uniquely determined by a matrix, including the multiplicity; 例如eigenvalue: 0,1,1,3,5 -> 1 is eigenvalue of A with multiplicity 2, 随机选两个垂直的eigenvalue=1的向量表示span的平面]



Building a Matrix with Specified Eigenvectors

In applications, given a matrix (a sample covariance matrix), you want to extract the eigenvectors and eigenvalues; When you’re learning the math, it’s more intuitive to go in opposite direction, you know what eigenvectors and eigenvalues you want, you want to create the matrix that has those eigenvectors and eigenvalues.

create the matrix:

Choose n mutually orthogonal **unit dim-n vectors** v1,..., vn (specify an orthonormal coordinate system)

Let V = [v1 v2 ... vn], which is n*n matrix

unit, orthogonal -> $V^{\mathsf{T}}V=I$ (非对角线垂直得0，对角线平方得1) $\Rightarrow V^\top=V^{-1}\Rightarrow VV^\top=I$

V is orthonormal matrix: used as linear trans to some space, acts like [rotation when det(V)=1] or [reflection(rotate + mirror image) when det(V)=-1]？？？

Choose some eigenvalues $\lambda_i$

![image-20240223132124964](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402231321073.png)

Definition of eigenvector: $AV=V\Lambda $ 	(Av=$\lambda$v 的矩阵形式)

$AVV^\top=V\Lambda V^\top $ -> Theorem: ![image-20240223132522134](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402231325186.png)

得到的A矩阵有之前想要的eigenvectors/values

This is a matrix factorization called the eigendecomposition (every real, symmetric matrix has one)

example: eigenvector normalized (1,1), (1,-1), eigenvalue 2,-1/2 (上面的两张图)

![image-20240223133527323](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402231335357.png)

[For "given a matrix, extract eigenvectors and eigenvalues" which is difficult, take math 221 Matrix Computations / Numerical Linear Algebra, you can use library to compute this]

Observe: $A^2=V\Lambda V^\top V\Lambda V^\top=V\Lambda^2V^\top $ (V^T^V约掉); $A^{-2}=V\Lambda^{-2}V^{\top}$

[squaring a matrix squares its eigenvalues without changing its eigenvectors]

定义matrix square root

Given a symmetric PSD matrix $\boldsymbol{\Sigma}$ -> symmetric square root A = $\boldsymbol{\Sigma^{1/2}}$

(PSD -> eigenvalue > 0, 有平方根)

- compute eigenvectors/values of $\boldsymbol{\Sigma}$
- take square roots of $\boldsymbol{\Sigma}$’s eigenvalues
- A = ![image-20240223213700723](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232137832.png) (same eigenvectors, changed eigenvalues) (另一个平方根矩阵加个负号 ![image-20240223213938029](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232139067.png)，一般不用)



the way to visualize a symmetric matrix -> Visualizing Quadratic Forms

[shows how applying the matrix affects the length of a vector]

quadratic form of M is $x^\top Mx$



左图circles, transformed by A, 得到右图椭圆 (quadratic form of a matrix)；z是2d向量，A是上面计算的2*2矩阵

[The same matrix A as above, stretch along the direction with eigenvalue 2 and shrinks along the direction with eigenvalue 1/2]

由图中从左到右箭头的变化可知 q2(Az) = q1(z)

q1是z-space -> q2是x-space

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232159569.png" alt="image-20240223215940530" style="zoom:50%;" />

想要设计tansfer到右图的A矩阵 / Given $x^\top Mx$ (右图),  backward engineer which vector I chose

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402240145796.png" alt="image-20240224014529645" style="zoom:30%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232212988.png" alt="image-20240223221210908" style="zoom:50%;" />

contours 轮廓；coordinate axes指的是标准的x轴y轴等，椭圆轴对称

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232217988.png" alt="image-20240223221721950" style="zoom:50%;" />

![image-20240223222205729](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232222794.png)

A symmetric matrix M is

- positive definite if $w^\top\boldsymbol{M}w>0$ for all w$\neq$0 <-> all eigenvalues positive
- positive semidefinite if $w^\top\boldsymbol{M}w\geq0$ for all w <-> all eigenvalues nonnegative
- indefinite if 同时有正负 eigenvalue
- invertible if no zero eigenvalue

图一，只有一个minimum，eigenvalue均为正

图二，红线上都是minimum，该方向eigenvalue为零，另一个eigenvalue>0 (蓝线)

图三，交点是saddle point，eigenvalue一正一负   

![image-20240223225525168](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232255235.png)

Every squared matrix is **pos semidef**, including A^-2^. [Eigenvalues of A^-2^ are squared, cannot be negative.] 

An invertible matrix eigenvalue$\neq$0 -> If A^-2^ exists, it is **pos def**

->

- A^2 is pos semidef
- A^-2 (if exist) is pos def

For isosurfaces of $x^\top Mx$ for a pos semidef, singular(not invertible, has zero eigenvalue) M

-> isosurfaces are cylinders(横截面是椭圆的)柱体 instead of ellipsoids椭球体[at least one direction go infinity (图中横轴)]; unlike ellipsoidal isosurface where each isosurface is bounded

usually if there is some positive eigenvalue, you can take a cross section in which you see ellipsoids (图中竖切面)

These cylinders have ellipsoidal cross sections spanning the directions with nonzero eigenvalues, but they run in straight lines along the directions with zero eigenvalues

![image-20240223234737399](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232347458.png)

![image-20240304005059649](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040050727.png)

![image-20240304005115206](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403040051246.png)



ANISOTROPIC各向异性 GAUSSIANS

multivariate Gaussian/normal distribution (different variances along different directions)  -> covariance matrix

![image-20240223235133925](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232351995.png)

注意公式中 $\boldsymbol{\Sigma}^{-1}$的quadratic form

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232352034.png)

- $\boldsymbol{\Sigma}$ is the d*d SPD covariance matrix (def, not semidef？？？？？？？)
- $\boldsymbol{\Sigma}^{-1}$ is the d*d SPD precision matrix

函数分成两部分：q(x)是quadratic form；n(x)只做了scale, dont change isosurface

![image-20240223235740217](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232357265.png)

左图是q(x), mapping from feature space (2d) to a real number

中间图是n(x)，simple, monotonic decreasing, convex fn, >0

n(x) turn quadratic fn into gaussian PDF, 左侧quadratic的最低点到右侧变成gaussian的最高点。

由绿线可见isocontour没有变，但是isovalue变了

左侧走向无穷，映射到右侧，无限接近于0 (PDF > 0 )

![image-20240224002718819](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402240027854.png)

![image-20240224003514051](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402240035084.png)



Covariance

Let R, S be random variables—column vectors or scalars; 计算Cov是outer product; E[R]=$\mu_R$, E[S]=$\mu_S$; 算出d*d Cov matrix

![image-20240224003634485](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402240036530.png)

If R is a vector, covariance matrix for R is: (R向量的d个元素)

Cov(R1,R2) = Cov(R2,R1) -> symmetric 

![image-20240224003651824](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402240036856.png)

For a Gaussian $R\sim\mathcal{N}(\mu,\Sigma)$, one can show Var(R) = $\boldsymbol{\Sigma}$

[In hw2, statisticians didn’t just arbitrarily decide to call $\boldsymbol{\Sigma}$ a covariance matrix, they discovered that if you find the covariance of the normal distribution by integration, it turns out that the covariance is $\boldsymbol{\Sigma}$.]

pairwise independent表示vector R的元素两两独立，Cov项均为0 -> diagonal

![image-20240301180501199](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403011805232.png)

n()不改变轮廓，f(x)轮廓同q(x)，q(x)轮廓由$\boldsymbol{\Sigma^{1/2}}$的特征值/向量决定

$\boldsymbol{\Sigma^{-1}}$ is diagonal -> ellipsoids are axis-aligned, radii是$\boldsymbol{\Sigma^{1/2}}$的特征值，即 Var(R)=$\boldsymbol{\Sigma}$ 对角线上特征值的平方根

![image-20240301180520421](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403011912068.png)

参考：

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403011904478.png" alt="image-20240301190414435" style="zoom:50%;" />

![image-20240301190321974](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403011903065.png)

[So when the features are independent, you can write the multivariate Gaussian PDF as a product of univariate Gaussian PDFs. When they aren’t, you can do a change of coordinates to the eigenvector coordinate system, and write it as a product of univariate Gaussian PDFs in eigenvector coordinates. You did something very similar in Q6.2 of Homework 2？？？？？]



# LEC9 Anisotropic Gaussian: MLE, QDA/LDA, terms: centering, decorrelating, whitening

ANISOTROPIC GAUSSIANS AND GDA

standard deviations 标准差

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403021459918.png" alt="image-20240302145939777" style="zoom:100%;" />



Maximum Likelihood Estimation for Anisotropic Gaussians

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403021513954.png)

同之前

![image-20240302183315599](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403021833681.png)

对一些点进行高斯MLE的效果案例

![image-20240302183727571](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403102309449.png)

$\hat{\Sigma}_{\mathbb{C}}$ is pos semidef (not always pos def) [If there are some zero eigenvalues, the standard version of QDA just doesn’t work. We can try to fix it by eliminating the zero-variance dimensions (eigenvectors). Homework 3 suggests a way to do that.？？？？？？？]

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050239400.png" alt="image-20240305023935279" style="zoom:50%;" />



Revisit QDA&LDA

[Conflicting notation warning: capital X represents a random variable, but later it will represent a matrix.]

QDA

![image-20240305024659885](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050246975.png)

假设half training point in class C, half in class D, prior = 50% 因此左图只写了$f_{X\mid Y=\mathbf{C}}(x)\mathrm{~}\&f_{X\mid Y=\mathbf{D}}(x)$

QC - QD -> hyperbola: 一个eigenvector方向上升，一个eigenvector方向下降 (hyperbola decision boundary双曲线决策边界 (indefinite) is not possible with isotropic各向同性 Gaussians)

 左图哪个类别的曲线高就选哪类 ；QC - QD图哪个大于0的地方选C；右图可以算出判断出错概率($s(Q_{\mathbf{C}}-Q_{\mathbf{D}})=0.5$)

![image-20240305025715801](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050257863.png)

下图数据是计算机按高斯分布生成，因此知道真实parameter

粉线是用真实parameter画出的分解线，绿线是用估计parameter画出的分解线

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050300121.png" alt="image-20240305030024056" style="zoom:20%;" />

Multi-class QDA (anisotropic Voronoi diagram): [红线部分是同一个高斯分布裂成两块]

![image-20240305031035472](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050310559.png)



LDA

One $\Sigma$ for all classes. Decision fn is

![image-20240305031312071](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050313223.png)

[the decision function is linear and the decision boundary is a hyperplane.]

![image-20240305031512777](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050315850.png)

Multi-class LDA: choose class C that maximizes the linear discriminant fn

![image-20240305031544032](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050315087.png)

[Note that we use a linear solver to compute $\mu_{\mathbf{C}}^{\top}\boldsymbol{\Sigma^{-1}}$ just once, so the classifier can evaluate test points quickly.]

![image-20240305032112930](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050321014.png)

[Mathematica messed up the top left plot a bit; there should be no red in the left corner, nor blue in the right corner.]

![image-20240305032343345](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050323403.png)

[The real-world distributions almost surely aren’t Gaussians, but LDA still works reasonably well.]

？？？？

![image-20240305032533127](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050325205.png)

When the Bayes optimal boundary is linear, as at left, LDA gives a more stable fit whereas QDA may overfit. When the Bayes optimal boundary is curved, as at right, QDA often gives you a better fit. [the Bayes optimal decision boundary is purple (and dashed), the QDA decision boundary is green, the LDA decision boundary is black (and dotted).]

![image-20240305032809766](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050328838.png)

Change loss -> asymmetric loss; 选取10%, 50%, 90% Posterior prob 的分界线

new feature: polynomial, edge detector, word2vec

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403050336347.png)

[LDA & QDA are the best method in practice for many applications. In the STATLOG project, either LDA or QDA were among the top three classifiers for 10 out of 22 datasets. But it’s not because all those datasets are Gaussian. LDA & QDA work well when the data can only support simple decision boundaries such as linear or quadratic, because Gaussian models provide stable estimates. See ESL, Section 4.3.]



Some Terms

Let X be n*d design matrix of sample pts, Each row i of X is a sample pt $X_i^{\mathsf{T}}$

1. centering X ($\dot{X}$): subtract $\boldsymbol{\mu}^{\mathsf{T}}$ from each row of X, $X\to\dot{X}$ ($\boldsymbol{\mu}^{\mathsf{T}}$ is the mean of all rows of X, the mean of all rows of $\dot{X}$ is zero)

Let R be drawn from uniform distribution on sample pts. Sample covariance matrix is $\operatorname{Var}(R)=\frac1n\dot{X}^\top\dot{X}$

[compute covariance matrix for QDA: For design matrix $X_C$ that contains only the pts of class C, you have $\boldsymbol{\hat{\Sigma_C}}=\frac{1}{n_C}\dot{X_C^\top}\dot{X_C}$]

2. decorrelating $\dot{X}$: applying rotation Z = $\dot{X}$V, where $\mathrm{Var}(R？？？？？？？？)=V\Lambda V^\top $ [rotates the sample points to eigenvector coordinate system]

-> Z is decorrelated data, Var(Z) = $\Lambda$ 

[Z has diagonal covariance. If $\boldsymbol{X_i}\thicksim\boldsymbol{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$, then approximately, $Z_i\sim\mathcal{N}(0,\Lambda)$] 

[Proof: $\mathrm{Var}(\mathbb{Z})=\frac1n\mathbb{Z}^\top\mathbb{Z}=\frac1nV^\top\dot{X}^\top\dot{X}V=V^\top\mathrm{Var}(\mathbb{R})V=V^\top V\Lambda V^\top V=\Lambda $]

3. whitening X ($X\to W$) = sphering $\dot{X}$ = centering + sphering

sphering $\dot{X}$: applying transform $W=\dot{X}\operatorname{Var}(R)^{-1/2}$ [Recall that $\Sigma^{-1/2}$ maps ellipsoids to spheres]

Then W has covariance matrix $\mathbf{I}$, [If $X_i\sim\mathcal{N}(\mu,\Sigma),$, then approximately, $W_i\sim\mathcal{N}(0,I)$]

[whitening 相当于归一化，对SVM和NN都有好处；whitening is built in for QDA/LDA, 使用QDA/LDA不用进行preprocess ]

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403101709143.png" alt="image-20240310170944067" style="zoom:50%;" />



# LEC10 Regression menu, Least-squares linear reg, Logistic reg

REGRESSION

[QDA and LDA don’t just give us a classifier; they also give us the probability that a particular class label is correct. So QDA and LDA do regression on probability values]

x is data pts, w is model parameter, h is hypothesis (the fn we learn)

![image-20240305174425791](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403051744921.png)

menu

![image-20240305184105135](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403051841231.png)

[recall that LDA produces a posterior probability function with expression (3). So the logistic function seems to be a natural form for modeling certain probabilities. If we want to model posterior probabilities, sometimes we use LDA; but alternatively, we could skip fitting Gaussians to points, and instead just try to directly fit a logistic function to a set of probabilities.]

Cross-entropy的y和z必须在0 ~ 1，sigmoid的结果在0 ~ 1；aggregate all loss fn into a big one -> cost fn

maximum loss: optimize the max loss, sensitive to outlier

cost fn (c) 中的$\omega$是loss之间的权重，(d)(e)中的w是model parameter

![image-20240305184413106](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403051844153.png)



LEAST-SQUARES LINEAR REGRESSION (Gauss, 1801)

Linear regression fn (1) + squared loss fn (A) + cost fn (a)

![image-20240305192524962](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403051925169.png)

![image-20240305192549723](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403051925901.png)

Usually n > d

![image-20240305192651146](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403051926205.png)

Recall fictitious dimension trick [from Lecture 3]: rewrite h(x) = x · w + $\alpha$ as ![image-20240305192833442](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403051928644.png)

Now X is an n*(d + 1) matrix; w is a (d + 1)-vector. [We’ve added a column of all-1’s to the end of X]

![image-20240305193312496](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403051933707.png)

Find w that minimizes $\|Xw-y\|^2=\mathbf{RSS}(w)$, for **residual sum of squares**

Optimize by calculus:

![image-20240305194359834](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403051944060.png)

$X^{\mathsf{T}}X$ is always pos-semidef(not always pos-def), If $X^{\mathsf{T}}X$ is singular, problem is underconstrained  (never overconstrained). (underconstrained means has many solution, solution is a whole subspace, not one single sol; never overconstrained means has at least one sol).

[because the sample pts all lie on a common subspace (through the origin)？？？？？？]

![image-20240310182214815](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403101822003.png)

![image-20240310182226615](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403101822726.png)

 When $X^{\mathsf{T}}X$ pos-def: (solve the normal equation to get w, never calculate inverse which is slow)

![image-20240305200705264](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403052007358.png)

[We never compute X^+^ directly, but we are interested in the fact that w is a linear transformation of y.]

[X is not square, so X can’t have inverse. However, pseudoinverse is defined, if $X^{\mathsf{T}}X$ is invertible, then X^+^ is a “left inverse.”]

Observe: $X^+X=(X^\top X)^{-1}X^\top X=I\quad\Leftarrow(d+1)\times(d+1)$ explains the name “left inverse”

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403052018459.png" alt="image-20240305201823280" style="zoom:40%;" />

![image-20240305201747829](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403052017909.png)

[if n > d+1, then H is singular, rank d+1 or less (low rank)？？？？？？; ideally, H should be identity matrix]

![image-20240305204019343](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403052040447.png)

[In discussion section 6, we’ll address how to handle the underconstrained case where $X^{\top}X$ is singular]

[Apparently, least-squares linear regression was first posed and solved in 1801 by the great mathematician Carl Friedrich Gauss, who used least-squares regression to predict the trajectory of the planetoid Ceres. A paper he wrote on the topic is regarded as the birth of modern linear algebra.]



LOGISTIC REGRESSION (David Cox, 1958)

Logistic regression fn (3) + logistic loss fn (C) + cost fn (a). Fits “probabilities” in range [0, 1].

Usually used for classification. The input yi’s can be probabilities, but in most applications they’re all 0 or 1.

- QDA, LDA: generative models 
- logistic regression: discriminative model (directly predict posterior prob)

With X and w including the fictitious dimension ($\alpha$ is w’s last component)

![image-20240306011443244](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403060114374.png)

These loss functions are always convex

![image-20240306012351102](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403060123195.png)

J(w) is convex! Solve by gradient descent

![image-20240306012551489](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403060125609.png)

[compute $s(\gamma)$ to accelerate calculation]

![image-20240306012945552](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403060129631.png)

![image-20240306013004078](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403060130124.png)

梯度的负号到GD里负负得正

![image-20240306014621651](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403060146752.png)

[Works best if we shuffle pts in random order, process one by one. For very large n, sometimes converges before we visit all pts!]

Starting from w = 0 works well in practice [start from anything work well]

![image-20240306015014067](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403060150160.png)

If sample pts are linearly separable -> w · x = 0 separates them, scaling w to have infinite length causes s(Xi ·w)->1 for pt in class C, s(Xi ·w)->0 for pt not in class C, and J(w)->0 [as $||w||\rightarrow\infty $] 

[Moreover, making w grow extremely large is the only way to get the cost function J to approach zero.]

In above picture, the cost function J(w) has no finite local minimum, but gradient descent will “converge” to a solution, in the sense that the cost J will get arbitrarily close to zero, though of course the weight vector w will never become infinitely long. Mathematically speaking, w diverges, J(w) converge to zero

[A 2018 paper by Soudry, However, Nacson, Gunasekar, and Srebro shows that gradient descent applied to logistic regression eventually converges to the maximum margin classifier, but the convergence is very, very slow. A practical logistic regression solver should use a different optimization algorithm.]



# LEC11 Least-squares poly reg, weighted Least-squares  reg, Newton's method+Logistic reg, ROC curves

LEAST-SQUARES POLYNOMIAL REGRESSION

Replace each Xi with feature vector $\mathbf{\Phi}(X_i)$ with all terms of degree 0...p, Otherwise just like linear or logistic regression

2-feature data: $\Phi(X_i)=[X_{i1}^2\quad X_{i1}X_{i2}\quad X_{i2}^2\quad X_{i1}\quad X_{i2}\quad1]^\top $

[Notice that we’ve added the fictitious dimension “1” here, so we don’t need to add it again to do linear/logistic regression. This basis covers all polynomials quadratic in Xi1 and Xi2]

Log. reg. + quadratic features = same form of posteriors as QDA, Very easy to overfit!

两种overfit

![image-20240306030338015](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403060303147.png)

US population 预测 -> extrapolation badly，polynomial通常不擅长外推 

[k-nearest neighbour classifier 擅长外推，因为不会跳出已有数据范围]

![image-20240306030352263](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403060303309.png)

左图train pts, 右图test pts, test图中degree-10 regression does decent extrapolation for a short distance,

![image-20240306033851320](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403060338495.png)

 

WEIGHTED LEAST-SQUARES REGRESSION

Linear regression fn (1) + squared loss fn (A) + cost fn (c)

[The idea of weighted least-squares is that some sample points might be more trusted than others, or there might be certain points you want to fit particularly well. So you assign those more trusted points a higher weight. If you suspect some points of being outliers, you can assign them a lower weight.]

$\Omega$ is diagonal

![image-20240306034427204](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403060344351.png)

find the minimum by setting the gradient to zero, which leads us to solve for w in normal equations: $X^{\mathsf{T}}\boldsymbol{\Omega}Xw=X^{\mathsf{T}}\boldsymbol{\Omega}y$



NEWTON’S METHOD

Iterative optimization method for smooth fn J(w). (faster than GD) 

[We’ll use Newton’s method for logistic regression. Newton’s method does not work for Perceptron Algorithm]

Idea: You’re at point v. Approximate J(w) near v by quadratic fn. Jump to its unique critical pt. Repeat until bored

goal: find minimum of blue curve J

初始在蓝曲线上找一点，找到通过该点，在该点坡度相同的黄色抛物线 (second order approximation of blue curve)；跳转到黄色抛物线的最低点 

![image-20240307171205637](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403071712749.png)

Once newton's method get close to region whose surrounding looks quadratic, it jumps to minimum very fast

![image-20240307172007282](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403071720404.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403071734597.png)

[do not compute a matrix inverse directly. It's faster to solve a linear system of equations by Cholesky factorization/conjugate gradient method]

![image-20240307173844589](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403071738649.png)

Warning: Doesn’t know difference between minima, maxima, saddle pts. 

If multiple critical pt, starting pt must be “close enough” to desired critical pt.

[The closer J is to quadratic, the faster Newton’s method tends to converge.]

[Newton’s method is superior to blind gradient descent for some optimization problems for several reasons. First, it tries to find the right step length to reach the minimum, rather than just walking an arbitrary distance downhill (learning rate). Second, rather than follow the direction of steepest descent, it tries to choose a better descent direction.

Nevertheless, it has some major disadvantages. The biggest one is that computing the Hessian can be quite expensive, and it has to be recomputed every iteration. It can work well for low-dimensional weight spaces, but you would never use it for a neural network, because there are too many weights. Newton’s method also doesn’t work for most nonsmooth functions. It particularly fails for the perceptron risk function, whose Hessian is zero, except where the Hessian is not even defined.]



LOGISTIC REGRESSION WITH NEWTON’S METHOD

![image-20240307180932526](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403071809626.png)

derive the Hessian, $\begin{aligned}\sum_{i=1}^ns_iX_i\end{aligned}$ 求导

![image-20240307180952750](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403071809797.png)

si is 0~1 -> $\Omega$ is always pos-def, $X^\top\boldsymbol{\Omega}X$ is always pos-semidef -> J is convex

[The logistic regression cost function is convex, so Newton’s method finds a globally optimal point if it converges.]

$\Omega$ & s are fns of w

![image-20240307202526779](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403072025977.png)

[Notice that this looks a lot like weighted least squares, but the weight matrix $\Omega$ and the right-hand-side vector y-s change every iteration. So we call it An example of iteratively reweighted least squares (相当于weight changes at every iteration)

weighted least-squares regression中big weight in $\Omega$ means putting more emphasis on a point, 与weighted least-squares regression相反的是, a small weight in $\Omega$ causes the Newton iteration to put more emphasis on a point when it computes e (由于仅左侧有$\Omega$，除到右边)]

LDA vs. Logistic Regression

Advantages of LDA: 

- For well-separated classes, LDA stable; log. reg. surprisingly unstable (LR适合两种点混在一起的情况)
- $>$2 classes easy & elegant; log. reg. needs modifying (softmax regression) [see Discussion 6] 
- LDA slightly more accurate when classes nearly normal, especially when not a lot of data

Advantages of log. reg.:

- More emphasis on decision boundary (More emphasis on some pts than others); always separates linearly separable pts 
- More robust on some non-Gaussian distributions (e.g., distribution with large skew, more pt on one side than another) 
- Naturally fits labels between 0 and 1 [usually probabilities

[Correctly classified points far from the decision boundary have a small effect on logistic regression; misclassified points far from the decision boundary ha ve the biggest effect. By contrast, LDA gives all the sample points equal weight when fitting Gaussians to them. Weighting points according to how badly they’re misclassified is good for reducing training error, but it can also be bad if you want stability or insensitivity to bad data.]

linearly separable data set with a very narrow margin, LR vs. LDA

![image-20240307211719576](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403072117668.png)



ROC CURVES (for test sets) (receiver operating characteristics)

generate this curve by trying every probability threshold; for each threshold, measure the false positive & true positive rates and plot a point. after trained, run on test set/validation set

Shows rate of false positives vs. true positives. We assume there is a knob we can turn to trade off these two types of error. Here the knob is the posterior probability threshold (>threshold 就预测 +ve) for Gaussian discriminant analysis or logistic regression.]

false negative rate = 1 - sensitivity; specificity = 1 - false positive rate

斜直线：random classifier，不管输入是什么，按概率给输出；如中点就是不看输入，50%预测+ve，50%预测-ve 

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403072122733.png" alt="image-20240307212244677" style="zoom:50%;" />

[A rough measure of a classifier’s effectiveness is the area under the curve. For a classifier that is always correct, the area under the curve is one. For the random classifier, the area under the curve is 1/2, so you’d better do better than that.]

IMPORTANT: In practice, the trade-off between false negatives and false positives is usually negotiated协商 by choosing a point on this plot, based on real test data, and NOT by taking the choice of threshold that’s best in theory



# LEC12 Statistical justification for regression, bias-variance decomp

[4 levels ML: the application, the model, the optimization problem, the optimization algorithm. last two lectures about regression were at the bottom two levels: optimization. But why did we pick these cost functions? Today is the second level, the model. I will describe some models, how they lead to those optimization problems, and how they contribute to underfitting or overfitting.]

STATISTICAL JUSTIFICATIONS FOR REGRESSION

Typical model of reality: 

- sample points come from unknown prob. distribution: $X_{i}\sim D$

- y-values are sum of unknown, non-random fn + random noise: $\begin{aligned}\forall X_i,\quad&y_i=g(X_i)+\epsilon_i,&\epsilon_i\sim D'\end{aligned}$  

  (D' has mean zero; fn g means ground-truth)

[reality is described by a function g. g is unknown but not random; it represents a consistent relationship between X and y that we can estimate. noise is independent of X (this is not realistic in practice, but that’s all we’ll have time to deal with this semester) (Also notice that this model leaves out systematic errors, like when your measuring device adds one to every measurement, we usually can’t diagnose systematic errors from data alone.)]

Goal of regression: find h that estimates g

expected value of label Y for any particular fixed training pt x (definition of ground-truth)

![image-20240307222555704](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403072225859.png)

Least-Squares Cost Function from Maximum Likelihood

![image-20240307222805981](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403072228204.png)

[We treat g as a “distribution parameter.” If the noise is normally distributed, maximum likelihood tells us to estimate g by least-squares regression.]

[However, I’ve told you in previous lectures that least-squares is very sensitive to outliers. If the error is truly normally distributed, that’s not a big deal, especially when you have a lot of sample points. But in the real world, the distribution of outliers often isn’t normal. Outliers might come from wrongly measured measurements, data entry errors, anomalous events, or just not having a normal distribution. When you have a heavy-tailed distribution of noise, for example, least-squares isn’t a good choice.]



Empirical 经验 Risk 即训练误差，bias，不考虑overfit

The risk for hypothesis h is expected loss R(h) = E[L] over all training pt  (X, Y) in some joint distribution

[for Gaussian discriminant analysis (generative model), we can estimate the joint probability distribution for X and Y and derive the expected loss. But now assume we don’t have generative model, we approximate the distribution in a very crude way: we pretend that the sample points are the distribution.]

Discriminative model: we don’t know X’s distribution D. How can we minimize risk?

Empirical distribution: the discrete uniform distribution over the sample pts

Empirical risk: expected loss under empirical distribution

![image-20240307232639839](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403072326085.png)

[The hat on the R indicates that $\hat{\mathbb{R}}$ is only a cheap approximation of the true, unknown statistical risk R. Often, this is the best we can do. For many but not all distributions, the empirical risk converges to the true risk in the limit as n->∞. Choosing h that minimizes $\hat{\mathbb{R}}$ is called empirical risk minimization.]

Takeaway: this is why we [usually] minimize the sum of loss fns



Logistic Loss from Maximum Likelihood

What cost fn should we use for probabilities (number of 0~1)? 

Actual probability pt Xi is in class C is yi (label); predicted prob is h(Xi).

Imagine $\beta$ duplicate copies of Xi, with yi$\beta$ in class C, and (1-yi)$\beta$ not (actually there is only one Xi, 这里是为了表示它有70%在classC，有30%不在)

[Let’s use MLE to choose the hypothesis most likely to generate these labels for these sample points. The following likelihood is the probability of generating these labels **in a particular fixed order**.]

![image-20240309142636639](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403091426758.png)

logistic loss fn是负的，因此前面多出一个负号

![image-20240307234807133](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403072348277.png)

Takeaway: Max likelihood -> minimize P logistic losses. [explains where the weird logistic loss function comes from.]



THE BIAS-VARIANCE DECOMPOSITION

There are 2 sources of error in a hypothesis h: (why h does not come out g)

![image-20240308001308478](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403080013614.png)

![image-20240308001522816](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403080015989.png)

h is a vector of weights

![image-20240308001906882](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403080019040.png)

z is particular fixed point (no randomness, test pt, not necessarily training pt)

![image-20240308002456643](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403080024701.png)

decompose risk fn into bias and variance

randomness: h is random var (randomness inherited from X,y which come from some joint distribution？？？？？, we use training data to choose weight), $\gamma$ is random for added noise 

![image-20240308002540973](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403080025041.png)

![image-20240308003754954](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403080037014.png)

pointwise version of the bias-variance decomposition

曲线是gt，从曲线上任取20个带噪的点(not necessarily on curve)，fit 一条直线，重复50次(模拟无限次)；右图bias图中红线是50条线的平均，查看z点的三种误差来源

- bias is difference between the black and red curves
- At center right, the variance is the expected squared difference between a random black line and the red line (at test point z)
-  At lower right, the irreducible error is the expected squared difference between a random test point and the sine wave.

![image-20240308004714169](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403080047310.png)

Mean version: let $z\sim D$ (from the same distribution D as training pt) be random variable; take mean over D of bias^2^, variance. 

->

- Underfitting = too much bias
- Most overfitting caused by too much variance
- Training error reflects bias but not much variance; test error reflects both
- For many distributions, variance->0 as n->∞ (训练样本增多)
- If h can fit g exactly, for many distributions bias->0 as n->∞ (h要有足够的复杂度) 
- If h cannot fit g well, bias is large at “most” points – Adding a good feature reduce bias; adding a bad feature rarely increase it
- Adding a feature usually increases variance [don’t add a feature unless it reduces bias more] 
- Can’t reduce irreducible error
- Noise in test set affects only Var($\epsilon$); noise in training set affects only bias & Var(h) 
- We can’t precisely measure bias or variance of real-world data [we can't know g exactly and our noise model might be wrong] 
- But we can test learning algs by choosing g & making synthetic data

横轴复杂度逐渐增加

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403080112075.png" alt="image-20240308011213983" style="zoom:67%;" />



Example: Least-Squares Linear Reg

For simplicity, no fictitious dimension. [our linear regression function has to be zero at the origin.]

![image-20240308153602592](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403081536765.png)

Lin. reg. computes weights

![image-20240308153952780](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403081539921.png)

BIAS is $\operatorname{E}[h(z)]-g(z)=\operatorname{E}[w^\top z]-\nu^\top z=z^\top\operatorname{E}[w-\nu]=z^\top\operatorname{E}[X^+e]=z^\top\operatorname{E}[X^+]\operatorname{E}[e]=0$

Warning: This does not mean h(z) - g(z) is always 0! Sometimes +ve, sometimes -ve, mean over training sets is 0. 

[Those deviations偏差 from the mean are captured in the variance.？？？？？]

[When the bias is zero, a perfect fit is possible. But when a perfect fit is possible, not all learning methods give you a bias of zero; here it’s a benefit of the squared error loss function. With a different noise or a different loss function, we might have a nonzero bias even fitting a linear h to a linear g.？？？？？]

VARIANCE is $\operatorname{Var}(h(z))=\operatorname{Var}(w^\top z)=\operatorname{Var}(\nu^\top z+(X^+e)^\top z)=\operatorname{Var}(z^\top X^+e)$

？？？？？？？？？

[This is the dot product of a vector $z^\top X^+$ with an isotropic 各向同性, normally distributed vector e. The dot product reduces it to a one-dimensional Gaussian along the direction $z^\top X^+$, so this variance is just the variance of the 1D Gaussian times the squared length of the vector $z^\top X^+$.]

$\begin{aligned}
&=\sigma^2\left\|z^\top X^+\right\|^2=\sigma^2z^\top(X^\top X)^{-1}X^\top X(X^\top X)^{-1}z
&=\sigma^2z^\top(X^\top X)^{-1}z
\end{aligned}$

If we choose coordinate system so D has mean zero, then $X^\top X\to n\operatorname{Var}(D)$ as n -> ∞, so for $z\sim D$

$\mathrm{Var}(h(z))\approx\sigma^2\frac dn$ [where d is the dimension—the number of features per sample point.]

![image-20240308162356933](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403081623033.png)

？？？？？？？？？

Takeaways: 

- Bias can be zero when hypothesis function can fit the real one! [a nice property of the squared error loss function.] 
- Variance portion of RSS (overfitting) 
  - decreases as 1/n (sample points)
  - increases as d (features) or O(d^p^) if you use degree-p polynomials

[I’ve used linear regression because it’s a relatively simple example. But the bias-variance trade-off applies to many learning algorithms, including classification as well as regression. But for most learning algorithms, the math gets a lot more complicated than this, if you can do it at all. Sometimes there are multiple competing bias-variance models and no consensus on which is the right one.]



# LEC13 Ridge regression, MAP, Feature selection, Lasso

RIDGE REGRESSION aka Tikhonov Regularization

Least-squares linear regression + $\ell_2$-penalized mean loss. (1) + (A) + (a) + (d)

![image-20240309154747600](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403091547815.png)

w' is w with component $\alpha$ replaced by 0 (w的最后一列fictitious dim置零)

X has fictitious dimension but we DON’T penalize $\alpha$ ($\alpha$ is just changing the coordinate system, we don't want to favor any one coordinate system？？？？)

a regularization term, aka a penalty term, for shrinkage: to encourage small $||w^{\prime}||$

暂时省略$\alpha$ ->

1. Guarantees pos-def normal equation; no underconstrained, always unique solution.

[Standard least-squares linear regression yields singular normal equations when the sample points lie on a common hyperplane in feature space—for example, when d > n.]

fitting a plane to green pts, 绿点在一条直线上，因此过该直线的无数平面都是可行解

![image-20240310004808431](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100048542.png)

Pos-semidef (ill-posed) -> regularization -> pos-def (well-posed)

![image-20240310005151109](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100051147.png)

2. Reduces overfitting by reducing variance

Example: Input X1 = (0, 0) with label 0; X2 = (1, 1) with label 0; X3 = (0.51, 0.49) with label 1. Linear reg gives 50x1-50x2. [This linear function fits all three points exactly.一个三维空间中过该三点的二维平面] Big weights!

[label间最大差值➗点之间最小距离 = 正常weight的大小，这里1/0.5 = 2，而weight = 50，是异常的；Weights this big would be justified if there were big differences between labels, or if there were small distances between points, but neither is true. Large weights imply that tiny changes in x can cause huge changes in y. Consider that the labels don’t differ by more than 1 and the points are separated by distances greater than 0.7. So these disproportionately large weights are a sure sign of overfitting.]

penalize large weight. When you have large variance and a lot of overfitting, it implies that your problem is close to being ill-posed (X3 = (0.5, 0.5)就是ill-posed), 这种情况下penalize可以帮助减少overfit

可视化 J(w)

The ridge regression solution lies where a red isocontour just touches a blue isocontour tangentially. $\lambda$ 增大，按绿线移动; shrinks w and helps to reduce overfitting. [绿线通常贴近椭圆的长轴而非短轴，因为长轴是change slowly的direction]

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100132616.png)

solve it

![image-20240310014026011](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100140103.png)

where I' ((d+1)*(d+1 )) is identity matrix with 右下角元素 set to zero. [Don’t penalize the bias term $\alpha$]

[$X^{\top}X+\lambda I^{\prime}$ is always positive definite for $\lambda$ > 0, assuming X ends with a column of 1’s.]

solve norm eq for w, thus $h(z)=w^\top z$

![image-20240310031142357](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100311517.png)

![image-20240310031314038](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100313118.png)

Ideally, features should be “normalized” to have same variance, then can equally penalize them

![image-20240310032419858](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100324947.png)

[Alternative: use asymmetric penalty by replacing I' w/other diagonal matrix. For example, if you use polynomial features, you could use different penalties for monomials of different degrees]



Bayesian Justification for Ridge Reg

trust small weight vector w' more than big w' -> Assign a prior probability on w': $w^{\prime}\sim\mathcal{N}(0,\varsigma^2)$ with PDF $f(w^{\prime})\propto e^{-\|w^{\prime}\|^2/(2\varsigma^2)}$

[This prior prob says that we think weights close to zero are more likely to be correct.] 

Apply MLE to maximize the posterior prob:

![image-20240310033133822](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100331898.png)

[We are treating w and y as random variables, but X as a fixed constant—it’s not random.]

This method (using MLE, but maximizing posterior) is called maximum a posteriori (MAP). 

[A prior probability on the weights is another way to understand regularizing ill-posed problems.]



FEATURE SUBSET SELECTION

[Some of you may have noticed as early as Homework 1 that you can sometimes get better performance on a spam classifier simply by dropping some useless features.]

All features increase variance, but not all features reduce bias. 

Idea: Identify poorly predictive features, ignore them (weight zero). 

- Less overfitting, smaller test errors. 
- 2nd motivation: Inference. Simpler models convey interpretable wisdom.

Sometimes it’s hard: Different features can partly encode same information. Combinatorially hard to choose best feature subset.

Alg: Best subset selection. Try all 2^d^ - 1 nonempty subsets of features. [Train one classifier per subset.] Choose best classifier by (cross-)validation. Slow.

Other heuristics

![image-20240310034824180](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100348312.png)

![image-20240310035022798](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100350845.png)

[Forward stepwise is a better choice when you suspect only a few features will be good predictors; e.g., spam. Backward stepwise is better when most features are important. If you’re lucky, you’ll stop early.]



LASSO (Robert Tibshirani, 1996) (“Least absolute shrinkage and selection operator”)

Least-squares linear regression + $\ell_1$-penalized mean loss. (1) + (A) + (a) + (e)

[it often naturally sets some of the weights to zero]

![image-20240310040343267](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100403374.png)

- ridge reg: isosurfaces of $||w^{\prime}||^2$ are hyperspheres. 
- The isosurfaces of $||w^{\prime}||_1$ are cross-polytopes. 

The unit cross-polytope is the convex hull of all the positive & negative unit coordinate vectors.

![image-20240310040557848](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100405936.png)

When the red isocontour touches the tip 尖 of the diamond, the weight w1 gets set to zero

对于上右图三位情况，可能有0/1/2个w分量set to zero [For example, in 3D, if the red isosurface touches a sharp vertex of the cross-polytope, two of the three weights get set to zero. If it touches a sharp edge of the cross-polytope, one weight gets set to zero. If it touches a flat side of the cross-polytope, no weight is zero]

![image-20240310041534610](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100415683.png)

that’s what we want to happen to weights that don’t have enough influence. [This doesn’t always happen—for instance, the red isosurface could touch a side of the diamond instead of a tip of the diamond.]

![image-20240310041325119](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100413170.png)

对于predict power很大的w分量不容易go to zero，反之容易；Add feature always increase variance, sometimes reduce bias, predict power is reduce bias more than increase variance

[This shows the weights for a linear regression problem with 10 weight分量. As lambda increases, more and more of the weights become zero. Only four of the weights are really useful for prediction; they’re in color. 灰色的六个是useless feature

Statisticians used to choose by looking at a chart like this and trying to eyeball a spot where the re aren’t too many predictors and the weights aren’t changing too fast. But nowadays they prefer validation.]

![image-20240310042014902](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403100420014.png)

Sometimes sets some weights to zero, especially for large $\lambda$. 

Algs: subgradient descent, least-angle regression (LARS), forward stagewise

[Lasso can be reformulated as a quadratic program, but it’s a quadratic program with 2^d^ constraints, because a d-dim cross-polytope has 2^d^ facets. In practice, special-purpose optimization methods have been developed for Lasso. I’m not going to teach you one, but if you need one, look up the last two of these algorithms. LARS is built into the R Language for statistics.] 

[As with ridge regression, you should probably normalize the features first before applying Lasso.]



# LEC14 decision tree

DECISION TREES: Nonlinear method for classification

Uses tree with 2 node types: 

- internal nodes test feature values (usually just one) & branch accordingly 
- leaf nodes specify class h(x)

![image-20240323153316143](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403231533370.png)

![image-20240323153330868](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403231533014.png)

- Cuts x-space into rectangular cells 
- Works well with both categorical and quantitative features (特别是categorical feature, 相对于其他模型更适合)
- Interpretable result (inference，可以画出决策树图，可解释性) 
- Decision boundary can be arbitrarily complicated (can be as complicated as you want)

![image-20240323154322164](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403231543236.png)

Greedy, top-down learning heuristic (cannot prove get best tree, get best will be NP hard):

Let S $\subseteq$ {1, 2,..., n} be subset of training point indices. Top-level call: S = {1, 2,..., n}.

X_ij, point i, feature j

if 中return的是leaf node，else中return的是internal node，记录了splitting feature&value

[not necessarily binary split]

![image-20240323155552868](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403231555023.png)

(*) How to choose best split?

- Try all splits. [All features, and all splits within a feature] 

- let J(S) be the cost of set S . 

- Choose the split that 

  - minimizes J(S_l) + J(S_r); 

  - minimizes weighted average $\frac{|S_l|J(S_l)+|S_r|J(S_r)}{|S_l|+|S_r|}$ (|S_l|表示S_l的点的个数 set cardinality）

    [更常用，有可能左边一万点，右边一百点]

choose cost J(S): Measure the entropy

Let Y be a random class variable, and suppose P(Y = C) = pC. The surprise of Y being class C is $-\log_2p_\mathbb{C}$ [Always nonnegative.] 

- event w/prob 1 gives us zero surprise [我知道这是class C，一点也不惊讶]
- event w/prob 0 gives us infinite surprise

[In information theory, the surprise is equal to the expected number of bits of information we need to transmit which events happened to a recipient who knows the probabilities of the events. Often this means using fractional bits, which may sound crazy, but it makes sense when you’re compiling lots of events into a single message; e.g. a sequence of biased coin flips.]

The entropy of an index set S is the average surprise [when you draw a point at random from S]

![image-20240323182720667](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403231827814.png)

如果一半C一半D，需要1bit来表示；如果n pts all different class， 需要log2(n) bit来表示

![image-20240323183632045](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403231836135.png)

[The entropy is the expected number of bits of information we need to transmit to identify the class of a sample point in S chosen uniformly at random]

- Left: entropy H(pC) when there are only two classes. (pD = 1-pC)
- Right: entropy H(pC, pD) when there are three classes. (pE = 1-pC-pD) Observe that the entropy is strictly concave

![image-20240323192056832](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403231920940.png)

J(S) -> H, Weighted avg entropy after split is $\begin{aligned}H_{\text{after}} & = \frac { \left | S _ { l }\right|H(S_{l})+\left|S_{r}\right|H(S_{r})}{\left|S_{l}\right|+\left|S_{r}\right|}\end{aligned}$

Choose split that (minimize Hafter) maximize information gain $\boldsymbol{H(S)-H_{\mathrm{after}}}$ (information gain always >= 0)

![image-20240323193150573](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403231931704.png)

Info gain 在两种情况下为0，其余情况均>0

- one child is empty
- for all C, $\begin{aligned}P(y_i=\mathbf{C}|i\in S_l)=P(y_i=\mathbf{C}|i\in S_r)\end{aligned}$ (如20C 10D -> 10C 5D + 10C 5D)

left: concave 凹, 只要H(Sl)与H(Sr)不是曲线上同一点 -> info gain > 0 (不是直线上中点因为是weighted, H(s)必然在Hafter竖直上方)

right: 如果定义 J(pC) = % of pts misclassified -> concave but not strictly concave -> 只要 J(Sl) J(Sr) 都在同一侧直线上, info gain=0(J(s)必然在Jafter竖直上方) -> doesn't work well, shouldn't use this definition

![image-20240323194839302](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403231948467.png)

[By the way, the entropy is not the only function that works well. Many concave functions work fine, including the simple polynomial p(1- p)] 

More on choosing a split: 

- For binary feature xi: children are xi = 0 & xi = 1. 
- If xi has 3+ discrete values: split depends on application. [use multiway splits/cluster into binary splits.] 
- If xi is quantitative (连续的): sort all pts of S on feature xi; 每两个不同xi值之间任选一根划分线即可 (下图的四根线在划分上一样)

​		[radix sort in linear time if n is huge] 

![image-20240323202209232](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403232022357.png)

Clever bit: As you scan sorted list from left to right, you can update entropy in O(1) time per point!

每根划分线都需要计算左侧右侧X和C的数量，四个数值；第一根划分线需要linear time，第二根及之后划分线只需要O(1) (如第一根划分线到第二根划分线经过一个C，只需要左侧C+1，右侧C-1)

![image-20240323202609787](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403232026853.png)

![image-20240323202926470](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403232029583.png)

Algs & running times: (n is number of training pts)

![image-20240323203049491](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403232030619.png)



# LEC15 decision tree variation, ensemble learning, bagging, random forest

DECISION TREE VARIATIONS

Decision Tree Regression: Creates a piecewise constant regression fn (not continuous)

虽然stupid但是是最常用的 Decision Tree Regression

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403241911318.png)

Leaf stores labe: $\mu_S=\frac1{|S|}\sum_{i\in S}y_i$, the mean label for training pts $i\in S$ (一个box五个点，该box的label就是五个点label的平均值)

Cost $\begin{aligned}J(S)=\mathrm{Var}(\{y_i:i\in S\})=\frac{1}{|S|}\sum_{i\in S}(y_i-\mu_S)^2\end{aligned}$

We choose the split that minimizes the weighted average of the variances of the children after the split

***

Stopping Early (back to classification)

We do not need to keeps subdividing treenodes until every leaf is pure (数据很精确时可以go to pure leaves, 数据量大时没必要)

- Limit tree depth (for speed) 
- Limit tree size (big data sets) 
- Pure tree may overfit 
- Given noise or overlapping distributions, pure leaves tend to overfit; better to stop early and estimate posterior probs

[When you have strongly overlapping class distributions, refining the tree down to one training point per leaf is absolutely guaranteed to overfit, giving you a 1-nearest neighbor classifier. It’s better to stop early, then classify each leaf node by taking a vote of its training points; this gives you a classifier akin to a k-nearest neighbor classifier.]

如横轴身高，竖轴性别

![image-20240324192839125](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403241928251.png)

Instead of returning the majority class, each leaf could return a posterior probability histogram

![image-20240324193149340](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403241931398.png)

Leaves with multiple points return 

- a majority vote or class posterior probs (classification)
- an average (regression)

Select stopping condition(s)

- Next split doesn’t reduce entropy/error enough (dangerous; pruning剪枝 is better) 
- Most of node’s points (e.g., > 90%) have same class [to deal with outliers & overlapping distribs] 
- Node contain few training points (如< 10) [especially for big data] or Box edges are **all** tiny [deep resolution may be pointless]
- Depth too great [risky if there are still many training points in the box] 
- use validation to decide whether splitting the node lowers your validation error [slowest but most effective ]

[But if your goal is to avoid overfitting, it’s generally even more effective to grow the tree a little too large and then use validation to prune it back]

***

Pruning (after building the tree)

greedily remove each split whose removal improves valid performance. [do validation once for each split considering reversing.] 

[The reason why pruning often works better than stopping early: a split that doesn’t seem to make much progress is followed by a split that makes a lot of progress. If you stop early, you’ll never find out. Pruning is simple and highly recommended when you have enough time]

![image-20240325231530011](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403252315102.png)

左图横轴typo: tree size -> # leaf node

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403252317582.png" alt="image-20240325231701511" style="zoom:30%;" />

It might seem expensive to do validation once for each split we consider reversing. But you can do it pretty cheaply

![image-20240325235102700](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403252351828.png)

***

Multivariate Splits

Find non-axis-aligned splits with other classification algs (SVMs, logistic regression, GDA) or by generating them randomly (随机生成几个split选最好的那个) -> gain better classifier at cost of worse interpretability or speed

左图每个split 都是 single variate (fast)；右图split是multivariate

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403252356296.png)

[A good compromise is to set a limit on the number of features]

***

ENSEMBLE 合唱 LEARNING

Decision trees are 

- fast, simple, interpretable, easy to explain, invariant under scaling/translation (坐标轴变化不影响模型), robust to irrelevant features (自动忽略不好的feature，不会去用)
- not the best at prediction [Compared to previous methods], High variance [Though we can achieve very low bias]

[suppose we take a training data set, split it into two halves, if the two trees pick different features for the very first split at the root of the tree, then it’s quite common for the trees to be completely different. So decision trees tend to have high variance. we can reduce the variance of decision trees by taking an average answer of a bunch of decision trees]

We can take average of output of 

- different learning algs 
- same learning alg on many training sets [if we have tons of data] 
- **bagging**: same learning alg on many random subsamples of one training set 
- **random forests**: randomized decision trees on random subsamples

Metalearner takes test point, feeds it into all T learners, returns majority class/average output (for all alg, not only decision tree)

- Regression algs: take median or mean output [of all the weak learners] 
- Classification algs: take majority vote OR average posterior probs

tips

Use learners with low bias (e.g., deep decision trees). 

High variance & some overfitting are okay. Averaging reduces the variance! [Each learner overfit in its own unique way.] 

Averaging sometimes reduces bias & increases flexibility a bit, but not reliably. e.g., linear classifiers -> average -> nonlinear decision boundary  [Averaging reduces variance more than bias, so get the bias small before averaging] 

Hyperparameter settings usually different than 1 learner. [pick hyperparameter with low bias high variance] [number of trees is not hyperparameter. When random forests work, usually the variance drops monotonically with the number of trees, and the main limit on the number of trees is computation time.]

![image-20240505222751287](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405052227437.png)

***

Bagging = Bootstrap AGGregatING (Leo Breiman, 1994 at Berkeley)

[Bagging is a randomized method for creating many different learners from the same data set. It works well with many learning algorithms. One exception seems to be k-nearest neighbors; bagging degrades it.]

Given n-point training sample, generate random subsample of size n' by sampling with replacement放回. (Some points chosen multiple times; some not chosen)

very usually, n' = n, 63.2% on average are chosen

![image-20240326004038951](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403260040123.png)

Build learner. Points chosen j times have greater weight:

[If a point is chosen j times, we want to treat it the same way we would treat j different points all bunched up infinitesimally close together无限紧密地聚集在一起] 

![image-20240326004612115](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403260046208.png)

Repeat until T learners.

***

Random Forests: bagging + randomized tree

[bagging isn't randomized enough, often the decision trees look very similar. need randomized tree]

![image-20240326005211914](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403260052072.png)

[split tend to “decorrelate” the trees, when you take average of the trees, you will have less variance than a single tree. Averaging works best when you have very strong learners that are also diverse.] 

Sometimes test error drops even at 100s or 1,000s of decision trees! 

Disadvantages: slow; loses interpretability/inference.

![image-20240326010633296](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403260106403.png)

[Random forest classifiers for 4-class spiral螺旋 data. Each forest takes the average of 400 trees. The top row uses trees of depth 4. The bottom row uses trees of depth 12. From left to right, we have axis-aligned splits, splits with lines with arbitrary rotations, and splits with conic sections圆锥曲线. Each split is chosen to be the best of 500 random choices.]

![image-20240326010734205](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403260107265.png)

[In these examples, all the splits are axis-aligned. The top row uses trees of depth 4. The bottom row uses trees of depth 12. From left to right, we choose each split from 1, 5, or 50 random choices. The more choices, the less bias and the better the classifier.]



# LEC16 Kernel

Recall featurizing map $\mathbf{\Phi}:\mathbb{R}^d\to\mathbb{R}^D$ d表示低维，D表示高维

degree-p polynomials -> $D\in\Theta(d^p)$ features [100 feature&degree-4 polynomial, -> 4 million feature]

KERNELS -> use features without computing them

Observation: In many learning algs, 

- the weights can be written as a linear comb of training points (ridge/logistic regression, SVM, perceptron)

- we can use inner products of $\mathbf{\Phi}(x)$’s only -> don’t need to compute $\mathbf{\Phi}(x)$, compute O(d) instead of O(D)

  ![image-20240509153502441](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405091535396.png)

suppose optimal weight vector $w=X^\top a=\sum_{i=1}^na_iX_i$ for some $a\in\mathbb{R}^n$

Substitute this identity into alg and optimize dual weights/para a (n维) instead of primal原始 weights w (D维)

***

Kernel Ridge Regression

we should peanalize bias term $\alpha$/center pts to make $\alpha$ = 0, to satisfy: weight can be written as a linear combo of training point

[By centering, the actual bias won’t usually be exactly 0, but it will be close enough]

Center X and y so their means are zero: $X_i\leftarrow X_i-\mu_X,\quad y_i\leftarrow y_i-\mu_y,\quad X_{i,d+1}=1$  [don't center the 1's!]

This lets us replace $I^{\prime}$ with $I$ in normal equations: $(X^\top X+\lambda I)w=X^\top y$

Suppose a is a solution to $(XX^\top+\lambda I)a=y.$ [Always has a solution if $\lambda$ > 0]

[注意式1是 $X^\top X$；式2是$XX^\top$]

式2两边左乘$X^\top$ -> $X^\top y=X^\top XX^\top a+\lambda X^\top a=(X^\top X+\lambda I)X^\top a$ 

-> 解式2得到a, w = $X^\top a$ is a solution to 式1, and w is a linear comb of training points

![image-20240326162229192](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403261622338.png)

a is a dual solution; solves the dual form of ridge regression:

![image-20240326162531331](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403261625582.png)

[We obtain this dual form by substituting $w=X^\mathrm{T}a$ into the original ridge regression cost function.]

testing: weighted sum of inner products between training pts $Xi^T$ and test pts $z$

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403261630880.png)

Let $k(x,z)=x^\top z$ be kernel fn [Later, we’ll replace x and z with $\Phi(x)$ and $\Phi(z)$, and that’s where the magic will happen.]

Let $K=XX^\top $ be n * n kernel matrix. Note $K_{ij}=k(X_i,X_j)$

if X doesn't have rank d, then K is singular -> probably no solution if $\lambda$ = 0. [we choose positive $\lambda$ to fix this] 

K is always singular if n > d + 1. [But you only want to use the dual form when d > n 如poly feature blow up to be more than n. But K could still be singular when d > n.] 

前两行是training

dual ridge regression produces the same predictions as primal ridge regression (with a penalized bias term)! The difference is the running time; the dual algorithm is faster if d > n, because the primal algorithm solves a d * d linear system, whereas the dual algorithm solves an n * n linear system.]

![image-20240326173125687](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403261731779.png)

***

The Kernel Trick (aka Kernelization)

[compute a polynomial kernel without actually computing the features]

The polynomial kernel of degree p is $k(x,z)=(x^\top z+1)^p$

monomial 单项式; $x^{\top}z$ = x1z1 + x2z2 

![image-20240326173841296](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403261738526.png)

[Notice the factors of $\sqrt{2}$. If you try higher polynomial degree p, you’ll see a wider variety of these constants, they don’t matter much, because the implicit primal weights w will scale themselves to compensate]

计算$\Phi(x)$ and $\Phi(z)$再取inner product很慢O(D); 计算$(x^\top z+1)^p$很快O(d)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403261754204.png)

![image-20240326180525985](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403261805259.png)

***

Kernel Logistic Regression (similar to kernel perceptron, so we skip it)

原始算法

![image-20240326181326375](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403261813559.png)

Dualize with $\boldsymbol{w}=\boldsymbol{\Phi}(X)^\top\boldsymbol{a}$

$w\leftarrow w+\epsilon\Phi(X)^{\top}(y-s(\Phi(X)w))$ 两边同除$\Phi(X)^\top$ ($w=\Phi(X)^\top a$) -> $a\leftarrow a+\epsilon(y-s(\Phi(X)w))$

Let $\begin{aligned}K=\Phi(X)\Phi(X)^\top\end{aligned}$  [The n * n kernel matrix; but we don’t compute $\mathbf{\Phi}(X)$—we use the kernel trick]

Note that $\begin{aligned}Ka=\Phi(X)\Phi(X)^\top a=\Phi(X)w\end{aligned}$ which appears in the algorithm above

最后test时$\sum_{i=1}^na_ik(X_i,z)$即可给出分类结果，$s\left(\sum_{i=1}^na_ik(X_i,z)\right)$给出posterior prob；s apply component-wise to vector 

![image-20240326182800546](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403261828625.png)

Training for j iterations:

![image-20240326183502066](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403261835159.png)



Alternative training: stochastic gradient descent (SGD). Primal logistic SGD step is $w\leftarrow w+\epsilon\left(y_i-s(\Phi(X_i)^\top w)\right)\Phi(X_i)$

Dual logistic SGD maintains a vector $q=Ka\in\mathbb{R}^n$ (倒二行->倒一行), Note that $\begin{aligned}q_i=(\Phi(X)w)_i=\Phi(X_i)^\top w\end{aligned}$

![image-20240327130344402](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403271303633.png)

[SGD updates only one dual weight ai per iteration; that’s a nice benefit of the dual formulation. We cleverly update q = Ka in linear time instead of performing a quadratic-time matrix-vector multiplication.]

![image-20240327140201041](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403271402137.png)

[Alternative testing: If # of training points and test points both exceed D/d, classifying with primal weights w may be faster. This appies to ridge regression as well.]

![image-20240327140307141](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403271403202.png)

***

Gaussian Kernel/radial basis fn kernel

[we can do fast computations in spaces with exponentially large dimensions, why don’t we generate infinite-dim feature]

$\mathbf{\Phi}:\mathbb{R}^d\to\mathbb{R}^\infty $ such that 

![image-20240327140628908](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403271406079.png)

here’s the feature vector that gives you this kernel for d = 1

![image-20240327141140859](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403271411013.png)

[This is an infinite vector, and $\Phi(x)\cdot\Phi(z)$ is a series that converges to k(x,z). Nobody actually uses $\Phi(x)$ directly, nobody cares about it; we just use kernel function k(·, ·)]

[it’s best not to think of points in a high-dimensional space. Instead, think of the kernel k as a measure of how similar or close together two points are to each other ]

![image-20240327152146568](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403271521744.png)

[A hypothesis h that is a linear combination of Gaussians centered at four training points, two with positive weights and two with negative weights. If you use ridge regression with a Gaussian kernel, your “linear” regression will look something like this. 很 平滑]

![image-20240327152356629](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403271523703.png)

- Gives very smooth h [In fact, h is infinitely differentiable; it’s $ C^\infty\text{-continuous}$] 
- Behaves somewhat like k-nearest neighbors, but smoother
- Oscillates振荡 less than polynomials (depending on $\sigma$) [poly oscillates crazily]
- k(x,z) interpreted as a similarity measure. Maximum when z = x; goes t o 0 as distance increases. 
- Training points “vote” for value at z, but closer points get weightier vote.

[The “standard” kernel k(x,z) = x · z assigns more weight to training point vectors that point in roughly the same direction as z. By contrast, the Gaussian kernel assigns more weight to training points near z.]



Choose $\sigma$ by validation (trades off bias vs. variance):

larger $\sigma$ -> wider Gaussians & smoother h -> more bias & less variance

The decision boundary (solid black) of a softmargin SVM with a Gaussian kernel. Observe that in this example, it comes reasonably close to the Bayes optimal decision boundary (dashed purple). The dashed black curves are the boundaries of the margin. The small black disks are the support vectors that lie on the margin boundary

![image-20240327155647275](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202403271556424.png)

The polynomial and Gaussian kernels are the two most popular

[There are many other kernels are defined directly as kernel functions without worrying about. 

Not every function can be a kernel function, a function is qualified only if it always generates a positive semidefinite kernel matrix, for every sample.

Not every featurization leads to a kernel function that can be computed faster than $\mathbf{\Theta}(D)$ time, the vast majority cannot]



# LEC17 Neural network, backpropagation

in 1969, Marvin Minsky and Seymour Papert published a book called “Perceptrons”, 下图XOR的01无法用Perceptron划分开, “Frank, you’re telling us this machine is going to be conscious of its own existence but it can’t do XOR?” -> first AI winter, 10 years

![image-20240406143533608](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061435363.png)

But actually adding one new feature can solve this

or we can: (A linear comb of a linear comb is still linear comb) we need to add nonlinearity between linear comb (neuron), nonlinearity could be as simple as clamping the output so it can’t go below zero

logistic function is [0,1], nice for other neurons input; is smooth, has well-defined gradients and Hessians; is a good model for post prob

we can produce XOR, we can simulate logic circuits

![image-20240406151925544](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061519644.png)

Network with 1 Hidden Layer (using logistic function, 2 output; softmax for >2 output)

$V_{i}^{\top}$ is a row in matrix, $V_{i}$ is transpose it to get a column vector 

x is (d+1)*1

![image-20240406165059511](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061650586.png)

write neural network in matrix form

![image-20240406171308704](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061713839.png)

Training -> stochastic or batch gradient descent

loss fn for one pt, cost fn aggregate them; find the weight matrices V and W that minimize J

![image-20240406172401559](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061724622.png)

Usually there are many local minima, cost function is not even close to convex

start by setting all weights to zero and do GD on the weights 

-> there’s no difference between one hidden unit and any other hidden unit. The GD has no way to break this symmetry 

-> we start with random weights

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061811144.png" alt="image-20240406181153002" style="zoom:25%;" />

[random weights shouldn’t be too big, because if a unit’s output gets too close to zero or one, it can get “stuck,” (a modest change in the input values causes barely any change in the output value). Stuck units tend to stay stuck]

![image-20240406173853067](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061738163.png)

stochastic gradient descent (one pt each step)

- Naive gradient computation: O(edges^2^) 
- Backpropagation: O(edges) [dynamic programming, computing solutions of subproblems first]

1. review: chain rule

![image-20240406181517966](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061815069.png)

先计算forward pass, 再计算backward pass

![image-20240406181903450](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061819521.png)

if a unit’s output goes to more than one unit

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061830865.png)

![image-20240406184030761](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061840871.png)

2. The Backpropagation Alg

review:

![image-20240406165059511](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061849715.png)

![image-20240406171308704](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061851286.png)

forward pass:

![image-20240406184644779](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061846846.png) here Vi is only one row of V; similar to Wj；最后一行中间过程类似，省略了

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061842393.png" alt="image-20240406184212337" style="zoom: 40%;" />

backward pass:

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061852478.png)

有红箭头的是vector，其他均为scalar

![image-20240406185612647](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404061856733.png)



# LEC18 NN intuition, vanishing gradient

natural world & engineering world: the airplane we make do not flap their wings; bird do not have jet engine; for engineering, we work with the stuff we already have: we have ..., what can we do with them?

4 way to approximate fn

piecewise constant (integral) / piecewise linear (affine) -> local; Taylor/fourier -> global

![image-20240414160215544](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404141602670.png)

question: can we use neuro net to express these 4 method? (must use same non-linearity, so we can speed with GPU)

[the '1' can be used for shift left and right]

![image-20240415004647204](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404150046313.png)

1. piecewise constant 

   - x -> h: activation fn = sgn(x); h -> z, just sum up, no activation fn; have expression power

   - z -> h can backdrop; h -> x can't backdrop (gradient = 0); only height of impulse can change, location of impulse are fixed
   - maybe initialize with many random fixed location, small width impulses, so can't change location doesn't matter
   - cannot get deep, cannot backprop through one non-linearity 

![image-20240415003834831](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404150038979.png)

2. piecewise linear/affine (most popular)

   - similarly, activation fn = Relu, impressed by diodes 

   - derivative = 0 for input < 0; derivative = 1 for input >= 0, activated; some input are activated while others are not (backprop doesn't change value); dead relu -> all input < 0, frozen

     [recognize SVM, hinge loss, D.B. only depend on some pts, not others]

   ![image-20240415010401704](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404150104866.png)



2种non-linearty

- saturating non-linearty (sigmoid, tanh, logistics)
- non-saturating non-linearty (Relu, keep going up)

***

THE VANISHING GRADIENT PROBLEM for sigmoid

unit output s is close to 0/1 for most training points, $s^{\prime}=s(1-s)\approx0$, GD changes s very slowly. Unit is “stuck.” Slow training.

more layers -> more problematic, can't deep

![image-20240415014006582](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404150140690.png)

using Relu can solve this

[The output of a ReLU can be arbitrarily large, creating the danger that it might overwhelm units downstream. This is called the “exploding gradient problem,” especially for RNN]



Initializing Weights

- all weight < 0 -> Relu dead; all weight > 0 -> no linearty
- weight too small -> breaking symmetry takes a long time; weight too large -> vanishing gradient for sigmoid/exploding gradient for ReLUs

a rule of thumb for initializing weights into both sigmoid and ReLU units

For unit with fan-in (# input edge to a neuro) $\eta$, initialize each incoming edge to random weight with mean 0, std. dev. $\frac1{\sqrt{\eta}}$

-> Xavier/He initialization

Idea: input layer data are standardized, all the value in same hidden layer are also standardized with initial weight 



for DL with many hyperpara -> follow other people for most empirical para, search small number of para yourself



output unit

Regression: linear output, no activation fn; squred-error loss

Classification: to choose from k  2 classes, use softmax (nicely differentiable by computers)

[directly use max{output value} is differentiable, but derivative to non-winner is 0]

![image-20240415022726447](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404150227567.png)

[single training point, choose one label to be 1 and the others to be 0 (one-hot encoding)

But one-hot encoding has disadvantage: each prediction zi can never be exactly 1 or 0. If you choose target labels such as 0.1 or 0.9, a neural network with lots of layers can “interpolate” the labels -> z = y for every training point, the cost function achieves its global minimum]

To fix vanishing gradient problem, use cross-entropy loss fn instead of squared error [prevents the vanishing gradient problem in output units, but it can’t be applied to hidden units]

![image-20240415023521444](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404150235594.png)

Wi: row i of the last weight matrix of network 

![image-20240415025320712](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404150253770.png)

![image-20240415024757080](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404150248229.png)

[Notice that both $\nabla_WL$ and $\nabla_hL$ are linear in the error z-y. So the softmax unit with cross-entropy loss does not get stuck when the softmax derivatives are small. This is related to the fact that the cross-entropy loss goes to infinity as the predicted value zi approaches zero or one. The vanishing gradient of the softmax unit is compensated for by the exploding gradient of the cross-entropy loss]

backpropagation for ReLU hidden units, a k-class softmax output, the cross-entropy loss function, and $\ell_{2}$ regularization

![image-20240415030242681](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404150302839.png)



Two-class classification: softmax k = 2 is sigmoid 

![image-20240415023731177](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404150237239.png)



[neuro biology 略]



# LEC19 faster training, double descent, better generalization, CNN

lr is often one of the most important thing to tune (rather than batch size)

Optimizer

- SGD, require memory one-time proportional to para; analogy: proportional control in control theory, LR circuit for circuit

- SGD with momentum (momentum level) -> can use larger lr, require two-time memory; analogy: PI control, LCR circuit (critical damping)

[life advice: people may think they understand things well (actually do not understand due to limited knowledge), then feel weird, then truely understand again

![image-20240416024343853](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404160243039.png)

]

- Adam: require three-time memory



Fast training

- review

  - Fix vanishing gradient problem
  - SGD (on one pt)

  - Normalize: NN is an example of an optimization algorithm whose cost function tends to have better-conditioned Hessians if the input features are normalized, so it may converge to a local minimum faster

    [the nonlinearity of a sigmoid or ReLU unit falls where the linear combination of values coming in is close to zero. Centering makes it easier for the first layer of hidden units to be in the nonlinear operating region 指分界处0附近]

    ![image-20240417195214503](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404171952653.png)

    ![image-20240417195227838](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404171952895.png)

- new

  - adaptive learning schedule

    [size of training set has weak effect on the best choice of lr. So use a subsample to quickly estimate a good lr, then apply it to all your training points]

  - Different lr for each layer of weights. 

    [Commonly there are large variations in the magnitudes of the gradients in different layers of edges]

  - Emphasizing schemes: repeat the rare examples more often

    [NN learn redundant examples quickly, and rare examples slowly]

    [emphasizing schemes can backfire if you have bad outliers]

    - Stochastic: present examples from rare classes more often, or w/bigger lr
    - Can do the same for high-loss examples. [E.g., perceptron alg. presents only misclassified examples？？？]
    - Batch: cost fn is a weighted average of training pts. [Examples from rare classes be more heavily weighted]

  -  Acceleration schemes: Adam, AdaGrad, AMSGrad, RMSprop, momentum


***

DOUBLE DESCENT

Theory: bias-variance trade-off

Practice: large is better, train until zero loss and go on 

[Hidden layers are wide enough + numerous enough -> network output correct label for every training pt -> global minimum]

![image-20240416143011938](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404161435600.png)

[Double descent has also been observed in decision trees and even in linear regression where we add random features to the training points (thereby adding more weights to the linear regression model)]

[The currently accepted explanation for double descent, per Nakkiran et al., is that “at the interpolation threshold . . . the model is just barely able to fit the training data; forcing it to fit even slightly-noisy or misspecified labels will destroy its global structure, and result in high test error. However for over-parameterized models, there are many interpolating models that fit the training set, and SGD is able to find one that ‘absorbs’ the noise while still performing well on the distribution.”]

[very high level: a tall matrix and a thin matrix have same sigular value]

***

BETTER GENERALIZATION

- review

  - L2 regularization / weight decay: add $\lambda||w||^2$ to cost/lost fn 

    [w includes all the weights in all the weight matrices, rewritten as a vector]

    [we suspect that overly large weights are spurious. With NN, it’s not clear whether penalizing is bad or good. Penalizing the bias terms has the effect of drawing each ReLU or sigmoid closer to the center of its nonlinear operating region]

    ![image-20240416135835414](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404172030708.png)

    Bayes optimal boundary (purple); decision boundary (black)

    ![image-20240417203222031](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404172032152.png)

  - Ensemble of Bagging + random initial weights of each network (slow)

  - Dropout: for each iteration, randomly pick some output (node) of every layer and set to 0

    Dropout emulates an ensemble in one network [force NN to learn on subset of the features]

    ![image-20240417203414700](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404172034780.png)

    [It seems to work well to disable each hidden unit with prob 0.5, and to disable input units with a smaller prob]

    [give some advantages of ensemble, but faster to train. Hinton et al., “Improving neural networks by preventing co-adaptation of feature detectors.”]

***

CNN

![image-20240505175641062](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405051756198.png)

Vision: 200 * 200 image = 40,000 pixels, connect all to 40,000 hidden -> 1.6 billion connections, overparametrized & slow

[double descent is cool, computational power is limited]

Process image: 100 * 100 -> 100 * 100 * 1 -> 50 * 50 * 4 -> ... -> 1 * 1 * 10000 

(come from hand-design filter banks: edge/corner detectors etc)

![image-20240416145625921](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404161456040.png)

ConvNet ideas:

- Local connectivity: A hidden unit connects only to a small patch of units in previous layer / a small area of input image (only in early layers, in later layers we still use fully connected)

- Shared weights: Groups of hidden units share same set of weights, called a mask / filter / kernel

  [No relationship to the kernels of Lecture 16]

  We learn several masks, each mask operates on every patch/small area of image (exploit repeated structure in images, audio)

  Masks * patches = hidden units in first hidden layer

  Convolution: the same linear transformation applied to different patches of the input by shifting the mask around

  [Shared weights is like regularization, fewer weights & It’s unlikely that a weight will become spuriously large if it’s used in many places]

In CNN, hidden units are features learned from multiple patches simultaneously, then applying those features everywhere

Yann LeCun’s LeNet 5 (layer 1-4 are just hardcoded max-functions with no weights and nothing to train)

- layer 1 and 3 are convolutional layers in which groups of units share weights
- Layers 2 and 4 are pooling layers that make the image smaller

![image-20240417210825899](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404172108074.png)

[neural net research was popular in the 60’s, but the 1969 book Perceptrons killed interest in them throughout the 70’s. They came back in the 80’s, but interest was partly killed off a second time in the 00’s by SVM. SVMs work well for a lot of tasks, they’re much faster to train, and they have only one hyperparameter, whereas neural nets take a lot of work to tune. The event that brought attention back to neural nets was the ImageNet Image Classification Challenge in 2012. The winner of that competition was a neural net, and it won by a huge margin, about 10%. It’s called AlexNet, and it’s surprisingly similarly to LeNet 5, in terms of how its layers are structured. However, there are some new innovations that led to their prize-winning performance, in addition to the fact that the training set had 1.4 million images: they used ReLUs, dropout, and GPUs for training.]

![image-20240417211656411](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404172116542.png)



# LEC20 Unsupervised learning, PCA

UNSUPERVISED LEARNING

no labels, discover structure in the data

- Clustering: partition data into groups of similar/nearby points. 
- Dimensionality reduction: data often lies near a low-dimensional subspace/manifold in feature space (data point do not use all dim of feature space; matrices have low-rank approximations), identify a continuous variation from sample point to sample point.
- Density estimation: fit a continuous distribution to discrete data (such as MLE Gaussian)

***

PRINCIPAL COMPONENTS ANALYSIS (PCA) (Karl Pearson, 1901) [Dimensionality reduction]

Goal: Given sample points in R^d^, find k directions that capture most of the variation (reduce from dim d to dim k)

![image-20240426161551100](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404261615265.png)

Reduce mnist from 784D to 2D, you can still seperate 0 form 1 (reduce to 20D can tell 10 digit apart pretty well)

![image-20240426161820493](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404261618550.png)

Why? 

- computation cheaper, e.g., regression. 

- Remove irrelevant dim to reduce overfitting in learning alg (reduce variance). 

  [Like subset selection from LEC13, but the final feature we reduce to aren’t axis-aligned, they’re linear combo of input feature (just can tell pts apart as much as possible, has no special meaning)]

- Find a small basis for representing variations in complex things, e.g., faces, genes



From now on, assume X is centered: $\sum_iX_i=0$ (mean=0)

Let X be n*d design matrix. [No fictitious dim]

1. if one principal direction

let w be a unit vector, The orthogonal projection of point x onto vector w is $\tilde{x}=(x\cdot w)w$  ($x \cdot w$ give scalar of x on w)

If w not unit, $\tilde{x}=\frac{x\cdot w}{\|w\|^2}w$

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404271631546.png" alt="image-20240427163147387" style="zoom:50%;" />

[pick the best direction w, then project all the data down onto w]

2. Given orthonormal (mutually orthogonal, length 1) directions v1,..., vk, $\tilde{x}=\sum_{i=1}^k(x\cdot\nu_i)\nu_i$

For example, k=2, span一个平面, 即把所有点投影到这个平面上

![image-20240427164248480](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404271642616.png)

Often we just

- want the k principal coordinates坐标值: $x · v_i$   (in principal component space)

- don’t want the projected point in R^d^: $(x\cdot\nu_i)\nu_i$

$X^{\top}X$ is square, symmetric, positive semidefinite, d*d matrix

Let $0\leq\lambda_1\leq\lambda_2\leq\ldots\leq\lambda_d$ be its eigenvalues

Let v1, v2,..., vd be corresponding orthogonal unit eigenvectors/principal components

[the most important principal components will be the ones with the greatest eigenvalues, show in three different ways]

1. PCA derivation 1: Fit a Gaussian to data with MLE. Choose k Gaussian axes of greatest variance

we choose the direction with widest Gaussian (then choose secoond widest)

![image-20240428181620498](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404281816638.png)

Recall that MLE estimates a covariance matrix $\hat{\Sigma}=\frac{1}{n}X^{\top}X$ [Presuming X is centered]

PCA alg:

- Center X

- Optional: Normalize X. Units (测量单位) of measurement different? 

  - different: Normalize
  - not different: Usually don’t. [same units of measurement, variance difference is usually meaningful] 

- Compute unit eigenvectors/values of $X^{\top}X$

- Choose k (how many principal coordinates to use) 

  [Optional: based on the eigenvalue sizes]

-  For the best k-dimensional subspace, pick eigenvectors $\nu_{d-k+1},\ldots,\nu_d$ (choose the largest ones) 

- Compute the k principal coordinates x · vi for each training/test point

  [we can project the original, un-centered data OR we can project the centered data (translate test data same as training data, test data should subtract mean of training data)]

17 features, 17 eigenvectors

![image-20240428222355736](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404282223882.png)

Projection of 4D (mutual orthogonal) data to 2D subspace 

(each pt represent a city, each city has 4 features: urbanpop -> population)

![image-20240428223609124](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404282236268.png)

[not normalized -> rare occurrences like murder have little influence on principal directions. Which is better? depend on whether you think low-frequency event like murder and rape should have a larger influence; 

OR choose k & normalize or not:  use validation to decide when regression/classification]

2. PCA derivation 2: Find direction w that maximizes sample variance of projected data

 把点投影到w上后宽度越大越好

![image-20240428232140402](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404282321535.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404282323126.png" alt="image-20240428232345941" style="zoom:40%;" />

see Rayleigh quotient -> smell eigenvectors, If w is an eigenvector vi of $X^{\top}X$, Ray. quo. = $\lambda_i$

maximize -> w=vd achieves maximum variance $\lambda_d/n$  [For proof, look up “Rayleigh quotient” in Wikipedia]

pick k direction -> k largest eigenvalues

3. PCA derivation 3: Find direction w that minimizes mean squared projection distance

Minimizing mean squared projection distance = maximizing variance

![image-20240428234047829](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404282340922.png)

difference between PCA & linear reg: 

- In linear regression, the projection direction is always vertical
- In PCA, the projection direction is orthogonal to the projection hyperplane

![image-20240428234430778](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404282344873.png)



Application: PCA on biology

input bool feature dim = 309,790 projected to largest two principal components

SNP for the genes of different Europeans, the projected genotypes resemble the geography of Europe closely

![image-20240429001046071](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404290010265.png)



Eigenfaces

X contains n images of faces, d pixels each

Face recognition: Given a query face, compare it to all training faces; find nearest neighbor in R^d^

[works best if you have several training photos of each person]

Problem: Each query takes $\Theta(nd)$ time

Solution: Run PCA on faces. Reduce to much smaller dimension d' -> nearest neighbors takes O(nd') time

[We’ll talk about speeding up nearest-neighbor search at LEC25]

[If you have 500 stored faces with 40,000 pixels each, and you reduce them to 40 principal components, then each query face requires you to read 20,000 stored principal coordinates instead of 20 million pixels.]

The “average face” is the mean used to center the data; eigenface 0 is the largest eigenvalue

![image-20240429001741041](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404290017201.png)

K是# principal coordinate

![image-20240429002523114](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404290025282.png)

For best results, equalize the intensity distributions first

![image-20240429002632758](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404290026835.png)

[Eigenfaces encode both face shape and lighting. Ideally, we would factor out lighting and analyze face shape only. Some people say that the first 3 eigenfaces are usually all about lighting, and you sometimes get better by dropping the first 3 eigenfaces.] [Blanz and Vetter use PCA in a more sophisticated way for 3D face modeling. They take 3D scans of people’s faces and find correspondences between peoples’ faces and an idealized model. For instance, they identify the tip of your nose, the corners of your mouth, and other facial features. Instead of feeding an array of pixels into PCA, they feed the 3D locations of various points on your face into PCA.]



# LEC21 SVD, k-means clustering/Lloyd's algo, Hierarchical Clustering

- Singular Value Decomposition (SVD) -> not symmetric and not square

- eigendecomposition -> square, symmetric matrix

  [nonsymmetric matrices don’t eigendecompose nicely, and non-square matrices don’t have eigenvectors at all]

Problems in PCA solved by SVD: 

- Computing $X^{\top}X$ takes $\Theta(nd^2)$ time (d很大的时候可以优化到d^2^以下)
- $X^{\top}X$ poorly conditioned (large eigenvalue >> small eigenvalue) -> algo for eigenvector numerically inaccurate/slower

Every X has a singular value decomposition $X=UDV^\top $

usually n > d

all ui & vi are ortho+normal

![image-20240429145121102](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291451278.png)

Diagonal entries $\delta_1,\ldots,\delta_d$ of D are nonnegative singular values of X; # nonzero singular value is equal to the rank of X

[If the centered sample points span a subspace of dim r, there are r nonzero singular values and rank X = r]

[we can get negative sigular value to positive by flip the sign of ui, so we can always get positive/zero singular value]

![image-20240429151138463](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291511643.png)

[there are 4 versions of SVD, we do not care the singular vector with singular value 0 (carry no info), so we use reduced form]

[if n<d, U & D are n * n, V is n * d]



Fact: vi (the right sigular vectors in $V^\top $ or column of V) is an eigenvector of $X^{\top}X$ w/eigenvalue $\delta_i^2$ (in diagonal of D)

Proof: $X^\top X=VDU^\top UDV^\top=VD^2V^\top $ (eigendecomposition of $X^{\top}X$)

[why SVD is more numerically stable: the ratios between singular values are smaller than the ratios between eigenvalues]



Fact: We can find the k greatest singular values & corresponding vectors in O(ndk) time, faster than computing $X^{\top}X$

[we can save time by computing some of the singular vectors we need, not computing all of them]

[approximate, randomized algorithm that are even faster, producing an approximate SVD in O(nd log k) time, https://code.google.com/archive/p/redsvd/]

Important: Row i of UD gives the principal coordinates of sample point Xi -> $\forall i,\forall j,X_i\cdot\nu_j=\delta_jU_{ij}$

[Proof: $XV=UDV^\top V=UD,\mathrm{so}(XV)_{ij}=(UD)_{ij}$]

***

CLUSTERING

- Discovery: Find songs similar to songs you like; determine market segments 
- Hierarchy: Find good taxonomy of species from genes 
- Quantization: Compress a data set by reducing choices 
- Graph partitioning: Image segmentation; find groups in social networks

![image-20240429160224543](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291602740.png)



k-Means Clustering aka Lloyd’s Algorithm (Stuart Lloyd, 1957)

![image-20240429163942155](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291639320.png)

optimal sol is NP-hard O(nk^n^) (Try every partition)

k-means heuristic:

- (1) yj’s are fixed; update µi’s
- (2) µi’s are fixed; update yj’s

Halt when step (2) changes no assignments

[optimize one variable is fast, optimize them simultaneously is NP-hard]

- Step (1): One can show (calculus) the optimal µi is the mean of the points in cluster i

- Step (2): The optimal y assigns each point Xj to the closest center µi

  [If there’s a tie平局, stay in the previous cluster, so algo can halt]

example: step1: assigns each point Xj to the closest center µi

![image-20240429165542845](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291655988.png)

[Both steps decrease objective fn unless they change nothing. Therefore, the algo never returns to a previous assignment. Hence algo must terminate. Usually very fast in practice. Finds a local minimum, often not global]

![image-20240429170147073](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291701163.png)

Getting started: 

- Forgy method: choose k random sample points to be initial µi’s; go to (2). 
- Random partition: randomly assign each sample point to a cluster; go to (1). 
- k-means++: like Forgy, but biased distribution [Each center is chosen with a preference for points far from previous centers, try to get centers far from each other, work well both in practice and theory]

For best results, run k-means multiple times with random starts

数值是final objective fn值

![image-20240429170952454](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291709671.png)

Why did we choose that objective fn to minimize? Partly because it is equivalent to minimizing the following fn:

![image-20240429171252295](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291712399.png)

[training pt with index j, m both in cluster i; this objective function is equal to twice the previous one]



Normalize the data first? Sometimes yes, sometimes no

[If some features are much larger than others, they will dominate the Euclidean distance. 

different units of measurement -> normalize them

same unit of measurement -> not normalize them]



k-Medoids Clustering

[Generalizes k-means beyond Euclidean distance]

specify dissimilarity fn d(x, y) between points x, y

[Can be arbitrary, ideally satisfies triangle inequality d(x, y) < d(x,z) + d(z, y). Sometimes people use the $\ell_{1}$/$\ell_{\infty}$ norm. Sometimes people specify a matrix of pairwise distances between the input points]

Replace mean with medoid, the sample point that minimizes total distance to other points in same cluster. The medoid of a cluster is always one of the input points

[Suppose you have a database that tells you how many of each product each customer bought. You’d like to cluster together customers who buy similar products for market analysis. But if you cluster customers by Euclidean distance, you’ll get a big cluster of all the customers who have only ever bought one thing. So Euclidean distance is not a good measure of dissimilarity. Instead, it makes more sense to treat each customer as a vector and measure the angle between two customers. If there’s a large angle between customers, they’re dissimilar.]

***

Hierarchical Clustering (a group of algo)

[k-means do not know how to choose number k, hierarchical clustering can solve this]

Creates a tree; every subtree is a cluster. [So some clusters contain smaller clusters]

- Bottom-up/agglomerative clustering: start with each point a cluster; repeatedly fuse融合 pair
- Top-down/divisive clustering: start with all pts in one cluster; repeatedly split it

![image-20240429172951112](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291729261.png)

[input is a point set, agglomerative clustering > divisive clustering; input is a graph, divisive clustering > agglomerative clustering]

We need a distance fn for clusters A, B:

![image-20240429173813244](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291738372.png)

[The first three work for any distance function, the centroid linkage only works for Euclidean distance

there’s a variation of the centroid linkage that uses the medoids instead of the means, medoids are defined for any distance function and are more robust to outliers]



Greedy agglomerative alg: Repeatedly fuse the two clusters that minimize d(A, B). Naively takes O(n^3^) time

[for complete/single linkage exist sophisticated algo called CLINK and SLINK, which run in O(n^2^) time. package is called ELKI]

Dendrogram: Illustration of the cluster hierarchy (tree) in which the vertical axis encodes all the linkage distances

x-axis has no meaning, y-axis is linkage distance

最底下每个leaf node表示一个sample pt

d(A, B)表示fuse A & B后的d值

![image-20240429175522987](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291755627.png)

三张图都是同一张，人为选择虚线的高度，图二的虚线以下分成了两个cluster，图三的虚线以下分成了三个cluster

![image-20240429175148459](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291751597.png)

complete linkage gives the best-balanced dendrogram (most popular)

single linkage gives a very unbalanced dendrogram sensitive to outliers (especially near the top of the dendrogram)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291758192.png" alt="image-20240429175838101" style="zoom:50%;" />

Warning: centroid linkage can cause inversions where a parent cluster is fused at a lower height than its children. 

[So statisticians don’t like it, but centroid linkage is popular in genomics基因组学] 



# LEC22 high-dim space: vectors have same length and orthogonal angle, Random projection, pseudoinverse & SVD

THE GEOMETRY OF HIGH-DIMENSIONAL SPACES

Consider a 1-dim random point $p\sim\mathcal{N}(0,I)\in\mathbb{R}^d$

[it seems that it has length 0 with high red prob, length 1 with lower blue prob, length 2 with low green prob]

![image-20240429182451376](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291824506.png)

But in high-dim, that intuition is completely wrong, vast majority of random pts are at approximately same distance from mean. They lie in a thin shell

$\|p\|^2=p_1^2+p_2^2+\ldots+p_d^2$

Each component pi is sampled independently from a univariate normal distribution with mean zero and variance one (上图)

The square of a component $p_i^2$, as well as $\|p\|^2$, come from a chi-squared卡方 distribution

![image-20240429183640219](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291836358.png)

![image-20240429183803839](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404291838936.png)

For large d, $\left\|p\right\|$ is concentrated in a thin shell around radius $\sqrt{d}$  with a thickness proportional to $\sqrt[4]{2d}$

[The mean value of $\left\|p\right\|$ isn’t exactly $\sqrt{d}$, but it is close, because the mean of $\left\|p\right\|^2$ is d and the standard deviation is much smaller. Likewise, the standard deviation of $\left\|p\right\|$ isn’t exactly $\sqrt[4]{2d}$, but it’s close]

[if d is about a million, imagine a million-dimensional egg whose radius is about 1000, and the thickness of the shell is about 67 (10 times the standard deviation 6.7, enough to catch all pts). The vast majority of random pts are in eggshell, not inside the egg]

[a hiding statistical principle: Suppose you want to estimate the mean of a distribution—in this case, the distribution $\chi^2(1)$. The standard way to do that is to sample many numbers from the distribution and take their mean. The more numbers you sample, the smaller the standard deviation, the more accurate your estimate is. When we sample a vector from a milliondimensional normal distribution and compute its length, that’s exactly what we’re doing. large number law hide in high-dem data]



What about a uniform distri instead of normal distri? Consider concentric spheres同心球 of radii r & $r-\epsilon $

![image-20240429232840019](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404292328360.png)

[In high dimensions, almost every point chosen uniformly at random in the outer ball lies outside the inner ball]

![image-20240429233817275](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404292338475.png)

Conclusion:

- Random points from uniform distribution in ball: nearly all are in thin outer shell

-  Random points from Gaussian distribution in ball: nearly all are in some thin shell

Consequences:

- In high dim, sometimes the nearest neighbor and 1000th-nearest neighbor don’t differ much
- k-means clustering and nearest neighbor classifiers are less effective for large d

[distances are less differentiated from each other in high-dim space. algo depend highly on distances don't work well in high-dim]

***

Angles between Random Vectors

What is the angle $\theta$ between a random $p\sim\mathcal{N}(0,I)\in\mathbb{R}^d$ and an arbitrary $q\in\mathbb{R}^d$

[every $p_i\sim\mathcal{N}(0,I)$]

Without loss of generality, set $q=\begin{bmatrix}1&0&0\ldots0\end{bmatrix}^\top $

![image-20240430000614150](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404300006335.png)

[If d is large, $\text{COS }\theta $ is very close to zero; $\theta$ is very close to 90]

[In high-dim spaces, two random vectors are almost always very close to orthogonal. an arbitrary vector is almost orthogonal to the vast majority of all the other vectors]

[A former CS 189 head TA, Marc Khoury, has a nice short essay entitled “Counterintuitive Properties of High Dimensional Space”, which you can read at https://marckhoury.github.io/blog/counterintuitive-properties-of-high-dimensional-space]

***

RANDOM PROJECTION

An alternative to PCA as dim reducing preprocess for clustering, classification, regression

Approximately preserves distances between points!

[project onto a random chosen subspace, sometimes it preserves distances better than PCA (PCA preserve variance). algo like k-means clustering and nearest neighbor classifiers will give similar results to what they would give in high-dim, with much faster running time. We often project a high-dimensional space to a medium-dimensional space]

[PCA常常出现两个不同的点project到同一个点的情况，但是random就几乎不会出现 -> preserve distance]

[ ] means ceiling (in theory), smaller k works better in practice

![image-20240430003929972](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404300039120.png)

[The multiplication by $\sqrt{d/k}$ helps preserve the distance, kind of make up for the coordinates we throw away]

![image-20240430004533168](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404300045235.png)

after q,w project to $\hat{q}$,$\hat{w}$, distance的变化within lower&upper bound的概率大于$1-2\delta $

[In practice, experiment with k to find the best speed-accuracy trade-off. 

If you want all inter-sample-point distances to be accurate, you should set $\delta$ smaller than 1/n^2^, so you need a subspace of dimension $\Theta(\log n)$

Reducing $\delta$ doesn’t cost much (because of the logarithm), but reducing $\epsilon$ costs more. You can bring 1,000,000 sample points down to a 10,000-dimensional space with at most a 6% error in the distances

What is remarkable about this result is that the dimension d of the input points doesn’t matter]

project pts in 100 000-dim space down to 1000-dim

![image-20240430005938402](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404300059488.png)

[Why does this work? A random projection of q-w is like taking a random vector and selecting k components. The mean of the squares-of-thos-k-components approximates the mean for the whole population.]

[How do you get one uniformly distributed random projection direction? choose each component of v from a univariate Gaussian distribution $\mathcal{N}(0,I)$, then normalize the vector to unit length $v/||v||$

How do you get a random subspace? choose k random directions v1,v2,...,vk, then use Gram–Schmidt orthogonalization to make them mutually orthonormal

Interestingly, Indyk and Motwani show that if you skip the expensive normalization and Gram–Schmidt steps, random projection still works as well, because random vectors in a high-dim space are nearly equal in length and nearly orthogonal to each other]

***

[unsupervised learning is done, go to supervised till end of semester]

THE PSEUDOINVERSE AND THE SVD

1. the psuedoinverse of diagonal matrix

Let D be a diagonal n*d matrix

its pseudoinverse D^+^: transpose D and replace every nonzero entry with its reciprocal

![image-20240430173348357](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301733522.png)

$DD^+D=D$ and $D^+DD^+=D^+$

If D is square, $D^2D^+=D$

If D is square & no zeros on the diagonal, D+ is the inverse of D, $DD^+ = D^+D = I$

2.  the pseudoinverse of a matrix in general

Let X be any n*d matrix. Let $X=UDV^{\top}$ be its SVD, rank D = rank X

The Moore–Penrose pseudoinverse of X is $X^+=VD^+U^\top $

Observe:

![image-20240430181233449](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301812627.png)

Rank = dim row space = dim col space

think of X as a linear function that maps row X to col X, ignore the null spaces of X and X^T^, then that linear function is a bijection (one to one) whose inverse is the pseudoinverse X^+^

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301821297.png" alt="image-20240430182147109" style="zoom:50%;" />

Linear function $f$ : $\operatorname{row}X\to\operatorname{col}X$, $p\mapsto Xp$ is a bijection

Its inverse is $f^{-1}$ : $q\mapsto X^+q$

Let r = rank X

- The r right singular vectors vi with $\delta_i\neq0$ are an orthonormal basis for row X (图中v1, v2)
- The r left singular vectors ui with $\delta_i\neq0$ are an orthonormal basis for col X (图中u1, u2)

$X\nu_i=\delta_iu_i\quad X^+u_i=\frac1{\delta_i}\nu_i$



assume X is a 4*3 matrix, dim row = dim col = rank = 2 as the figure above

X first project pt in R^3^ to the R^2^ row X, then map pt in row X to pt in col X

simliar for X^+^, first project pt in R^4^ to the R^2^ col X, then map pt in col X to pt in row X

[map a 3-dim space down to 2-dim, it can’t be a bijection, so X doesn’t have an inverse, just a pseudoinverse]

![image-20240430195323000](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301953282.png)



Now we can show pseudoinverse always gives a good solution in least-squares linear regression, even when $X^TX$ is singular

[form DIS6]

![image-20240430195632272](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404301956394.png)

If normal equation have multiple sol, w = X^+^y is the least-norm sol (the best sol), it minimizes $\left|\left|w\right|\right|$ among all sol  [proof at DIS6]

[This way of solving the normal equations is very helpful when $X^TX$ is singular because n < d or the sample points lie on a subspace of the feature space]

if X has a 0.0001 singular value, the reciprocal of that singular value will be very large and have a very large effect on w. We should treat it as 0 sometimes. Ridge regression implements this policy to some degree, see DIS12]

![image-20240430201010522](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404302010780.png)



# LEC23 Learning theory, dichotomie, shatter function, VC dim

LEARNING THEORY: WHAT IS GENERALIZATION?

human generalize far better than machine, human only need to be trained on few data

- A range space/set system is a pair (P, H)

- P is set of all possible test/training points (can be infinite)

- H is hypothesis class, a set of hypotheses/ranges/classifiers) [eg. the set of all linear classifier]

  h is one hypothesis, a subset h$\subseteq $P that specifies which points h predicts are in class C

  [each h is a 2-class classifier & a set of pt, and H is a set of sets of pt]

1. power set, k number, 每个number有在/不在class中两种情况，一共2^k^ (can learn all possible h)

![image-20240504154036491](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405041540713.png)

![image-20240504154738683](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405041547861.png)

[The power set classifier sounds very powerful, but it can’t generalize at all. It can classify test points any way you like, that means it has learned nothing about the test points from the training points

the linear classifier can learn only two hypotheses that fit this training data (划分线在中间问号的左侧或右侧),  it can generalize]

Suppose all training pts & test pts (pt, label) are drawn independently from same prob distri D defined on domain P

Let h$\in$H be a hypothesis, h predicts a pt x is in class C if x$\in$h

The **true risk/generalization error R(h)** of h is the prob that h misclassifies a random pt x from D (x$\in$C but x$\notin$h or vice versa)

[the risk is the mean test error for test points drawn randomly from D. For a particular test set, sometimes the test error is higher, sometimes lower, but on average it is R(h). If you had an infinite amount of test data, risk = test error]

Let X$\subseteq $P be a set of n train pts drawn from D, The **empirical risk/training error $\hat{R}(h)$** is % of X misclassified by h

h misclassifies each train pt w/prob R(h), so total misclassified has a binomial distri. As $n\to\infty $,  $\hat{R}(h)$ better approximates R(h)

risk of misclassification is R(h)=25% for 20 pt and 500 pt (binomial distri), 横轴是# misclassified train pt; more pt -> narrow peak

[infinite train data, this distri would become infinitely narrow and the training error would always be equal to the risk]

![image-20240504161446611](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405041614805.png)

Hoeffding’s inequality tells us prob of bad estimate:

![image-20240504162003671](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405041620843.png)

[prob of a number drawn from a binomial distribution will be far from its mean. If n is big enough, then it’s very unlikely]

$\epsilon$ = 0.1, 横轴是training pts

![image-20240504162145807](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405041621930.png)

Idea for learning alg: go over all h in H, **choose $\hat{h}\in H$ that minimizes $\hat{R}(h)$! Empirical risk minimization**

[In practice is often NP-hard, in theory, we pretend that we have the computational power to try every hypothesis]

Problem: if too many hypotheses, some h with high R(h) will get lucky and have very low $\hat{R}(h)$ (high test err, low train err)

[central idea of learning theory: we don't want learning algorithm to have the largest class of hypotheses, because some of them will get lucky and have high test err, low train err. That’s another way to understand what “overfitting” is. 

More precisely, the problem isn’t too many hypotheses. Usually we have infinitely many hypotheses, and that’s okay. The problem is too many dichotomies]

***

Dichotomies

A dichotomy of X is X$\cap $h (those train pts that is h predict class C)

[Think of each dichotomy as a function assigning each train pt to class C or class Not-C]

green line is hypothesis, those are 3 dichotomy

![image-20240504190414536](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405041904727.png)

This is not a dichotomy for linear classifier because no linear classifier can do this classification

![image-20240504190555158](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405041905421.png)

There are infinite # hypothesis, but finite # dichotomy (2 hypothesis may have the same dichotomy)

For n train pts, there could be up to 2^n^ dichotomies. The more dichotomies there are, the more likely it is that one of them will get lucky and have misleadingly low empirical risk]

Extreme case: if H allows all 2^n^ possible dichotomies, must have $\hat{R}(\hat{h})$=0, overfit

If **H induces $\Pi$ dichotomies**, $\Pr(\text{at least one dichotomy has }|\hat{R}-R|>\epsilon)\leq\delta\text{, where }\delta=2\Pi e^{-2\epsilon^2n}$

[prob of at least one dichotomy is a bad estimate (get lucky, $\hat{R}$ < R)]

firstly, fix $\delta $, with prob $\geq1-\delta $, for every h $\in$ H: $|\hat{R}(h)-R(h)|\leq\epsilon=\sqrt{\frac1{2n}\ln\frac{2\Pi}\delta}$

smaller $\Pi$ (# possible dichotomies), larger n (# train pts), train error closer to true risk & test error (这是两个概念)

[Smaller $\Pi$ means less likely to overfit, less vriance but more bias. This doesn’t necessarily mean the risk will be small. If our hypothesis class H doesn’t fit the data well, both the training error and the test error will be large]

Let **$h^*\in H$ minimize true risk $R(h^*)$**, best classifier

$|\hat{R}(h)-R(h)|\leq\epsilon=\sqrt{\frac1{2n}\ln\frac{2\Pi}\delta}$

-> $R(\hat{h})\leq\hat{R}(\hat{h})+\epsilon $ ; $\hat{R}(h^*)\leq R(h^*)+\epsilon $ ; [$\hat{R}(\hat{h})\leq\hat{R}(h^*)$]

-> $R(\hat{h})\leq\hat{R}(\hat{h})+\epsilon\leq\hat{R}(h^*)+\epsilon\leq R(h^*)+2\epsilon,\quad\epsilon=\sqrt{\frac1{2n}\ln\frac{2\Pi}\delta}$

[with enough training data and a limit on the number of dichotomies, empirical risk minimization usually chooses a classifier close to the best one in the hypothesis class]

**Sample Complexity**

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405041939111.png" alt="image-20240504193922821" style="zoom:35%;" />

[If $\Pi$ is small, we won’t need too many train pts to choose a good classifier. 

if $\Pi$ = 2^n^ this inequality says n has to be bigger than n. So the power set classifier can’t learn much or generalize at all

We need to severely reduce $\Pi$, the number of possible dichotomies. One way to do that is to use a linear classifier]

***

The Shatter Function & Linear Classifiers

[How many ways can you divide n points into two classes with a hyperplane?]

H (eg. all linear classifier), X is a particular set of train pts

\# of dichotomies: $\begin{array}{rcl}\prod_H(X)&=&|\{X\cap h:h\in H\}|&&\in[1,2^n]\text{ where }n=|X|\end{array}$ (# train pt)

for all possible train sets of size n

shatter function: $\begin{array}{rcl}\prod_H(n)&=&\max_{|X|=n,X\subseteq P}\prod_H(X)\end{array}$

Example: Linear classifiers in plane. H = set of all halfplanes. $\Pi_H(3)=8$

[flip C/N to get another 4]

![image-20240504200534020](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405042005334.png)

$\prod_H(4)=14$ (at most 14, not 16, Radon’s Theorem)

dichotomies that halfplanes cannot learn (no linear classifier can handle):

![image-20240504200724371](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405042007622.png)

Surprising fact: for all range spaces, either $\Pi_H(n)$ is polynomial in n, or $\Pi_H(n)=2^n\quad\forall n\geq0$

[Imagine that you have n points (train + test). 

Either a range space permit every possible dichotomy and the train points don’t help you classify the test points at all; 

or the range space permits only a polynomial subset of the 2^n^ possible dichotomies

poly & 2^n^ 的巨大鸿沟是一片无人区,  No shatter function ever occupies the no-man’s-land between polynomial and 2^n^

so once you have labeled the train pts, you have usually cut down the # ways you can classify the test points dramatically]

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405042019512.png" alt="image-20240504201959381" style="zoom:40%;" />

Corollary: linear classifiers need only $n\in\Theta(d)$ training pts for training error to accurately predict risk or test error

[The constant hidden in $\Theta $ can be quite large. For example, if you choose $\epsilon$ = 0.1 and $\delta $ = 0.1, then setting n = 550d will always suffice. (For very large d, n = 342 d will do)]

[This sample complexity applies even if you add polynomial features or other features, but you have to count the extra features in d. So the number of training points you need increases with the number of polynomial terms]

***

VC Dimension

The Vapnik–Chervonenkis dimension of (P, H)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405042032653.png" alt="image-20240504203225232" style="zoom:50%;" />

Say that H shatters a set X of n pts if $\prod_H(X)=2^n$

![image-20240505200551416](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405052005585.png)

[for small train set, often can shatter, if your H space has a good limit, then you'll reach a threshold where you cannot shatter]

VC(H) is size of largest X that H can shatter [X is a point set for which all 2^n^ dichotomies are possible]

en指e*n

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405042040835.png" alt="image-20240504204035611" style="zoom:50%;" />

[VC dimension is an upper bound on the exponent of polynomial. This theorem is useful because often we can find an easy upper bound on VC dimension. You just need to show that for some number n, no set of n points can have all 2^n^ dichotomies.]

Corollary: O(VC(H)) training pts suce for accuracy. [Again, the hidden constant is big]

eg. linear classifier in R^2^

![image-20240504205719974](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405042057158.png)

[The VC dimension doesn’t always give us the tightest bound. In this example, the VC dimension promises $\prod_H(n)$ is at worst cubic in n; but Cover’s Theorem says it’s quadratic in n. In general, linear classifiers in d dimensions have VC dimension d + 1, which is one dimension looser than the exponent Thomas Cover proved

That’s not a big deal, as the sample complexity and the accuracy bound are both based on the logarithm of the shatter function. So if we get the exponent wrong, it only changes a constant in the sample complexity

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405042106714.png" alt="image-20240504210655544" style="zoom:25%;" />

In practice, VC dim is easier to get than complicated exact Cover Theorem]

[takeawway: If you limit the hypothesis class, your artificial child will only need to look at O(d) cows to learn the concept of cows. If you don’t, your artificial child will need to look at every cow in the world, and every non-cow too]

![image-20240505201904674](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405052019850.png)



# LEC24 AdaBoost, NN(Nearest neighbor)

ADABOOST (Yoav Freund and Robert Schapire, 1997)

AdaBoost (“adaptive boosting”) is an ensemble method for classification (or regression)

- reduces bias [ensemble/random forests reduce variance]
- trains learners on weighted sample points [like bagging]
- uses different weights for each learner
- increases weights of misclassified training points
- gives bigger votes to more accurate learners.

Input: n*d design matrix X, vector of labels $y\in\mathbb{R}^n$ with yi = ±1

Ideas: 

- Train T classifiers G1,...,GT

- Weight for training point Xi in Gt grows according to how many of G1,...,Gt1 misclassified it

  [Moreover, if Xi is misclassified by very accurate learners, its weight grows even more] 

  [And, the weight shrinks every time Xi is correctly classified] 

- Train Gt to try harder to correctly classify training pts with larger weights

- Metalearner is a linear combination of learners. For test point z, $M(z)=\sum_{t=1}^T\beta_tG_t(z)$

  Each Gt is ±1, but M is continuous. Return sign of M(z).

[G tend to be any unlinear classifirer, decision trees work very well, linear classifier usually not work well]

[how to assign different weight to training pts. For example, 

- in regression we usually modify the risk function by multiplying each point’s loss function by its weight

- In a soft-margin SVM, we modify objective fn by multiplying each point’s slack by its weight

- To weight points in decision trees, we use a weighted entropy where instead of computing the proportion of points in each class, we compute the proportion of weight in each class

  In practice, our individual learners are often classification algorithms like decision trees that don’t explictly try to minimize any loss function at all]

In iteration T, what classifier GT and coeffient $\boldsymbol{\beta}_{T}$ should we choose? Pick a loss fn L(prediction, label)

1. GT

[just the metalearner loss fn, the loss of classifier G not specified]

![image-20240430233521209](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404302335525.png)

AdaBoost metalearner uses exponential loss function

![image-20240430233601643](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404302336740.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404302343374.png" alt="image-20240430234337181" style="zoom:30%;" />

[The exponential loss function has the advantage that it pushes hard against badly misclassified points]

第二行 -> 第三行，第1～T-1个learner都训练好了，看作一个constant $w_{i}^{(T)}$，我们正在训练第T个learner

第五行, (1) all pts + (2) misclassified pts; ignore (1), optimize GT with (2) (also can ignore coeff $(e^{\beta_T}-e^{-\beta_T})$ which is positive)

![image-20240430234547197](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404302345363.png)

What GT minimizes the risk? The learner that minimizes the sum of $w_i^{(T)}$ over all misclassified pts Xi

$w_i^{(T)}$ is weight, one for each training pt (i:1～n)

observation: each learner’s weights are related to the previous learner’s weights (we can use this to speed up)

[this is also a special benefit for exponential loss fn]

![image-20240501002400855](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405010024039.png)

[in practice, boosting is usually used with short, imperfect decision trees instead of tall, pure decision trees, because adaboost reduce bias, we should ensure it not overfit]

2. $\boldsymbol{\beta}_{T}$

第二行 -> 第三行：同除$e^{-\beta_{T}}\sum_{i=1}^{n}w_{i}^{(T)}$

![image-20240501005515615](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405010055930.png)

- If errT = 0, T = ∞ [So a perfect learner gets an infinite vote] 
- If errT = 1/2, T = 0 [So a learner with 50% (same as random) weighted training error gets no vote at all]

[A learner with 60% errT (T < 0) is just as useful as a learner with 40% errT; the metalearner just reverses the signs of its votes]

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405010103490.png" alt="image-20240501010328358" style="zoom:40%;" />

example of a linear classifier

![image-20240501011957593](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405010119729.png)

Why boost decision trees? [As opposed to other learning algorithms?] Why short trees? 

- Boosting reduces bias reliably, but not always variance. AdaBoost trees are impure/short to reduce overfitting

  [The AdaBoost variance is more complicated: it often decreases at first, because successive trees focus on different features, but often it later increases. Sometimes boosting overfits after many iterations, and sometimes it doesn’t] 

- Fast. [Short trees]

-  No hyperparameter search needed

  [UC Berkeley’s Leo Breiman called AdaBoost with decision trees “the best off-the-shelf classifier in the world.”] – Easy to make a tree beat 45% training error [or some other threshold] consistently

- Easy to get a threshold like <45% in training error, and stop, for all learners [or some other threshold] 

- AdaBoost + short trees is a form of subset selection

  [Features that don’t improve the metalearner’s predictive power enough aren’t used at all]

- Linear decision boundaries don’t boost well

More about AdaBoost:

- Posterior prob can be approximated: $P(Y=1|x)\approx1/(1+e^{-2M(x)})$

-  Exponential loss is vulnerable to outliers; for corrupted data, use other loss

  [Loss functions have been derived for dealing with outliers, they have more complicated weight computations.] 

- If every learner beats error µ for µ < 50%, metalearner training error will eventually be zero. [proof in hw7]

[AdaBoost paper and its authors, Freund and Schapire, won the 2003 Godel Prize, a prize for outstanding papers in TCS]

the picture, AdaBoost with stumps (depth-one decision trees). 

- At left, training error eventually drops to zero, average loss (which is continuous, not binary) continues to decay 

- At right, test error drops to 5.8% after 400 iterations, even though each learner has an error rate of about 46%. AdaBoost with more than 25 stumps outperforms a single 244-node decision tree

  [In this example no overfitting is observed]

![image-20240501015356892](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405010153044.png)

![image-20240506131810034](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405061318163.png)

***

NEAREST NEIGHBOR CLASSIFICATION

Idea: Given query point q, find the k training pts nearest q. Distance metric of your choice

- Regression: Return average label of the k pts
- Classification: Return class with the most votes from the k pts OR return histogram of class prob

large k -> underfit; small k -> overfit

the ideal k depends on how dense your data is. As your data gets denser, the best k increases

![image-20240501140210476](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011402774.png)

Bayes risk is the best (min) you can achieve 

![image-20240501140920518](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011409693.png)

[requirements of this theorem: training pts and the test pts have to be drawn independently from the same prob distri (IID); The theorem applies to any separable metric space, so it’s not just for the Euclidean metric.]

![image-20240501141649999](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011416246.png)



# LEC25 kNN(k-nearest neighbor), speed up: Voronoi Diagram, k-d tree

Exhaustive k-NN Alg

query pt -> test pt

Given query point q:

- Scan through all n training pts, computing (squared) distances to q

- Maintain a max-heap with the k shortest distances seen so far

  [Whenever you encounter a training point closer to q than the pt at the top of the heap, you remove the heap-top point and insert the better point. a heap will speed up keeping track of the kth-shortest distance]

Time to train classifier: 0

Query time: O(nd + n log k) [expected O(nd + k log n log k) if random pt order]

[It’s a cute theoretical observation that you can slightly improve the expected run time by randomizing the point order so that only expected O(k log n) heap operations occur. But in practice you’ll probably lose more from cache misses than you’ll gain from fewer heap operations]

Can we preprocess training pts to obtain sublinear query time?

- 2–5 dimensions: Voronoi diagrams (BSP trees)

- Medium dim (up to ~30): k-d trees (easier to implement than Voronoi diagrams)
- Large dim: exhaustive k-NN, but can use PCA or random projection locality sensitive hashing [still research]

***

Voronoi Diagrams (support 1-NN only)

[in practice only use for 2-dim]

Let X be a point set. The Voronoi cell of $w\in X$ is: $\mathrm{Vor~}w=\{p\in\mathbb{R}^d:\|p-w\|\leq\|p-\nu\|\quad\forall\nu\in X\}$

[A Voronoi cell is always a convex polyhedron or polytope] 

![image-20240501145635879](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011456040.png)

Size (# of vertices of the polyhedrons) $\in O(n^{\lceil d/2\rceil})$ (Ceiling)

[This upper bound is tight when d is a small constant. As d grows, the tightest asymptotic upper bound is somewhat smaller than this, but the complexity still grows exponentially with d]

but often in practice it is O(n)

[we leave out a constant that may grow exponentially with d. so if d =1000, actually is O(n) = O(d^1000^ n)]

Point location: Given query point $q\in\mathbb{R}^d$, find the point $w\in X$ for which $q\in\text{Vor }w$

- 2D: **trapezoidal map** can perform search on a Voronoi diagram effiently

  O(n log n) time to compute V.d. and a trapezoidal map for pt location; O(log n) query time 

- dD (d>3): Use **binary space partition tree (BSP tree)** for pt location. 

  [it’s difficult to characterize the running time of this strategy, although it is often logarithmic in 3–5 dim]

Voronoi Diagrams support 1-NN only

[for k-NN, there is something called an order-k Voronoi diagram that has a cell for each possible k nearest neighbors. But nobody use for two reasons

- the size of an order-k Voronoi diagram is $\Theta(k^2n)$ in 2D, and worse in higher dimensions
- there’s no software available to compute one]

[There are also Voronoi diagrams for other distance metrics, like $\ell_{1}$ and $\ell_{\infty}$ norms]

***

k-d Trees

“Decision trees” for NN search

same: each treenode in a k-d tree represents a rectangular box in feature space, we split a box by choosing a splitting feature & value belonging to a training point in the box

Differences: [the way to choose spliting feature&value different, not entropy any more]

- Choose splitting feature

  - Choose splitting feature w/greatest width: feature i in $\max_{i,j,k}(X_{ji}-X_{ki})$ [slow]

    [if we draw a sphere around the query point, we want it not intersect very many boxes of the decision tree. So it helps if the boxes are nearly cubical, rather than long and thin]

    split along the longest axis -> i-axis 

    ![image-20240501162516232](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011625403.png)

  - another cheap alternative: rotate through the features

    [assume 4 feature: split on first feature at depth 1, second feature at depth 2, third feature at depth 3, fourth feature at depth 4, first feature at depth 5,...

    This builds the tree faster by a factor of O(d)] 

- choose splitting value: 

  - median point for feature i

    [guarantees short tree: $\lfloor\log_2n\rfloor $ tree depth; O(nd log n) tree-building time for gratest width, or just O(n log n) time if rotate through features]

    ![image-20240501163919716](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011639884.png)

  - midpoint $\frac{X_{ji}+X_{ki}}2$

    [guarantee nicely shaped boxes, but it could unbalance your tree]

    ![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011640731.png)

  - alternate between medians at odd depths and centers at even depths

    [also guarantees an O(log n) depth]

- Each internal node stores a training pt, Usually the splitting point

  [decision tree store every training pt in leaf node; k-d tree have pts in internal nodes, so when searching the tree, we often stop searching earlier -> faster] 

root node represent the entire plane/feature space

![image-20240501164937866](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011649984.png)

![image-20240501165217553](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011652759.png)

[after the tree is built, the classification algorithm is also different from decision tree. you usually have to visit multiple leaves of the tree to find the nearest neighbor]

We sometimes use an approximate nearest neighbor algorithm to save time, instead of demanding exact nearest neighbor:

![image-20240501165907790](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011659005.png)

Query alg. maintains: 

- one-nearest neighbor found so far (or k-nearest): start at ∞ and **goes down**

- to maintain unexplored subtrees, use Binary min-heap, keyed by **distance from q**: start from closest box and **goes up**

  [distance is not q to a traininig pt in B, but q to vertice of box]

![image-20240501171230764](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011712924.png)

when the "goes down" meet the "goes up", we can stop, because no more box store closer pts than we've already found

notice q & nearest pt so far, there are still pts inside circle whose box haven't be explored; 右下角蓝色box will never be explored

![image-20240501173545983](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011735152.png)

min key in priority queue = the distance to the closest box we have not explored yet

![image-20240501174440594](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011744787.png)

For k-NN, replace “r” with a max-heap holding the k nearest neighbors

[k-d trees are not limited to the Euclidean $\begin{pmatrix}\ell_2\end{pmatrix}$ norm, works with any $\ell_{p}$ norm for $p\in[1,\infty]$]

Why $\epsilon$-approximate NN? In figure is the worst case, have to visit every node in the k-d tree to find exact nearest neighbor, slow

![image-20240501175015524](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011750648.png)

[In practice, approximate nearest neighbor sometimes improves the speed by a factor of 10 or even 100; This is especially true in high dim—remember that in high-dim space, the nearest point often isn’t much closer than a lot of other points.]

Software: ANN (U. Maryland), FLANN (U. British Columbia), GeRaF (U. Athens)

***

Example: im2gps, a modern research uses 1-NN and 120-NN search to solve problem by James Hays and Prof. Alexei Efros (UCB)

Goal: given a query photograph, determine where on the planet the photo was taken -> geolocalization

They evaluated both 1-NN and 120-NN. treat each photograph as one long vector is too expensive, they reduced each photo to a small descriptor made up of a variety of features that extract the essence of each photo

With 120-NN, their most sophisticated implementation came within 64 km of the correct location about 50% of the time

***

![image-20240501180139180](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202405011801446.png)











 







# HW1

![image-20240125153658806](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401260736928.png)

![image-20240215031012589](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150310642.png)

![image-20240215031024563](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402150310604.png)

# HW2

- [Linear algebra quick reference](https://laurentlessard.com/teaching/ece532/cheat_sheet.pdf)
- [Math 54 "cheat sheet"](https://math.berkeley.edu/~peyam/Math54Fa11/Cheat Sheets/Cheat Sheet (regular font).pdf)
- [Matrix cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
- [Matrix differentiation](https://atmos.washington.edu/~dennis/MatrixCalculus.pdf)
- [Probability for Data Science (chapters 8, 13, 17, and 20)](http://prob140.org/textbook/content/README.html) 
- [General proof techniques and proof techniques for linear algebra & probability](http://snap.stanford.edu/class/cs224w-2016/recitation/proof_techniques.pdf)
- [Math for Machine Learning (linear algebra, calculus, and probability)](http://gwthomas.github.io/docs/math4ml.pdf)
- [EECS 127 Course Reader](https://eecs127.github.io/assets/notes/eecs127_reader.pdf)

Question 1

P(A) = E[1{A}], 1{·} is the indicator function

![image-20240207020737888](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402070207978.png)

通过样本均值sample mean估计真实均值true mean

Jensen's inequality: 假设f是一个凸函数，X是一个随机变量，对于任意的随机变量X，有![image-20240207023817485](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402070238560.png)

reverse Jensen's inequality: if *f* is a concave function, then  ![image-20240207030550251](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402070305333.png)![image-20240207030604452](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402070306496.png)

组合 ![image-20240207035257777](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402070352889.png)

排列 ![image-20240207035314167](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402070353191.png)

Question 3

elementary matrix operations do not change a matrix’s rank.

![image-20240206023107080](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402060231203.png)

![image-20240206171125300](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402061711359.png)

Question 4

ℓp-norm（Lp范数）：![image-20240206012512317](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402060125425.png)

Frobenius范数（Frobenius norm）是矩阵的一种范数：所有元素的平方和的平方根 ![image-20240206012422420](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402060124501.png)

Question 5

moment-generating function 唯一确定随机变量的分布



![image-20240221175503080](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211755185.png)



![image-20240221175551262](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211755310.png)

![image-20240221175604787](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402211756837.png)



# HW3

![image-20240223025934769](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402230259901.png)

如果积分上限是∞，下限是x，结果是 -f(x)

![image-20240223212050797](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202402232120832.png)



# HW6

When using BatchNorm, you should make sure to freeze the normalization layer (i.e. normalize with estimated population statistics instead of batch statistics) by calling model.eval() before evaluation. When normalizing with fixed statistics the input batch size doesn't make a difference anymore. Remember to un-freeze with model.train() at the beginning of each epoch.



![image-20240419002217899](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202404190022185.png)
