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

![   ](https://cdn.jsisdelivr.net/gh/yuhengtu/typora_images@master/img/202311261522342.png)

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

gradient descent gets closer and closer forever, but never exactly reaches the true minimum. We call this behavior “convergence.” The last question of HW2 will give you some understanding of why convergence happens under the right conditions.？？？？？？/

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

V is orthonormal matrix: used as linear trans to some space, acts like [rotation when det(V)=1] or [reflection(rotate + mirror image) when det(V)=-1]？？？？？？？

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




