## 1 large margin classification
### 1.1 optimization objective
+ Alternative view of logistic regression

我们对逻辑回归模型进行某些修改，来得到SVM。**First Step**，我们先回忆一下逻辑回归模型，如下图：

![graph_of_LRModel](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/graph_of_LRModel.png)

下图显示了：给定一个样本实例(x,y),代价函数与z值的分析图：

![LRModel_with_one_single_sample](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/LRModel_with_one_single_sample.png)

上图中的紫色的折线代表新的代价函数。**在LR中，当观察到一个正样本y=1时，试图给$\mathbf{\theta }^\mathrm{T}x$设置很大的值，这意味着代价函数
$-log\frac{1}{1+e^{-(\mathbf{\theta }^\mathrm{T}x)}}$将变得很小。同理，当观察到一个负样本y=0时，试图给$\mathbf{\theta }^\mathrm{T}x$设置很小的值，这意味着代价函数
$-log(1-\frac{1}{1+e^{-(\mathbf{\theta }^\mathrm{T}x)}})$将变得很小.**
+ support vector machine

通过对LR的代价函数进行标记上的替换，我们得到了支持向量机的代价函数。如下图：

![cost_function_for_SVM](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/cost_function_for_SVM.png)

在LR中，代价函数可以写为`A + λB`的形式，给与λ更大的值意味着给与B更大的权重；相对应的，在SVM中，代价函数的形式变为`cA + B`,给定c非常小的值意味着给与B更大的权重。我们也可以认为$c = \frac{1}{\lambda }$,但是，需要注意：这两个代价函数并不相等。<br>
**SVM使用数据直接学习决策函数f(x)作为预测，是一种判别式模型。**
### 1.2 large margin intuition
+ SVM decision boundary

代价函数为：
$$min\ c\sum_{i=1}^{m}\[y^{(i)}cost_{1}(\mathbf{\theta }^\mathrm{T}x^{(i)})+(1-y^{(i)})cost_{0}(\mathbf{\theta }^\mathrm{T}x^{(i)})]+\frac{1}{2}\sum_{i=1}^{m}(\theta \_{j})^{2}$$
假设给定 c 一个非常大的值，为使得代价函数值最小，我们希望找到一个 使得第一项为0的最优解。此时，代价函数最小化问题转换成一个条件约束最优化问题，如下：
$$ min\ \frac{1}{2}\sum_{i=1}^{m}(\theta \_{j})^{2}, $$
$$ s.t.\  \mathbf{\theta }^\mathrm{T}x^{(i)}\geq 1,\ if\  y^{(i)}=1,$$
$$\mathbf{\theta }^\mathrm{T}x^{(i)}\leq -1,\  if\  y^{(i)}=0.$$
通过下图，可以直观地理解SVM 又被称为最大间隔分类器：

![intuition_to_Large_Margin](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/intuition_to_Large_Margin.png)
+ Large margin classifier in presence of outliers

如果给定正则化项c非常大的值，SVM的决策边界(也可以理解为最大间隔分离超平面)将会拟合甚至那些异常值，导致产生过拟合问题。
### 1.3 mathematics behind large margin classification
+ 1.3.1 vector inner product向量内积(点乘)

存在两个向量：u和v，有向量内积为：$\vec{u}\cdot \vec{v} = \left \| u \right \|\cdot \left \| v \right \|cos\angle (u,v)$. 可以看出向量内积的运算结果为一个实数，即一个标量。当两个向量非零且正交（夹角等于90°）时，向量的内积为0.|u·v|≤|u|·|v|，当夹角为0时等号成立。向量内积的几何意义：
> 1. 表征或计算两个向量的夹角
> 2. 向量v在向量u方向上的投影

![vector_inner_production](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/vector_inner_production.png)

图中，||u||表示向量u 的模长，p表示向量v在u上的投影。
> 延伸知识：向量外积(叉乘)

向量叉乘定义：向量a和b的叉乘是一个向量，其长度|a×b|=|a||b|sin∠(a,b)，向量方向正交于a，b。在三维几何中，2个向量的外积结果又被称为**法向量**，其垂直于a和b构成的平面。在二维空间中，外积向量的长度|a×b|等于向量a和b所围平行四边形的面积。

+ 1.3.2 SVM decision boundary

![SVM_decision_boundary](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/SVM_decision_boundary.png)

> 从向量内积和向量模长的视角来看，SVM的优化问题可以写为：

$$ min\ \frac{1}{2}\sum_{i=1}^{m}(\theta \_{j})^{2} = \frac{1}{2}\left \| \theta  \right \|^{2}, $$
$$ s.t.\ \mathbf{\theta }^\mathrm{T}x^{(i)} = p^{(i)}\cdot \left \| \theta  \right \| \geq 1,\ if\  y^{(i)}=1,$$ 
$$ \mathbf{\theta }^\mathrm{T}x^{(i)} = p^{(i)}\cdot \left \| \theta  \right \|\leq -1,\  if\  y^{(i)}=0.$$ 
此处向量θ是垂直于决策边界的法向量。当y=1时，我们希望$ p^{(i)}\cdot \left \| \theta  \right \| \geq 1$，如果$ p^{(i)}$ 的值太小，意味着$\left \| \theta  \right \|$需要取较大的值，这不是我们的优化目标，因此上面的左图并不是我们的最优解；相反，右图得到较大的$ p^{(i)}$ ，相应的，我们可以得到较小的$\left \| \theta  \right \|$值。

> 从最大几何间隔分离超平面的角度来看：

首先超平面表示为$\mathbf{\theta }^\mathrm{T}x + b = 0$,其中 w 为垂直于超平面的法向量，箭头指向的一方为超平面的正方向，反之为负方向；b表示位移项，决定了超平面与原点之间的距离，当b=0时，超平面经过原点。于是我们有样本空间中任一点 x 到超平面(Θ，b)的距离可写为：
$$ r=\frac{\left \|  \mathbf{\theta }^\mathrm{T}x + b \right \|}{\left \| w \right \|}, $$
结合上面的向量内积的视角，可以看到：分子是特征向量x与参数向量Θ的内积的绝对值，可写为：$\left \| \theta  \right \|\left \| x \right \|cos\angle (\theta ,x)$，分母为参数向量Θ的模长，所以，r其实就是p，即向量x在向量Θ
上的投影。**间隔最大化 margin maximum**的思想是：存在超平面(Θ，b)能够将训练数据正确分开，即有
![公式10.1](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/%E5%85%AC%E5%BC%8F10.1.gif)

记上式为（式10.1）,。这里选择+1和-1是为了方便计算，因为不管取何值，最终都可以约为1.其中，距离超平面最近的几个训练样本使得等号成立，他们又被称为“支持向量（support vector）”。两个异类支持向量到超平面的距离之和（即为最大间隔）表示为：
$$ r=\frac{2}{\left \| w \right \|}. $$
上述公式由如下推导得出：
设存在两个异类支持向量:  $x_{1}$ 为正类,  $x_{2}$ 为负类。想要使得r最大，等价于使得下式最大：
$$ r=\frac{\mathbf{\theta }^\mathrm{T}\cdot (x_{2}-x_{1})}{\left \| w \right \|} $$
此式子又等价于：
$$ r=\frac{1-b-(-1-b)}{\left \| w \right \|}=\frac{2}{\left \| w \right \|}. $$
此处需要注意**存在一个约束条件：需要满足上述公式10.1，此公式可以简化为
$y_{i}(\mathbf{\theta }^\mathrm{T}x_{i}+b)\geq +1$。**
欲找到最大间隔分离超平面，即找到满足约束条件的参数Θ和b，使得r最大。公式描述如下：
$$ max_{(\theta ,b)}\ \frac{2}{\left \| w \right \|}\\\\
s.t.\ y_{i}(\mathbf{\theta }^\mathrm{T}x_{i}+b)\geq +1 $$
上述公式等价于：
$$ min_{(\theta ,b)}\ \frac{1}{2}\left \| w \right \|^{2}\\\\
s.t.\ y_{i}(\mathbf{\theta }^\mathrm{T}x_{i}+b)\geq +1 $$

以上即为**支持向量机(SVM)的基本型**，为方便接下来的叙述，记为（式10.2）。
+ 1.3.3 原始问题的对偶化（dual problem）

首先，**引入凸优化问题的概念**：凸优化问题也被称作 约束最优化问题，公式描述如下：
$$ min_{w}\ f(w)\\\\
s.t.\ g_{i}(w)\leq 0, i=1,2,\cdots ,k\\\\
     h_{i}(w) = 0, i=1,2,\cdots ,k $$
其中目标函数f(x)和约束函数 $ g_{i}(w) $ 都是 $ \Re ^{n} $ 上的连续可微的凸函数,约束函数 $ h_{i}(w) $ 是 $ \Re ^{n} $ 上的仿射函数（一阶多项式函数，形如a·x+b）。**当目标函数f（x）是二次函数且约束函数$ g_{i}(w) $是仿射函数时，上述凸最优化问题成为凸二次规划（convex quadratic programming）问题**。
故，上小节中的（式10.2）是一个凸二次规划问题，**应用拉格朗日对偶性，可以将原始问题对偶化，通过求解对偶问题得到原始问题的最优解**。
> a. 构建拉格朗日函数，**使约束最优化问题变为无约束最优化问题**

为原始问题中的每一个约束条件引进拉格朗日乘子（ Lagrange multiplier），记为 $ \alpha \_{i}\geq 0 $, $ \alpha =(\alpha \_{1},\alpha \_{2},\cdots ,\alpha \_{m}), $  得到拉格朗日函数为：
$$ L(w,b,\alpha )=\frac{1}{2}\left \| \left \| w \right \|\right \|^{2}+\sum_{i=1}^{m}\alpha \_{i}(1-y_{i}(\mathbf{ w }^\mathrm{T}x_{i}+b)) =\frac{1}{2}\left \| \left \| w \right \|\right \|^{2}-\sum_{i=1}^{m}\alpha \_{i}y_{i}(\mathbf{ w }^\mathrm{T}x_{i}+b)+\sum_{i=1}^{m}\alpha \_{i}. $$

>b.  **原始无约束最优化问题 → 对偶问题**：·

引入拉格朗日函数后，原始问题变为：求在w，b满足约束条件下的，以下极小极大问题：
$$ min_{(w,b)}\ max_{(\alpha )}\ L(w,b,\alpha ), $$
根据拉格朗日对偶性，**原始问题的对偶问题是极大极小问题**：
$$ max_{(\alpha )}\ min_{(w,b )}\ L(w,b,\alpha ). $$
`step 1:求 min L(w,b,α)`
求L(w,b,α)关于w和b的偏导数，并令其为0，可得：
$$ w = \sum_{i=1}^{m}\alpha \_{i}y_{i}x_{i}, $$
$$ 0 = \sum_{i=1}^{m}\alpha \_{i}y_{i}. $$
代入拉格朗日函数后消去w 和 b后，得到：
$$ min_{(w,b )}\ L(w,b,\alpha )=-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha \_{i}\alpha \_{j}y_{i}y_{j}(x_{i}\cdot x_{j})+\sum_{i=1}^{N}\alpha \_{i}. $$
`step 2:求 min L(w,b,α)对 α 的极大，即是对偶问题：`
$$ max_{(\alpha )}-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha \_{i}\alpha \_{j}y_{i}y_{j}(x_{i}\cdot x_{j})+\sum_{i=1}^{N}\alpha \_{i}\\
s.t.\ \sum_{i=1}^{N}\alpha \_{i}y_{i}=0\\
\alpha \_{i}\geq 0,i=1,2,\cdots ,N $$

上述式子等价于：
$$ min_{(\alpha )}\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha \_{i}\alpha _{j}y_{i}y_{j}(x_{i}\cdot x_{j})+\sum_{i=1}^{N}\alpha \_{i}\\
s.t.\ \sum_{i=1}^{N}\alpha \_{i}y_{i}=0\\
\alpha \_{i}\geq 0,i=1,2,\cdots ,N $$
上式记为（式10.3）,是对偶最优化问题。
> c. 求解$ \alpha ^{\ast }, 
\mathbf{\alpha ^{\ast }=(\alpha ^{\ast }\_{1},\alpha ^{\ast }\_{2},\cdots ,\alpha ^{\ast }\_{N}) }^\mathrm{T} $.

根据以下定理：<br>
**考虑原始问题和对偶问题。假设函数f(x)和$ g_{i}(w) $是凸函数，$ h_{i}(w) $是仿射函数；并且假设不等式约束$ g_{i}(w) $是严格可行的，即存在w，对所有i有$ g_{i}(w) < 0 $,则存在 $ w^{\ast },\alpha ^{\ast },\beta ^{\ast } $,使得 $ w^{\ast } $ 是原始问题的解，$ \alpha ^{\ast },\beta ^{\ast } $是对偶问题的解。**<br>
有：存在下标j，使得$ \alpha_{j}> 0 $,得到原始最优化问题的解如下：
$$ w^{\ast }=\sum_{i=1}^{N}\alpha \_{i}^{\ast }y_{i}x_{i}\\\\
b^{\ast }=y_{j}-\sum_{i=1}^{N}\alpha \_{i}^{\ast }y_{i}(x_{i}\cdot x_{j)} $$
**证明：**<br>
根据定理：
>  $ w^{\ast },\alpha ^{\ast },\beta ^{\ast } $是原始问题和对偶问题的解 的充分必要条件是：$ w^{\ast },\alpha ^{\ast },\beta ^{\ast } $满足KKT(Karush-Kuhn-Tucher)条件:
条件1～3：分别求 $ L(w^{\ast },\alpha ^{\ast },\beta ^{\ast })$ 对三个变量的偏导数，并令其为0；<br>
条件4：$ \alpha ^{\ast }g_{i}(w^{\ast })=0 $<br>
条件5：$ g_{i}(w^{\ast })\leq 0 $<br>
条件6：$ \alpha ^{\ast }\geq 0 $<br>
条件7：$ h_{i}(w) = 0 $<br>
根据条件3，有
$$ \alpha ^{\ast }\_{i}(y_{i}(w^{\ast }\cdot x_{i}+b^{\ast })-1)=0 $$
又$$ 
## 2 Kernels
### 2.1 Kernels Ⅰ
### 2.2 Kernels Ⅱ
## 3 SVMs in practice - using an SVM 


