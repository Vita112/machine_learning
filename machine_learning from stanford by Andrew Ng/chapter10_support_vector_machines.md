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
+ vector inner product

![vector_inner_production](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/vector_inner_production.png)

图中，|||u|表示向量的模长，p表示向量v在u上的投影。
+ SVM decision boundary

![SVM_decision_boundary](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/SVM_decision_boundary.png)

> 从向量内积和向量模长的视角来看，SVM的优化问题可以写为：
$$ min\ \frac{1}{2}\sum_{i=1}^{m}(\theta \_{j})^{2} = \frac{1}{2}\left \| \theta  \right \|^{2}, $$
$$ s.t.\ \mathbf{\theta }^\mathrm{T}x^{(i)} = p^{(i)}\cdot \left \| \theta  \right \| \geq 1,\ if\  y^{(i)}=1,$$ 
$$ \mathbf{\theta }^\mathrm{T}x^{(i)} = p^{(i)}\cdot \left \| \theta  \right \|\leq -1,\  if\  y^{(i)}=0.$$ 
此处向量θ是垂直于决策边界的法向量。当y=1时，我们希望$ p^{(i)}\cdot \left \| \theta  \right \| \geq 1$，如果$ p^{(i)}$ 的值太小，意味着$\left \| \theta  \right \|$需要取较大的值，这不是我们的优化目标，因此上面的左图并不是我们的最优解；相反，右图得到较大的$ p^{(i)}$ ，相应的，我们可以得到较小的$\left \| \theta  \right \|$值。

> 从最大几何间隔分离超平面的角度来看：首先超平面表示为$\mathbf{\theta }^\mathrm{T}x + b = 0$,其中 w 为垂直于超平面的法向量，箭头指向的一方为超平面的正方向，反之为负方向；b表示位移项，决定了超平面与原点之间的距离。于是我们有样本空间中任一点 x 到超平面(Θ，b)的距离可写为：

## 2 Kernels
### 2.1 Kernels Ⅰ
### 2.2 Kernels Ⅱ
## 3 SVMs in practice - using an SVM 


