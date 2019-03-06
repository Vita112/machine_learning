## 1 from logistic regression to linear classifier
+ logistic regression

logistic regression 是一个0/1分类模型，它将输入特征的线性组合作为自变量，自变量取值范围从负无穷至正无穷，使用ligistic函数（或sigmoid函数）将结果映射到（0,1）上，
映射后的值g(z)也被认为是`y=1`的概率。观察lr图像可以发现：当z趋近于正无穷时，y≈1；当z趋近于负无穷时，y≈0.**模型的目标是**：学习得到参数Θ，使得正样例(y=1)的特征组合$ \mathbf{\theta }^\mathrm{T}x\gg 0 $, 负样例(y=0)的特征组合$ \mathbf{\theta }^\mathrm{T}x\ll 0. $
+ linear classifier

我们对上述logistic regression中特征组合做一些变形，即令$ \mathbf{\theta }^\mathrm{T}x=\mathbf{w }^\mathrm{T}x+b,y\in {-1,+1} $，得到线性分类函数如下：
![linear_classifier](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/linear_classifier.gif)
## 2 how to get SVM
SVM，通俗来讲是一种二分来模型，其基本模型定义为特征空间上的间隔最大的线性分类器，其学习策略是间隔最大化，最终可转化为凸二次规划问题的求解。
### 2.1　optimal hyper plane
定义超平面为：$$ f(x)=\mathbf{w }^\mathrm{T}x+b $$.关于超平面的直观理解是：它将特征空间中的数据点分为2类，当f(x)＞0时，点x在超平面的一侧，对应的y是1；当f(x)＜0时，点x在超平面的另一侧，对应的y是-1；位于超平面上的点x，有f(x)=0.<br>
**我们的问题是：如何确定最优超平面，即如何找到距离两边的数据的间隔最大的超平面**?为此，以下先导入functional margin 和 geometrical margin的概念。
### 2.2 functional margin and geometrical margin
+ functional margin

在超平面w·x+b=0确定的情况下，|w·x+b|表示点x到超平面(w,b)的距离.因为超平面将数据分为两类：+1和-1，故可以使用y(w·x+b)的正负性来表示或者判别分类结果的正确性，
以下给出**函数间隔**的定义：<br>
> 对于给定的训练数据集T和超平面(w,b)，定义超平面(w,b)关于样本点$ (x_{i},y\_{i}) $的函数间隔为
$$  \hat{\gamma}\_{i}=y_{i}(\mathbf{w }^\mathrm{T}x_{i}+b)= y_{i}f(x)$$
定义超平面(w,b)关于训练数据集的函数间隔为超平面(w,b)关于数据集中所有样本点$ (x_{i},y\_{i}) $的函数间隔的最小值，即
$$ \hat{ \gamma }=min\hat{\gamma}\_{i} $$

由于如果成比例的改变w和b的值，函数间隔的值f(x)将受到影响，虽然此时，我们的超平面还是同一个。为解决这个问题，我们可以给法向量w加上一些约束条件，
因而引出了 真正定义空间中的点到超平面的距离——几何间隔。
+ geometrical margin

引子：<br>
假定对于一个点 x ，令其垂直投影到超平面上的对应点为 x0 ，w 是垂直于超平面的一个向量，为样本x到超平面的距离，如下图所示：
![to_deduce_geometrical margin](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/to_deduce_geometrical%20margin.jpg)
根据平面几何知识，我们有：
$$ x=x_{0}+\gamma \frac{w}{\left \| \left \| w \right \| \right \|} $$
推导后，得到
$$ \gamma =\frac{\mathbf{w }^\mathrm{T}x+b}{\left \| \left \| w \right \| \right \|}=\frac{f(x)}{\left \| \left \| w \right \| \right \|} $$
有
$$ \tilde{\gamma}\_{i} =y_{i}\gamma =y_{i}\frac{f(x)}{\left \| \left \| w \right \| \right \|}=\frac{\hat{\gamma \_{i}}}{\left \| \left \| w \right \| \right \|} $$
于是，定义**超平面(w,b)关于训练数据集的几何间隔**为超平面(w,b)关于数据集中所有样本点$ (x_{i},y\_{i}) $的几何间隔的最小值，即
$$ \tilde{\gamma }=min\tilde{\gamma}\_{i} $$
上述函数间隔和几何间隔的关系可以看出：
$$ \tilde{\gamma }=\frac{\hat{\gamma }}{\left \| \left \| w \right \| \right \|} $$
$$ y_{i}f(x)=\left | f(x) \right |=\left | \mathbf{w }^\mathrm{T}x+b \right | $$
### 2.3 maximum margin classifier
## 3 linearly separable problem
### 3.1 from primitive problem to dual problem
### 3.2 KKT conditions
### 3.3 3 steps to solve a dual problem
## 4 linearly non-separable problem
### 4.1 what is Kernel function?
### 4.2 several kinds of common Kernel Function
### 4.3 the essence of Kernel Function
## 5 using Relaxation variables to handle outliers


reference：<BR>
[CSDN:支持向量机通俗导论（理解SVM的三层境界）](https://blog.csdn.net/v_july_v/article/details/7624837)
