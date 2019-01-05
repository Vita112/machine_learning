## 1 classification and representation
### 1.1 classification
  邮件被分为spam/not spam，online transation被分为fraudulent/not fraudulent，一个rumor是malignant/benign，这几个例子都是分类问题，
  在分类问题中，我们试图预测的变量`y`只会出现两个结果：0 或者 1.
  > $$y\in{0,1}$$
    0：negative class   absence of something<br>
    1：positive class   presence of something<br>
    上面这种情况是一种两分类问题(binary classification problem)，后面我们会讲到多分类问题，即$$y\in({0,1,2,3})$$

+ 将线性回归算法应用于分类问题
   
  线性回归中，我们的假设函数是一个`linear line`，由于训练数据集中的y只有两个值：1和0，为达到最佳拟合， 我们设定hΘ(x)=0.5的点作为界限，
把输入特征分为两类，如同下图所显示的，在紫色线的左边部分，hΘ(x)>0.5,我们将其看作1;而在紫色线的右边部分，hΘ(x)<0.5，我们将其看作0.
这样似乎是可行的，拟合出的假设函数将确实将y值分为两类。但是，我们可能增加列另外一些特征输入，此时假设函数可能变为蓝色那条线。但是，很明显，将y分为0 和 1 两类的，应该是一条垂直于x轴的蓝色直线。
![apply_linear_regression_for_classification](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/apply_linear_regression_for_classification.png)

**以上，我们并不认为，linear regression可以适用于分类问题。分类问题的假设函数也不应该是线性的**
+ Logistic Regression
   在逻辑回归中，假设函数hθ(x)的值总是在区间\[0,1]。逻辑回归的名字可能有点迷惑，因为名字中含有`regression`，但这只是由一些
   历史问题造成的。它实际上是一个分类算法。
### 1.2 hypothesis representation
  在逻辑回归模型中，我们希望分类器的输出值在0到1之间，也就是说希望 $$0\leq h_{\theta }(x)\leq 1.$$在线性回归中，我们的假设函数为：$h_{\theta }(x) = \mathbf{\theta }^\mathrm{T}x,$现在我们对此稍加修改，得到逻辑回归的假设函数如下：
  $$z=\mathbf{\theta }^\mathrm{T}x,$$
  $$h_{\theta }(x) =g(z)=\frac{1}{1+e^{-(z)}},$$
  $$h_{\theta }(x) =g(z)=\frac{1}{1+e^{-(\mathbf{\theta }^\mathrm{T}x)}},$$
  
在逻辑回归模型中，假设函数有2种叫法:`Sigmoid function or Logistic function`.关于假设函数$h_{\theta }(x)$和参数θ的图像如下：

![logistic_regression_mode](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/logistic_regression_model.png)

+ interpretation of hypothesis output

![interpretation_of_hypothesis_output](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/interpretation_of_hypothesis_output.png)

上图解释了 `逻辑回归模型如何解决分类问题`，请务必理解图片中的内容。
### 1.3 decision boundary(决策边界)
  在上一节中，我们给出了逻辑回归中的假设函数，本节中，我们将使用`decision boundary`来帮助我们理解`逻辑回归中的假设函数是如何工作的`。
逻辑回归的公式为：$h_{\theta }(x) =g(\mathbf{\theta }^\mathrm{T}x)=\frac{1}{1+e^{-(\mathbf{\theta }^\mathrm{T}x)}},$ 结合上节中的图片，我们可以发现：
>如果$h_{\theta }(x)\geq 0.5$,即g(z)$\geq 0.5$,此时，$\mathbf{\theta }^\mathrm{T}x\geq 0$,预测结果g(z)=1的概率更大，g(z)更向1趋近。，
 如果$h_{\theta }(x)< 0.5$,即g(z)< 0.5,此时，$\mathbf{\theta }^\mathrm{T}x< 0$,预测结果g(z)=0的概率更大，g(z)更向0趋近。，

+ explanation of decision boundary

假设我们已经拟合好了参数，最终选择了$θ_0=-3$,$θ_1=1$,$θ_2=1$,于是我们得到了参数向量为θ等于\[-3,1,1]的转置。根据逻辑回归算法，
我们有以下结论：当-3+$x_1$+$x_2$>=0时，假设函数h(θ)>=0.5，预测结果`y=1`的概率更高，预测结果出现在$x_1$+$x_2$=3的右上方；当-3+$x_1$+$x_2$<0时，假设函数h(θ)<0.5，预测结果`y=0`的概率更高，预测结果出现在$x_1$+$x_2$=3的左下方。参考下图：

![decision_boundary](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/decision_boundary.png)

+ non-linear decision boundarises

  同样的，当假设函数是高阶多项式假设函数时，也可以得到决策边界。下图显示了两种更为复杂的决策边界，假设函数为一个高阶多项式特征变量。第一个图中，假定已经拟合好了参数，并得到了参数向量θ等于\[-1,0,0,1,1]的转置，此时的决策边界为一个半径为1的圆；下面的图显示了更复杂的决策边界。
  
![non-linear_decision_boundaries](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/non-linear_decision_boundaries.png)

## 2 logistic regression model
### 2.1 cost function
  在逻辑回归中也有代价函数，只是此时我们的代价函数与线性回归中的不同，这是因为
>如果使用线性回归中的代价函数，即$$J(\theta ) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta }(x^{(i)})-y^{(i)}))^{2},$$ 
其中，![hypothesis_function_for_linear_regression](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/hypothesis_for_multiple_variables.gif).我们将得到一个non-convex function，也就是说我们会得到多个local optima。

显然，我们并不希望这样。在逻辑回归中，我们的假设函数如下：
![cost_function_for_logistic_regression](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/cost_function_for_logistic_regression.png)
当y=1时，我们得到如下有关$J(\theta )$和$h_{\theta }(x) $的图像：
![cost_function_when_y=1](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/cost_function_when_y%3D1.png)
当y=0时，我们得到如下有关$J(\theta )$和$h_{\theta }(x) $的图像：
![cost_function_when_y=0](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/cost_function_when_y%3D0.png)
上图反映了如下对应关系：
![y取1或0时_hθ(x)与cost_function的关系](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/y%E5%8F%961%E6%88%960%E6%97%B6_h%CE%B8(x)%E4%B8%8Ecost_function%E7%9A%84%E5%85%B3%E7%B3%BB.png)
### 2.2 simplified cost function and gradient decent
在上一集中我们得到了二分类问题的代价函数，为方便我们对代价函数使用梯度下降算法，我们可以用一种更加简单的表达式来合并这两个式子。如下图
![simplified_logistic_regression_cost_function](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/simplified_logistic_regression_cost_function.png)
从概率论观点看，我们求 使得简化后的J(θ)得到最小值的 参数θ的值，其实等价于 求J(θ)的关于θ的最大似然估计，J(θ)是log-likelyhood function。
我们的任务是：最小化代价函数，即通过最小化训练集中的预测值和真实结果间的误差，得到参数θ，然后使用参数θ对新输入x进行预测。由于分类问题实际上是一个概率问题，所以我们也可以将 假设函数即我们的模型 看成训练数据的概率密度函数，即某个数据属于2分类中的哪一类的概率，公式表达如下：
$$h_{\theta }(x)=\frac{1}{1+e^{-\theta^{\mathrm{T}}x}}$$
$$P(y=1|x;θ)$$
+ 使用梯度下降算法求出最优θ值

我们知道，梯度下降的更新规则为：$$\theta _{j}:=\theta _{j}-\alpha \frac{\partial J(\theta )}{\partial \theta }$$，

所以需要先对J(θ)求偏导，J(θ)如下图:
![interpretation_logistic_regression_cost_function](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/interpretation_logistic_regression_cost_function.png)
然后，我们对J(θ)求关于θ的偏导数，如下图：

![derivativeForLogisticRegressionCostFunction](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/derivativeForLogisticRegressionCostFunction.png)

其中第(2)到(3)的推导过程，一定要自己推导出来！！于是我们得到逻辑回归梯度下降的更新规则为：

![gradient_decent_rule_in_logistic_regression](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/gradient_decent_rule_in_logistic_regression.png)

我们发现：逻辑回归的梯度下降更新规则与线性回归梯度下降规则是一样的，**但实际上它们是完全不同的，因为假设函数$h_θ(x)$是不一样的**，此处不写出$h_θ(x)$，这个必须背下来！
+ use feature scaling to make sure conversion
+ vectorized implementation

梯度下降规则的向量化表示为：$$\theta :=\theta -\alpha\mathbf{X}^\mathrm{T}(g(X\theta )-\vec{y})$$
+ 手写笔记

![mynotes-for-logistic-regression](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/mynotes-for-logistic-regression.jpg)
### 2.3 advanced optimization
如果用代码来实现梯度下降算法，我们首先需要编写两个代码，一个用于计算J(θ)，一个计算J(θ)的偏导数，然后将偏导数的结果代入梯度下降规则中，最后得到使得J(θ)最小的θ值。**只是，在写出了计算J(θ)和J(θ)偏导数的代码之后，我们的优化算法其实不只梯度下降一种**，实际上还可以使用
> conjugate gradient 共轭梯度算法
> BFGS算法 ：一种拟牛顿法，使用BFGS矩阵作为拟牛顿法中的对称正定矩阵的方法，是求解无约束非线性优化问题的常用方法之一
参考[BFGS算法](https://blog.csdn.net/itplus/article/details/21897443)，讲的很详细，需要时间去理解。
> L-BFGS算法

以上这三种算法都有一个主要的思想：他们是用一个智能的内部循环，也被称为线性搜索算法，他可以自动为学习速率α选择不同的值，然后选择一个最优的α。
其优缺点如下：
> 优点：不需要手动选择α；比梯度下降更快达到收敛
> 缺点：实现将更加复杂
example：
![octave_implementation_for_2_parameters](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/octave_implementation_for_2_parameters.png)
后面接着讲了 octave的实现过程。
## 3 multiclass classification



