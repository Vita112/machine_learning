关于过拟合问题，可以同时参考这篇git笔记：[overfitting](https://github.com/Vita112/machine_learning/blob/master/overfitting.md)
##  regularization
ameliorate改善 or reduce overfitting problem
### 1  the problem of overfitting
+ underfitting or high bias 欠拟合或高偏差
+ overfitting or high variance- 过分拟合训练数据，导致无法泛化到新的数据样本中，以致于无法对新样本进行预测.
**generalize**：how well a hypothesis applies to new examples which has not seen in the training set。
+ how to address？
当我们有很多特征值，但只有非常少的训练数据时，就会出现过拟合。**解决方法**主要有2种：
>1. reduce number of features:\[disadvantage:lose some information]<br>
-manually select which feature to keep<br>
-model selection algorithm which will be talk in later course
>2. regularization<br>
-keep all the features but reduce magnitude/values of parameters $θ_j$<br>
-works well when we have a lot of features, each of which contributes a bit to predicting y.
### 2 cost function
+ the idea of regularization
在线性回归中，我们的目标是最小化代价函数的均方误差，假设我们使得某些参数值变得很小，几乎趋近于0，那么这些参数对J(θ)的影响就会变得很小，因而避免造成过拟合。**正则化的思想**是：参数取更小的值，意味着更简单的假设模型，也就不易于发生过拟合。
+ 最小化所有参数
![shrink_all_parameters](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/shrink_all_parameters.png)
上图中，所有的参数相加项(正则化对象)没有包含$θ_0$，因为$θ_0$对于最终的代价函数的结果影响十分小。我们希望加入正则化项后，可以同时达成2个目标，并使其达到平衡：
>1. fit the training set well
>2. keep parameters small so that to get more simplier hypothesis

如果λ值很大，这意味着所有地参数$θ_j$都将趋向于0，导致模型欠拟合。
+ how do i think of this？

1.引入正则化思想的引例中，要注意：这是一个不太严谨的假设，只是为了方便更直观地理解正则化.<br>
2.为什么缩小了参数后，就可以得到更加简化地模型呢？<br>
3.如何选择正则项参数λ？后面课程中将会讲到一系列自动选择正则化参数的方法。
### 3 regularization linear regression
线性回归的两种基本学习算法：基于梯度下降 和 基于正规方程。正则化的线性回归代价函数如下：![regularilized_linear_regression](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/regularilized_linear_regression.png)
+ 正则化的梯度下降算法

![regularized_gradient_decent](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/regularilized_gradient_decent.png)
由于没有对$θ_0$进行正则化操作，因此第一个式子是$θ_0$的更新规则。第二个式子则是对J(θ)求偏导后得到的结果，合并同类项得到第三个式子，其中第一项中的$1- \alpha \frac{\lambda}{m}$是一个略小于1的数，表示每次对参数$θ_j$进行轻微的压缩；第二项与未正则化的梯度下降规则相同。
+ regularized normal equation

前面我们得到了最小化J(θ)正规方程法，它通过明确地取其关于$θ_j$的导数，并将他们设置为0，来最小化J(θ)，公式如下：
$$\theta =(\mathbf{X}^\mathrm{T}X)^{-1}\mathbf{X}^\mathrm{T}y，$$ 关于其详细的推导过程，在b站上有一位大神用白板推到过，如下图：
![use_MLE_to_get_optimal_θ](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/2.1LR-%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95%E5%8F%8A%E5%85%B6%E5%87%A0%E4%BD%95%E6%84%8F%E4%B9%89.png)
其中的思想是：首先使用最小二乘法得到关于参数θ/权重的代价函数的向量表示，然后使用**最大似然估计**，即对代价函数J(θ)求偏导并令其等于0，得到最小化J(θ)的θ值。<br>
最小二乘法也可以从概率论视角来解释，为假设函数f(θ)加上一个 服从均值为0，方差为$\sigma^{2}$的高斯噪声$\varepsilon$,于是关于给定x的y的概率分布(后验概率分布)为一个 服从均值为$\mathbf{\theta }^\mathrm{T}x$,方差为$\sigma^{2}$ 的高斯分布，对数似然后得到对数似然函数L(θ)=$logP(Y|X;\theta )$,然后求$ \hat{\theta }= argmaxL(\theta )$,结果与最小二乘法一样。如下图：
![2.2LR-LSE-概率视角-高斯噪声-MLE](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/2.2LR-LSE-%E6%A6%82%E7%8E%87%E8%A7%86%E8%A7%92-%E9%AB%98%E6%96%AF%E5%99%AA%E5%A3%B0-MLE.png)
>**正则化的正规方程**

需要在矩阵逆项中加入惩罚项，公式如下：
![regularized_normal_equation_for_θ](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/regularized_normal_equation_for_%CE%B8.png),在白板推导系列中为：
![2.3LR-正则化-ridgeRegression岭回归](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/2.3LR-%E6%AD%A3%E5%88%99%E5%8C%96-ridgeRegression%E5%B2%AD%E5%9B%9E%E5%BD%92.png)，概率论-贝叶斯视角下为：
![2.4LR-正则化-ridgeRegression-概率视角-高斯噪声高斯先验](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/2.4LR-%E6%AD%A3%E5%88%99%E5%8C%96-ridgeRegression-%E6%A6%82%E7%8E%87%E8%A7%86%E8%A7%92-%E9%AB%98%E6%96%AF%E5%99%AA%E5%A3%B0%E9%AB%98%E6%96%AF%E5%85%88%E9%AA%8C.png)
+ non-invertability(advanced/optional)

之前我们讨论过正规方程法中的不可逆问题，他是这样描述的：当样本数m$\leqslant$特征数n时，$\mathbf{X}^\mathrm{T}X)$将变得不可逆invertable，或者称为奇异的singular，导致这个矩阵退化be degenerate。
>正则化解决了不可逆的问题：如果λ是大于零的，那么逆项中的矩阵将变得可逆，
### 4 regularization logistic regression

同线性回归一样，我们为逻辑回归的代价函数增加惩罚项，得到公式如下：
![regularized_cost_function_for_logistic_regression](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/regularized_cost_function_for_logistic_regression.png)
然后 求J(θ)关于θ的偏导数，将偏导结果放到梯度下降更新规则中，我们得到如下θ的更新规则：
![gradient_decent_rule_for_regularized_logistic_function](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/gradient_decent_rule_for_regularized_logistic_function.png)
图中，方括号内部的内容就是对J(θ)求偏导后的结果，注意此处的假设模型$h_{\theta }(x)$。
+ advanced optimization algorithm 高级优化算法on octave

此处有一个函数需要解释一下：函数fminunc()返回的是 函数costFunction在无约束条件下的最小值，即为代价函数的最小值。下图为在octave上的实现：
![implementation_on_octave_for_advanced_optimization_algorithm_for_regulariezed_logistic_regression](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/implementation_on_octave_for_advanced_optimization_algorithm_for_regulariezed_logistic_regression.png)


+ 参考：[B站机器学习白板推导系列](https://space.bilibili.com/97068901/video)
