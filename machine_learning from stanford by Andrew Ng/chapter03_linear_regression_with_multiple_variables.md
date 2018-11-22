## gradient descent for multiple variables
+ hypothesis and cost function

hypothesis: 
![hypothesis_for_multiple_variables](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/hypothesis_for_multiple_variables.gif)

parameters: $$\theta _{0},\theta _{1},\cdots ,\theta _{n}$$

cost function: $J(\theta _{0},\theta _{1},\cdots ,\theta _{n})$

$$=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta }(x^{(i)})-y^{(i)})^{2}$$


我们约定x0=1，把参数θ0，θ1，……θn当作一个`n+1`维的向量(n+1 dimensional vector),把$$J(\theta _{0},\theta _{1},\cdots ,\theta _{n})$$看作 以一个`n+1`维的向量为参数 的函数，

+ gradient descent

repeat {
    $$\theta _{j}:=\theta _{j}-\alpha \frac{\partial J(\theta _{0},\cdots ,\theta _{n})}{\partial \theta _{j}}$$
    }              (simultaneously update for every j=0,……,n)

下面这张图片显示，即使是多于2个参数的情况下，他们的梯度算法其实是同一件事请。
![gradient_descent_for_multivariables](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/gradient_descent_for_multivariables.png)
+ 梯度下降算法中的实用技巧1-feature scaling

**idea:** make sure features are on a similar scale<br>
假设我们两个特征值，x1 = size of house(0-2000)， x2 = numbers of bedrooms(1-5),此时我们画出的轮廓图，会发现它由很多个 `长短半轴相差十分大`的椭圆构成，这意味着若想要达到全局最小处，需要花费很多时间，做多次梯度下降，并且有可能来回反复波动。<br>
**一种有效的优化方法Feature scaling**<br>
case1:<br>
x1,x2分别除以各自的最大取值数，使得0<= x1 <=1,0<= x2 <=1.一般说来，在进行featrue scaling 时，会尽量使$X_i$在\[-1, 1]之间，也就是说这个范围不要太大，也不要太小。<br>
case2:mean normalization(均值归一化)<br>
具体说就是，使用$x_i$-$μ_i$ 代替$x_i$，来是特征值具有为0的特征值。(replace $x_i$ with $x_i$-$μ_i$ to makefeatrues have approximately 0 mean.)注意：必要对$x_0$=1进行这一操作$μ_i$为平均值，公式如下：
$$x_{i}=\frac{x_{i}-\mu _{i}}{s}$$
> $μ_i$代表 average value of $x_i$ in training set. $s_i$ 代表the range of that featrue,the range指的是最大值减去最小值。

#通过特征缩放，可以使得梯度下降的速度变得更快，减少收敛到全局最小值的循环次数。#
+ 梯度下降算法中的实用技巧2-learning rate

  learning rate 是梯度下降算法的更新规则。在选好数据，设置好参数，运行梯度算法时，learning rate α其实可以看做是一种调试手段，帮助我们确认梯度算法是否工作正常。接下来将通过例子来看learning rate 具体是如何工作的。<br>
 下图是**关于代价函数J(Θ)和 a number of iterations of gradient descent的图像**，即代价函数随着迭代步数的增加而变化的曲线图，通过观察，可以判断算法是否已经收敛.
 
 ![]()
 其中纵轴代表J(Θ)的值， x轴代表算法迭代次数。 比如在运行100步后我将得到一个$Θ_j$值，不管这个$Θ_j$的是多少，我们画出他对应的J(Θ)的值，同样在运行200次后，也会得到一个J(Θ)的值。所以这条曲线显示的是`梯度下降算法迭代过程中，代价函数J(Θ)的值`,如果**算法正常工作，则每一步迭代之后，J(Θ)的值都应该下降，也就是说曲线应该是单调递减的。**当曲线逐渐趋于平坦时，表示算法基本已经**收敛**了。<br>
*对于特定的问题，梯度算法的迭代次数可以相差很大。我们提前很难判断需要多少次迭代。*<br>
**如何选择合适的learning rate？**<br>
case1:自动的收敛测试，即如果代价函数J(Θ)的下降小于一个很小的值ε，则认为他已经收敛了，例如选择ε=$e^(-3)$。但，选择一个合适的阈值ε是很困难的。<br>
case2：观察曲线图，调整learning rate.
```当α取值很大时，曲线图很有可能成单调上升趋势。随着迭代次数的增加，代价越来越大；
当α取值过小时，每一次的更新幅度会很小，将花费很多时间，进行多次迭代后，才能收敛
```

+ features and polynomial regression(特征和多项式回归)

>选择特征的方法 —— 多项式回归(polynomial regression)

在多变量线性回归中，我们的假设函数为
![hypothesis_for_multiple_variables](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/hypothesis_for_multiple_variables.gif)
可以看到，我们有多个特征、多个参数，此时一次线性函数已经无法很好的拟合我们的数据，于是我们考虑使用二次函数模型(quadratic function model)，但是随着观察拟合结果的变化，我们发现二次函数最终会降下来，但在我们这次的问题中，我们并不认为房子的价格在高到一定程度后会降下来。那我们在增加一项，使模型变为一个三次函数模型(cubic function model)，假设函数中包含一个三次方项(a  third-order term),此时拟合后的曲线是这样一条曲线。
![cubic function model]()

>我们选择什么样的模型对数据进行拟合，取决于我们想要解决什么样的具体问题，并且针对特定的问题，当你选择不同的角度来审视问题时，所选的特征(数据)也会产生差异。有时，也可以**自定义新的特征**，以帮助我们获得更好的模型。<br
**如何将模型与数据进行拟合？  答案是 使用多元线性回归的方法**<br>
    对我们的算法做一个简单的修改。比如之前的算法中，h(Θ)= Θ0 + Θ1x_1 + Θ2x_2 +Θx_3（x_1为房子的面积） ，我们对其进行修改，使得x_1为房子的面积，x_2为房子面积的平方，x_3为房子面积的立方。更改后的式子与原式是相等的（**这里需要理解一下为什么相等**）。再应用线性回归的方法，我们得到一个模型，可以用它来拟合我们的数据。当我们的特征范围十分大时，需要对数据进行**归一化处理**后，再使用梯度下降法来找到J(Θ)的最小值。归一化处理将各特征值变得具有可比性。<br>
**如何找到我们有意义的特征**<br>
    这里的有意义是指，对于我们想要预测结果来说，如何选择这特征将产生影响。为得到我们理想的拟合模型，除上述的建立一个三次模型方法外，我们也可以使用平方根项的方法。此时我们得到这样一条拟合曲线。
![the square root function model]()
曲线上升，然后到一定高度后开始变得平坦，但它永远不会下降。
  + 以上我们了解了多变量梯度下降算法，以及如何优化其中的偏导项，找到的J(Θ)的最小值。
