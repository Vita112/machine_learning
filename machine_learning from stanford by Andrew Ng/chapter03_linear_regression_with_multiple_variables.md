## gradient descent for multiple variables
+ hypothesis and cost function

hypothesis: 
![hypothesis_for_multiple_variables](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/hypothesis_for_multiple_variables.gif)

parameters: $$\theta _{0},\theta _{1},\cdots ,\theta _{n}$$

cost function: $J(\theta _{0},\theta _{1},\cdots ,\theta _{n})$

$$=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta }(x^{(i)})-y^{(i)})^{2}$$


我们约定x0=1，把参数θ0，θ1，……θn当作一个`n+1`维的向量(n+1 dimensional vector),把$$J(\theta _{0},\theta _{1},\cdots ,\theta _{n})$$看作 以
一个`n+1`维的向量为参数 的函数，

+ gradient descent

repeat {
    $$\theta _{j}:=\theta _{j}-\alpha \frac{\partial J(\theta _{0},\cdots ,\theta _{n})}{\partial \theta _{j}}$$
    }              (simultaneously update for every j=0,……,n)

下面这张图片显示，即使是多于2个参数的情况下，他们的梯度算法其实是同一件事请。
![gradient_descent_for_multivariables](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/gradient_descent_for_multivariables.png)
+ 梯度下降算法中的实用技巧1-feature scaling

**idea:** make sure features are on a similar scale
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


