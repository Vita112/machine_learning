## 1 classification and representation
### 1.1 classification
  邮件被分为spam/not spam，online transation被分为fraudulent/not fraudulent，一个rumor是malignant/benign，这几个例子都是分类问题，
  在分类问题中，我们试图预测的变量`y`只会出现两个结果：0 或者 1.
  > $$y\in{0,1}$$
    0：negative class   absence of something<br>
    1：positive class   presence of something<br>
    上面这种情况是一种两分类问题(binary classification problem)，后面我们会讲到多分类问题，即$$y\in({0,1,2,3})$$

+ 将线性回归算法应用于分类问题
   
  线性回归中，我们的假设函数是一个`linear line`，由于训练数据集中的y只有两个值：1和0，为达到最佳拟合， 我们设定h0(x)=0.5的点作为界限，
把输入特征分为两类，如同下图所显示的，在紫色线的左边部分，h0(x)<0.5,我们将其看作1;而在紫色线的右边部分，h0(x)>0.5，我们将其看作0.
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

上图解释了 ```逻辑回归模型如何解决分类问题```，请务必理解图片中的内容。
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



## 3 multiclass classification
