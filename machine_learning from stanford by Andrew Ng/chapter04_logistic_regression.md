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
  在逻辑回归模型中，我们希望分类器的输出值在0到1之间，也就是说希望 $$0\leq h_{\theta }(x)\leq 1$$。在线性回归中，我们的假设函数为：$h_{\theta }(x) = \mathbf{\theta }^\mathrm{T}x,$现在我们对此稍加修改，得到逻辑回归的假设函数如下：
  $$z=\mathbf{\theta }^\mathrm{T},x$$
  $$h_{\theta }(x) =g(z)=\frac{1}{1+e^{-(z)}},$$
  $$h_{\theta }(x) =g(z)=\frac{1}{1+e^{-(\mathbf{\theta }^\mathrm{T}x)}},$$
  
在逻辑回归模型中，假设函数有2种叫法:`Sigmoid function or Logistic function`.关于假设函数$h_{\theta }(x)$和参数θ的图像如下：

![logistic_regression_mode](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/logistic_regression_model.png)

+ interpretation of hypothesis output

![interpretation_of_hypothesis_output](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/interpretation_of_hypothesis_output.png)


### 1.3 decision boundray(决策边界)




## 2 logistic regression model



## 3 multiclass classification
