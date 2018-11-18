
## 1. Model Representation
+ reading materials
  我们使用$x^{(i)}$代表输入变量，也称特征变量；使用$y^{(i)}$代表输出变量，每一对（$x^{(i)}$，$y^{(i)}$）是一个训练样本。同时，使用X代表输入变量的空间，Y代表输出变量的空间，本例中，X=Y=R。<br>
  监督学习方法中，我们给出一个训练集，来学习一个函数h：X→Y，使得h(x)可以很好的预测出y的值。<br>
  上述过程，我们可以更形象化（pictorially）地来看：
```
training set → learning algorithm → function h
input:x → function → output：predicated y
```

当预测的目标变量是连续的，我们称这是一个回归学习问题；当y只取一些分散的值时，我们称之为分类问题。


+ 1.1 一个引例-预测房价

假设我们有一个数据集，数据集包含某市的住房价格。使用这个数据集，我们画出一个 关于不同住房面积和与之对应的房价的 关系图。
现在你的一个朋友有一个面积为a平米的房子想要出售，我们要做的是：如何预测朋友的房子将会卖多少钱?<br>
*通过监督学习算法， 我们可以使用数据集建立一个模型，根据模型来预测朋友房子的售价。*

+ 1.2 线性回归

```
上节课我们知道，有2种常见的监督学习方式：回归和分类。
回归指的是，我们根据之前的数据预测出一个`准确的`输出值。
分类问题中，由于输入值对应的输出值离散地分布在数据空间中，并形成明显的某几类，因而对于我们想要进行预测的输入值，我们不会得到
准确的某个值，而是得到属于哪一类的结果。

```
我们使用预测房价问题说明监督学习中的线性回归问题。例子中某市的房价数据被称为训练集（a training example），包含不同房屋的面积及其价格。
我们使用`m表示训练样本的数目`，即训练集中有多少对房屋面积及其价格的数据；接着使用$x^{(i)}$代表`输入变量`，$y^{(i)}$代表`输出变量`,一对（$x^{(i)}$，$y^{(i)}$）。
代表一个训练样本。由于对于每一个输入值$x^{(i)}$来说，数据集都给出了准确的输出值$y^{(i)}$，通过观察X到Y某种映射关系（使用学习算法学习），我们得到一个有关x，y的函数h（h=hypothesis）。到此通过监督学习，我们可以说建立起了一个关于房屋面积和价格的模型，使用这个模型，可以得出给定输入值对应的输出值。<br>
这个模型也被成为**线性回归模型（Linear Regression Model）:**
$$h_{\theta }(x)=\theta _{0}+\theta _{1}*x$$
由于模型中只有一个变量x，又被称为`单变量线性回归`。 

## 2. Cost Function
+ cost function 

cost function用于评价假设函数(hypothesis function)的准确性,具体采用平方差的方式计算。在上一节中我们知道了假设函数为：
$h_{\theta }(x)=\theta _{0}+\theta _{1}*x$,

$Θ_i$ 是模型参数。每给出一对 $Θ_0$ 和 $Θ_1$ 的值，我们将得到一条直线拟合。我们的目标是：如何选择 $Θ_0$ 和 $Θ_1$ 得到最佳拟合直线，以使得输出变量y到假设函数 直线的垂直距离的和最小。<br>
此处我们需要使用代价函数来求出最佳的$Θ_0$ 和 $Θ_1$:

![cost function](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/cost_function_chapter02.gif)

其中，m是样本数量。这个函数又被称为`squared error function` 或者`mean squared error`。除以2是为了梯度下降（gradient descent）,也有利于导数项的减少。
+ 单变量代价函数

为更直观的了解代价函数的作用和功能，我们先从简化的假设函数开始。我们假设 $θ_0$ 为0，有
$h_{\theta }(x)=\theta _{1}*x$,
$J(\theta _{1})$

$$=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y_{i}}-y_{i})^{^{2}}=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta }(x_{i})-y_{i})^{2}$$
现在我们开始对 $Θ_1$ 进行取值，分别画出假设函数h(x)和代价函数J($θ_1$)的函数。假设我们的训练样本中包含了3个点，即(1,1) (2,2) (3,3)，当θ1=1时，

$h_{\theta }(x)=x$, $h_{\theta }(x_{i})=y_{i}$, 
$J(1)$=0.
在$J(θ_1)$的图像中，横轴代表θ1，于是我们在(1,0)画一个点。如下图：

![θ1=1](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/%CE%B8%3D1.png)

当θ1=0时，如下图：

![θ1=0](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/%CE%B8%3D0.png)

当θ1=0.5时，

$h_{\theta }(x)=0.5x$, $0.5x_{i}=y_{i}$,如下图：

![θ1=0.5](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/%CE%B8%3D0.5.png)

当θ1=-0.5时，假设函数斜率为负，函数单调递减。如下图：

![θ1=-0.5](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/%CE%B8%3D-0.5.png)

继续的，我们可以给θ1赋很多值，以得到$J(θ_1)$的函数图像，
![J(θ1)的图像](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/3%E4%B8%AA%CE%B8%E7%9A%84%E5%8F%96%E5%80%BC%E5%AF%B9%E5%BA%94%E7%9A%84%E6%8B%9F%E5%90%88%E7%9B%B4%E7%BA%BF.png)
得出代价函数的最小值，相应的，我们以此时的$θ_1$为斜率画出的直线就是我们要求的最佳拟合直线。
+ cost function J($θ_0$,$θ_1$)

当代价函数有两个参数时，假设函数是一条不通过原点的直线，我们的任务仍然是最小化 J($θ_0$,$θ_1$)，找到那条最佳拟合直线。下图显示了我们用到的函数和参数：

![functions and parameters](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/functions%20and%20parameters.png)

由于代价函数有两个参数，因此我们绘制出其图像为一个三维曲面，如下图所示，

![J(θ0,θ1)](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/J(%CE%B80%EF%BC%8C%CE%B81).png)

在三维曲面上，我们并不容易直观的找到最小的J($θ_0$,$θ_1$)，为更直观的观察J($θ_0$,$θ_1$)的变化，我们使用contour plots，于是hθ(x)和J($θ_0$,$θ_1$)的图像可以表示如下：

![J(contour plots)](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/contour_plots.png)

图中右侧称为轮廓图，上面标出的紫色的三个点表示J($θ_0$,$θ_1$)值相同。轮廓图有一圈圈的椭圆形构成，每一个圈表示J(θ0,θ1)相同的所有点的集合。我们要找的最小值因该是轮廓图的 一系列同心椭圆形的中心点。图中红色的点的θ0=800，θ=-0.15(大约)，对应左边的蓝色直线，很显然这并不是最优拟合直线，这个点离轮廓图的中心点也远。下图已经十分接近最小点了。

![close to minimizeJ(θ0,θ1)](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/close%20to%20minimizeJ(%CE%B80%2C%CE%B81).png)

以上的图形帮助哦我们直观的了解了假设函数和代价函数。以及如何找到J($θ_0$,$θ_1$)的最小值，得到最佳拟合直线。
下一题视频中，讲介绍一种学习算法，自动找出能使代价函数最小化的参数θ0和θ1的值。

## 3. Gradient descent
use gradient discent to get the minimun of cost function J($θ_0$,$θ_1$)，to estimate the parameters of the hypothesis function 
梯度下降算法不仅用于线性回归，它广泛应用于机器学习的其他方面。下面讲述使用梯度下降算法最小化代价函数J。下面是问题概述：
 > have some function J($θ_0$,$θ_1$)<br>
   want minJ($θ_0$,$θ_1$)<br>
   
事实上，梯度下降算法可应用于更一般的函数，比如cost function可以取多个参数，n=0，1，2，3，……n，求出J(θ0，θ1，……，θn)的最小值。为简洁起见，以下只讲2个参数的情况，构想如下:
>  _outline:_<br>
   start with some θ0，θ1<br>
   keep changing θ0，θ1 to reduce J($θ_0$,$θ_1$) until we hopefully end up at a minimum
 
首先初始化θ0和θ1，比如使他们都等于0；然后一点点改变θ0和θ1，以使得J($θ_0$,$θ_1$)变小，直到找到J的最小值(可能是局部最小值)。
+ 梯度下降算法如何工作
 
 ![graph of J(θ0,θ1)](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/graph%20of%20J(%CE%B80%2C%CE%B81).png)
 
 上面的三维曲面图可以看作一个公园中的两座山，现在我们在坐标所在位置，我们要做的是：360°环顾四周，选择一个方向，能够使我们迈着步子快速的走到山底。<br>
 *选择每一步走多长，朝什么方向移动？*<br>
 起始点的选择会影响局部最小值。如下图：
 
![different local minimum](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/different%20local%20minimum.png)

**梯度下降算法定义**

![gradient descent algorithm](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/gradient%20descent%20algorithm.png)
 
 也就是是说，将不断重复
 
 $$\theta _{j}:=\theta _{j}-\alpha \frac{\partial J(\theta _{0},\theta _{1}))}{\partial \theta _{j}}$$
 
 直到收敛，过程中不断更新参数θj，j=0，1,represents the feature index number.(j is a iteration)
 ```
 符号:=表示赋值(assignment),是一个赋值运算符。符号=表示声明(truth assetion).
 α是一个数字，表示学习速率(learning rate)，控制我们以多大的幅度更新参数θj。
 the partial derivative of J(θ0,θ1) determine the direction in which the step is taken
 ```
**one subtlety about gradient descent-update simultaneously**

需要同时更新θ0和θ1，计算公式的右边部分，赋值到左边，同时更新θ0和θ1.使用temp·来储存计算结果，同时赋值给θj。下图是两种更新方式：
![update θj simultaneously](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/update%20%CE%B8j%20simultaneously.png)

下一小节进入公式的微分项中，计算这个微分项。
### 3.1 gradient descent intuition
+ minJ($θ_1$),$θ_1$∈R

   $θ_1$：=$θ_1$ - α$\frac{\det J(\theta _{1})}{\det \theta _{1}}$

每使用一次$\frac{\det J(\theta _{1})}{\det \theta _{1}}$对$θ_1$求导一次，就更新$θ_1$的值。求导后的值有两种情况：当为负数时，更新后的$θ_1$的值将增大；当为正数时，$θ_1$的值将减小。总之，在不断的更新调整中，最终我们得到一个$θ_1$使得J($θ_1$)的值最小。当α(learning rate)太小，梯度下降将很慢；当α太大，我们有可能跳过最小值点，导致无法收敛，甚至是分散。<br>
当$θ_1$的初始值在一个局部最低点，此处求导后得到0，因此，此时$θ_1$不会更新。它使你的解始终保持在局部最优点。<br>**as we approach a local minimum, gradient descent will automatically take smaller steps.so no need to decrease α over time.* *这是因为在θ1不断接近最低点的过程中，导数是越来越接近0的。就是说随着梯度下降法的运行，$θ_1$移动的幅度会`自动地`变得越来越小，直到最后收敛到局部极小值。<br>下图说明了这样一个过程：

![converge to a local minimum while taking smaller step automatically](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/converge%20to%20a%20local%20minimum%20while%20taking%20smaller%20step%20automatically.png)
接下来，将回到代价函数的本质，即之前所讲的`平方误差函数`，结合梯度下降法，得到机器学习的`第一个算法`——**线性回归算法**。

### 3.2 gradient descent for linear regression
+ gradient descent algorithm and linear regression modle

![gradient descent algorithm and linear regression modle](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/gradient%20descent%20algorithm%20and%20linear%20regression%20modle.png)

接下来，我们将梯度下降法应用到代价函数中，最小化`平方误差代价函数`。先看微分项$\frac{\partial }{\partial \theta _{j}}J(\theta _{0},\theta _{1})$

$$=\frac{\partial \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^{2}}{\partial \theta _{j}}$$

当j=0, 
![θ0 of gradient descent](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/%CE%B80%20of%20gradient%20descent.gif)
当j=1，

![θ1 of gradient descent](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/%CE%B81%20of%20gradient%20descent.gif)
