## 1 motivations
### 1.1 non-linear hypotheses
一个事实：现实生活中，我们需要非线性模型来解决更加复杂的问题。
+ 引例
假设现在有一个监督学习下的分类问题，数据只包含2个特征，此时使用高阶多项式作为我们的假设模型，构造一个包含很多非线性项的逻辑回归函数$h_{\theta }(x)=g(z)=g(\frac{1}{1+e^{-(\mathbf{\theta }^\mathrm{T}x)}})$,
z是一个高阶非线性函数，g(z)是sigmoid函数，这个模型可以将训练数据分为两类，完成分类任务。**但现实世界中，特征数总是大于2的，有时可能非常多**，比如假设有n=100个特征,此时特征二次项的个数为：
$\frac{n^{2}}{2!}$=5000，三次项特征数为$\frac{n^{3}}{3!}$=17000。如果我们使用逻辑回归算法模型，如此多的特征项极易导致过拟合问题，而且计算量庞大computationally expensive。我们发现：
随着初始特征数的增加，高阶多项式的项数以几何级数的速度增加。如果我们舍弃特征项的某些部分，比如只取二次项特征的一个子集，假设只取$x_{1}^{2},x_{2}^{2},\cdots ,x_{100}^{2}$，最后我们将得到一个类似椭圆的分类决策边界，显然这不是一个
好的模型，因为它忽略了太多的特征信息。
### 1.2 neurons and the brainx_{1}^{2}
## 2 neural networks
### 2.1 model representation Ⅰ
### 2.2 model representation Ⅱ
## 3 applications
### 3.1 examples and intuitions Ⅰ
### 3.2 examples and intuitions Ⅱ
### 3.3 multiclass classification
