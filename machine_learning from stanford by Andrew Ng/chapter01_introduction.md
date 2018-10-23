## 1. what is machine learning?
+ 1.1 机器学习的2种定义

> Arthur Samuel： the field of study that gives computers the ability to learn 
without being explicitly learned.
（在进行特定编程的情况下，给与计算机学习能力的领域。）

Samuel编写了一个西洋棋程序(checkers playing program)，编程者自己并不是下棋高手，但他让程序自己与自己下了上万盘棋，通过观察哪种布局会赢，哪种会输后，逐渐明白了棋盘布局
的好坏，并且水平超过了Samuel。程序具有足够的耐心，在与自己下棋的过程中，积累了丰富的经验，于是水平不断提高。

> Tom Mitchell: a computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. 
( 一个程序被认为能从经验E中学习，解决任务 T，达到性能度量值P，当且仅当，有了经验E后，经过P评判， 程序在处理 T 时的性能有所提升)

运用tom 的定义，我们可以这样来解释西洋棋程序：<br>
E: the experience of having the program play games against itsself.<br>
T: play checkers<br>
P: the probability that wins the next game

+ 1.2 两种主要的学习算法

```
supervised learning: teach computer how to do sth
unsupervised learning: let computer learn by itsself
```
  
other algorithms like: reinforcement learning, recommender system.<br>
more detailed definition of the above 2 algorithms is as follows:  

## 2. supervised learning
在监督学习中，数据的输入和输出都是确定的，对于每一个输入的变量，都存在与之对应的正确输出。
观察输入变量和输出变量，我们可以得到一个关于输入和输出之间的对应关系。
监督学习问题又可分为“回归”问题和“分类”问题。在回归问题中，所有输入变量对应的输入值是连续，
我们可以使用某种函数来描述它；在分类问题中，输入变量得到的输出值被明显的划分为几类，在这几类
输出值间有着明显的界限，我们在一个离散型输出值中预测结果。
+ 2.1 引例-房价预测-回归问题

![房价预测](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/02_%E6%88%BF%E4%BB%B7%E9%A2%84%E6%B5%8B_%E5%9B%9E%E5%BD%92%E9%97%AE%E9%A2%98.jpg)

上图的房价预测是一个监督学习算法中回归问题的例子。从图可以看出，对于每一个给定的输入值，其输出值在图表中的分布是连续的。
观察这种关系，我们可以大致估算出，在模型数据的输入值范围内，某一个未给定输出值的输入值的输出值。

+ 2.2 肿瘤性质预测-分类问题

![](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/02_%E8%82%BF%E7%98%A4%E8%89%AF%E6%80%A7%E6%81%B6%E6%80%A7%E9%A2%84%E6%B5%8B_%E5%88%86%E7%B1%BB%E9%97%AE%E9%A2%98.jpg)

上图显示的是一个分类问题，使用所获取的数据，我们得到一个关于 肿瘤的大小与其是良性/恶性的 离散型分布图。这个图帮助我们
预测未给定确定输出值的 输入值的预测输出结果。

+ 2.3 多特征分类问题

![](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/02_%E5%A4%9A%E7%89%B9%E5%BE%81%E5%88%86%E7%B1%BB%E9%97%AE%E9%A2%98.jpg)

在现实世界中，一个实际问题往往不知包含2到3个特征，有时甚至包含无穷多个特征，这时，
我们的算法也必须要能够处理多个特征，多个属性。在后来的章节中，将会介绍一种叫做
`支持向量机`的算法，支持电脑处理无穷多的特征。

## 3. unsupervised learning
 在无监督学习算法中，所有的数据都是一样的，数据没有被给定属性或标签。我们需要做的，就是
 从这个数据集中`找到某种结构`。我们使用聚类的方法。
 
 + 3.1 google news：Cluster
 
 task：搜索网页上的成千上万条新闻，然后按照不同的主题，`自动地`将它们聚合在一起显示出来。<br>
另一个是关于基因芯片的例子。假设有一组不同个体的基因数据集，我们的任务是：根据这些数据把不同的个体
归入不同的类或不同类型的人。数据不会告诉你有哪几类人，也不会告诉你哪个人属于哪一类。我们必须通过对
所给的数据进行cluster，得到某种数据结构，然后`自动地`按得到的结构对个体进行分类。

+ 3.2 应用

  + 组织大型地计算机集群
  + 分析社交网络
  + 自动将客户分到不同的细分市场
  
+ 3.3 鸡尾酒就会问题(cocktail party problem algorithm)

在一个鸡尾酒宴会上，同一时间有多个人同时说话，可能还混杂有音乐/笑声等其他声音，要如何识别单个人的说话内容？
我们假设有两个人在一个屋子里，在屋子的不同位置摆放了两个麦克风， 现在让这两个人同时说话，
使用麦克风记录下他们的声音。我们如何分理出这两个人的声音？

```
[W,s,v] = svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x')
```
这枚课程要求使用octave编程环境，在octave中实现算法。
