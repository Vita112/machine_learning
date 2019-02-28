参考博文链接1：[Adaboost算法的原理及推导](https://blog.csdn.net/v_july_v/article/details/40718799)<br>
链接2：[Adaboost算法原理分析及实践+代码](https://blog.csdn.net/guyuealian/article/details/70995333)<br>
链接3：[详解boosting系列算法——Adaboost](https://blog.csdn.net/weixin_38629654/article/details/80516045)<br>
链接4：[统计学习那些事儿](https://cosx.org/2011/12/stories-about-statistical-learning)<br>
链接5：[统计学习精要(The Elements of Statistical Learning)](http://www.loyhome.com/%E2%89%AA%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E7%B2%BE%E8%A6%81the-elements-of-statistical-learning%E2%89%AB%E8%AF%BE%E5%A0%82%E7%AC%94%E8%AE%B0%EF%BC%88%E5%8D%81%E5%9B%9B%EF%BC%89/)<br>
链接6：[直观理解Adaboost算法过程及特征(有图)](https://blog.csdn.net/m0_37407756/article/details/67637400)<br>
[PPT1_decission_tree_and_adaboost]()<br>
[PPT2_variation_principle_and_adaboost]()
## 1 提升（boosting）方法
### 1.1 基本思路
   boosting方法在分类问题中，通过改变训练样本的权重，学习多个基本分类器，最后通过线性组合提高分类性能，得到最终分类器。首先引入两个概念：**strongly learnable**和**weakly learnable**，关于强可学习：a problem is learnable or strongly learnable if there exists an algorithm that outputs a learner *h* in polynomial time such that for all δ ＞0， ε≤0.5,![strongly_learnable](https://github.com/Vita112/machine_learning/blob/master/img/strongly_learnable.gif)

也就是说，存在一个多项式算法，能够有很大的把握得到一个误差很小的模型。关于弱可学习：一个概念，存在一个多项式的学习算法能够学习他，学习的正确率仅比随机猜测的稍好一些，称这个概念为弱可学习的。1990年，Schapire证明，在PAC学习的框架下，强可学习与弱可学习是可以等价的。在boosting方法中，需要解决2个问题：
 > ① *在每一轮，如何改变训练数据的权重分布；* <br>
 > ② *如何将弱分类器组合成一个强分类器。*<br>
下面的adaboost方法较好的解决了这2个问题。
## 2 Adaboost 原理
### 2.1 什么是Adaboost？
Adaboost，英文全称为‘Adapltive Boosting’(自适应增强)，是一种将弱学习器提升为强学习器的集成学习算法。它通过改变训练样本的权值，学习多个分类器，然后将分类器进行线性组合成强分类器。具体的，**改变训练数据的权重分布**：提高前一轮训练中被错误分类的数据的权值，降低正确分类数据的权值，使得被错误分类的数据在下一轮训练中更受关注；然后根据不同分布调弱学习算法得到一系列弱分类器；再**使用加权多数表决的方法，将弱分类器进行线性组合**，具体组合方法是：误差率小的分类器，增大其权值；误差率大的分类器，减小其权值。

**Adaboost算法步骤：**
>+ 1. 初始化训练数据的权值分布。当样本数为N时，每一个训练样本的初始权值为：1/N.
>+ 2. 训练弱分类器。在构造下一个训练集中，降低被正确分类的样本的权值，提高被错误分类的样本的权值。权值更新后的样本集被用于训练下一个分类器，训练过程迭代。
>+ 3. 使用加权多数表决方法，组合训练得到的弱分类器，得到强分类器。提高分类误差率小的弱分类器的权重，降低分类误差率大的弱分类器的权重。

### 2.2 Adaboost算法流程
给定一个训练数据集 $T=\{{(x_{1},y_{1}),(x_{2},y_{2}),\cdots ,(x_{N},y_{N})\}}$, 其中 $y_{i}\subset \{{-1,1\}}.$
流程如下：
+ 步骤1：初始化训练样本的权值分布，如下：
$$ D_{1}= (w_{11},w_{12},\cdots ,w_{1i},\cdots ,w_{1N}),\\
w_{1i}=\frac{1}{N}, i=1,2,\cdots ,N $$
+ 步骤2：反复学习多个弱分类器，多轮迭代执行以下操作。m表示迭代次数，且m = 1，2，……，M
> a. 使用具有权重分布$D_{m}$的训练数据集学习，得到基本分类器：
$$G_{m}(x):\chi \rightarrow {-1,+1},$$
b. 计算$G_{m}(x)$在训练数据集上的分类误差率：
$$e_{m}=P(G_{m}(x)\neq y_{i})=\sum_{i=1}^{N}w_{mi}I(G_{m}(x)\neq y_{i}),$$
c. 计算$G_{m}(x)$的系数：
$$\alpha \_{m}=\frac{1}{2}log\frac{1-e_{m}}{e_{m}},$$

**通过公式我们有：当 $e_{m}\leq \frac{1}{2}$ 时,随着 $e_{m}$ 越小, $\alpha \_{m}$ 会逐渐增大**.
也就是说，给予在训练数据集上误差率小的弱分类器较高的权值。<br>
d. 更新训练数据集的权重分布：
$$ D_{m+1}= (w_{m+1,1},w_{m+1,2},\cdots ,w_{m+1,i},\cdots ,w_{m+1,N}), $$
$$ w_{m+1,i}=\frac{w_{mi}}{Z_{m}}exp(-\alpha \_{m}y_{i}G_{m}(x_{i})), i=1,2,\cdots ,N $$
$$Z_{m}=\sum_{i=1}^{N}w_{mi}exp(-\alpha \_{m}y_{i}G_{m}(x_{i})).$$
上面，$Z_{m}$是规范化因子，我们也可以通过以下步骤将其化简：
$$Z_{m}=\sum_{i=1}^{N}w_{mi}exp(-\alpha \_{m}y_{i}G_{m}(x_{i}))\\\\
=\sum_{i=1}^{N}w_{mi}e^{-\alpha \_{m}}I(G_{m}(x_{i})= y_{i})+\sum_{i=1}^{N}w_{mi}e^{\alpha \_{m}}I(G_{m}(x_{i})\neq y_{i})\\\\
=(1-e_{m})e^{-\alpha \_{m}}+e_{m}e^{\alpha \_{m}},$$
又
$$\alpha \_{m}=\frac{1}{2}log\frac{1-e_{m}}{e_{m}},$$
代入上式化简后得到：
$$Z_{m}=2\sqrt{e_{m}(1-e_{m})}$$

+ 步骤3：使用加权多数表决方法，得到基本分类器的线性组合：
$$f(x)=\sum_{m=1}^{M}\alpha \_{m}G_{m}(x),$$
于是，最终的分类器$G(x)$:
$$G(x)=sign(f(x))=sign(\sum_{m=1}^{M}\alpha \_{m}G_{m}(x)).$$

### 2.3 AdaBoost 算法的训练误差分析
+ 定理1 (AdaBoost 的训练误差界)AdaBoost 算法最终分类器的训练误差界为：
$$\frac{1}{N}\sum_{i=1}^{N}I(G(x_{i})\neq y_{i})\leq \frac{1}{N}\sum_{i}exp(-y_{i}f(x_{i}))=\prod_{m}Z_{m}.$$
**证明**：<br>
上式前半部分：<br>
当$G(x_{i})\neq y_{i}$时，$y_{i}f(x_{i})<0$,于是，$exp(-y_{i}f(x_{i}))\geq 1$.<br>
上式后半部分：<br>
![to_get_error_boundary_of_AdaBoost_algorithm](https://github.com/Vita112/machine_learning/blob/master/img/to_get_error_boundary_of_AdaBoost_algorithm.gif)

**这个定理说明：可以在每一轮选取适当的$G_m$，使得$Z_m$最小，从而使得训练误差下降最快**.

## 3 加法模型和前向分步算法
### 3.1 加法模型 additive model
考虑加法模型
$$f(x)=\sum_{m=1}^{M}\beta \_{m}b(x;\gamma \_{m}),$$
其中，$b(x;\gamma \_{m})$ 为基函数，$\gamma \_{m}$ 为基函数的参数, $\beta \_{m}$ 为基函数的系数。在给定训练数据及损失函数L(y,f(x))的条件下，学习加法模型成为经验风险极小化，即损失函数极小化问题：
$$min_{(\beta \_{m},\gamma \_{m})}\sum_{i=1}^{N}L(y_{i},\sum_{m=1}^{M}\beta \_{m}b(x;\gamma \_{m}))$$

我们可以使用forward stagewise algorithm 简化这个复杂的优化问题。
具体地，从前向后，每一次只学习一个基函数及其系数，即每步只需优化如下损失函数：

$$min_{(\beta ,\gamma)}\sum_{i=1}^{N}L(y_{i},\beta b(x;\gamma))$$

### 3.2 前向分步算法forward stagewise algorithm
> input：训练数据集$T={(x_{1},y_{1}),(x_{2},y_{2}),\cdots (x_{N},y_{N})};$ 损失函数L(y,f(x));基函数集{b(x;γ)}<br>
> ouput:加法模型 f(x)

+ 1. 初始化$f_{0}(x)=0$
+ 2. 对于m=1,2,……,M
> a. 极小化损失函数
$$(\beta \_{m},\gamma \_{m})=arg\ min_{(\beta ,\gamma )}\sum_{i=1}^{N}L(y_{i},f_{m-1}(x_{i})+\beta b(x_{i};\gamma ))$$
得到参数$\beta \_{m},\gamma \_{m}).$<br>
> b. 更新
$$f_{m}(x)=f_{m-1}(x_{i})+\beta_{m} b(x_{i};\gamma_{m}).$$

+ 3. 得到加法模型
$$f(x)=f_{M}(x)=\sum_{m=1}^{M}\beta_{m} b(x_{i};\gamma_{m}).$$

### 3.3 前向分步算法与AdaBoost算法
AdaBoost algorithm 是前向分步算法的特例，其中，模型是由基本分类器组成的加法模型，损失函数为exp损失函数(exponential loss function):
$$L(y,f(x))=exp(-yf(x)).$$在AdaBoost算法中，第m轮迭代后的模型可以表示为：
$$f_{m}(x)=f_{m-1}(x)+\alpha \_{m}G_{m}(x).$$
**我们的优化目标是:通过前向分步算法得到的$\alpha \_{m}$, $G_{m}(x)$，使得$f_{m}(x)$在训练数据及上的指数损失函数最小**，即
$$(\alpha \_{m},G_{m}(x))=arg\ min_{(\alpha ,G)}\sum_{i=1}^{N}exp\[-y_{i}(f_{m-1}(x_{i})+\alpha G(x_{i}))],$$
通过简化，我们得到
$$(\alpha \_{m},G_{m}(x))=arg\ min_{(\alpha ,G)}\sum_{i=1}^{N}\bar{w_{mi}}exp\[-y_{i}\alpha G(x_{i}))],$$
其中，$\bar{w_{mi}}$可以表示为：

![bar{w_m-1,i}](https://github.com/Vita112/machine_learning/blob/master/img/bar%7Bw_m-1%2Ci%7D.gif)

**下面分别求出使得损失函数最小的$\alpha \_{m}$, $G_{m}(x)$，** 首先，
+ 1. 求$G_{m}^{\*}(x)$：

实际上，求最优的$G_{m}^{\*}$，就是AdaBoost算法的基本分类器，因为它是使得 第m轮加权训练数据分类误差最小的基本分类器。公式表示如下：
$$G_{m}^{\*}(x)=arg\ min(G)\sum_{i=1}^{N}\bar{w_{mi}}I(y_{i}\neq G(x_{i})).$$
+ 2. 求$\alpha \_{m}^{\*}$, 即 求损失函数$L(y_{i},f(x_{i}))$对$\alpha \_{m}$的偏导数，并令其为零。首先给出指数损失函数如下：

![adaboost_Loss_function](https://github.com/Vita112/machine_learning/blob/master/img/adaboost_Loss_function.gif)

对 $\alpha \_{m}$ 求偏导并使导数为0，有
$$(e_{m}-1)e^{-\alpha \_{m}}+e_{m}e^{\alpha \_{m}}=0,$$
即
$$e_{m}-1+e_{m}e^{2\alpha \_{m}}=0,$$
$$e^{2\alpha \_{m}}=\frac{1-e_{m}}{e_{m}},$$
最后得到，
$$\alpha \_{m}^{\*}=\frac{1}{2}log\frac{1-e_{m}}{e_{m}}.$$
我们发现，求解出来的最优的$\alpha \_{m}^{\*}$就是 AdaBoost算法中的基本分类器的系数。
+ 3. 权值更新

在简化后的损失函数中，我们令
$$\bar{w_{mi}}=exp\[-y_{i}f_{m-1}(x_{i})],$$
根据上文中的推导，我们很容易得到
$$\bar{w_{m+1,i}}=\bar{w_{m,i}}exp\[-y_{i}\alpha \_{m}G_{m}(x)].$$
这与adaboost算法的样本权值更新规则一致，只相差规范化因子，因而**等价**。
