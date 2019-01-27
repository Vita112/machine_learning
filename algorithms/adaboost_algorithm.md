参考博文链接1：[Adaboost算法的原理及推导](https://blog.csdn.net/v_july_v/article/details/40718799)<br>
链接2：[Adaboost算法原理分析及实践+代码](https://blog.csdn.net/guyuealian/article/details/70995333)<br>
链接3：[详解boosting系列算法——Adaboost](https://blog.csdn.net/weixin_38629654/article/details/80516045)<br>
链接4：[统计学习那些事儿](https://cosx.org/2011/12/stories-about-statistical-learning)<br>
链接5：[统计学习精要(The Elements of Statistical Learning)](http://www.loyhome.com/%E2%89%AA%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E7%B2%BE%E8%A6%81the-elements-of-statistical-learning%E2%89%AB%E8%AF%BE%E5%A0%82%E7%AC%94%E8%AE%B0%EF%BC%88%E5%8D%81%E5%9B%9B%EF%BC%89/)<br>
链接6：[直观理解Adaboost算法过程及特征(有图)](https://blog.csdn.net/m0_37407756/article/details/67637400)<br>
[PPT1_decission_tree_and_adaboost]()<br>
[PPT2_variation_principle_and_adaboost]()
## 1 Adaboost 原理
### 1.1 什么是Adaboost？
Adaboost，英文全称为‘Adapltive Boosting’(自适应增强)，是一种将弱学习器提升为强学习器的集成学习算法。它通过改变训练样本的权值，学习多个分类器，然后将分类器进行线性组合成强分类器。
具体的，提高前一轮训练中被错误分类的数据的权值，降低正确分类数据的权值，使得被错误分类的数据在下一轮训练中更受关注；然后**根据不同分布调弱学习算法**得到一系列弱分类器；再将弱分类器进行线性组合，具体组合方法是：误差率小的分类器，增大其权值；误差率大的分类器，减小其权值。
**Adaboost算法步骤：**
>+ 1. 初始化训练数据的权值分布。当样本数为N时，每一个训练样本的初始权值为：1/N.
>+ 2. 训练弱分类器。在构造下一个训练集中，降低被正确分类的样本的权值，提高被错误分类的样本的权值。权值更新后的样本集被用于训练下一个分类器，训练过程迭代。
>+ 3. 组合训练得到的弱分类器，得到强分类器。提高分类误差率小的弱分类器的权重，降低分类误差率大的弱分类器的权重。

### 1.2 Adaboost算法流程
给定一个训练数据集 $T=\{{(x_{1},y_{1}),(x_{2},y_{2}),\cdots ,(x_{N},y_{N})\}}$,其中$y_{i}\subset \{{-1,1\}}$.流程如下：
+ 步骤1：初始化训练样本的权值分布，如下：
$$D_{1}= (w_{11},w_{12},\cdots ,w_{1i},\cdots ,w_{1N}),\\w_{1i}=\frac{1}{N},i=1,2,\cdots ,N$$
+ 步骤2：多轮迭代，m表示迭代次数且m = 1，2，……，M

