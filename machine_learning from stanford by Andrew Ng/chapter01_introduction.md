## what is machine learning?
+ 机器学习的2种定义

> Arthur Samuel： the field of study that gives computers the ability to learn 
without being explicitly learned.
（在进行特定编程的情况下，给与计算机学习能力的领域。）

Samuel编写了一个西洋棋程序(checkers playing program)，编程者自己并不是下棋高手，但他让程序自己与自己下了上万盘棋，通过观察哪种布局会赢，哪种会输后，逐渐明白了棋盘布局
的好坏，并且水平超过了Samuel。程序具有足够的耐心，在与自己下棋的过程中，积累了丰富的经验，于是水平不断提高。

> Tom Mitchell: a computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. 
( 一个程序被认为能从经验E中学习，解决任务 T，达到性能度量值P，当且仅当，有了经验E后，经过P评判， 程序在处理 T 时的性能有所提升)
运用tom 的定义，我们可以这样来解释西洋棋程序：<br>
E: the experience of having the program play games against itsself.
T: play checkers
P: the probability that wins the next game

+ 两种主要的学习算法

```
supervised learning: teach computer how to do sth
unsupervised learning: let computer learn by itsself
```
  
others like: reinforcement learning, recommender system.<br>
more specificate definition about main 2 types above are as follow:  

## supervised learning

