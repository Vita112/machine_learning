介绍一种 给定训练集，为神经网络拟合参数的算法。
## 1 cost function
+ define a few variables
```
L = total number of layers in the network,神经网络的总层数
$s_{l}$ = number of units (not counting bias unit) in layer l，第l层中未包含偏置项的神经计算单元数
K = number of output units/classes,输出单元数/输出类数
```
+ cost function

$h_{\Theta }(x)\_{k}$  代表 第k类输出的假设函数，是一个k-dimensional vector。在神经网络中，代价函数是**逻辑回归代价函数的泛化**，为对比，写下逻辑回归的代价函数如下：
![cost_function_for_regularized_logistic_regression]()
在神经网络中，我们的代价函数稍微有些复杂，如下：
![cost_function_for_NN]()
由于可能面临多分类问题，公式中含有nested summations，嵌套求和先内部遍历求和，然后外部遍历求和。上式中第一项，内部求和求k从1到K的所有每一个逻辑回归算法的代价函数，然后按照输出的顺序，依次相加，其中$y_{k}^{(i)}$表示标签i对应的标注分类。第二项类似于逻辑回归中的正则化项，不同于二分类发问题，权重矩阵$\Theta ^{(L-1)}$不再是一个1×($s_{l}$+1)的矩阵，而应该是K×($s_{l}$+1)的矩阵，也就是 对应于output layer的参数矩阵$\Theta ^{(L-1)}$是当前矩阵，其行数等于K(最终输出层的节点数，即分类数)，行数等于 （当前层的节点数+1）。
+ note

double sum:只是单纯把 在输出层的每一个单元的逻辑回归代价 相加起来<br>
triple sum:只是单纯把 在整个网络中的所有的个体Θ的平方 相加起来<br>
triple sum中的i不代表第i个训练样本
## 2 backpropagation
### 2.1 backpropagation algorithm

### 2.2 backpropagation intuition 
## 3 backpropagation in practice
### 3.1 implementation note:unrolling parameters
### 3.2 gradient checking
### 3.3 random initialization
### 3.4 putting it together
## 4 application of neural networks-autonomous driving
