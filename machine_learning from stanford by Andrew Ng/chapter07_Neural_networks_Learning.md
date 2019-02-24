介绍一种 给定训练集，为神经网络拟合参数的算法。
## 1 cost function
+ define a few variables

> L = total number of layers in the network,神经网络的总层数<br>
> $s_{l}$ = number of units (not counting bias unit) in layer l，第l层中未包含偏置项的神经计算单元数<br>
> K = number of output units/classes,输出单元数/输出类数
+ cost function

$h_{\Theta }(x)\_{k}$  代表 第k类输出的假设函数，是一个k-dimensional vector。在神经网络中，代价函数是**逻辑回归代价函数的泛化**，为对比，写下逻辑回归的代价函数如下：

![cost_function_for_regularized_logistic_regression](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/cost_function_for_regularized_logistic_regression.png)

在神经网络中，我们的代价函数稍微有些复杂，如下：

![cost_function_for_NN](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/cost_function_for_NN.png)

由于可能面临多分类问题，公式中含有nested summations，嵌套求和先内部遍历求和，然后外部遍历求和。上式中第一项，内部求和求k从1到K的所有每一个逻辑回归算法的代价函数，然后按照输出的顺序，依次相加，其中$y_{k}^{(i)}$表示标签i对应的标注分类。第二项类似于逻辑回归中的正则化项，不同于二分类发问题，权重矩阵$\Theta ^{(L-1)}$不再是一个1×($s_{l}$+1)的矩阵，而应该是K×($s_{l}$+1)的矩阵，也就是 对应
于output layer的参数矩阵$\Theta ^{(L-1)}$是当前矩阵，其行数等于K(最终输出层的节点数，即分类数)，行数等于 （当前层的节点数+1）。
+ note

double sum:只是单纯把 在输出层的每一个单元的逻辑回归代价 相加起来<br>
triple sum:只是单纯把 在整个网络中的所有的个体Θ的平方 相加起来<br>
triple sum中的i不代表训练样本i
## 2 backpropagation
### 2.1 backpropagation algorithm
“backpropagation”是神经网络的专业术语，用于最小化代价函数，即求得$min_{\Theta }J(\Theta )$.直观来理解的话，就是首先计算最后一层的误差，然后再一层一层反向求出各层的误差，直到倒数第二层（第一层不存在误差）。我们将此转换为求代价函数偏导数问题，即求$\frac{\partial J(\Theta )}{\partial \Theta \_{i,j}^{(l)}}$.**接下来讲解如何求这个偏导数**：讲义图为

![compute_partial_derivative_of_J(Θ)_in_NN](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/compute_partial_derivative_of_J(%CE%98)_in_NN.png) 

假设给定一个标注训练集$\\{(x^{(1)},y^{(1)})\cdots (x^{(m)},y^{(m)})\\}$,
+ 对于所有的l，i，j，设定 $\Delta \_{i,j}^{(l)}:= 0$；for training example t=1 to m:
+ 1.set $a^{(1)}:=x^{(t)}$
+ 2.perform forward propagation to compute $a^{(l)}$ for l=2,3,……,L。
假设我们的网络总共只有四层，即L=4，且只有一个训练实例($x^{(1)},y^{(1)}$), 输出类别K=4，如下图：

![forward_propagation_in_NN](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/forward_propagation_in_NN.png)

Forward propagation：：
>+ $a^{(1)} =x$
>+ $a^{(2)}= g(z^{(2)})$,  $z^{(2)}=\Theta ^{1}a^{1}$,  add $a\_{0}^{2}$
>+ $a^{(3)}= g(z^{(3)}),  z^{(3)}=\Theta ^{2}a^{2}$,  add $a\_{0}^{3}$
>+ $a^{(4)}= h_{\Theta }(x)=g(z^{(4)}),  z^{(4)}=\Theta ^{3}a^{3}$
+ 3.using $y_{j}^{(t)}$（实际值），compute $\delta \_{j}^{(L)}=a_{j}^{(L)}-y_{j}^{(t)}$, $\delta \_{j}^{(L)}$代表*激活单元的预测与实际值之间的误差*， 其维度为(输出层个数×1).
> 为得到最后一层之前的所有Δ值，我们使用一个公式来从右往左反向计算
+ 4.compute $\delta ^{(L-1)},\delta ^{(L-2)},\cdots ,\delta ^{(2)}$,
    $$\delta ^{(l)}=((\Theta ^{(l)})^\mathrm{T}\delta ^{(l+1)}).\ast a^{(l)}.\ast (1-a^{(l)}),$$
> 上式可解释如下：第l层的δ值等于 下一层(l+1 layer)的δ值乘以l层的Θ矩阵，然后再点乘函数${g}'$,它是关于$z^{(l)}$的导数，${g}'(z^{(l)})=a^{(l)}.\*(1-a^{(l)})$.假设我们现在只有一个训练实例($x^{(1)},y^{(1)}$),首先有代价函数为：
$J(\theta )=-ylogh(x)-(1-y)log(1-h(x))$，计算误差就是求J(θ)关于z的偏导数，计算公式为：
$$\delta^{(l)}=\frac{\partial J(\theta )}{\partial z^{(l)}},$$ 使用链式法则，我们先推导$\delta^{(3)}$,$\delta^{(2)}$:
$$\delta^{(3)}=\frac{\partial J(\theta )}{\partial z^{(3)}}=\frac{\partial J(\theta )}{\partial a^{(4)}}\cdot \frac{\partial a^{(4)}}{\partial z^{(4)}}\cdot \frac{\partial z^{(4)}}{\partial a^{(3)}}\cdot \frac{\partial a^{(3)}}{\partial z^{(3)}},$$
![errorCalculationInBPA](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/errorCalculationInBPA.png)

通过上述过程，我们发现反向传播的过程可以从**偏导数的链式求导**过程得到：
$$\frac{\partial J(\theta )}{\partial a}\cdot\frac{\partial a}{\partial z},$$同样我们可以得到结果a-y.
+ 5.我们设置Δ的更新公式：
$$\Delta \_{i,j}^{(l)}:=\Delta \_{i,j}^{(l)} + a\_{j}^{(l)}\delta \_{i}^{(l+1)},$$ 
因此，我们Δ矩阵更新公式为：
>+ $D_{i,j}^{(l)}:=\frac{1}{m}(\Delta \_{i,j}^{(l)}+\lambda \Theta \_{i,j}^{(l)}),$if j ≠ 0.
>+ $D_{i,j}^{(l)}:=\frac{1}{m}\Delta \_{i,j}^{(l)},$ if j = 0.

大写的Δ的矩阵D被当作一个**累加器**，当我们不断迭代时，它将l层的所有最终迭代值加起来，最终计算我们的偏导数。因此我们有
$$\frac{\partial J(\Theta )}{\partial \Theta \_{i,j}^{(l)}}=D_{i,j}^{(l)}$$


### 2.2 backpropagation intuition 
#### 2.2.1 forward propagation
首先，我们复习一下前向传播到底做了哪些工作：
![a_closer_look_at_into_forward_propagation](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/a_closer_look_at_into_forward_propagation.png)
如图，神经网络有2个隐藏层，$z^(i)$代表输入层经过加权变换后得到的值，$a^(i)$代表sigmoid激活函数，经过压缩后的值属于(0，1)，经过最后一次加权后再使用sigmoid函数操作，得到最终的输出结果。图中给出了 使用前向传播计算 $z_{1}^{(3)}$的步骤。
#### 2.2.2 what is backpropagation doing？
+ 假设我们只有一个样本实例($x^{(i)},y^{(i)}$):
![backpropagation_with_a_single_example](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/backpropagation_with_a_single_example.png)

一种直观理解反向传播的思想是：计算所有的$\delta \_{j}^{(l)}$，可以把这些项看作是激励值的误差。更正式的说法是：$\delta \_{j}^{(l)}$是cost(i)关于$z_{j}^{(l)}$的偏导数。此时的神经网络图如下：
![forward_propagation_and_backpropagation](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/forward_propagation_and_backpropagation.png)

我们以$\delta \_{2}^{(2)},\delta \_{2}^{(3)}$的计算为例：$$\delta \_{2}^{(2)}=\Theta \_{12}^{(2)}\delta \_{1}^{(3)}+\Theta \_{22}^{(2)}\delta \_{2}^{(3)},$$
$$\delta \_{2}^{(3)}=\Theta \_{12}^{(3)}\delta \_{1}^{(4)}$$
## 3 backpropagation in practice
### 3.1 implementation note:unrolling parameters matrices into vectors
![example_in_octave](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/example_in_octave.png)

![learning_algorithm_implementation_with_octave](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/learning_algorithm_implementation_with_octave.png)
### 3.2 gradient checking
每次在实现反向传播，或者其他类似的梯度下降算法时，都可以使用梯度检查，以确信这些算法的正确性。
+ 当θ是一个向量参数时：
 ![partial_derivative_respect_to_θ_vector](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/partial_derivative_respect_to_%CE%B8_vector.png)
 
 ![gradient_descent_checking_implementation_with_octave](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/gradient_descent_checking_implementation_with_octave.png)
 
 使用梯度检查时，有一点需要注意:在训练分类器之前，应确保关掉你的梯度检查，这是因为*运行梯度检查的计算量十分大，会导致程序运行很慢*，而反向传播算法是一个比梯度检验更快的计算导数的方法，因此，*一旦你确定了反向传播的实现是正确的，要确定在训练算法时关掉梯度检验。*
### 3.3 random initialization
在逻辑回归中，将θ的初始值设置为0是可行的，但这在训练神经网络的时候，却并不可行。此时，所有的权重都相同，阻止了神经网络进行有效的学习。

![zero_initialization_case](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/zero_initialization_case.png)

为解决这个问题，在神经网络的训练中，使用随机初始化参数的方法，以打破上图的对称性(symmetry breaking).如下图：

![random_initialization](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/random_initialization.png)
### 3.4 putting it together
+ training a neural network-neural network architecture
![neural network architecture](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/neural%20network%20architecture.png)
+ steps to train a neural network
> 1. randomly initialize weights
> 2. implement forward propagation to get $h_{\theta }(x^{(i)})$ for any $x^{(i)}$
> 3. implement code to compute cost function J(Θ)
> 4. implement backpropagation to compute partial derivatives $\frac{\partial J(\Theta )}{\partial \Theta \_{jk}^{l}}$
在第4步中，具体地，我们使用一个for循环遍历m个样本点，对每一个样本点，我们都使用向前传播和反向传播进行迭代，得到神经网络中每一层中每一个单元对应的激励值$a^{(l)}$和δ值。
> 5. use gradient checking to compare $\frac{\partial J(\Theta )}{\partial \Theta \_{jk}^{l}}$ vs. using numerical estimate of gradient of J(Θ) 
> 6. use gradient descent or advanced optimization method with backpropagation to try to minimize J(Θ) as a function of parameters Θ
## 4 application of neural networks-autonomous driving
just see the video
