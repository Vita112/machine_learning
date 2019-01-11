介绍一种 给定训练集，为神经网络拟合参数的算法。
## 1 cost function
+ define a few variables

> L = total number of layers in the network,神经网络的总层数<br>
> $s_{l}$ = number of units (not counting bias unit) in layer l，第l层中未包含偏置项的神经计算单元数<br>
> K = number of output units/classes,输出单元数/输出类数
+ cost function

$h_{\Theta }(x)\_{k}$  代表 第k类输出的假设函数，是一个k-dimensional vector。在神经网络中，代价函数是**逻辑回归代价函数的泛化**，为对比，写下逻辑回归的代价函数如下：
![cost_function_for_regularized_logistic_regression]()
在神经网络中，我们的代价函数稍微有些复杂，如下：
![cost_function_for_NN]()
由于可能面临多分类问题，公式中含有nested summations，嵌套求和先内部遍历求和，然后外部遍历求和。上式中第一项，内部求和求k从1到K的所有每一个逻辑回归算法的代价函数，然后按照输出的顺序，依次相加，其中$y_{k}^{(i)}$表示标签i对应的标注分类。第二项类似于逻辑回归中的正则化项，不同于二分类发问题，权重矩阵$\Theta ^{(L-1)}$不再是一个1×($s_{l}$+1)的矩阵，而应该是K×($s_{l}$+1)的矩阵，也就是 对应
于output layer的参数矩阵$\Theta ^{(L-1)}$是当前矩阵，其行数等于K(最终输出层的节点数，即分类数)，行数等于 （当前层的节点数+1）。
+ note

double sum:只是单纯把 在输出层的每一个单元的逻辑回归代价 相加起来<br>
triple sum:只是单纯把 在整个网络中的所有的个体Θ的平方 相加起来<br>
triple sum中的i不代表第i个训练样本
## 2 backpropagation
### 2.1 backpropagation algorithm
“backpropagation”是神经网络的专业术语，用于最小化代价函数，即求得$min_{\Theta }J(\Theta )$.我们将此转换为求代价函数偏导数问题，即求$\frac{\partial J(\Theta )}{\partial \Theta \_{i,j}^{(l)}}$.**接下来讲解如何求这个偏导数**：讲义图为
![compute_partial_derivative_of_J(Θ)_in_NN]() 

假设给定一个标注训练集$\\{(x^{(1)},y^{(1)})\cdots (x^{(m)},y^{(m)})\\}$,
+ 对于所有的l，i，j，设定 $\Delta \_{i,j}^{(l)}:= 0$；for training example t=1 to m:
+ 1.set $a^{(1)}:=x^{(t)}$
+ 2.perform forward propagation to compute $a^{(l)}$ for l=2,3,……,L
假设我们的网络如下：
![forward_propagation_in_NN]()
进行前向传播后有：
>+ $a^{(1)} =x$
>+ $a^{(2)}= g(z^{(2)})$,$z^{(2)}=\Theta ^{1}a^{1}$,add $a\_{0}^{2}$
>+ $a^{(3)}= g(z^{(3)}),z^{(3)}=\Theta ^{2}a^{2}$,add $a\_{0}^{3}$
>+ $a^{(4)}= h_{\Theta }(x)=g(z^{(4)}),z^{(4)}=\Theta ^{3}a^{3}$
+ 3.using $y_{j}^{(t)}$，compute $\delta \_{j}^{(L)}=a_{j}^{(L)}-y_{j}^{(t)}$, $\delta \_{j}^{(L)}$的维度为(输出层个数×1).
> 为得到最后一层之前的所有Δ值，我们使用一个公式来从右往左反向计算
+ 4.compute $\delta ^{(L-1)},\delta ^{(L-2)},\cdots ,\delta ^{(2)},\delta ^{(l)}=((\Theta ^{(l)})^\mathrm{T}\delta ^{(l+1)}).*a^{(l)}.*(1-a^{(l)})$
> 上式可解释如下：第l层的δ值等于 下一层(l+1 layer)的δ值乘以l层的Θ矩阵，然后再点乘函数${g}'$,它是关于$z^{(l)}$的导数，${g}'(z^{(l)})=a^{(l)}.\*(1-a^{(l)})$.假设我们现在只有一对样本数据(x,y),反向传播的过程其实可以从**偏导数的链式求导**过程得到：
$$\frac{\partial J(\theta )}{\partial a}\cdot\frac{\partial a}{\partial z},$$同样我们可以得到结果a-y.
+ 5.我们设置Δ的更新公式：$$\Delta _{i,j}^{(l)}:=\Delta _{i,j}^{(l)}+a_{j}^{(l)}\delta \_{i}^{(l+1)},$$ 
因此，我们Δ矩阵更新公式为：
>+ $D_{i,j}^{(l)}:=\frac{1}{m}(\Delta \_{i,j}^{(l)}+\lambda \Theta \_{i,j}^{(l)}),$if j ≠ 0.
>+ $D_{i,j}^{(l)}:=\frac{1}{m}\Delta \_{i,j}^{(l)},$ if j = 0.

大写的Δ的矩阵D被当作一个**累加器**，当我们不断迭代时，它将l层的所有最终迭代值加起来，最终计算我们的偏导数。因此我们有
$$\frac{\partial J(\Theta )}{\partial \Theta \_{i,j}^{(l)}}=D_{i,j}^{(l)}$$


### 2.2 backpropagation intuition 
## 3 backpropagation in practice
### 3.1 implementation note:unrolling parameters
### 3.2 gradient checking
### 3.3 random initialization
### 3.4 putting it together
## 4 application of neural networks-autonomous driving
