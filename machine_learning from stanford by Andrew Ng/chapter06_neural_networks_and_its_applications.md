## 1 motivations
### 1.1 non-linear hypotheses
一个事实：现实生活中，我们需要非线性模型来解决更加复杂的问题。
+ 引例

假设现在有一个监督学习下的分类问题，数据只包含2个特征，此时使用高阶多项式作为我们的假设模型，构造一个包含很多非线性项的逻辑回归函数$h_{\theta }(x)=g(z)=g(\frac{1}{1+e^{-(\mathbf{\theta }^\mathrm{T}x)}})$,
z是一个高阶非线性函数，g(z)是sigmoid函数，这个模型可以将训练数据分为两类，完成分类任务。**但现实世界中，特征数总是大于2的，有时可能非常多**，比如假设有n=100个特征,此时特征二次项的个数为：<br>
$\frac{n^{2}}{2!}$=5000，三次项特征数为$\frac{n^{3}}{3!}$=17000。如果我们使用逻辑回归算法模型，如此多的特征项极易导致过拟合问题，而且计算量庞大computationally expensive。我们发现：<br>
随着初始特征数的增加，高阶多项式的项数以几何级数的速度增加。如果我们舍弃特征项的某些部分，比如只取二次项特征的一个子集，假设只取$x_{1}^{2},x_{2}^{2},\cdots ,x_{100}^{2}$，最后我们将得到一个类似椭圆的分类决策边界，显然这不是一个好的模型，因为它忽略了太多的特征信息。
+ car detectation task

可以看成是 通过图片的像素点亮度矩阵来告诉我们它代表了汽车的哪个部位。**基本思想**是：使用一个带标签的样本集，其中包括汽车和汽车以外的物体的图片，使用学习算法对样本集进行训练，以得到一个分类器，这个分类器在接受新的样本输入时，能够识别出 它是不是一辆汽车。
> zoom目标图片中的一小部分，我们假设它是一个50×50的像素图片，于是我们得到n=2500个像素点，特征向量x是一个包含所有2500个像素点的像素强度值，这个例子是灰度图片grayscale images的情况，**当使用RGB彩色图片时**，那么我们将得到n=75000个像素点。此时，二次项特征的个数将达到$\frac{n^{2}}{2!}\approx 3,000,00$,这个数字太大了，对于每个样本来说，要发现并表示所有这300万个项的计算成本太高！！
```
像素强度值pixel indensity values：告诉我们图片中每个像素点的亮度值或称灰度值(表示色彩的强烈程度)，
在典型的计算机图片表示方法中，取值范围在0~255之间。
```
### 1.2 neurons and the brain
+ 起源：算法试图模仿mimic人脑

一个假设：我们只需要一个学习算法，就可以完成大脑所作的所有五花八门的事情。下面将通过2个例子来证明这个假设的合理性：
+ auditory cortex learn to see

我们使用耳朵接受声音信号，把声音信号传到听觉皮层cortex，因此我们听到了声音。而之所以听觉皮层能够听到声音，是**因为在耳朵和听觉皮层之间有一个连接神经，正是这个神经连接使得大脑对声音做出反映；那么如果我们切断这个神经，同时把它re-wire重新接到一个动物的大脑上，此时，眼睛的视神经上的信号将传到听觉皮层，结果证明：听觉皮层auditory cortex learn to see！！amazing！！**

+ somatosensory cortex learn to see

躯体感觉皮层的作用是 处理人体的触觉的，如果我们做一个和刚才类似的重接实验，我们会发现躯体感觉皮层也能学会看！！这些实验被称为神经重接实验neur-rewiring experiments。类似实验还有很多，他们的原理大致都是,**使用一个能够接受外部信息的东西，比如人体的某个器官，将其连接到身体的某个senor中，
那么，这个sensor将学会这个接受外部信息的东西的功能(此解释太不专业了！！)。** 比如一些失明的人可以通过学会解读从环境反弹回来的声波模式，也就是声纳，
来了解周围的环境。

+ 从某种意义上来说，如果大脑中有一块脑组织可以处理光、声音、触觉信号，那么也许存在一种学习算法，可以同时处理视觉、听觉和触觉。我们的任务不是编写成千上万的程序来完成大脑所做的这些复杂的事情，而是找到一些近似的，大脑的学习算法，实现它，通过自学，算法学习如何处理不同类型数据。
## 2 neural networks
### 2.1 model representation Ⅰ
首先通过一张图看一下大脑的神经元如何工作：
![neuron_in_brain](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/neuron_in_brain.png)

一个神经元主要有3个部分组成：细胞主体；树突dendrite，一定数量的输入神经，接受来自其他神经元的信息；轴突axon，输出神经，将信息/信号传递给其他神经元。**神经元是一个计算单元，它从输入神经接受一定数目的信息，进行计算，然后将结果通过轴突传到其他节点或其他神经元**.在人工神经网络中，我们使用一个非常简单的模型来模拟神经元的工作：将神经元模拟为一个逻辑单元，逻辑单元接受输入信息，并输出计算结果。在分类问题中，这个逻辑单元也被称为以sigmoid函数或者逻辑函数为激励函数的人工神经元.**神经网络就是不同的神经元组合在一起的集合，通过下图解释：**
![layers_in_nn](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/layers_in_nn.png)

layer1代表输入层，在这一层输入特征项x1，x2，x3；，layer3代表输出层，这一层神经元计算并输出假设模型$h_{\Theta }(x)$的最终结果。layer2被称作隐藏层，隐藏层有时不只一层，任何非输入且非输出层都属于隐藏层，在监督学习中我们无法在训练集中看到隐藏层的值。下面解释神经网络如何完成计算，图如下：
![how_NN_computes.](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/how_NN_computes.png)
> 首先，对标记符号解释：$a_{i}^{(j)}$:第j层的第i个单元的激活函数，此处是一个sigmoid/逻辑激励函数；$\Theta ^{j}$：控制从第j层到第j+1层的函数映射的权重矩阵。其次，**必须理解$\Theta ^{j}$的矩阵维度**：一般说来，$\Theta ^{j}\in \mathbb{R}^{s_{j+1}\times (s_{j}+1})$, $s_{j}$表示第j层的节点数.

+ 以上我们从数学上定义了NN:即定义一个函数$h_{\Theta }(x)$表示输入x到输出y的映射，不同的参数对应不同的假设，给出不同的假设模型。*下一节预告：深入理解假设的作用；使用例子演示假设如何计算。*
### 2.2 model representation Ⅱ
接上一小节中对神经网络过程的描述，我们发现隐藏层激活g函数的输入是一个加权线性组合，它进行的是矩阵向量操作，即$\Theta ^{j}x_{i}$ , $x_i$是我们的特征输入。我们使用z来表示这个加权线性组合，则
$$z^{(j)}=z_{i}^{(j)}=\mathbf{\left (z_{1}^{(j)},z_{2}^{(j)},\cdots ,z_{n}^{(j)}\right )}^\mathrm{T},$$
$z_{i}^{(j)}$表示第j层上第i个单元的加权线性组合，即为$$z_{i}^{(j)}=\Theta _{i}^{(j-1)}a_{i}^{(j-1)},$$
其维度表示为$z ^{j}\in \mathbb{R}^{s_{j}\times 1},$ 
于是，$a^{(j)}=g(z^{(j)})$ 的维度与 $z ^{j}$ 的维度相同，为统一记号表示，我们将初始特征向量x记作 $a^{1}$ 。这些记号使我们能够更加清楚地理解神经网络的计算过程。此节中介绍一种**前向传播算法**：从输入层的$a^{1}$层开始向前传播到第1个隐藏层，使用激励函数计算得到输出$a^{2}$，然后继续向前传播至第2个隐藏层，重复上次的步骤，直到最终达到输出层。同逻辑回归类似的是：神经网络也使用了sigmoid算法作为激活函数，**不同的是**：神经网络中，并不是使用sigmoid函数一次就得到结果，而是在网络的每一层都使用sigmoid函数对参数进行训练，然后将训练结果作为输入，“喂”给网络的下一层。以下为图片直观表示：

![neural_networks_model_representation](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/neural_networks_model_representation.png)
+ what will be talked in the next 2 videos？

为何层层训练后，就可以学习更加复杂的假设？video 1<br>
NN如何利用隐藏层计算更复杂的特征，并输入到最后的输出层？

## 3 applications
### 3.1 examples and intuitions Ⅰ
这一节尝试从逻辑运算的视角解释解释神经网络的工作原理。假设我们有二进制的输入特征x1和x2，它们要么取0值，要么取1值。下面的简化图中我们只取了4个样本点，被分为正负两类样本，分类规则是*逻辑运算中的异或非规则*。
>+ XOR: exclusive OR gate，**异或门**，数学符号为：⊕，a⊕b:如果a、b值不同，则结果为1；如果a、b值相同，则结果为0.
>+ XNOR: not XOR gate，**异或非门**。逻辑规则为：如果a、b值相同，则结果为1；如果a、b值不同，则结果为0.

![non-linear_classification_examle_XOR_XNOR](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/non-linear_classification_examle_XOR_XNOR.png)
我们的任务是：**使用神经网络模型拟合数据，找到决策边界来区分正负样本**。
+ simple example：AND

假设我们有二进制的输入特征x1和x2，且$x_{1},x_{2}\in\\{{0,1}\\}$,$y=x_{1} AND x_{2}$.下图显示：为特征项加入特定的权重后，使用逻辑真值表进行运算，我们**发现假设模型$h_{\Theta }(x)$其实正在做“AND”与运算”**。
![simple_example_AND](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/simple_example_AND.png)
+ simple example：OR

![simple_example_OR](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/simple_example_OR.png)
### 3.2 examples and intuitions Ⅱ
内容:**演示神经网络如何计算非线性假设函数**。
+ simple example：NOT

idea：**给与 希望取非运算的变量 一个绝对值大的负数，作为权值**。比如例子中，x1的权值为-20，得到了对x1进行非运算的效果。
+ compute x1 XNOR x2

在开始之前，我们结合上述三种运算，得到一个这样的权重矩阵$\Theta ^{(1)}$:
![The_Θ_matrices_for_AND_NOR_OR](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/The_%CE%98_matrices_for_AND_NOR_OR.png)，再次结合运算，我们得到XNOR logical operator，当且仅当x1和x2同时为0或者1时，输出1.
![XNOR_logical_operator](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/XNOR_logical_operator.png).从第1层到第2层，我们使用矩阵$\Theta ^{(1)}$进行AND和NOR运算，
$$\Theta ^{(1)}=\begin{bmatrix}
-30 &20  &20 \\\\ 
 10&-20  &-20 
\end{bmatrix},$$
从第2层到第3层，我们使用矩阵$\Theta ^{(2)}$进行OR运算。
$$\Theta ^{(2)}=\begin{bmatrix}
-10 &20  &20 
\end{bmatrix}.$$
现在，我们使用神经网络进行了 XNOR运算，网络的隐藏层有2个节点。**演示图如下：**
![summarize_for_XNOR_using_NN](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/summarize_for_XNOR_using_NN.png)
结合逻辑真值表，最终输出$h_{\Theta }(x)$的值。
### 3.3 multiclass classification
在多分类问题中，我们使用向量表示模型最终输出的结果，![resulting_classs](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/resulting_classs.png)。下图例子中我们的input是代表不同事物的图片，经过隐藏层的计算得到一个向量，使用到最后一层的参数矩阵Θ左乘该向量，得到四个4维向量。
![examples_of_multiclassification](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/examples_of_multiclassification.png)
每一层的神经网络计算步骤如下：![NN_computing_for_multiclassification](https://github.com/Vita112/machine_learning/blob/master/machine_learning%20from%20stanford%20by%20Andrew%20Ng/img/NN_computing_for_multiclassification.png)

## 4 summary

得益于计算机硬件性能，包括计算速度、计算能力和存储空间等的跨越式提升，神经网络在沉寂一段时间后再次引起大众关注，并在ml领域取得了极大的突破，得到了很好的性能表现。总结要点如下：
+ a node in nueral networks is a neural computing unit,which receives data from the former layer and run it with our hypotheses model.
+ we can't observe the computing processes of neural networks
+ allow us to deal with complex problems having large number of features
