+ 时间复杂度

  不是指程序解决问题时需要多长的时间，而是指**当问题规模扩大后，程序运行所需时间的长度增长得有多快。**对于高速处理数据的计算机来说，当数据规模扩大到原来的数百倍后，
  程序运行的时间是否还是一样，或者也跟着慢了百倍，甚至慢更多。
  >+ O(1)：也称常数级复杂度，不管数据有多大，程序处理花的时间始终是那么多；
  >+ O(n)：数据规模变得有多大，程序运行花费的时间也跟着变得有多长，例如找n个数中的最大值；
  >+ O(n^2):数据扩大2倍，时间变慢4倍，比如冒泡排序、插入排序等；
  >+ O(a^n):指数级复杂度，数据规模变大时，所需时间长度成几何阶数上涨，比如一些穷举类算法；
  
  复杂度一般被分为两种级别:一种是多项式级别的复杂度，比如O(1)、O(log(n))等，因为规模n出现在底数的位置；另一种是O(a^n)、O(n!)
  型的复杂度，是非多项式级的，其复杂程度计算机往往不能承受。当我们选择的**通常都是多项式级复杂度**的算法。
  
+ P问题polynomial problem和NP问题Non-deterministic Polynomial problem
  
  概念：如果一个问题**可以找到一个多项式时间复杂度的算法来解决它**，那么这个问题就属于P问题。也就说所有可以在多项式时间内求解的判定问题构成P类问题，判定问题是 判断是否有一种能够解决某一类问题的可运行的算法的 研究课题。
  NP问题不是非P类问题，它是指可以**在多项式时间里验证一个解**的问题，也可以说，在多项式时间里猜出一个解的问题。
  
  关于NP问题的一个例子：
  >我RP很好，在程序中需要枚举时，我可以一猜一个准。现在某人拿到了一个求最短路径的问题，问从起点到终点是否有一条小于100个单位长度的路线。
  它根据数据画好了图，但怎么也算不出来，于是来问我：你看怎么选条路走得最少？我说，我RP很好，肯定能随便给你指条很短的路出来。然后我就胡乱画了几条线，
  说就这条吧。那人按我指的这条把权值加起来一看，嘿，神了，路径长度98，比100小。于是答案出来了，存在比100小的路径。别人会问他这题怎么做出来的，他就可以说，
  因为我找到了一个比100小的解。在这个题中，找一个解很困难，但验证一个解很容易。验证一个解只需要O(n)的时间复杂度，也就是说我可以花O(n)的时间把我猜的路径的长度加出来。
  那么，只要我RP好，猜得准，我一定能在多项式的时间里解决这个问题。

存在另一种情形：目前还没有办法在多项式时间里验证一个解的问题。Hamilton回路问题是：给定一个图，问能否找到一条经过每个顶点一次且恰好一次(不遗漏也不重复)，最后又走回来的路
路。满足这个条件的路径叫做Hamiton回路。如果我们将它换成另一个问题：问给定的图中，是否不存在Hamiton回路？这个问题没法在多项式时间里进行验证，因为除非你试过所有的路(n!)，否则，
你不能断定“没有Hamiton回路”。

+ P类问题、NP问题和NPC问题

很显然，所有的P类问题都是NP问题，因为能多项式的解决一个问题，必然能多项式的验证一个问题。但，不是所有的NP问题都属于P问题。
通常情况下，NP 问题是有可能找到多项式算法，但同时人们相信：存在至少一个 不可能有多项式级复杂度的NP问题。所以P≠NP。
在研究NP问题的过程中，找到一类非常特殊的NP问题，即NPC问题(Non-deterministic Polynomial completible problem)。
> 在说明NPC问题之前，先引入一个概念——约化（reducibility），又被称为归约。简单说就是，一个问题A可以约化为问题B的含义是，可以用问题B的解法来解决问题A，或者说问题A可以“变成”问题B。
举个例子，求解一个一元一次方程和求解一个一元二次方程时，我们可以把前者约化为后者。约化的规则是：两个方程的对应项系数不变，一元二次方程的二次项系数为0。在“问题A约化为问题B”中有一个重要的直观意义：
B的时间复杂度高于或等于A的时间复杂度，也就是说问题A不一定比问题B难。*约化具有传递性*：问题A可以约化为问题B，问题B可以约化为问题C，那么，问题A一定可约化为问题C。这里“可约化”是指 可以“多项式地”约化polynomial-time reducible，
即变换输入的方法是 能在多项式级的时间里完成的。从归约的定义中，我们看到当问题A归约为问题B时，时间复杂度增加了，问题应用的范围也增大了。

再回到我们的P、NP问题，以及约化的传递性，我们有一个想法：是否有可能找到一个时间复杂度最高，且能够“通吃”所有NP问题的 一个超级NP问题？答案是肯定的！**也就是说，
存在这样一个NP问题，所有的NP问题都可以约化成它。**并且，这种问题不止一个，他有很多个，构成一类问题，这一类问题就是**NPC问题**。
>+ 同时满足以下2个条件的问题就是NPC问题：<br>
首先，它是一个NP问题；然后，所有的NP问题都可以约化为它。


reference：[解读P问题、NP问题、NPC问题的概念CSDN](https://blog.csdn.net/sp_programmer/article/details/41749859)