## gradient descent for multiple variables
+ hypothesis and cost function

hypothesis: $$h_{\theta }(x)= \mathbf{\theta }^\mathrm{T}x=\theta _{0}x_{0}+\theta _{1}x_{1}+\cdots +\theta _{n}x_{n}$$

parameters: $$\theta _{0},\theta _{1},\cdots ,\theta _{n}$$

cost function: $$J(\theta _{0},\theta _{1},\cdots ,\theta _{n})=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta }(x^{(i)})-y^{(i)})^{2}$$


我们约定x0=1，把参数θ0，θ1，……θn当作一个`n+1`维的向量(n+1 dimensional vector),把$$J(\theta _{0},\theta _{1},\cdots ,\theta _{n})$$看作 以
一个`n+1`维的向量为参数 的函数，

+ gradient descent

repeat {
    $\theta _{j}:=\theta _{j}-\alpha \frac{\partial J(\theta _{0},\cdots ,\theta _{n})}{\partial \theta _{j}}$
    }              (simultaneously update for every j=0,……,n)
