# 1、OT
> 最优运输被广泛应用于多个领域，包括计算流体力学，多幅图像之间的颜色转移或图像处理背景下的变形，计算机图形学中的插值方案，以及经济学、通过匹配和均衡问题等。此外，最优传输最近也引起了生物医学相关学者的关注，并被广泛用于单细胞RNA发育过程中指导分化以及提高细胞观测数据的数据增强工具，从而提高各种下游分任务的准确性和稳定性。
> 
> 当前，许多现代统计和机器学习问题可以被重新描述为在两个概率分布之间寻找最优运输图。
>- 例如，领域适应旨在从源数据分布中学习一个训练良好的模型，并将该模型转换为采用目标数据分布。
>- 另一个例子是深度生成模型，其目标是将一个固定的分布，例如标准高斯或均匀分布，映射到真实样本的潜在总体分布。在最近几十年里，OT方法在现代数据科学应用的显著增殖中重新焕发了活力，包括机器学习、统计和计算机视觉。

## 1.1、Monge问题

Gaspard Monge首次形式化描述了最优运输，即将一个土堆$\alpha$重新排列另一个土坑$\beta$的最佳运输方法

$$
\alpha=\sum_{i=1}^n a_i\delta_{x_i}\ \text{and}\ \beta=\sum_{j=1}^mb_j\delta_{y_j}\tag{1}
$$


$a_i$表示土堆$x_i$的供应量，$b_j$表示土坑$y_j$的需求量，$\delta$是狄拉克函数，mass是一个土堆，如$a_i\delta _{x_i}$，规定一个土堆只能运输到一个土坑

Monge问题的寻找一个运输代缴最少的映射$T:\{x_1,...,x_n\}\to\{y_1,...,y_m\}$，将一个分布的土堆运输到另一个分布中，并且$T$满足:
$$
\forall j\in\{1,...,m\},b_j=\sum_{i:T(x_i)=y_j}a_i\tag{2}
$$


约束条件简写为$T\#\alpha=\beta$，称为mass conservation constraint，$T\#$称为push-forward操作。

mass conservation constraint：
- 当n=m时，可以实现$1x\leftrightarrow1y$，即一对一，
- 当n>m时，可以实现$1x\to1y,1x\leftarrow多y$，
- 当n<m时，该问题无解

Monge问题的形式化表达如下：
$$
\min_T\{\sum_{i=1}^nc(x_i,T(x_i)):T\#\alpha=\beta\}\tag{3}
$$
其中$c(x_i,T(x_i))$表示从$x_i$到$T(x_i)$的代价函数

缺点：
- $n\lt m$时，不存在可行解，
- push-forward操作导致式（3）的可行域非凸，时间复杂度为$n!$

## 1.2、Kantorovich问题

Leonid Kantorovich重新形式化了Monge的问题，对mass conservation constraint进行修改，允许一个土堆填充到多个土坑

**定义**：

给定一个代价矩阵$\bold C\in\mathbb R^{n\times m}$，离散概率分布向量$\alpha\in\mathbb R^n,\beta\in\mathbb R^m$，求解从α分布到β分布映射的最小代价矩阵$\bold P^*$可以被量化为
$$
\min_{p\in U(\alpha,\beta)}\big<\bold C,\bold P\big>\stackrel{def}=\sum_{i,j}C_{i,j}P_{i,j}\tag{4}
$$

$$
U(\alpha,\beta)=\big\{P\in\mathbb R_+^{n\times m}|P\bold1m=\alpha,P^\top\bold1n=\beta\big\}\tag{5}
$$

其中，

- $\bold C$和$\bold P$的矩阵内积最小化，即运输代价最小的两点应分配更大的权重$P_{i,j}$

- kantorovich问题松弛了mass conservation constraint，允许多$x\to$多$y$匹配
- $\bold P^*$的行和等于α向量，列和等于β向量，即供应方输出的货量总和等于货物持有量，需求方输入的货量总和等于需求期望量
- $\bold P^*$的行和列分别等于两个概率分布向量，$\bold P^*$可以理解为联合概率分布矩阵，α和β也常称为"边缘分布向量"

## 1.3、对比Monge问题和Kantorovich问题

- Kantorovich问题松弛了Monge问题的mass conservation constraint，引入了mass splitting思想，即源点$x_i$可以被分散运输到不同地方
- Kantorovich问题松弛了Monge问题中的确定性运输，转而考虑概率运输
- Monge问题不一定有可行解，而Kantorovich问题是一个凸问题和线性规划问题，具有可行解
- Monge问题局限应用场景，而Kantorovich问题灵活应用于ren'yi

# 2、边缘分布最优传输MDOT

# 3、联合分布最优传输JDOT

# 4、深度联合分布最优传输DeepJDOT