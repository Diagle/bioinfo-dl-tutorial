[toc]

# OT for DA

## 一、背景知识

## 1.1、测度

### 1.1.1、概率测度

#### 可积函数（概率函数）

设$\mathbb {X,Y}$为非空集合。$P(\mathbb X)$为定义在集合$\mathbb X$上的概率测度的集合，$\mu\in P(\mathbb X),v\in\mathbb Y$，若函数$\phi:\mathbb X\to\mathbb R$在测度$μ$下的期望是收敛的，则称 $\phi$关于$μ$ 是可积的。即$\mathbb E_{X\sim\mu}[\phi(x)]=\int_\mathbb X|\phi(x)|d\mu(x)\lt\infty$

关于$μ$可积函数的空间记为：
$$
L^1(\mu)=\big\{\varphi:X\to\mathbb R|\mathbb E_{x\sim\mu}[\varphi(x)]\lt\infty\big\}
$$
所讨论的数学对象能应用到离散、连续或混合概率测度的情形下。均值可以使用求和、积分或期望等一种或多种记号来表示和计算。为简化记号起见，我们将假设每一个概率测度都对应着一个概率密度函数或概率质量函数。

#### 耦合coupling（联合概率）

给定两个随机向量$\bf X,Y$，组合成一个具有与$\bf X,Y$相同边缘分布的非唯一联合分布的随机向量$\bf (X,Y)$。所有这些联合分布的空间$\Pi$构成了搜索空间。边缘分布为$\mu,v$的联合概率测度的集合记作$\Pi(\mu,v)$，也即可行计划的集合
$$
\begin{align}
\Pi(\mu,\nu)&=\bigg\{\pi\in P(X\times Y)|\forall A\subseteq X,B\subseteq Y,
\\

&\int_{A\times Y}d\pi(x,y)=\int_Ad\mu(x),\int_{X\times B}d\pi(x,y)=\int_Bdv(y)\bigg\}
\end{align}
$$

$X\times Y$ 是定义联合分布的笛卡尔积空间。第一个条件确保联合分布的 x 边缘分布为 $\mu$。第二个条件确保联合分布的 y 边缘分布为 $\nu$。

**离散且有限的矩阵形式**

设$\tilde a、\tilde b$分别为$\mu,v$对应的概率质量函数，$\tilde a、\tilde b$的支撑有限，即$d_\mathbb X=|\mathbb X|\lt\infty$且$d_\mathbb Y=|\mathbb Y|\lt\infty$。定义概率单纯性$\Sigma_d=\{x\in[0,1]^d|x^T\mathbf 1=1\}$。设$a\in\Sigma_{d_\mathbb X}$和$b\in\Sigma_{d\mathbb Y}$为$\tilde a,\tilde b$的向量表示。即对于一些分配$\{x_i | 1 ≤ i ≤ d\mathbb X\} = \mathbb X$，$\{y_i|1\le i\le d_\mathbb Y=\mathbb Y\}$，设$a_i=\tilde a(x_i),b_i=\tilde b(y_i)$。

设$\Pi(\mu,v)$对应于有限维度的矩阵$U(a,b)$，表示边缘$u、v$的构成的联合概率质量函数。定义为
$$
U(a,b)=\{P\in[0,1]^{d_\mathbb X\times d_{\mathbb Y}}|\bf P1=a,P^\top1=b\}
$$

请注意，对任一$P\in U(a,b)$，在某些边缘密度质量函数为$\tilde a、\tilde b$的随机变量$(x,y)$的联合概率质量函数下， $P$的第ij个元素满足$P_{ij}=Pr(X=x_i,Y=y_i)$

**经验测度**

当面对真实世界数据和问题时，常常不能得到分布$\mu,v$的准确形式。反而，具有样本$\{X_i\}^n_{i=1}\sim\mu$和$\{y_j\}_{j=1}^m\sim v$。在这种情况下使用经验测度颇为合适
$$
\mu_n=\frac1n\sum_{i=1}^n\delta_{x_i},\nu_n=\frac1m\sum_{j=1}^m\delta_{y_j}
$$
其中 $\delta_z$ 表示集中在 $z$ 处的单位质量。

#### 概率指标度量

概率指标是比较概率度量的函数。可以是提出的指标（如Wasserstein距离）或者散度（如KL散度）。其量化两个分布的差异。在概率论中，有两个方向的概率指标类别，积分概率指标（IPMs）与$f-$散度。

**积分概率指标（IPMs）**

对于同一族函数$\mathcal F$，IPM$d_\mathcal F$定义分布$P、Q$间的分布为
$$
d_\mathcal F(P,Q)=\underset{f\in\mathcal F}{\mathrm{sub}}|\underset{x\sim P}{\mathbb E}[f(x)]-\underset{x\sim Q}{\mathbb E}f(x)|
$$
因此，IPMs基于不同P-Q衡量分布间的距离。对$\mathcal F=\mathrm{Lip}_1$，$\mathcal L^*_{KR}$是IPM。另一个重要的指标是最大平均差异（MMD），定义$\mathcal F=\{f\in\mathcal H_k:|f|_{\mathcal H_k}\le 1\}$，其中$\mathcal H_k$是核为$k:\mathbb R^d\times\mathbb R^d\to\mathbb R$再生希尔伯特空间。这些指标在生成式模型和域适应中发挥了重要作用。后者领域中第三重要的距离就是$\mathcal H-$距离。

**$f-$散度**

相对地，$f$-散度基于P和Q的概率值测度之间的差异。对于凸的，较低的半连续函数$f:\mathbb R^+\to\mathbb R$且$f(1)=0$
$$
D_f(P||Q)=\underset{x\in Q}{\mathbb E}\bigg[f(\frac{P(x)}{Q(x)})\bigg]\tag{8}
$$
其中，$P(x)$表示P的密度。$f-$散度的一个例子就是KL散度，$f(u)=u\log(u)$

**比较**

正如[20]中提出的，IPMs有两个性质优于f-散度。
- 第一，即使当P和Q具有非联合的支撑时，$d_\mathcal F$也有定义。例如，在训练开始时，GANs生成了较差的样本，所以$P_{model}、P_{data}$具有非联合的支撑。这种情况下，IPMs提供一个有意义的指标，无论$P_{model}$有多差，$D_f(P_{model}||P_{data})=+\infty$。
- 第二，IPMs计算样本依赖空间的几何形状。例如，考虑高斯分布的流形$\mathcal M=\{\mathcal N(\mu,\sigma^2):\mu\in\mathbb R,\sigma\in\mathbb R_+\}$，带欧式基础代价的Wasserstein距离为
$$
W_2(P,Q)=\sqrt{(\mu_P-\mu_Q)^2+(\sigma_P-\sigma_Q)^2}
$$
而KL散度与hyperbolic有关。如F 1所示。总的来说，分布间差异指标的选择严重影响学习算法的成功与否。每一指标/散度的选择都在概率分布的空间中引入了不同的几何，因此改变了ML问题的背后优化。

### 1.1.2、非负的非归一化测度

大部分情况下，我们感兴趣的是概率测度间的距离。除非另外说明，所有的测度均为概率测度。某些例子中的这些距离可以推广到包含非负且有限、不一定是概率测度的测度空间中。在这种情况下，期望记号 $\mathbb{E}_{x\sim\mu}(\cdot)$ 并不适用，改用积分 $\int_X (\cdot)\,\mathrm{d}\mu(x)$ 或求和的形式。用$\mathcal{M}_+(X)$ 表示定义在 X 上的有限、非负且未归一化测度的空间。



## 1.2、数学形式

### 1.2.1、Monge形式（最优映射）

给定概率测度$μ$和$v$，最优运输映射$\bf{t}^\star:\mathbb X\to \mathbb Y$定义为服从$μ$的随机变量$X$映射到服从$v$的随机变量$Y$，以$\bf X$和$\bf t^*(X)=Y$最低期望代价的方式。Monge形式为

$$
M_c(\mu,\nu)=\inf_{t\in T_{\mu,\nu}}\int_{\mathbb X}c(x,t(x))d\mu(x)=\inf_{t\in T_{\mu,\nu}}\mathbb E_{x\sim\mu}[C(X,t(X))]\tag{1}
$$

其中$T_{\mu v}=\{t:\mathbb X\to \mathbb Y|t_\#(\mu)=v\}$表示所有可能映射的集合。$t_\#(\mu)$是$\mu$在$t$下的*pushforward*测度。概率测度的语境下，$t_\#(\mu)$为随机变量$t(X)$的概率测度。C表示从$X$传输到$Y$单位质量的单位成本。$C$的选择取决于应用并受到领域先验知识和数据的影响。

**局限**：可能不存在映射$\bf t$使得$t_{\#}=v$。例如$u$表示2个状态的概率质量，无法使用确定性函数映射到3个状态的概率质量的测度。

### 1.2.2、Kantorovitch形式（最优计划）

给定$\mu、v$和某些代价函数$c$，KantorovichOT问题（§2 离散样本的一般形式）定义为

$$
\begin{aligned}
\mathcal K_c(\mu.\nu)&=\inf_{\pi\in\Pi(\mu,\nu)}\int_{\mathbb{X\times Y}}c(x,y)d\pi(x,y)\\
&=\inf_{\pi\in\Pi(\mu,\nu)}\mathbb E_{(x,y)\sim\pi}[c(x,y)]
\end{aligned}
\tag{2}
$$

任何达到该下确界的 $\pi^*$ 都称为一个最优传输计划（OT plan）。本例中$X、Y$为离散随机变量，其概率质量向量为$a、b$ ，改写为
$$
\mathcal K_{c}(\mu,\nu)=\min_{P\in U(a,b)}\big<C,P\big>\tag{3}
$$
$C=[C(x_i,y_j)]_{ij}$为包含$X$和$Y$间配对代价的矩阵。（式2）作为原始形式，可以写成其他相等的形式（如对偶形式）。Brenier 定理（见文献 [1, 定理 2.1]）表明：在欧几里得空间中，Kantorovich 问题与 Monge 问题是等价的，并且最优传输映射可以表示为某个凸函数的梯度。此外，一些工作 [43] 在 Kantorovich 的表述中引入了 Monge 映射这一变量，从而联合学习最优耦合以及传输映射的近似。



## 1.3、最优化理论



## 二、常见的OT形式

### 2.1、正则化OT

**公式推导**

原始OT形式的一些计算和统计限制可以通过在原始对象上（式2）加入正则化项来缓解。熵正则化形式有在计算上的优势和机器学习的兼容性。正则化OT问题定义为：
$$
\mathcal{OT}_{\Omega,\lambda}(\mu,\nu)=\inf_{\pi\in\Pi(\mu,\nu)}\int_{\mathbb X\times Y}c(x,y)d\pi(x,y)+\lambda\Omega(\pi)\tag{5}
$$

其中$\Omega$是正则化运算符，$\lambda$是正则化系数。



正则化离散形式OT问题由下式给出：
$$
\mathcal{OT}_{\Omega,\lambda}(\mu,\nu)=\inf_{P\in(a,b)}\big<C,P\big>+\lambda\Omega(P)\tag{6}
$$
$\Omega$的选择之一是联合分布与参考分布$m_1$和$m_2$的积测度之间的KL散度，即$\Omega(P)=KL(P||m_1\times m_2)=\sum_{ij}P_{ij}\log\frac{P_{ij}}{Q_{ij}}$
当$m_1$和$m_2$是均匀分布时，正则化项(忽略常数)被称为**熵正则化**：
$$
\Omega(P)=-H(P)=-\sum_{ij}P_{ij}(\log P_{ij}-1)
$$

其中，$H$是Boltzmann-Shannon熵函数。增加一个负的Boltzmann-Shannon熵函数就得到著名的熵正则化OT：
$$
\mathcal{OT}_{\Omega,\lambda}(\mu,\nu)=\inf_{P\in(a,b)}\big<C,P\big>-\lambda H(P)\tag{7}
$$
是原始问题（式3）带有额外的严格凸正则化项（负的Boltzmann-Shannon熵）的版本。这种正则化使最优计划偏向于均匀分布。熵-正则化OT(优化目标值即已知的Sinkhorn”距离“)可重写为最小化一个KL散度的形式：

$$
\mathcal{OT}_{\lambda}(\mu,\nu)=\lambda\inf_{P\in U(a,b)}\text{KL}(P||K)\tag{8}
$$
其中，$K$（即Gibbs 核）是对缩放后的ground cost逐元素取负指数得到的，即$K_{ij} = \exp(−C{ij}/\lambda)$。已知KL散度为$P$在固定$K$下的严格凸函数，即存在一个唯一的$P$最小化以上表达式且对不同的$K$不同。需要注意的是，Sinkhorn 距离尽管并非严格意义上的“距离”，但当代价函数 $c(x,y)$ 关于 x 和 y 对称时，它在 $\mu$ 和 $\nu$ 之间是对称的，即$OT_λ(μ,ν)=OT_λ(ν,μ).$

**Sinkhorn–Knopp 算法**

尽管正则化项可以通过引入额外先验并结合具体应用需求来加以解释 [50]，熵正则化所带来的计算优势才是其最主要、最具吸引力的优点。具体而言，由 Cuturi 在 GPU 时代推广开的式 (7) 的表述 [51]，可以通过一种称为 Sinkhorn–Knopp 算法 [52] 的迭代矩阵缩放方法高效求解，该算法通常简称为 Sinkhorn 算法（算法 1）。该算法将最优传输计划的计算过程简化为一系列矩阵–向量乘法操作。刻画测度$\mu$和$v$的差异的核矩阵记为$K_{ij}=e^{\frac{C_{ij}}{\lambda}}$，其中，C是ground cast矩阵，λ为式6中的正则化系数。算法寻找满足以下条件的唯一最优解：
$$
P_{\lambda}=\text {diag}(\mu)K\text{diag}(v)\tag{Sinkhorn 理论}
$$
$$
P_\lambda\mathbf 1_m=a,P_\lambda^\top\mathbf1_n=b\tag{最优条件 }
$$
寻找满足以上要求的向量$\mu$和$v$仅通过重复替代缩放（算法1第3行）直到收敛。收敛到给定容差 $\delta$ 所需的迭代次数由代价矩阵 $C$ 中元素相对于 $\lambda$ 的尺度所决定。迭代仅仅应用于核矩阵$K$（或其转置）到向量$\mu,\nu$。Sinkhorn 算法在其基本形式下，每次迭代在点数规模上的最坏情况复杂度为二次复杂度（当然也存在许多可行的改进方法，例如 [54]），并且具有线性收敛率。这使得它相比于用于求解无正则化形式的线性规划方法要高效得多，也更具可扩展性。

![image-20251225200432414](assets\image-20251225200432414.png)

**Sinkhorn 散度**

尽管Sinkhorn”距离“为具有计算吸引力。式7中仍存在两个局限：
- 正则化OT最优值并非一种距离

- 该形式在最小化解中引入了一种称为熵偏置（entropic bias）的偏差[55]。

这些问题可以通过 Sinkhorn 散度 [55], [56] 来解决；这是一种基于 Sinkhorn“距离”的另一种正则化最优传输变体，其定义如下：
$$
\overline{\mathcal{OT}}=\mathcal{OT}_\lambda(\mu,\nu)-\frac12\bigg(\mathcal{OT}_{\lambda}(\mu,\mu) +\mathcal{OT}_{\lambda}(\nu,\nu)\bigg)\tag{9}
$$
Sinkhorn 散度对输入是光滑且可微的，并且相比于常规的最优传输（OT），具有更好的样本复杂度 [57]。虽然其需要计算三个$\mathcal{OT}_\lambda$值，而原始只需计算一个，但用以计算$\mathcal{OT}_\lambda(\mu,\mu)$和$\mathcal{OT}_\lambda(\nu,\nu)$的系统Sinkhorn算法要快于正则Sinkhorn算法。Altschuler et al.对Sinkhron及其更快的变种称之为Greenkhron的计算复杂度进行了分析。将在 §5.3 and §5.5讨论Sinkhorn变种。

**局限**

熵正则化OT的一个已知的局限是设置正则化强度$\lambda$。显著的计算加速通常依赖于较高程度的正则化（即较大的$\lambda$），但这可能导致过度扩散（即最优传输计划变得模糊）的问题 [68]。例如，在点对齐案例中（§2），较强的正则化会导致稠密匹配，大多数点之间会彼此相连。这会削弱其作为对应关系的解释性。另一方面，在理论上，当 $\lambda$非常小时可以得到高度集中的稀疏传输计划，但在实际中这往往会引发数值不稳定问题 [2], [71]。



### 2.2、不平衡OT和部分OT

**动因**

Kantorovich 形式（式2）中的mass conservation constraint要求两概率分布之间的总质量相等。而不平衡 OT（Unbalanced OT, UOT）指的是放宽这一约束的形式化方法，从而允许对任意（非归一化）测度或部分质量进行传输。这在类似多目标追踪和密集统计中很有用。熵正则化可以组合到UOT和POT中。使得能够使用Sinkhorn算法且加速计算。式2中的平衡最优传输（balanced OT）表述在存在离群点时可能表现出很强的非鲁棒性。由于该方法试图将 $\mu$ 中的全部质量都传输到 $\nu$，哪怕只有一个被污染的数据点，也可能使最优传输代价任意增大。而不平衡最优传输（UOT）一方面可以通过为异常值分配较小的质量来降低这种敏感性。这使得 UOT 更适合处理包含受污染或噪声数据的机器学习应用。能够检测并且不传输离群点的最优传输（OT）变体通常建立在非平衡最优传输（unbalanced OT, UOT）表述之上。

**不平衡OT**

将OT拓展到任意正测度的一个方式为采用*marginals relaxation*，即去除式 (2) 中的硬性边际约束，并通过增加一个软正则化项来惩罚质量变化。这通常通过 **Csiszar 散度** $D_\phi$ 实现
$$
\mathcal{UOT}_\phi(\mu,\nu)=\inf_{\pi\in\mathcal M_+(\mathbb{X\times Y})}\int_{\mathbb{X\times Y}}c(x,y)d\pi(x,y)+\lambda_1D_{\phi}(\pi_1|\mu)+\lambda_2D_\phi(\pi_2|\nu)
$$

其中 $M_+(\mathbb{X\times Y})$表示$\mathbb{X\times Y}$上有限非负测度的空间，$D_\phi$是由$\phi$引导的散度，衡量边缘分布($\pi_1,\pi_2$)与测度$(u,v)$之间质量变化的程度，通过($\lambda_1,\lambda_2$)控制惩罚强度。采用替换后带惩罚的限制的一个明显结果就是$\pi$的边缘分布不再需要等于（$\mu,\nu$）。$D_\phi$的具体例子包括KL散度和Total Variation。最近，[78]提出了一种基于最大均值差异（MMD）的正则化，并证实了修改后的形式具有更实用的性质如样本维度无关的复杂度。

**部分OT**

另一种延申OT到非平衡测度的方式通常使用在离散案例中，即部分分配*partial assignment*。这种形式被称为部分最优传输（Partial OT, POT）。当从 $n$ 个点传输到 $m$ 个点时，可以通过将计划矩阵$P \in \mathbb{R}^{n \times m}$扩展为 $\tilde P \in \mathbb{R}^{(n+1) \times (m+1)}$，以及将代价矩阵 $C$ 扩展为 $\tilde C \in \mathbb{R}^{(n+1) \times (m+1)}$ 来调整形式。新增的行和列被称为“垃圾桶*dustbin*”或“虚拟点*dummy point*”，用于吸收未匹配点的质量，实际上将非平衡问题转换为平衡问题，这样就能重新利用式2的传统计算。

**局限**

对质量变体参数（$\lambda_1、\lambda_2$）的选择敏感。此外，当集成正则化，微调正则化系数充满挑战且依赖于具体应用。

### 2.3、OT计算的延申

#### 2.3.1、投影 OT

**动因**

加速最优传输（OT）计算的一种方法是利用低维投影的思想，通过汇总在数据的 1D 投影上计算的 OT 距离来实现。所刺激该方法的原因是现实中在不变/1D案例中求解OT问题计算廉价。具体而言，我们可以利用分位函数 $F_\mu^{-1}$ 和 $F_\nu^{-1}$ 来表示一维测度 $\mu$ 与 $\nu$ 之间的 p-Wasserstein 距离，其形式如下：
$$
\mathcal W_p(\mu,\nu)=\int_0^1c^p(F_\mu^{-1}(t),F_v^{-1}(t))dt^{\frac1p}\tag{11}
$$
其中$c^p(.,.)$表示ground cost函数p次方。当$n$和$v$分别为$n$和$m$个样本的经验分布时，式11能够通过简单排序以复杂度$\mathcal O(n\log(n)+m\log(m))$高效计算。

**形式和特性**

切片 Wasserstein（Sliced Wasserstein, SW）的提出的动因是利用上述计算效率，SW是高维分布在一维投影熵无限多Wasserstein距离的平均值。对任意$\mu,\nu$，其SW距离形式化为：
$$
\mathcal{SOT}_p^p(\mu,\nu)=\mathbb E_{0\sim \text{Unif}(S^{d-1})}\big[\mathcal W_p^p(g_{\theta_\#}\mu,g_{\theta_\#}\nu)\big]\tag{12}
$$

其中，$g_{\theta}(X)$表示由参数$\theta$定义的线性投影映射。$g_\#\theta$表示$u$的pushforward测度，即随机变量$g_\theta(X)$的概率测度，$X$为概率测度$\mu$的随机变量。$\text{Unif}(\mathbb S^{d-1})$表示均匀分布在d维单位球体$\mathbb S^{d-1}$的表面上。由于测度在所选方向上的投影（$g_{\theta_\#\mu}、g_{\theta_\#\nu}$）是一维的，因此式 (12) 中的 $W_p^p$ 可以利用式 (11) 高效地计算。实际上，获得无限多的投影是不可能的，所以使用蒙特卡洛方法进行近似，替代为有限数量$L$随机投影方向上计算的平均值：
$$
\hat{\mathcal{SOT}}_p^p(\mu,\nu)=\frac1L\sum_{l=1}^L\mathcal W_p^p(g_{\theta_l\#}\mu,g_{\theta_l\#}\nu)\tag{13}
$$
**应用**

SW易于计算并且具有相似于Wassertein距离的理论性质。因此其在许多ML应用作为更好的替代。一些例子包括核定义、生成模型、池化、神经文本总和、模型选择、3D点云表示学习和集合、以及域适应[97]。此外，还用减缓诸如Wasserstein barycenter和Gromov Wasserstein代价高昂的OT形式的计算边界。

**局限** 

SW的拓展研究大不如集中于式13相关的两个问题：
- 如何确定更好的投影方向$\theta_l$
- 较之于线性$g_\theta$更好的映射
第一个问题的动机在于：随机选取的 $\theta_l$ 会带来更高的投影复杂度。换句话说，为实现式13很好的近似，需要更大数量的投影$L$。可以考虑一些富含信息的投影来克服该问题而非使用全随机。这种思想下，Max Sliced Wasserstein（Max-SW）支持简单”最佳方向“。对于$p=2$，定义如下：
$$
\max\mathcal{SOT}^2_2(\mu,\nu)=\max_{\theta\in\mathbb S^{-1}}\mathcal W_2^2 (g_{\theta\#}\mu,g_{\theta\#}\nu)\tag{14}
$$
其中最佳方向为在投影测度间产生最大距离的方向。然而，寻找这一最佳方向并非易事。在实践中，通常以使投影后测度均值差异最大的方向来进行近似替代。  分布式切片 Wasserstein（Distributional Sliced Wasserstein，DSW）[99] 指出，仅关注最重要的单一方向（例如 Max-SW）会忽略其他潜在的重要方向。因此，DSW 通过在单位球面上寻找一组“重要方向”的分布，在 SW 与 Max-SW 之间取得了一种折中。

第二个问题认为SW限制集中于线性$g_\theta$。刺激该问题的原因是假说非线性映射在高维背景下更出色。生成式切片Wasserstein（GSW）形式化了该思想。组合具有更高投影复杂度的非线性投影也有可能，最大生成切片Wasserestein（max-GSW）为这种趋势的范例。



#### 2.3.2、结构化OT





## 三、域适应

### 3.1、浅度域适应

### 3.2、深度域适应

### 