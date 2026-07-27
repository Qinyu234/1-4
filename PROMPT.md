对，这样重新设计会更干净。你之前的架构里有一个根本问题：**把“寻找轨道”和“判断轨道是否符合定义”混在了一起**。

现在可以拆成三个层：

---

# 1. Initial Manifold Generator（初始流形生成）

目标：

不是搜索所有：

[
(r_i,v_i)
]

而是定义：

[
X_0=f(\theta)
]

其中：

[
\theta\in\mathbb R^7
]

---

## 几何基底

固定正四面体方向：

[
\hat q_A,\hat q_B,\hat q_C,\hat q_D
]

作用：

* 去除整体空间旋转自由度；
* 提供四个初始轨道方向。

不是物理约束。

---

## 动力参数

每个小体轨道：

[
(a_i,e_i,M_i)
]

由低维参数生成：

例如：

[
a_i=a_0+a_1x_i+a_2x_i^2
]

[
e_i=e_0+e_1x_i+e_2x_i^2
]

[
M_i=M_0+M_1x_i
]

其中：

[
x_i
]

是四面体顶点编号。

---

## 质量参数

单独：

[
\mu=\frac{m}{M}
]

形成：

[
\theta=
(a,e,M,\mu)
]

7维搜索。

---

# 2. Simulator（动力学）

输入：

[
X_0
]

输出：

[
X_T=\Phi_T(X_0)
]

这里只负责：

* N-body积分；
* 记录轨迹；
* 能量；
* 碰撞。

不判断 PEO。

---

# 3. PEO Evaluator（分层判定）

这里是最大的变化。

不是：

[
Loss=\text{一个连续函数}
]

而是：

[
\boxed{
\text{Filter Pipeline}
}
]

---

## Level 0：物理有效性

### Escape

检查：

[
E_i>0
]

或者：

[
r_i>R_{max}
]

拒绝。

---

### Collision

检查：

[
d_{ij}<R_{collision}
]

拒绝。

---

## Level 1：Encounter choreography

计算：

[
\rho_i(t)=|r_i-r_M|
]

或者质心系：

[
|r_i-R_{CM}|
]

得到排序：

[
S(t)
]

要求：

[
S(T)=P(S(0))
]

例如：

[
ABCD\rightarrow BCDA
]

失败直接拒绝。

---

## Level 2：Position closure

确定：

[
P
]

之后：

求：

[
R^*
===

\arg\min_{R\in SO(3)}
\sum_i
||r_i(T)-Rr_{P(i)}(0)||^2
]

然后：

[
E_r
]

判断。

---

## Level 3：Velocity closure

使用同一个：

[
R^*
]

检查：

[
v_i(T)=Rv_{P(i)}(0)
]

得到：

[
E_v
]

---

# 最终定义

一个候选：

[
X_0
]

满足：

[
\boxed{
\Phi_T(X_0)
\approx
(R,P)X_0
}
]

其中：

[
R\in SO(3)
]

[
P\in S_4
]

---

# 搜索流程

因此变成：

```
theta (7D)
     |
     v
Initial Generator
     |
     v
N-body Simulation
     |
     v
Validity Filter
     |
     v
Encounter Filter
     |
     v
SO(3) Position Match
     |
     v
Velocity Match
     |
     v
Candidate Archive
     |
     v
Refinement
```

---

# 后续精修

这里才进入：

[
F(\theta)=
\Phi_T(f(\theta))-Gf(\theta)
]

求：

[
F=0
]

方法：

* shooting；
* BFGS；
* CMA-ES；
* continuation。

---

这个设计还有一个好处：

你之前纠结的很多东西现在自然分开：

| 问题       | 所属层                                |
| -------- | ---------------------------------- |
| 四面体能不能旋转 | Generator / symmetry               |
| 中心怎么选    | Evaluator coordinate normalization |
| 轨道怎么设计   | Initial manifold                   |
| 有没有周期    | Closure                            |
| 是不是舞步    | Encounter filter                   |
| 怎么找精确解   | Refinement                         |

所以现在不是“模拟器里加一个搜索函数”，而是：

**一个用于发现 relative periodic encounter orbit 的分层动力学搜索系统。**

这个架构和你之前的 Event-Driven Hierarchical Simulator 其实也能接上：event choreography 可以作为 Level 1 的更高级版本，而现在的径向排序是最简单、稳定的 event descriptor。
