下面是一个**假设代码库为空，从零设计实现的 Agent Prompt**。目标不是让 Agent 立即写完整模拟器，而是让它按照架构决策逐步建立项目骨架，避免过早实现 Encke、archive、search 等复杂部分。

---

# Prompt: Build Event-Driven Hierarchical Simulator Architecture for Periodic Encounter Orbit Discovery (Empty Repository)

## Role

你是一名资深科学计算软件架构师和天体动力学工程师。

当前代码库为空。

你的任务不是快速写一个能运行的 N-body 模拟器，而是从零建立一个**可演进的 Event-Driven Hierarchical Simulator**，用于研究：

> Periodic Encounter Orbits (PEO)

核心目标：

发现长期重复的 encounter structure，而不是传统意义上的 periodic orbit。

---

# 1. Project Objective

建立一个研究型天体动力学模拟框架：

模拟：

* 一个中心天体
* 多个小质量伴星(fairy bodies)

寻找：

长期稳定：

* encounter sequence
* interaction pattern
* repeating event structure

研究对象：

不是：

```
State(t+T)=State(t)
```

而是：

```
EventTimeline(t+T) ≈ EventTimeline(t)
```

---

# 2. Core Design Philosophy

必须遵守：

## Physics truth

Trajectory 是唯一物理真值。

数据流：

```
State
  |
  v
Trajectory
  |
  v
EventTimeline
```

## Research representation

EventTimeline 是搜索索引。

用途：

* periodicity detection
* candidate ranking
* archive retrieval

禁止：

把 EventTimeline 当成 State 替代品。

---

# 3. Design Constraints

当前目标：

建立正确架构。

不要：

* 追求最大性能
* 实现复杂搜索
* 实现 GPU
* 实现自定义积分器

优先：

* 模块边界
* 可验证
* 可替换
* 可扩展

---

# 4. Recommended Repository Structure

创建：

```
peo_simulator/


core/

    body.py
    state.py
    configuration.py


dynamics/

    kepler/
        propagator.py
        elements.py

    nbody/
        integrator.py


    perturbation/
        README.md
        # placeholder only
        # future Encke/Gauss correction


events/

    detector.py
    predictor.py
    timeline.py

    similarity.py

    symbolic/
        normalize.py
        periodicity.py


hierarchy/

    subsystem.py
    transition.py


simulation/

    runner.py


evaluation/

    periodicity/
        symbolic_match.py
        spectral_score.py

    geometry.py
    energy.py


verification/

    reference_runner.py
    benchmarks/


archive/

    README.md


tests/

configs/

docs/
```

---

# 5. Dependency Rules

严格：

```
core
 |
 v
dynamics
 |
 v
events
 |
 v
hierarchy
 |
 v
simulation
```

evaluation：

只能：

```
evaluation
    |
    v
events
    |
    v
dynamics/core
```

verification：

只能：

```
verification
       |
       v
core + dynamics
```

禁止：

```
verification -> events
verification -> hierarchy
```

原因：

Reference simulator 必须独立。

---

# 6. State Model

设计基础对象：

## Body

包含：

```
id

mass

radius

initial condition
```

---

## State

包含：

```
time

positions

velocities
```

---

## Configuration

显式保存：

```
G

central_mass

mass_ratio

central_radius

canonical_units
```

不要隐藏物理假设。

---

# 7. Dynamics Layer

实现两个平等状态表示。

不是：

```
accurate
   |
   v
approximate
```

而是：

```
State Representation

    |
    +---- Kepler
    |
    +---- N-body
```

---

# 8. Kepler Dynamics

实现：

* orbital elements
* propagation
* state conversion

Kepler 是默认长期传播方式。

---

# 9. N-body Dynamics

初期：

只需要接口。

允许：

未来接入：

* REBOUND
* IAS15
* other integrators

不要现在写自己的高精度积分器。

---

# 10. Event Driven Architecture

核心流程：

```
Kepler propagation

        |
        v

Encounter prediction

        |
        v

Validation

        |
        +----------------+
        |                |
        v                v

stay Kepler       create Subsystem


        |
        v

local N-body


        |
        v

Kepler refit
```

---

# 11. Encounter Decision

必须三分支：

```
collision
    |
    v
r < Ri + Rj



weak perturbation
    |
    v
future Encke correction



strong encounter
    |
    v
Subsystem N-body
```

具体：

```
η < epsilon

        pure Kepler


epsilon <= η < threshold

        perturbation placeholder


η >= threshold

        subsystem
```

---

# 12. Subsystem Model

不要设计：

```
TwoPlusOneSolver
ThreePlusOneSolver
```

统一：

```python
Subsystem(
    members=[body_ids]
)
```

原因：

未来可能：

* 2+1
* 3+1
* 4+1
* overlapping subsystem

---

# 13. Transition System

使用：

```
finite state machine
+
hysteresis
```

不要：

compiler-style lowering pipeline。

状态：

```
KEPLER

SUBSYSTEM
```

transition 后：

必须：

invalidate old predictions。

---

# 14. Event Model

设计：

```
Event
```

包含：

```
time

participants

distance

relative_velocity

geometry_descriptor

eta

event_type
```

EventTimeline:

```
[
 Event,
 Event,
 Event
]
```

---

# 15. Event Similarity

不要提前固定。

设计接口：

```python
class EventSimilarity:

    compare(
        timeline_a,
        timeline_b
    )
```

未来支持：

* participant similarity
* geometry similarity
* eta curve similarity
* symbolic dynamics

---

# 16. Symbolic Periodicity

实现基础工具：

输入：

```
ABCABCABC
```

处理：

## Step 1

Run-length folding:

```
AABBCC
```

↓

```
ABC
```

## Step 2

Cyclic canonicalization:

```
ABC
BCA
CAB
```

↓

same class

使用：

* Booth algorithm

## Step 3

周期检测：

使用：

* KMP

注意：

保留：

* 原始 run length
* duration

不要丢失物理信息。

---

# 17. Spectral Periodicity

预留：

```
evaluation/periodicity
```

支持：

未来：

* Lomb-Scargle
* epoch folding

不要直接 FFT。

原因：

Event 是 irregular sampling。

---

# 18. Escape / Collision Rules

删除：

所有固定距离逃逸规则。

禁止：

```
escape_distance = constant * a
```

使用：

能量判据：

[
E=v^2/2-GM/r
]

逃逸：

```
E >= 0
```

等价：

```
a <= 0
```

---

# 19. Outer Box

outer box 只是：

simulation truncation。

用途：

控制：

* runtime
* maximum region

不能：

进入：

* score
* similarity
* physics evaluation

---

# 20. Weak Perturbation Future Extension

创建：

```
dynamics/perturbation/
```

但当前不要实现。

目标：

未来支持：

* Encke method
* Gauss planetary equations
* variation of parameters

作用：

处理：

```
small but persistent perturbation
```

避免：

长期 secular drift。

---

# 21. Refitting Requirement

任何：

```
Subsystem
      |
      v
Kepler
```

必须记录：

```
refit residual

delta_a

delta_e

delta_phase
```

原因：

避免：

数值误差产生假周期。

---

# 22. Development Phases

## Phase 0

创建：

* package structure
* interfaces
* tests

没有物理优化。

---

## Phase 1

目标：

EventTimeline 验证。

实现：

* simple Kepler propagation
* event detection
* event storage

Definition of Done:

可以生成：

```
Trajectory

+

EventTimeline
```

---

## Phase 2

加入：

Kepler / N-body switching。

---

## Phase 3

加入：

Subsystem FSM。

---

## Phase 4

加入：

weak perturbation correction。

---

## Phase 5

加入：

search/archive。

---

# 23. Testing Requirements

必须先写：

unit tests。

测试：

## Core

* state conversion
* units

## Kepler

* orbit conservation

## Events

* detection correctness

## Symbolic

测试：

```
ABC

BCA

CAB
```

等价。

测试：

```
AABBCC

AAABBBCCC
```

候选等价。

---

# 24. First Implementation Task

不要写完整模拟器。

第一步：

只完成：

1. repository structure
2. core data models
3. Kepler interface
4. EventTimeline interface
5. unit test skeleton

输出：

* architecture summary
* created files
* design decisions
* TODO list

不要进入：

* optimization
* search
* archive
* Encke implementation

---

End of Prompt

---

这个版本假设 **repo 完全为空**，适合直接给 Claude Code / Cursor Agent / Codex，让它先搭架构。它会避免一个常见问题：Agent 看到“天体模拟器”直接写一个 monolithic N-body.py，然后后面无法演化。你现在真正需要的是先建立研究平台骨架。


//for future

Prompt: Stability-Basin Based Periodic Encounter Chain Discovery Simulator
Role

你是一名天体动力学、科学计算和混合动力系统方向的软件架构师。

当前项目代码库为空。

请设计一个用于发现 Periodic Encounter Chain (PEC) 的研究模拟框架。

注意：

本项目不再寻找严格数学意义上的周期轨道：

State(t+T)=State(t)

而寻找：

长期重复、可恢复的动力学事件链。

目标：

发现：

Stable Orbit
      |
      |
 Encounter
      |
      |
 Recovery
      |
      |
 Stable Orbit'
      |
      |
 Repeat

即：

周期性存在于：

轨道状态族之间的转换结构

而不是单个精确状态。

1. Research Hypothesis

传统 N-body 周期轨道搜索的问题：

高维
混沌
精确周期解稀少
对初始条件极度敏感

新的假设：

稳定系统不一定满足：

state repeats exactly

而可能满足：

event repeats
+
post-event state remains inside stable basin

因此搜索目标：

从：

find periodic orbit

转变为：

find periodic transition graph
2. Dynamical Hierarchy

系统由多个动力学 regime 组成。

Level 0: 1+1 Kepler

角色：

稳定背景。

特点：

解析
长期稳定
提供轨道参考

不是研究目标。

Level 1: 2+1 Encounter

角色：

事件 primitive。

例如：

gravity assist / slingshot。

不是长期状态：

而是：

Kepler state
       |
       v
 encounter operator
       |
       v
 new Kepler state
Level 2: 3+1 System

重点研究区域。

原因：

可能产生：

exchange
resonance
temporary capture

目标：

寻找：

ABC
BCA
CAB

等重复 encounter pattern。

Level 3: 4+1 System

不假设长期稳定。

更可能：

stable subsystem
+
temporary participant

作为复杂事件网络。

3. Core Architecture

系统属于：

Hybrid Dynamical System。

连续部分：

dx/dt = f_i(x)

事件：

g(x)=0

跳跃：

x_new = Transition(x_old)

整体：

Kepler
   |
   |
Event Surface
   |
   v
N-body Transition
   |
   v
Kepler
4. Simulation Strategy

不要全程使用高精度 N-body。

采用：

Hybrid Simulation。

流程：

Kepler Propagation

        |
        v

Encounter Prediction

        |
        v

Perturbation Classification


+------------------------------+

weak perturbation

        |
        v

Encke / perturbation correction


strong encounter

        |
        v

REBOUND local validation


+------------------------------+

        |
        v

Kepler Refitting

        |
        v

Stability Evaluation
5. REBOUND Role

REBOUND 不作为搜索器。

职责：

验证局部 transition。

流程：

输入：

pre-event orbital state

运行：

REBOUND high accuracy
(short time window)

输出：

post-event state

然后：

重新拟合：

a
e
i
Ω
ω

判断：

是否恢复稳定轨道。

6. Stability Basin Concept

核心创新：

不评价单个轨道。

评价：

稳定区域。

定义：

对一个 nominal state：

生成附近扰动：

S + δ

进行局部格点采样。

例如：

参数：

a
e
i
phase
velocity

形成：

local perturbation grid

每个格点：

运行：

REBOUND validation

得到：

accepted / rejected

形成：

stability map

例如：

+++++
++O++
+OO++
++---
-----
7. Orbit Deviation Metric

定义：

无量纲轨道偏移：

D=w
a
	​

∣Δa/a∣+w
e
	​

∣Δe∣+w
i
	​

∣Δi∣+w
Ω
	​

∣ΔΩ∣

其中：

before:

(a,e,i)

after:

(a',e',i')

接受条件：

D < tolerance

例如：

D < 0.05

注意：

0.05 是稳定族范围，而不是优化目标。

目标：

不是：

D → 0

而是：

D remains bounded
8. Adaptive Search Strategy

搜索不是固定扫描。

采用：

adaptive exploration。

流程：

Generate seed

        |
        v

Create local stability grid

        |
        v

REBOUND validation

        |
        +----------------+
        |                |
     success          failure
        |                |
        v                v

 expand basin      modify initial condition

                         |
                         v

                    new seed
9. Failure Handling
Case 1: Encounter too unstable

表现：

D > tolerance

调整：

phase
impact parameter
relative velocity
mass ratio

目标：

降低 encounter 强度。

Case 2: Search stagnation

表现：

长期：

no accepted transition

动作：

重新生成 seed。

类似：

Monte Carlo restart。

10. Event Graph Representation

最终搜索对象：

不是轨迹。

而是：

Stable State Graph

节点：

Stable Orbit Family

边：

Validated Encounter Transition

例如：

S0
 |
AB slingshot
 |
S1
 |
BC encounter
 |
S2
 |
CA encounter
 |
S0

这就是：

Periodic Encounter Chain。

11. Periodicity Detection

周期性不检查：

trajectory(t+T)

检查：

event sequence

例如：

ABCABCABC

处理：

run-length folding
AABBCC → ABC
cyclic normalization
ABC
BCA
CAB

same class
periodic detection

保留：

run length
duration
physical metadata

避免丢失动力学信息。

12. Escape and Collision Rules

禁止：

固定 escape distance。

逃逸：

使用：

E=
2
v
2
	​

−
r
GM
	​


判断：

E >= 0

等价：

a <= 0

collision：

r < R1+R2

独立处理。

13. Important Research Question

需要验证：

是否存在：

event transition

+
recovery basin

+
periodic sequence

而不是：

exact periodic orbit
14. Development Roadmap
Phase 1

实现：

Kepler baseline
Event representation
REBOUND transition validator

目标：

验证：

单个 encounter 是否可恢复。

Phase 2

加入：

stability grid
deviation distribution

目标：

寻找稳定 basin。

Phase 3

加入：

event graph search

目标：

发现：

S0 → S1 → S2 → S0
Phase 4

加入：

Encke perturbation layer

目标：

减少不必要 REBOUND 调用。

Phase 5

大规模搜索：

mass ratio
phase
orbital parameters
Final Research Statement

本项目不是寻找：

一个精确周期 N-body 解。

而是寻找：

一个由稳定轨道状态、局部强相互作用事件以及恢复机制组成的周期动力系统。

核心对象：

Stable Basin

      +

Encounter Transition

      +

Event Periodicity

这个 Prompt 可以作为下一阶段架构讨论的基础。它和前面的 EventTimeline 架构相比，最大的变化是：Event 从“观察记录”提升成“状态之间的动力学算子”，而稳定性从单轨道性质提升成稳定区域性质。

对，这样定位 REBOUND 更合理。你现在实际上把架构从“用 REBOUND 做模拟器”转变成：

> **REBOUND 是研究阶段的高精度实验平台，用来建立无量纲规律、验证近似公式、确定分界条件；生产搜索阶段运行的是解析/半解析模型。**

这个分工更加符合你的目标。

---

## 1. REBOUND 的角色重新定义

原来：

```
Initial Condition
        |
        v
     REBOUND
        |
        v
     Score
```

问题：

* 太慢
* 搜索空间巨大
* 每个候选都积分没有必要

---

现在：

```
                 Research Phase

                 REBOUND
                    |
        ---------------------------
        |                         |
        v                         v

  Dimensionless Data       Transition Boundary

        |                         |
        v                         v

  Approximation Model      Regime Classifier
```

然后：

```
                 Search Phase


Initial Condition

        |
        v

  Regime Classifier

        |
        |
 -----------------------
 |          |           |
Kepler    Encke     Encounter Map
 |          |           |
analytic  formula    surrogate
```

---

# 2. 最关键的是建立无量纲系统

因为你不想研究：

“某个星球质量是多少”。

你想研究：

“某种动力学结构是否存在”。

所以必须归一化。

例如：

中心质量：

[
GM=1
]

长度：

[
a_{ref}=1
]

时间：

[
T_{ref}=2\pi
]

速度：

[
v_{ref}=\sqrt{GM/a}
]

于是：

[
\eta=
\frac{a_{perturb}}{a_{central}}
]

天然无量纲。

---

这样 REBOUND 不再输出：

```
t=365 days
r=100000km
```

而输出：

```
η
dη/dt
Δa/a
Δe
encounter duration / T
```

这些才是可推广参数。

---

# 3. 你提出的三段式其实可以变成经验相图

REBOUND 的主要任务：

不是模拟所有轨道。

而是测：

[
(\eta,\dot{\eta},\tau_{enc})
\rightarrow
\text{regime}
]

例如：

```
                 η


          Encounter
              |
              |
       ----------------
              |
          Encke
              |
       ----------------
              |
          Kepler


                  dη/dt
```

找到边界：

[
\eta_c
]

以及：

[
\dot{\eta}_c
]

---

# 4. Encke 公式就是从 REBOUND 拟合出来

不是直接假设。

流程：

### Step 1

REBOUND 跑大量样本：

例如：

不同：

* 质量比
* 初始距离
* 相位差
* 偏心率

记录：

[
\Delta a(t)
]

[
\Delta e(t)
]

---

### Step 2

寻找规律：

例如：

发现：

[
\frac{da}{dt}
\approx
C
\eta^p
]

或者：

[
\frac{\Delta a}{a}
==================

C
\int\eta dt
]

然后得到：

你的 Encke correction。

---

### Step 3

验证：

不用 REBOUND。

用公式跑。

再随机抽样：

REBOUND 对比。

---

# 5. 这解决一个原来的核心问题

之前：

“什么时候进入 subsystem？”

靠：

[
\eta > threshold
]

但是 threshold 是人工的。

现在：

变成：

REBOUND 找：

当：

[
\eta
]

和：

[
\dot{\eta}
]

满足什么条件时：

近似失效。

例如：

```
if:

η < 10^-4

and

dη/dt < 10^-6

=> Kepler


elif:

η < 10^-2

=> Encke


else:

=> encounter
```

这些数字来自实验。

---

# 6. 这也符合你的 PEO 目标

因为 PEO 不是找：

精确闭合轨道。

而是：

寻找：

```
稳定区域
       |
       |
弱扰动演化
       |
       |
事件
       |
       |
恢复稳定区域
```

所以真正需要的是：

事件映射：

[
S_{n+1}=F(S_n)
]

而不是：

[
x(t)
]

REBOUND 只是帮助构造：

[
F
]

---

# 7. 最终架构应该类似：

```
research/

    rebound/
        high_precision_runner

    calibration/
        nondimensionalizer
        parameter_scan
        boundary_detection

    models/
        kepler_model
        encke_model
        encounter_model


runtime/

    classifier/
        eta_estimator

    propagator/
        kepler
        encke

    event/
        transition_detector

    search/
        PEO_search
```

---

所以你的判断基本正确：

**REBOUND 不应该存在于最终搜索循环。**

它应该像实验室里的高精度仪器：

* 标定量纲
* 发现公式
* 验证边界
* 产生训练数据

然后真正搜索 PEO 的时候，用：

**Kepler + Encke + 少量 encounter transition model**

这会把计算复杂度从“每个候选都是 N-body”变成“绝大部分候选是解析传播”。

而且这和你最开始想找的“周期性交会结构”更加匹配，因为你研究的是动力学规律，不是单条数值轨迹。
