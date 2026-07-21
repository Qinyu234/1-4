# Architecture Review: Event-Driven Hierarchical Simulator for PEO Discovery

## 0. 总体判断

这个方向是对的，而且不是新发明——它本质上是**混合辛积分器（hybrid symplectic integrator）**思路在"轨道搜索"场景下的再表达。REBOUND 的 `mercurius`、Chambers 的 MERCURY、行星子星盘模拟里常见的"drift-close encounter-drift"结构，都是同一个物理直觉的软件实现：大部分时间是解耦的 Kepler 漂移，只有近距离交会才值得上贵的直接积分。

你现在做的新东西，是把这个"省算力的工程技巧"**提升为研究对象本身**——不是为了积分更快，而是因为你怀疑：真正周期性重复的东西，是交会结构（event 序列），而不是笛卡尔状态。这个转变是合理的，但也带来一个这份文档目前还没讲清楚的核心风险，我放在第 5 节重点谈。

---

## 1. 关键问题：EventTimeline 应该是比 State 更高层的对象吗？

答案是：**数据依赖方向和"研究优先级方向"应该分开看，这是两条不同的箭头。**

- **数据流方向（生成关系）**：`State → Trajectory → Event`。Event 永远是从连续状态推导出来的，不能反过来。Trajectory（不管是纯 Kepler 段还是局部 N-body 段拼起来的）是唯一的物理 ground truth，任何 Event 都必须能追溯到产生它的那一段状态。
- **使用/研究方向（查询优先级）**：`Event 优先于 Trajectory`。当你做周期性判定、做 archive 检索、做候选轨道去重时，你应该先在 Event 序列这一层做比较，只有需要复核物理有效性时才回落到 Trajectory。

所以准确的说法不是"EventTimeline 取代 State 成为核心"，而是：

> **Trajectory 是生成层的真值（ground truth），EventTimeline 是研究层的索引（index）。二者不是替代关系，是"存储 vs 检索"的关系。**

这个区分很重要，因为如果把 EventTimeline 当成唯一真值，你会丢失复核能力（比如发现"周期性"其实是离散化伪影时，没法回去看原始轨迹）；如果把 Trajectory 当成唯一真值又不给 Event 一等公民地位，你会退化回"周期性只能在连续状态上算"的老范式，白做了这次架构转型。

---

## 2. 这份设计文档目前最大的一个缺口

你在第 12 节问"周期性应该在 trajectory 上算还是 event 上算"，但整份文档没有回答一个更前置的问题：

> **"Event(t+T) ≈ Event(t)" 里的这个 "≈" 具体是什么？**

这不是细节问题，是整个架构能不能成立的关键。可能的定义至少有四种，选哪种会导致完全不同的实现：

| 相似性定义 | 含义 | 风险 |
|---|---|---|
| 参与者集合相同 | 同样几个天体发生交会 | 太粗，容易把物理上完全不同的交会误判为"重复" |
| 交会几何类型相同 | 同样的接近距离量级、相对速度方向 | 需要定义几何分箱（binning），分箱粒度本身就是自由参数 |
| η 曲线形状相似 | 扰动强度随时间的轮廓相近 | 计算量大，但物理意义最强 |
| 子系统跃迁序列相同（符号动力学） | 类似三体问题里对周期轨道做符号分类（如 Broucke 的工作） | 优雅，但符号表的粒度选择依然是隐藏参数 |

**建议**：不要在架构里"预先决定"用哪种相似性度量，而是把它做成一个显式的、可替换的 `EventSimilarity` 策略接口，在 Stage 1（只加事件记录、不改变力学）阶段，用你已有的已知混沌三体/四体测试用例，对这四种定义做敏感性扫描——看你关心的"周期性结论"在哪种定义下稳定、哪种下一致性差。这比现在就把 EventTimeline 焊死成某一种数据结构更重要。

这也直接回答你在第 4 节问的"trajectory 是否只是验证artifact"：不完全是——trajectory 还承担着"当相似性度量选错时的复核数据源"这个角色，不能被丢弃或降级为纯粹的可选项。

---

## 3. 逐条设计判断

### 3.1 State-driven vs Time-driven（第3节）
判断：**方向正确，但"predict next event → jump directly"这一步有个隐藏假设**——即预测本身是廉价且可靠的。对于三体以上的近距交会，"下一次交会何时发生"的预测本身在临界区域（η 接近阈值时）是病态的（ill-conditioned），预测误差会随时间指数放大。建议不要把"预测"做成一次性的解析跳跃，而是做成"粗预测 + 细化"两步：先用解析近似给出一个时间窗口，再在窗口内用低成本采样确认，而不是直接跳到预测时刻——否则会在阈值附近反复"跳过头又跳回来"。

### 3.2 Reference vs Research Simulator（第4节）
判断：**分离正确，而且应该更彻底**——Reference Simulator（REBOUND）不应该依赖 events/、hierarchy/ 任何模块，只依赖 core/ 和 dynamics/。这样它才能始终作为独立的、不被你自己架构里的 bug 污染的真值来源。第13节的模块边界目前没有强制这条依赖规则，建议加上。

### 3.3 Kepler 作为默认状态（第6节）
优势成立。但"累积相位误差"不只是"potential issue"，它是**必须有监控机制的一等公民问题**：建议每个 Subsystem 折叠回 Kepler 时，强制做一次"refit 残差"记录（refit 前后椭圆根数的差异），这个残差本身应该进入 Event/统计流，而不是被丢弃——因为如果某条候选轨道的"周期性"恰好靠误差累积凑出来的，你需要能在事后查出来。Kepler 表示应该活在 `dynamics/kepler` 里，作为一个**状态层（state representation）**而不是"近似层"——因为它在架构里和 N-body 状态是平权的两种表示，不是"精确 vs 近似"的从属关系，这样后面替换积分器时接口更干净。

### 3.4 Prediction/Validation 分离（第7节）
判断正确，且这个分离本身就是防止"过早优化"的好设计——η 阈值可以在不改动预测逻辑的情况下单独调参。风险点：**pairwise 预测无法捕捉"三体同时近距"的情况**（A-B、A-C、B-C 各自都低于阈值，但 A、B、C 同时挤在一起时联合扰动超过阈值）。建议 validation 阶段除了 pairwise η，再加一个"局部密度/联合扰动"检查，否则会系统性漏掉某一类交会。

### 3.5 Subsystem 而非 TwoPlusOneSolver（第8节）
这是文档里最好的一个决定，明确支持。`Subsystem(members=[...])` 用显式列表就够了，**不需要一开始做成隐式/涌现的图结构**——等你真的遇到"子系统需要合并/重叠"这种显式列表表达不了的情况时再重构，现在做是过度设计。

### 3.6 状态机 vs 编译器式 lowering pipeline（第9节）
判断：**用简单的显式 FSM + 滞回（hysteresis）就够，不要做"compiler-like lowering pipeline"**。lowering pipeline 这个隐喻意味着"多阶段、可组合、可插入新pass"，但你目前只有一种跃迁逻辑（基于η的扩张/收缩），没有第二个需要复用的 pipeline 场景。这是本文档里第二明显的过度设计信号（第一个是原来的 TwoPlusOneSolver/ThreePlusOneSolver，已经被你自己纠正了）。

### 3.7 跃迁稳定性（第10节）
滞回思路正确。补充一点：跃迁不应该只由触发跃迁的那一对天体的η决定——**子系统成员变化后，必须让所有涉及旧成员的"待验证预测"失效并重新计算**，否则会出现"用过期预测触发的跃迁"这种难以复现的 bug（这是软件风险，不是物理风险，但很容易在调试时把两者混淆）。

---

## 4. 模块边界修订建议（对第13节）

```
core/            body, state, configuration        [无变化]

dynamics/        kepler, nbody                      [无变化，两者应平权]

events/
    predictor
    detector
    timeline
    similarity       <-- 新增：显式的 Event 相似性策略，见第2节

hierarchy/       subsystem, transition              [无变化]

simulation/      runner                             [无变化]

evaluation/      geometry, periodicity, energy_stability, encounter_pattern
                                                     <-- 明确列出评估器，依赖 events/ + core/，不依赖 hierarchy/

verification/    benchmark
                 <-- 依赖规则：只能依赖 core/ + dynamics/，禁止依赖 events/ hierarchy/

archive/         candidate_store, refit_residual_log
                                                     <-- 新增，但 Stage 1-2 阶段先留空/占位，不要提前实现完整schema
```

依赖方向：`core → dynamics → events → hierarchy → simulation`；`evaluation` 和 `verification` 都只向左依赖，不向右依赖，二者互相之间也不应该有依赖。

---

## 5. 过度设计 / 应推迟的部分（回答第15节问题4、5）

**现在就该砍掉或推迟的：**
- `events/` 里现在没有但很容易被诱惑先做的"通用事件相似性框架"——先用最简单的"参与者集合+η峰值"两个字段做 baseline，等 Stage 1 的敏感性扫描结果出来后再决定要不要上更复杂的度量。
- Section 9 的 lowering pipeline 隐喻——用显式 FSM。
- Section 11 的完整 archive schema（trajectory/parameters/event序列/稳定性指标全部存）——在你不知道"周期性判定"这件事本身是否可靠之前，存什么字段都是猜的。先只存 event 序列 + 触发它的随机种子/初始条件，能复现就够，其余字段等 Stage 2 有信号后再加。
- Stage 4 的自定义子系统积分器——REBOUND 的 IAS15 已经是成熟的近距交会积分器，没有证据表明它是瓶颈之前不要重新发明。

**应该保留、且是本次架构真正新东西的：**
- Event 作为一等对象（但明确它是"索引"不是"替代真值"）
- Subsystem 的显式列表表示
- Kepler/N-body 的平权状态表示

---

## 6. 演进路线评估（对第14节）

Stage 1→5 的顺序是合理的，风险从低到高排列正确。唯一建议的调整：**把"Event相似性度量的敏感性扫描"明确插入 Stage 1 的验收标准里**，作为进入 Stage 2 的门槛（gate），而不是让它隐含在"验证事件表示是否有意义"这句模糊的话里。也就是说 Stage 1 的 Definition of Done 应该是：

> 对至少 3-4 个已知混沌测试用例，在至少 3 种不同 η 阈值 / 相似性定义下，事件序列的"看起来周期"的结论是否稳定。如果结论随参数剧烈翻转，说明需要先解决第2节的问题，再进入 Stage 2。

---

## 7. 风险清单（对第15节问题6、7）

**科学风险**
1. **符号化伪周期（symbolic aliasing）**——最大风险，见第2节。
2. Kepler 折叠相位误差在长时间尺度上悄悄累积，导致真实存在的共振被"折叠掉"（假阴性）——需要第3.3节提到的 refit 残差监控。
3. Pairwise 预测漏掉三体同时近距——见3.4节。

**软件风险**
1. 混合模拟器产出的 trajectory 是异构的（Kepler 段 + N-body 段拼接），下游 evaluator/绘图/与 REBOUND 对比都需要统一的重采样/插值策略，否则"验证"这一步本身会引入新的数值误差来源，污染判断。
2. 跃迁触发后旧预测未失效导致的调试困难（见3.7节）。
3. 一旦 archive/search（Stage 5）在 Stage 1-2 结论还不稳固时提前实现，会产生大量基于不可靠周期性判据的"候选轨道"，浪费后续精修算力——这是纯工程风险但代价是研究时间。

---

## 结论

架构方向本质正确，且已经做对了最关键的两件事:把 Subsystem 做成统一抽象、把 Reference/Research 模拟器分离。真正决定这次转型成不成立的,不是代码结构,而是第2节那个还没回答的问题——"事件相似性"到底怎么定义。建议把 Stage 1 的目标从"验证事件记录是否有意义"收紧为"对相似性度量做敏感性分析",这是这份文档里唯一一个会直接决定后面所有工作是否白做的判断点。