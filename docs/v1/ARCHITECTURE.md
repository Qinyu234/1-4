# 架构拆分：Fairy Orbit v1

## 抽象拆分
| ID | 抽象 | 输入 | 输出 | 依赖 | 策略 |
|----|------|------|------|------|------|
| A1 | Body/System/gravity | masses, r, v, G | accelerations, E, L | — | DRY |
| A2 | Leapfrog | System, dt | updated System | A1 | DRY |
| A3 | tetra_rotation + IC | v_rad, v_tan, R, masses | System | A1 | DRY/Curry |
| A4 | SimulationRunner | System, dt, t_end | trajectory | A2 | Curry |
| A5 | OrbitEvaluator | trajectory | score, metrics | — | DRY |
| A6 | Grid search | param grids | ranked candidates | A3–A5 | Meta |
| A7 | Library store | candidate dict | JSON files | — | DRY |
| A8 | Visualization | trajectory / scan | plots | — | Meta |

## 目录（按领域组织）

| 模块/目录 | 职责 |
|-----------|------|
| `fairy_orbit/physics/` | Body, gravity, Leapfrog |
| `fairy_orbit/icgen/` | tetrahedron, generator |
| `fairy_orbit/simulation/` | runner |
| `fairy_orbit/analysis/` | evaluator |
| `fairy_orbit/search/` | grid, optimize stub |
| `fairy_orbit/library/` | JSON store |
| `fairy_orbit/visualization/` | plots |
| `experiments/` | first_grid_scan |
| `tests/` | pytest |

## 策略标注

| 抽象 ID | 策略 | 理由 |
|---------|------|------|
| A1–A2 | DRY | 无状态物理原语 |
| A3 | Curry | 参数→旋转矩阵链→System |
| A4 | Curry | 构造积分上下文再求值 |
| A5 | DRY | 纯函数评分 |
| A6–A8 | Meta | 批量枚举、落盘、出图 |
