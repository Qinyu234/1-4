# 任务清单：Fairy Orbit v1

> 顺序：Tasks → Tests → Code（coding-workflow）

## 验收
- [ ] 全部 pytest 通过
- [ ] `experiments/first_grid_scan.py` 可跑通并写出候选 / 图

## 任务

| ID | 策略 | 任务 | 交付 | Tests |
|----|------|------|------|-------|
| T1 | DRY | Body/System/forces | `physics/` | `tests/test_physics.py` |
| T2 | DRY | Leapfrog | `integrator.py` | `tests/test_integrator.py` |
| T3 | DRY | tetra_rotation | `icgen/tetrahedron.py` | `tests/test_tetrahedron.py` |
| T4 | Curry | 2D near-escape IC | `icgen/generator.py` | `tests/test_generator.py` |
| T5 | Curry | Runner | `simulation/runner.py` | `tests/test_runner.py` |
| T6 | DRY | OrbitEvaluator | `analysis/evaluator.py` | `tests/test_evaluator.py` |
| T7 | Meta | Grid + library | `search/`, `library/` | `tests/test_search_library.py` |
| T8 | Meta | First experiment | `experiments/first_grid_scan.py` | smoke via pytest or script |

## 实现顺序
1. T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8
