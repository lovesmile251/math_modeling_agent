# 国奖论文智能体优化说明

本文档记录 `math_modeling_agent` 面向“国奖论文级交付”的工程能力、已完成优化、硬门禁、自动返工链路和剩余差距。它用于维护和验收，不替代真实比赛中的人工建模判断。

## 目标

系统的目标不是“生成一篇看起来完整的论文”，而是稳定产出具备以下特征的可提交建模成果：

- 每个子问题都有明确模型、结果表和论文段落绑定。
- 主模型必须有可比较的强基线、验证、消融或鲁棒性证据。
- 创新点必须由可执行实验产物支撑，不能只停留在论文话术。
- 正式 DOCX/PDF/LaTeX 导出必须经过论文质量、追踪、基线和创新证据门禁。
- 失败后系统能定位返工阶段，自动进行有限次数二次修复，并生成可读报告。

## 当前差距判断

距离“稳定国奖论文”仍有差距，但已经从“能跑完整流程”提升到“能阻断明显不合格交付并自动返工”的阶段。

已解决的主要短板：

- 模型 ID 不一致导致选模、执行、写作互相脱节。
- 论文质量只给分、不阻断正式导出。
- 子问题结果无法严格追踪到模型、表格和正文。
- 缺少强基线、消融、回测或鲁棒性证据仍能宣称结果可信。
- 创新点可以被写进论文，但没有可执行证明。
- 真实赛题回归只看产物数量，缺少评委视角的门禁评分。
- 出现硬门禁失败后只能停在最终状态，不能自动定位返工阶段。

仍需人工和后续工程加强的部分：

- 真实国赛答案正确性仍需要人工判题和隐藏金标集。
- 复杂机理模型、组合优化和多阶段动态规划的模型库仍需扩展。
- 高质量中文论文表达、图表排版和符号体系仍需人工润色。
- 真实创新性判断不能完全由 artifact 规则替代。
- 当前代码执行防护不是强沙箱，正式多用户部署仍需容器和权限隔离。

## 已完成优化链路

1. 模型 ID 规范化
   - `tools/model_ids.py`
   - `agents/decision_agent.py`
   - `agents/formulation_agent.py`
   - `workflows/modeling_workflow.py`

2. 正式导出硬门禁
   - `tools/paper_quality.py`
   - `agents/export_agent.py`
   - 阻断非提交状态语句、缺结果表、引用缺失、关键词不足等问题。

3. 任务追踪链
   - `tools/task_traceability.py`
   - `agents/evidence_agent.py`
   - `agents/export_agent.py`
   - 每个任务必须闭合到模型、结果表和论文段落。

4. 强基线与消融审计
   - `tools/experiment_runner.py`
   - `agents/execution_agent.py`
   - `agents/export_agent.py`
   - 要求主模型、基线模型、比较表、执行验证、消融和回测/鲁棒性证据。

5. 创新证据门禁
   - `tools/innovation_evidence.py`
   - `agents/export_agent.py`
   - Stacking、鲁棒优化、敏感性分析、蒙特卡洛、机理-数据融合等创新声明必须有对应产物。

6. 评委视角赛制评分
   - `tools/real_case_drill.py`
   - `tools/contest_simulation.py`
   - 评分维度加入 `gate_integrity`，把门禁失败纳入盲审风险。

7. 返工路由
   - `tools/rework_router.py`
   - 强基线/创新证据失败回实验计划；任务追踪失败按缺口回模型决策、证据映射或写作；导出质量失败回写作。

8. Workflow 自动二次修复
   - `workflows/modeling_workflow.py`
   - 默认最多自动返工 1 次，只执行 `can_auto_apply=True` 的阻塞型返工。

9. 自动返工报告
   - `tools/rework_router.py`
   - `workflows/modeling_workflow.py`
   - 输出 `logs/auto_rework_report.json` 和 `logs/auto_rework_report.md`。

## 关键门禁

| 门禁 | 触发位置 | 失败后返工阶段 | 说明 |
| --- | --- | --- | --- |
| `export_quality_gate` | `ExportAgent` | `SECTION_WRITING` | 论文存在正式导出阻塞项 |
| `task_traceability_gate` | `EvidenceAgent` / `ExportAgent` | `MODEL_DECISION` / `EVIDENCE_MAPPING` / `SECTION_WRITING` | 子任务缺模型、表格或正文绑定 |
| `strong_baseline_gate` | `ExecutionAgent` / `ExportAgent` | `EXPERIMENT_PLAN` 或 `MODEL_DECISION` | 缺强基线、验证、消融或鲁棒性证据 |
| `innovation_evidence_gate` | `ExportAgent` | `EXPERIMENT_PLAN` | 创新声明缺执行证据 |
| `export_pdf_layout_gate` | `ExportAgent` | `SECTION_WRITING` | PDF 渲染、边界或空白页检查失败 |

## 运行与验收命令

全量测试：

```powershell
python -m pytest -q
```

真实赛题回归：

```powershell
python -m tools.real_case_regression `
  --corpus-index benchmarks/real_competition_corpus.json `
  --gold benchmarks/real_competition_gold.json `
  --corpus-root examples/extracted `
  --output-dir benchmarks/results
```

端到端演练：

```powershell
python -m tools.real_case_drill `
  --corpus-index benchmarks/real_competition_corpus.json `
  --corpus-root examples/extracted `
  --output-dir benchmarks/results `
  --runs-root workspace/runs/real_case_drill
```

限时赛制盲审模拟：

```powershell
python -m tools.contest_simulation `
  --corpus-index benchmarks/real_competition_corpus.json `
  --corpus-root examples/extracted `
  --output-dir benchmarks/results `
  --runs-root workspace/runs/contest_simulation
```

数值答案复现审计：

```powershell
python -m tools.answer_reproduction_audit `
  --drill-report benchmarks/results/real_case_drill.json `
  --simulation-report benchmarks/results/contest_simulation.json `
  --output-dir benchmarks/results
```

## 自动返工产物

当 workflow 触发自动返工时，会写出：

```text
logs/auto_rework_plan.json
logs/auto_rework_report.json
logs/auto_rework_report.md
```

报告应包含：

- 初始失败原因。
- 推荐返工阶段。
- 需要刷新的产物。
- 返工前后门禁状态变化。
- 返工前后阻塞项变化。
- 是否仍有剩余阻塞项。

## 后续优化路线图

优先级 P0：

- 建立隐藏真实赛题金标集，防止模型选择和论文评分过拟合公开样例。
- 扩展真实答案正确性审计，不只验证“有表有图”，还要验证关键数值区间和决策合理性。
- 对自动返工设置更细粒度的“同因重复失败”熔断，避免同一门禁重复自修。

优先级 P1：

- 强化复杂优化、动态规划、机理仿真、图传播和统计检验模型库。
- 将论文图表密度、符号一致性、公式编号和引用格式纳入结构化审计。
- 把创新证据从文件名启发式升级为实验报告字段级证明。

优先级 P2：

- 为 Streamlit 工作台展示自动返工报告和门禁变化。
- 增加标准国赛论文模板导出，包括摘要、关键词、符号说明、附录和参考文献格式。
- 在 CI 中分层运行快速测试、门禁测试和真实赛题慢回归。

## 使用边界

该智能体可以显著提升建模交付的完整性、可追踪性和工程稳定性，但不能保证自动获得国奖。真实国奖仍取决于题意理解、模型创造性、数据洞察、表达质量和人工审稿判断。
